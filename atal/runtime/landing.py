"""Getting a drone off the ground before its search and safely back down after it. These
steps are the run's own (direct simulator calls and setup commands); they do not go through
the planner or the action port."""
import asyncio
import time

from atal.core.telemetry import GroundState, ValidationStatus
from atal.flight.shutdown import safe_shutdown
from atal.runtime import rates


async def takeoff(adapter, vid, recorder, timeout_s):
    recorder.record("setup", step="takeoff", status="started", vehicle_id=vid,
                    note="direct simulator calls, not through the planner or port")
    d = adapter.drone(vid)
    d.enable_api_control()
    d.arm()
    await (await d.takeoff_async())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        s = adapter.get_snapshot(vid)
        if s.validation_status == ValidationStatus.VALID and s.ground_state == GroundState.AIRBORNE:
            recorder.record("setup", step="takeoff", status="airborne", vehicle_id=vid)
            return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"{vid} not airborne after takeoff")


async def descend_before_landing(ctx, drone) -> str:
    """Come down to just above the launch height, with no transition manager (one-drone
    runs). Returns the command's status.

    One retry, only when the descent stopped because telemetry was invalid mid-command
    (reason_code telemetry_lost, e.g. the pose topic and kinematics briefly disagreeing;
    the executor has already hovered). Any other failure is not retried: the shutdown
    then lands from wherever the drone is. Team runs with the managed airspace descend
    under their landing reservation instead (descend_reserved)."""
    vid = drone.vehicle_id
    target = round(drone.launch_altitude_m + rates.DESCEND_ABOVE_LAUNCH_M, 2)
    for attempt in (1, 2):
        r = await ctx.flight.change_altitude(vid, target, reason="setup: descend before landing")
        ctx.recorder.record("setup", step="descend", vehicle_id=vid, status=r.status.value,
                            reason_code=r.reason_code, attempt=attempt)
        if attempt == 1 and r.reason_code == "telemetry_lost":
            ctx.log(f"  {vid} descent stopped on invalid telemetry; retrying once")
            await asyncio.sleep(ctx.timing.poll(rates.DESCEND_RETRY_WAIT_S))
            continue
        return r.status.value
    return r.status.value


async def descend_reserved(ctx, drone) -> str:
    """Team runs with the managed airspace: the descent under the ACTIVE landing
    reservation (Recovery.landing_descent: inside its validity time, one same-reservation
    retry after lost telemetry, otherwise STUCK and a fresh grant by the cleanup
    deadline). "succeeded", or "stuck" / "not_cleared" (then the drone does not land)."""
    out = await drone.recovery.landing_descent(
        round(drone.launch_altitude_m + rates.DESCEND_ABOVE_LAUNCH_M, 2),
        ground_m=drone.launch_altitude_m, deadline_s=drone.cleanup.start())
    status = "succeeded" if out == "descended" else out
    ctx.recorder.record("setup", step="descend", vehicle_id=drone.vehicle_id, status=status,
                        reserved=True)
    return status


def land_timeout_s(ctx, drone) -> float:
    """Wall seconds allowed for the land command.

    Under a landing reservation: what is left of its validity time, and nothing more. A
    simulator timeout never extends the authorisation; if the land cannot finish inside
    it, the drone is not confirmed down (UNRESOLVED) and the reservation's keep-out keeps
    its column protected. Without a manager: the flat floor, or longer when the drone is
    still high (a failed descent leaves it at mission altitude, e.g. 160 m, which a flat
    120 s cannot cover). Unknown height: the floor."""
    rec = getattr(drone, "recovery", None)
    left = rec.landing_validity_left_s() if rec is not None else None
    if left is not None:
        return ctx.timing.poll(left)
    snap = ctx.adapter.get_snapshot(drone.vehicle_id)
    if snap is None or snap.validation_status != ValidationStatus.VALID:
        return rates.LAND_TIMEOUT_MIN_S
    height = max(0.0, snap.altitude_local_m - drone.launch_altitude_m)
    return max(rates.LAND_TIMEOUT_MIN_S,
               height / rates.LAND_SPEED_FLOOR_MPS + rates.LAND_TIMEOUT_MARGIN_S)


async def shut_down(ctx, drone, landing_cleared=None, on_touchdown=None):
    """Hover, land, and disarm only once the drone is confirmed on the ground. With the
    managed airspace, an airborne drone lands only with landing_cleared (its landing
    reservation is ACTIVE); without it, it neither lands nor disarms (UNRESOLVED)."""
    vid = drone.vehicle_id
    report = await safe_shutdown(ctx.adapter.drone(vid), adapter=ctx.adapter, vehicle_id=vid,
                                 log=lambda s: ctx.log(f"[{vid}] {s}"),
                                 confirm_poll_s=ctx.timing.poll(0.1),
                                 land_timeout_s=land_timeout_s(ctx, drone),
                                 landing_cleared=landing_cleared, on_touchdown=on_touchdown)
    ctx.recorder.record("shutdown", vehicle_id=vid, report=report)
    return report


async def clear_pads(ctx, pads, team_size: int, drone, res: dict):
    """Before any descent: hold the pad zone, or be outside it. A drone inside
    that cannot get the zone flies out to its edge; a drone that lands without
    the zone is a recorded contingency landing (it fails acceptance, except a dropped
    drone landing where it failed: "not_home").

    The reason is what happened, not who held the zone: "pad_zone_timeout" only when a
    wait for the zone actually timed out (here, or on the way home: the planner's
    return_home); otherwise "not_home" (it never came home, e.g. a dropout outside the
    zone). Until 2026-11-13 any landing while a teammate held the zone was labelled
    pad_zone_timeout, even with no wait."""
    vid = drone.vehicle_id
    if team_size < 2 or pads.holder == vid:
        return
    snap = ctx.adapter.get_snapshot(vid)
    xy = (snap.position_ned_m.x, snap.position_ned_m.y) if snap is not None else None
    timed_out = False
    if xy is not None and pads.inside(xy):
        wait_s = (rates.PAD_WAIT_S if drone.cleanup is None       # one cleanup deadline
                  else drone.cleanup.pad_zone_wait_s())
        if await pads.acquire(vid, timeout_s=wait_s / ctx.timing.k):
            return                                    # landing on its pad
        timed_out = True
        if not await drone.sector_executor.leave_pad_zone():
            ctx.recorder.record("pad_zone_exit_failed", vehicle_id=vid, held_by=pads.holder)
        snap = ctx.adapter.get_snapshot(vid)
        xy = (snap.position_ned_m.x, snap.position_ned_m.y) if snap else xy
    if res.get("contingency") is None:
        pr = res.get("planner")
        why = ("pad_zone_timeout" if timed_out
               or (pr and pr.return_home == "pad_zone_timeout") else "not_home")
        res["contingency"] = why
        ctx.recorder.record("contingency_landing", vehicle_id=vid, reason=why,
                            held_by=pads.holder, inside_zone=bool(xy and pads.inside(xy)),
                            position=xy)


async def _descend(ctx, drone, airspace) -> str:
    return (await descend_before_landing(ctx, drone) if airspace is None
            else await descend_reserved(ctx, drone))


async def fly_team_drone(ctx, pads, team_size: int, drone, res: dict, airspace=None):
    """One team drone's whole flight: take off, search (the planner runs until its search
    ends and it is back home), take its turn over the pads, descend, shut down. What
    happened is written into `res`.

    With the managed airspace (atal/runtime/airspace.py) the takeoff and climb wait for
    the drone's startup reservation (top-down), and the descent and landing need a landing
    reservation obtained above the pad while holding the zone (policy section 11). Without
    one the drone neither lands nor disarms."""
    vid = drone.vehicle_id
    cleared = None if airspace is None else False
    took_off = False
    try:
        if airspace is not None:
            snap = ctx.adapter.get_snapshot(vid)
            r = await airspace.startup(
                vid, (snap.position_ned_m.x, snap.position_ned_m.y), drone.launch_altitude_m)
            if r is None:
                raise RuntimeError("startup not granted: the drone stayed on its pad")
            drone.recovery.startup = r
        await takeoff(ctx.adapter, vid, ctx.recorder, timeout_s=30.0)
        took_off = True
        res["planner"] = pr = await drone.planner.run()
        ctx.log(f"  [{time.strftime('%H:%M:%S')}] {vid}: search ended: {pr.outcome} "
                f"({pr.reason}); return home: {pr.return_home}")
        await clear_pads(ctx, pads, team_size, drone, res)
        if airspace is not None:
            cleared = (drone.recovery.outcome != "unresolved"
                       and await drone.recovery.clear_landing(
                           ground_m=drone.launch_altitude_m, deadline_s=drone.cleanup.start()))
            res["landing_cleared"] = cleared
            if not cleared:
                ctx.recorder.record("landing_not_cleared", vehicle_id=vid)
                res["descend"] = "not_cleared"
                return
        res["descend"] = await _descend(ctx, drone, airspace)
        if airspace is not None and res["descend"] != "succeeded":
            cleared = False            # still up or stuck: it does not land (UNRESOLVED)
    except Exception as err:
        res["error"] = f"{type(err).__name__}: {err}"
        ctx.recorder.record("error", vehicle_id=vid, error=res["error"])
        ctx.log(f"  {vid} ERROR: {res['error']}")
    finally:
        if "descend" not in res or res["descend"] is None:
            try:                       # an error before the descent: same rule
                await clear_pads(ctx, pads, team_size, drone, res)
            except Exception as err:
                ctx.recorder.record("error", vehicle_id=vid,
                                    error=f"clear_pads: {type(err).__name__}: {err}")
            if (airspace is not None and took_off and "landing_cleared" not in res
                    and drone.recovery.outcome != "unresolved"):
                try:                   # the same landing rule on this path too
                    cleared = await drone.recovery.clear_landing(
                        ground_m=drone.launch_altitude_m, deadline_s=drone.cleanup.start())
                except Exception as err:   # noqa: BLE001 - recorded; no landing then
                    cleared = False
                    ctx.recorder.record("error", vehicle_id=vid,
                                        error=f"clear_landing: {type(err).__name__}: {err}")
                res["landing_cleared"] = cleared
                if cleared:
                    res["descend"] = await _descend(ctx, drone, airspace)
                    cleared = res["descend"] == "succeeded"
        touchdown = (None if airspace is None
                     else (lambda: airspace.manager.touchdown(vid)))
        res["shutdown"] = rep = await shut_down(ctx, drone, landing_cleared=cleared,
                                                on_touchdown=touchdown)
        drone.coordinator.state = "landed"
        if pads.holder == vid and not rep.grounded_confirmed:
            ctx.recorder.record("pad_zone_held_unconfirmed", vehicle_id=vid)
        else:
            pads.release(vid)                  # only after the disarm
        if airspace is not None:
            res["landing_times"] = landing_times(airspace.manager, drone)
            if rep.grounded_confirmed:
                await airspace.landed(vid, disarmed=rep.disarmed)


def landing_times(manager, drone) -> dict:
    """Sim seconds, kept apart: the landing request's queue wait (request to grant), the
    descent (start to confirmed touchdown), and the whole cleanup (its first pad-zone wait
    or landing request, to the end of shutdown)."""
    from atal.airspace.transition_manager import Kind, State
    rs = [x for x in manager.reservations.values()
          if x.vehicle_id == drone.vehicle_id and x.kind == Kind.LANDING]
    r = next((x for x in rs if x.state == State.COMPLETED), None)
    wait = drone.recovery.landing_wait or {}
    c = drone.cleanup
    return {"queue_wait_s": wait.get("queue_wait_s"),
            "descent_s": (round(r.ended_at - r.started_at, 2)
                          if r is not None and r.started_at is not None else None),
            "cleanup_total_s": (round((manager.now or 0.0) - c.started_at, 2)
                                if c.started_at is not None else None),
            "cleanup_deadline": round(c.deadline, 2) if c.deadline is not None else None,
            "landing_requests": len(rs)}
