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
    """Come down to just above the launch height. Returns the command's status."""
    r = await ctx.flight.change_altitude(
        drone.vehicle_id, round(drone.launch_altitude_m + rates.DESCEND_ABOVE_LAUNCH_M, 2),
        reason="setup: descend before landing")
    ctx.recorder.record("setup", step="descend", vehicle_id=drone.vehicle_id,
                        status=r.status.value)
    return r.status.value


async def shut_down(ctx, drone, landing_cleared=None, on_touchdown=None):
    """Hover, land, and disarm only once the drone is confirmed on the ground. With the
    managed airspace, an airborne drone lands only with landing_cleared (its landing
    reservation is ACTIVE); without it, it neither lands nor disarms (UNRESOLVED)."""
    vid = drone.vehicle_id
    report = await safe_shutdown(ctx.adapter.drone(vid), adapter=ctx.adapter, vehicle_id=vid,
                                 log=lambda s: ctx.log(f"[{vid}] {s}"),
                                 confirm_poll_s=ctx.timing.poll(0.1),
                                 landing_cleared=landing_cleared, on_touchdown=on_touchdown)
    ctx.recorder.record("shutdown", vehicle_id=vid, report=report)
    return report


async def clear_pads(ctx, pads, team_size: int, drone, res: dict):
    """Before any descent: hold the pad zone, or be outside it. A drone inside
    that cannot get the zone flies out to its edge; a drone that lands without
    the zone is a recorded contingency landing (it fails acceptance)."""
    vid = drone.vehicle_id
    if team_size < 2 or pads.holder == vid:
        return
    snap = ctx.adapter.get_snapshot(vid)
    xy = (snap.position_ned_m.x, snap.position_ned_m.y) if snap is not None else None
    if xy is not None and pads.inside(xy):
        wait_s = (rates.PAD_WAIT_S if drone.cleanup is None       # one cleanup deadline
                  else drone.cleanup.pad_zone_wait_s())
        if await pads.acquire(vid, timeout_s=wait_s / ctx.timing.k):
            return                                    # landing on its pad
        if not await drone.sector_executor.leave_pad_zone():
            ctx.recorder.record("pad_zone_exit_failed", vehicle_id=vid, held_by=pads.holder)
        snap = ctx.adapter.get_snapshot(vid)
        xy = (snap.position_ned_m.x, snap.position_ned_m.y) if snap else xy
    if res.get("contingency") is None:
        pr = res.get("planner")
        why = ("pad_zone_timeout" if (pr and pr.return_home == "pad_zone_timeout")
               or pads.holder not in (None, vid) else "not_home")
        res["contingency"] = why
        ctx.recorder.record("contingency_landing", vehicle_id=vid, reason=why,
                            held_by=pads.holder, inside_zone=bool(xy and pads.inside(xy)),
                            position=xy)


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
        res["descend"] = await descend_before_landing(ctx, drone)
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
                    res["descend"] = await descend_before_landing(ctx, drone)
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
