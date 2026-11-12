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


async def shut_down(ctx, drone):
    """Hover, land, and disarm only once the drone is confirmed on the ground."""
    vid = drone.vehicle_id
    report = await safe_shutdown(ctx.adapter.drone(vid), adapter=ctx.adapter, vehicle_id=vid,
                                 log=lambda s: ctx.log(f"[{vid}] {s}"),
                                 confirm_poll_s=ctx.timing.poll(0.1))
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
        if await pads.acquire(vid, timeout_s=rates.PAD_WAIT_S / ctx.timing.k):
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


async def fly_team_drone(ctx, pads, team_size: int, drone, res: dict):
    """One team drone's whole flight: take off, search (the planner runs until its search
    ends and it is back home), take its turn over the pads, descend, shut down. What
    happened is written into `res`."""
    vid = drone.vehicle_id
    try:
        await takeoff(ctx.adapter, vid, ctx.recorder, timeout_s=30.0)
        res["planner"] = pr = await drone.planner.run()
        ctx.log(f"  [{time.strftime('%H:%M:%S')}] {vid}: search ended: {pr.outcome} "
                f"({pr.reason}); return home: {pr.return_home}")
        await clear_pads(ctx, pads, team_size, drone, res)
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
        res["shutdown"] = await shut_down(ctx, drone)
        drone.coordinator.state = "landed"
        if pads.holder == vid and not res["shutdown"].grounded_confirmed:
            ctx.recorder.record("pad_zone_held_unconfirmed", vehicle_id=vid)
        else:
            pads.release(vid)
