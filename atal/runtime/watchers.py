"""Background tasks that run beside the drones for the whole run. They belong to the
experiment, not to any drone: they read the simulator's true state, record it, and end a
search when a limit is reached."""
import asyncio
import time

from atal.evaluation.sar_metrics import _is_impact


async def wait_or_stop(stop: asyncio.Event, period_s: float) -> None:
    """Sleep for period_s, or return at once when the run stops."""
    try:
        async with asyncio.timeout(period_s):
            await stop.wait()
    except asyncio.TimeoutError:
        pass


def collision_dict(c) -> dict:
    return c.model_dump(mode="json") if hasattr(c, "model_dump") else dict(c)


async def sample_truth(adapter, missions: dict, recorder, stop, period_s, on_breach,
                       on_impact=None):
    """Record every drone's true pose (`truth`), new collisions and the first geofence breach
    of each drone. on_breach(vid) and on_impact(vid, info) let the run stop that drone."""
    breached, seen = set(), {}
    while not stop.is_set():
        for vid in missions:
            try:
                for c in adapter.collision_log(vid):
                    if c.sim_time_ns is not None and (vid not in seen or c.sim_time_ns > seen[vid]):
                        recorder.record("collision", vehicle_id=vid, collision=c,
                                        sim_time_ns=c.sim_time_ns)
                        seen[vid] = c.sim_time_ns
                        info = collision_dict(c)
                        if on_impact is not None and _is_impact(info):
                            on_impact(vid, info)
            except Exception:
                pass
        poses = adapter.latest_poses()
        snap = {}
        for vid, m in missions.items():
            p = poses.get(vid)
            if p is None:
                continue
            snap[vid] = {"sim_time_ns": p[0], "north_m": round(p[1], 3),
                         "east_m": round(p[2], 3), "altitude_m": round(-p[3], 3)}
            out = m.geofence_excursion_m(p[1], p[2])
            if out > m.limits.geofence_margin_m and vid not in breached:
                breached.add(vid)
                recorder.record("geofence_breach", vehicle_id=vid, excursion_m=round(out, 2),
                                sim_time_ns=p[0])
                on_breach(vid)
        if snap:
            recorder.record("truth", sim_time_ns=max(v["sim_time_ns"] for v in snap.values()),
                            poses=snap)
        await wait_or_stop(stop, period_s)


async def watch_mission_time(adapter, vid, mission, planner, sector_executor, stop, period_s):
    """Mission time is SIM time from the start of the plan (termination.max_mission_time_s)."""
    while not stop.is_set():
        if planner.start_sim_s is not None:
            now = adapter.get_snapshot(vid).sim_time_s
            if now - planner.start_sim_s > mission.termination.max_mission_time_s:
                planner.on_time_up()
                sector_executor.interrupt("mission time limit")
                return
        await wait_or_stop(stop, period_s)


async def watch_sim_clock(adapter, stop, stall_s, period_s, on_stall, clock=time.monotonic):
    """Live only: if the sim clock (pose push topic) has not moved for stall_s of host
    time, mark the adapter stalled (every later snapshot raises SimStalled) and stop the
    sweeps, so the run winds down instead of waiting on a frozen sim. A pull request that
    never answers is caught by the adapter itself (request_timeout_s)."""
    last_ns, since = None, clock()
    while not stop.is_set():
        ns = max((p[0] for p in adapter.latest_poses().values()), default=None)
        if ns is not None and ns != last_ns:
            last_ns, since = ns, clock()
        elif clock() - since > stall_s and not getattr(adapter, "stalled", None):
            adapter.stalled = (f"sim clock stopped at {(last_ns or 0) / 1e9:.1f} s for "
                               f"{clock() - since:.0f} s")
            on_stall(adapter.stalled)
            return
        if getattr(adapter, "stalled", None):
            on_stall(adapter.stalled)
            return
        await wait_or_stop(stop, period_s)


async def inject_dropout(ctx, drone, at_s: float):
    """The dropout condition: at_s seconds of mission time into the search, this drone is
    told it has failed and stops searching."""
    vid = drone.vehicle_id
    while not ctx.stop.is_set():
        started = drone.planner.start_sim_s
        p = ctx.adapter.latest_poses().get(vid)
        if started is not None and p is not None and p[0] / 1e9 - started >= at_s:
            if drone.planner.on_dropout(f"injected dropout at {at_s:g} s"):
                ctx.log(f"  [{time.strftime('%H:%M:%S')}] {vid}: DROPOUT injected "
                        f"at {at_s:g} s of mission time")
                ctx.recorder.record("dropout_injected", vehicle_id=vid, at_s=at_s)
                drone.sector_executor.interrupt("dropout")
            return
        await wait_or_stop(ctx.stop, ctx.timing.poll(0.5))


async def stop_background(ctx, tasks: list) -> list[str]:
    """End the run's background tasks ((name, task) pairs) and return what went wrong in
    them. A background task that died is a run failure."""
    ctx.stop.set()
    errors = []
    results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
    for (name, _), res in zip(tasks, results):
        if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
            errors.append(f"{name}: {type(res).__name__}: {res}")
            ctx.recorder.record("background_error", task=name,
                                error=f"{type(res).__name__}: {res}")
    return errors
