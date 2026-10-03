"""
Three drones flying at the same time through the executor.

    python .\\test_multi_drone_live.py

Scene: sim_config/scene_three_drones.jsonc (Drone1/2/3 spawned 3 m apart).
One adapter loads the scene ONCE and serves all three vehicles; one executor
runs each drone's commands under that drone's own lock, so different drones
move concurrently while a single drone never gets overlapping commands.

Flight plan (each drone keeps its own altitude layer, so paths can't meet):
    setup    take off all three together (direct calls; executor takeoff is later)
    climb    Drone1 -> 6 m, Drone2 -> 8 m, Drone3 -> 10 m        (change_altitude)
    fan out  Drone1 north-west, Drone2 north, Drone3 north-east  (move_to_position)
    turn     each faces a different heading                      (rotate_to_heading)
    return   each flies back above its own launch point          (move_to_position)
    descend  each drops to 1.5 m above its launch height         (change_altitude)
    land     safe_shutdown for all three, concurrently

A status table for all drones prints every second while they fly.
Separation between every pair of drones is sampled at 20 Hz from the adapter's
pose cache (ground truth, evaluation only) for the whole run, and every
collision reported by the simulator is logged.

Exit code 0 only if every command for every drone SUCCEEDED, every drone landed
and disarmed, no pair came closer than MIN_SEPARATION_M, and there were no
drone-to-drone collisions or impacts.
"""
import asyncio
import sys
import time

from adapters.projectairsim_adapter import ProjectAirSimAdapter
from executor.projectairsim_executor import ProjectAirSimExecutor
from models.action import CommandStatus
from models.telemetry import GroundState, ValidationStatus
from utils.flight_safety import safe_shutdown
from utils.separation import SeparationTracker, assess, format_report

SCENE = "scene_three_drones.jsonc"
VEHICLES = ["Drone1", "Drone2", "Drone3"]
ALTITUDE = {"Drone1": 6.0, "Drone2": 8.0, "Drone3": 10.0}
# fan-out waypoints as offsets (north, east) from each drone's launch point
FAN_OUT = {"Drone1": (12.0, -10.0), "Drone2": (15.0, 0.0), "Drone3": (12.0, 10.0)}
TURN_TO = {"Drone1": 90.0, "Drone2": 180.0, "Drone3": 270.0}
SPEED = 3.0
MIN_SEPARATION_M = 2.0       # 3D; the 1.0 m position tolerance leaves margin at 3 m spacing
SEPARATION_SAMPLE_HZ = 20
DESCEND_ABOVE_LAUNCH = 1.5   # descend to launch altitude + this before landing
# (the spawn area in Blocks is ~2.7 m local, not 1.2, so this must be relative)


def row(s) -> str:
    return (f"  {s.vehicle_id:<7} {s.validation_status.value:<7} N={s.position_ned_m.x:6.1f} "
            f"E={s.position_ned_m.y:6.1f} alt={s.altitude_local_m:5.1f} hdg={s.heading_deg:5.1f} "
            f"gs={s.ground_speed_mps:4.1f} vs={s.vertical_speed_mps:+4.1f} {s.ground_state.value}")


async def monitor(adapter, stop: asyncio.Event, phase: dict):
    t0 = time.monotonic()
    while not stop.is_set():
        lines = []
        for vid in VEHICLES:
            try:
                lines.append(row(adapter.get_snapshot(vid)))
            except Exception as err:
                lines.append(f"  {vid:<7} snapshot error: {err}")
        print(f"--- t+{time.monotonic() - t0:5.1f}s  phase: {phase['name']}")
        print("\n".join(lines))
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


async def sample_separation(adapter, tracker, stop: asyncio.Event, phase: dict):
    while not stop.is_set():
        steps = phase.get("steps") or {}
        label = phase["name"] if not steps else ", ".join(f"{v}:{steps[v]}" for v in VEHICLES if v in steps)
        tracker.update(adapter.latest_poses(), phase=label)
        try:
            await asyncio.wait_for(stop.wait(), 1.0 / SEPARATION_SAMPLE_HZ)
        except asyncio.TimeoutError:
            pass


async def takeoff_all(adapter):
    """SETUP ONLY: direct Project AirSim calls, all three at once."""
    async def one(vid):
        d = adapter.drone(vid)
        d.enable_api_control()
        d.arm()
        await (await d.takeoff_async())
    await asyncio.gather(*(one(v) for v in VEHICLES))
    deadline = time.monotonic() + 10.0
    pending = set(VEHICLES)
    while pending and time.monotonic() < deadline:
        for vid in list(pending):
            s = adapter.get_snapshot(vid)
            if s.validation_status == ValidationStatus.VALID and s.ground_state == GroundState.AIRBORNE:
                pending.discard(vid)
        await asyncio.sleep(0.1)
    if pending:
        raise RuntimeError(f"not airborne after takeoff: {sorted(pending)}")


async def fly_mission(executor, vid, home, results, phase):
    """One drone's mission: runs its steps in order, stops at the first failure."""
    n0, e0, launch_alt = home
    dn, de = FAN_OUT[vid]
    alt = ALTITUDE[vid]
    steps = [
        ("climb", lambda: executor.change_altitude(vid, alt, speed_mps=2.0, reason="climb to layer")),
        ("fan out", lambda: executor.move_to_position(vid, n0 + dn, e0 + de, alt, SPEED, reason="fan out")),
        ("turn", lambda: executor.rotate_to_heading(vid, TURN_TO[vid], reason="turn")),
        ("return", lambda: executor.move_to_position(vid, n0, e0, alt, SPEED, reason="return")),
        # land_async descends at ~0.2 m/s; drop to a low hover first so landing is quick
        ("descend", lambda: executor.change_altitude(vid, round(launch_alt + DESCEND_ABOVE_LAUNCH, 2),
                                                     speed_mps=2.0, reason="descend")),
    ]
    for name, step in steps:
        phase.setdefault("steps", {})[vid] = name
        r = await step()
        results[vid].append((name, r))
        if r.status != CommandStatus.SUCCEEDED:
            phase["steps"][vid] = f"{name} FAILED"
            break
    else:
        phase["steps"][vid] = "done"


async def main() -> int:
    results = {v: [] for v in VEHICLES}
    phase = {"name": "connect"}
    with ProjectAirSimAdapter(vehicle_ids=VEHICLES, scene=SCENE) as adapter:
        await asyncio.sleep(0.5)
        executor = ProjectAirSimExecutor(adapter)
        stop = asyncio.Event()
        tracker = SeparationTracker(VEHICLES)
        mon = asyncio.create_task(monitor(adapter, stop, phase))
        sep = asyncio.create_task(sample_separation(adapter, tracker, stop, phase))
        try:
            homes = {}
            for vid in VEHICLES:
                s = adapter.get_snapshot(vid)
                homes[vid] = (s.position_ned_m.x, s.position_ned_m.y, s.altitude_local_m)
            phase["name"] = "takeoff (all)"
            await takeoff_all(adapter)
            phase["name"] = "missions (climb -> fan out -> turn -> return -> descend)"
            await asyncio.gather(*(fly_mission(executor, v, homes[v], results, phase) for v in VEHICLES))
        finally:
            phase["name"] = "safe_shutdown (all)"
            phase["steps"] = {}
            reports = await asyncio.gather(
                *(safe_shutdown(adapter.drone(v), adapter=adapter, vehicle_id=v,
                                log=lambda m, v=v: print(f"[{v}] {m}"))
                  for v in VEHICLES),
                return_exceptions=True)
            stop.set()
            await mon
            await sep
            collisions = {v: adapter.collision_log(v) for v in VEHICLES}

    print("\n================ RESULTS ================")
    ok = True
    for vid in VEHICLES:
        print(f"{vid}:")
        for name, r in results[vid]:
            mark = "OK  " if r.status == CommandStatus.SUCCEEDED else "FAIL"
            fix = f" [{r.corrections} corr]" if r.corrections else ""
            print(f"  {mark} {name:<8} {r.status.value:<9} {r.elapsed_s:6.2f}s  {r.reason}{fix}")
        done = len(results[vid]) == 5 and all(r.status == CommandStatus.SUCCEEDED for _, r in results[vid])
        ok = ok and done
    for vid, rep in zip(VEHICLES, reports):
        state = rep if isinstance(rep, Exception) else ("landed+disarmed" if rep.disarmed else "UNRESOLVED")
        print(f"  shutdown {vid}: {state}")
        ok = ok and not isinstance(rep, Exception) and rep.disarmed
    separation = assess(tracker, collisions, VEHICLES, MIN_SEPARATION_M)
    print()
    print(format_report(separation))
    ok = ok and separation.passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
