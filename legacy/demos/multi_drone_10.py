import asyncio
import sys
import time

from adapters.projectairsim_adapter import ProjectAirSimAdapter
from executor.projectairsim_executor import ProjectAirSimExecutor
from models.action import CommandStatus
from models.telemetry import GroundState, ValidationStatus
from legacy.fleet import run_all_or_cancel
from utils.flight_safety import safe_shutdown
from utils.separation import SeparationTracker, assess, format_report


SCENE = "scene_ten_drones.jsonc"

VEHICLES = [f"Drone{i}" for i in range(1, 11)]

# Four drones are assigned to the collision experiment.
CRASH_GROUP = ["Drone1", "Drone2", "Drone3", "Drone4"]
ROAM_GROUP = [v for v in VEHICLES if v not in CRASH_GROUP]

SPEED = 3.0
CRASH_ALTITUDE = 8.0
ROAM_ALTITUDE = 10.0
MIN_SEPARATION_M = 2.0
SEPARATION_SAMPLE_HZ = 20
DESCEND_ABOVE_LAUNCH = 1.5

# Separate destinations for the six roaming drones.
ROAM_OFFSETS = {
    "Drone5": (12.0, -12.0),
    "Drone6": (16.0, -5.0),
    "Drone7": (16.0, 5.0),
    "Drone8": (12.0, 12.0),
    "Drone9": (-10.0, -10.0),
    "Drone10": (-10.0, 10.0),
}

TURN_TO = {
    "Drone5": 45.0,
    "Drone6": 90.0,
    "Drone7": 135.0,
    "Drone8": 180.0,
    "Drone9": 225.0,
    "Drone10": 270.0,
}


def row(s) -> str:
    return (
        f"  {s.vehicle_id:<8} "
        f"{s.validation_status.value:<7} "
        f"N={s.position_ned_m.x:6.1f} "
        f"E={s.position_ned_m.y:6.1f} "
        f"alt={s.altitude_local_m:5.1f} "
        f"hdg={s.heading_deg:5.1f} "
        f"gs={s.ground_speed_mps:4.1f} "
        f"vs={s.vertical_speed_mps:+4.1f} "
        f"{s.ground_state.value}"
    )


async def monitor(adapter, stop, phase):
    t0 = time.monotonic()

    while not stop.is_set():
        lines = []

        for vid in VEHICLES:
            try:
                lines.append(row(adapter.get_snapshot(vid)))
            except Exception as err:
                lines.append(f"  {vid}: snapshot error: {err}")

        print(
            f"--- t+{time.monotonic() - t0:5.1f}s "
            f"phase: {phase['name']}"
        )
        print("\n".join(lines))

        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


async def sample_separation(adapter, tracker, stop, phase):
    while not stop.is_set():
        steps = phase.get("steps") or {}
        label = phase["name"]

        if steps:
            label += ", " + ", ".join(
                f"{v}:{steps[v]}" for v in VEHICLES if v in steps
            )

        tracker.update(adapter.latest_poses(), phase=label)

        try:
            await asyncio.wait_for(
                stop.wait(), 1.0 / SEPARATION_SAMPLE_HZ
            )
        except asyncio.TimeoutError:
            pass


async def takeoff_all(adapter):
    """Arm and take off all ten drones concurrently."""

    async def one(vid):
        drone = adapter.drone(vid)
        drone.enable_api_control()
        drone.arm()
        await (await drone.takeoff_async())

    await run_all_or_cancel(one(v) for v in VEHICLES)

    deadline = time.monotonic() + 15.0
    pending = set(VEHICLES)

    while pending and time.monotonic() < deadline:
        for vid in list(pending):
            snapshot = adapter.get_snapshot(vid)

            if (
                snapshot.validation_status == ValidationStatus.VALID
                and snapshot.ground_state == GroundState.AIRBORNE
            ):
                pending.discard(vid)

        await asyncio.sleep(0.1)

    if pending:
        raise RuntimeError(
            f"Drones did not become airborne: {sorted(pending)}"
        )


async def crash_mission(executor, vid, target, results, phase):
    """
    Send one collision-group drone toward the shared target.

    All four commands are launched concurrently by main().
    Actual contact depends on the simulation physics.
    """
    n, e, altitude = target

    phase.setdefault("steps", {})[vid] = "converge"

    result = await executor.move_to_position(
        vid,
        n,
        e,
        altitude,
        SPEED,
        reason="intentional collision convergence test",
    )

    results[vid].append(("converge", result))
    phase["steps"][vid] = (
        "converge complete"
        if result.status == CommandStatus.SUCCEEDED
        else "converge command failed"
    )


async def roam_mission(executor, vid, home, results, phase):
    """Fly to a distinct waypoint and then to a second waypoint."""
    n0, e0, launch_alt = home
    dn, de = ROAM_OFFSETS[vid]

    steps = [
        (
            "climb",
            lambda: executor.change_altitude(
                vid, ROAM_ALTITUDE, speed_mps=2.0,
                reason="roaming altitude",
            ),
        ),
        (
            "outbound",
            lambda: executor.move_to_position(
                vid,
                n0 + dn,
                e0 + de,
                ROAM_ALTITUDE,
                SPEED,
                reason="roam to separate waypoint",
            ),
        ),
        (
            "turn",
            lambda: executor.rotate_to_heading(
                vid, TURN_TO[vid], reason="roaming turn",
            ),
        ),
        (
            "second waypoint",
            lambda: executor.move_to_position(
                vid,
                n0 - dn * 0.5,
                e0 - de * 0.5,
                ROAM_ALTITUDE,
                SPEED,
                reason="continue roaming",
            ),
        ),
        (
            "descend",
            lambda: executor.change_altitude(
                vid,
                round(launch_alt + DESCEND_ABOVE_LAUNCH, 2),
                speed_mps=2.0,
                reason="descend before landing",
            ),
        ),
    ]

    for name, command in steps:
        phase.setdefault("steps", {})[vid] = name

        result = await command()
        results[vid].append((name, result))

        if result.status != CommandStatus.SUCCEEDED:
            phase["steps"][vid] = f"{name} failed"
            return

    phase["steps"][vid] = "roaming complete"


async def main() -> int:
    results = {v: [] for v in VEHICLES}
    phase = {"name": "connect", "steps": {}}

    with ProjectAirSimAdapter(
        vehicle_ids=VEHICLES,
        scene=SCENE,
    ) as adapter:
        await asyncio.sleep(0.5)

        executor = ProjectAirSimExecutor(adapter)
        stop = asyncio.Event()
        tracker = SeparationTracker(VEHICLES)

        mon = asyncio.create_task(monitor(adapter, stop, phase))
        sep = asyncio.create_task(
            sample_separation(adapter, tracker, stop, phase)
        )
        reports = []
        try:
            homes = {}

            for vid in VEHICLES:
                s = adapter.get_snapshot(vid)
                homes[vid] = (
                    s.position_ned_m.x,
                    s.position_ned_m.y,
                    s.altitude_local_m,
                )

            # Shared target is the center of the four collision drones'
            # launch positions. The four drones approach at one altitude.
            target_n = sum(homes[v][0] for v in CRASH_GROUP) / 4
            target_e = sum(homes[v][1] for v in CRASH_GROUP) / 4

            target = (target_n, target_e, CRASH_ALTITUDE)

            print("Collision group:", CRASH_GROUP)
            print("Roaming group:", ROAM_GROUP)
            print("Shared convergence target:", target)

            phase["name"] = "takeoff"
            await takeoff_all(adapter)

            phase["name"] = "collision and roaming missions"

            # All ten missions run concurrently. Each drone still uses
            # its own executor command lock.
            await run_all_or_cancel(
                [
                    *(
                        crash_mission(
                            executor, v, target, results, phase
                        )
                        for v in CRASH_GROUP
                    ),
                    *(
                        roam_mission(
                            executor, v, homes[v], results, phase
                        )
                        for v in ROAM_GROUP
                    ),
                ]
            )

        finally:
            phase["name"] = "safe shutdown"
            phase["steps"] = {}

            reports = await asyncio.gather(
                *(
                    safe_shutdown(
                        adapter.drone(v),
                        adapter=adapter,
                        vehicle_id=v,
                        log=lambda m, v=v: print(f"[{v}] {m}"),
                    )
                    for v in VEHICLES
                ),
                return_exceptions=True,
            )

            stop.set()
            await mon
            await sep

            tracker.finish()
            collisions = {
                v: adapter.collision_log(v) for v in VEHICLES
            }

    print("\n================ RESULTS ================")

    for vid in VEHICLES:
        print(f"\n{vid}:")

        for name, result in results[vid]:
            print(
                f"  {result.status.value:<9} "
                f"{name:<18} {result.elapsed_s:6.2f}s "
                f"{result.reason}"
            )

    print("\n================ SHUTDOWN ================")

    for vid, report in zip(VEHICLES, reports):
        if isinstance(report, Exception):
            print(f"{vid}: shutdown error: {report}")
        else:
            print(
                f"{vid}: "
                f"{'disarmed' if report.disarmed else 'UNRESOLVED'}"
            )

    print("\n================ COLLISIONS ================")

    for vid, events in collisions.items():
        print(f"{vid}: {events}")

    separation = assess(
        tracker, collisions, VEHICLES, MIN_SEPARATION_M
    )

    print("\n================ SEPARATION ================")
    print(format_report(separation))

    print(
        "\nTEST COMPLETE: inspect collision logs, "
        "separation results, and shutdown status."
    )

    # The separation test is expected to fail if the intended collisions
    # occur. Do not treat that expected failure as a safety PASS.
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
