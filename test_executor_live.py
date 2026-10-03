"""
Live check of ProjectAirSimExecutor (Pass 5, step 1) against the running simulator.

    python .\\test_executor_live.py

Sequence
    1. connect + load scene
    2. SETUP ONLY: arm and take off with direct Project AirSim calls
       (executor takeoff is a later Pass 5 command)
    3. executor.rotate_to_heading(initial heading + 90)   -> expect SUCCEEDED
    4. executor.change_altitude(current altitude + 2 m)   -> expect SUCCEEDED
    5. safe_shutdown(..., adapter=adapter) in finally, always

Exit code 0 only if both commands SUCCEEDED.
"""
import asyncio
import sys
import time

from adapters.projectairsim_adapter import ProjectAirSimAdapter
from executor.completion import heading_error_deg, horizontal_distance_m
from executor.projectairsim_executor import ProjectAirSimExecutor
from models.action import CommandStatus
from models.telemetry import GroundState, ValidationStatus
from utils.flight_safety import safe_shutdown

VEHICLE = "Drone1"


def show(label, s):
    print(f"[{label:<18}] t={s.sim_time_s:6.2f}s {s.validation_status.value:<7} "
          f"N={s.position_ned_m.x:6.2f} E={s.position_ned_m.y:6.2f} alt={s.altitude_local_m:5.2f}m "
          f"hdg={s.heading_deg:6.1f} gs={s.ground_speed_mps:4.2f} vs={s.vertical_speed_mps:+5.2f} "
          f"ground={s.ground_state.value}({s.ground_state_basis})")


def report(name, r):
    print(f"\n=== {name}: {r.status.value.upper()} "
          f"(sent={r.sent_to_simulator}, fallback={r.fallback_applied}, {r.elapsed_s:.2f} s)")
    print(f"    reason: {r.reason}")
    if r.start_snapshot and r.final_snapshot:
        show("start", r.start_snapshot)
        show("final", r.final_snapshot)


async def setup_takeoff(adapter):
    """Direct Project AirSim calls: setup only, not part of the executor."""
    d = adapter.drone(VEHICLE)
    d.enable_api_control()
    d.arm()
    await (await d.takeoff_async())
    # wait until the adapter agrees we are airborne with valid telemetry
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        s = adapter.get_snapshot(VEHICLE)
        if s.validation_status == ValidationStatus.VALID and s.ground_state == GroundState.AIRBORNE:
            show("after takeoff", s)
            return s
        await asyncio.sleep(0.1)
    raise RuntimeError(f"not airborne after takeoff: {s.ground_state.value} ({s.ground_state_basis})")


async def main() -> int:
    ok = False
    with ProjectAirSimAdapter(vehicle_ids=[VEHICLE]) as adapter:
        await asyncio.sleep(0.5)
        executor = ProjectAirSimExecutor(adapter)
        try:
            show("start", adapter.get_snapshot(VEHICLE))
            s0 = await setup_takeoff(adapter)

            # --- rotate ~90 deg from the initial heading
            target_hdg = (s0.heading_deg + 90.0) % 360.0
            r1 = await executor.rotate_to_heading(VEHICLE, round(target_hdg, 1),
                                                  reason="live test: rotate +90")
            report(f"rotate_to_heading({target_hdg:.1f})", r1)
            if r1.final_snapshot:
                f = r1.final_snapshot
                print(f"    heading error {heading_error_deg(target_hdg, f.heading_deg):+.2f} deg, "
                      f"drift {horizontal_distance_m(f, s0.position_ned_m.x, s0.position_ned_m.y):.2f} m")

            # --- climb 2 m, holding north/east
            if r1.status == CommandStatus.SUCCEEDED:
                s1 = adapter.get_snapshot(VEHICLE)
                target_alt = round(s1.altitude_local_m + 2.0, 2)
                r2 = await executor.change_altitude(VEHICLE, target_alt,
                                                    reason="live test: climb 2 m")
                report(f"change_altitude({target_alt:.2f})", r2)
                if r2.final_snapshot:
                    f = r2.final_snapshot
                    print(f"    altitude error {f.altitude_local_m - target_alt:+.2f} m, "
                          f"drift {horizontal_distance_m(f, s1.position_ned_m.x, s1.position_ned_m.y):.2f} m")
                ok = r2.status == CommandStatus.SUCCEEDED
        finally:
            print()
            await safe_shutdown(adapter.drone(VEHICLE), adapter=adapter, vehicle_id=VEHICLE)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
