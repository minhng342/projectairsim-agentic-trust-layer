"""
Diagnostic: does Simple Flight settle after move_to_position at every heading?

    python -m legacy.diagnostics.diag_heading_settle [--headings 315,45,0] [--modes rate,velocity]

Why: in the first live LLM run (scene_basic_drone.jsonc, drone spawned at heading
315), a 10 m move north overshot and then ORBITED the target in a slowly decaying
~1 m circle (period ~8 s) for over a minute, so the executor timed out. In the
three-drone scene (heading 0, and later 90/180/270) moves settled. Hypothesis:
position hold is poorly damped when the nose is not on a compass axis.

Result of the first diag run: 315 and 45 timed out orbiting, 0 and 90 settled.
Candidate fix: move_to_position's default yaw is "hold yaw RATE 0" (the heading
drifted 315 -> 318 while orbiting); ExecutorConfig.move_hold_heading=True commands
the start heading explicitly instead. --modes compares them:
    rate      Project AirSim default (yaw_is_rate=True, yaw=0)
    hold      yaw_is_rate=False, yaw=heading at command start
              (diag 2: no difference; 315/45 still orbit)
    velocity  ExecutorConfig.final_approach="velocity": after the move task ends,
              the executor commands velocity toward the target (v = 0.8/s * error,
              max 1 m/s) until settled, then hovers

No LLM. For each mode and heading: rotate to it, fly 10 m (alternately north and
back) at 3 m/s with the executor, then watch 10 s more and measure the residual
motion. Every command and the ground-truth trace are recorded in runs/.
"""
import argparse
import asyncio
import math
import sys
import time
from dataclasses import asdict, replace

from atal.core.action import CommandStatus
from atal.core.telemetry import GroundState, ValidationStatus
from atal.flight.executor import ExecutorConfig, ProjectAirSimExecutor
from atal.flight.shutdown import safe_shutdown
from atal.runtime.recorder import EvaluationRecorder
from atal.simulation.airsim import ProjectAirSimAdapter

VEHICLE = "Drone1"
LEG_M = 10.0
SPEED = 3.0
WATCH_S = 10.0


async def takeoff(adapter):
    d = adapter.drone(VEHICLE)
    d.enable_api_control()
    d.arm()
    await (await d.takeoff_async())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        s = adapter.get_snapshot(VEHICLE)
        if s.validation_status == ValidationStatus.VALID and s.ground_state == GroundState.AIRBORNE:
            return s
        await asyncio.sleep(0.1)
    raise RuntimeError("not airborne after takeoff")


async def watch(adapter, n, e, recorder, label):
    """Residual motion after the command returned: max speed, max/final distance."""
    max_speed = max_dist = 0.0
    last = None
    t_end = time.monotonic() + WATCH_S
    while time.monotonic() < t_end:
        s = adapter.get_snapshot(VEHICLE)
        d = math.hypot(s.position_ned_m.x - n, s.position_ned_m.y - e)
        max_speed, max_dist, last = max(max_speed, s.ground_speed_mps), max(max_dist, d), d
        recorder.record("diag_sample", label=label, dist_m=round(d, 3),
                        speed_mps=round(s.ground_speed_mps, 3), heading_deg=round(s.heading_deg, 1))
        await asyncio.sleep(0.1)
    return max_speed, max_dist, last


async def main(args) -> int:
    headings = [float(h) for h in args.headings.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    recorder = EvaluationRecorder(run_config={
        "script": "diag_heading_settle.py", "scene": args.scene, "headings": headings, "modes": modes,
        "leg_m": LEG_M, "speed_mps": SPEED, "watch_s": WATCH_S,
        "executor": asdict(ExecutorConfig())})
    print(f"Recording to {recorder.path}")
    rows, outcome = [], "aborted"
    try:
        with ProjectAirSimAdapter(vehicle_ids=[VEHICLE], scene=args.scene) as adapter:
            await asyncio.sleep(0.5)
            on_event = recorder.recorder_for(source="executor")
            ex = ProjectAirSimExecutor(adapter, on_event=on_event)
            try:
                s0 = await takeoff(adapter)
                home_n, home_e = s0.position_ned_m.x, s0.position_ned_m.y
                alt = round(s0.altitude_local_m + 4.0, 2)
                r = await ex.change_altitude(VEHICLE, alt, reason="diag: climb")
                print(f"climb to {alt}: {r.status.value} ({r.reason})")
                legs = [(m, h) for m in modes for h in headings]
                for i, (mode, h) in enumerate(legs):
                    ex = ProjectAirSimExecutor(adapter, on_event=on_event, config=replace(
                        ExecutorConfig(), move_hold_heading=(mode == "hold"),
                        final_approach="velocity" if mode == "velocity" else "none"))
                    r = await ex.rotate_to_heading(VEHICLE, h % 360.0, reason=f"diag: face {h:g}")
                    print(f"\n[{mode}] heading {h:g}: rotate {r.status.value}")
                    n = home_n + (LEG_M if i % 2 == 0 else 0.0)
                    r = await ex.move_to_position(VEHICLE, n, home_e, alt, speed_mps=SPEED,
                                                  reason=f"diag: move at heading {h:g}")
                    recorder.record("diag_move", mode=mode, heading=h, status=r.status.value,
                                    reason=r.reason,
                                    elapsed_s=r.elapsed_s, corrections=r.corrections)
                    print(f"  move: {r.status.value} in {r.elapsed_s:.1f} s, "
                          f"{r.corrections} corrections ({r.reason})")
                    if r.status != CommandStatus.SUCCEEDED:
                        await asyncio.sleep(0)   # executor already commanded a hover
                    vmax, dmax, dlast = await watch(adapter, n, home_e, recorder, f"{mode}:h{h:g}")
                    print(f"  next {WATCH_S:g} s: max speed {vmax:.2f} m/s, max dist {dmax:.2f} m, "
                          f"final dist {dlast:.2f} m")
                    rows.append((mode, h, r.status.value, r.elapsed_s, r.corrections, vmax, dmax,
                                 dlast))
                outcome = "done"
            finally:
                await safe_shutdown(adapter.drone(VEHICLE), adapter=adapter, vehicle_id=VEHICLE)
    finally:
        recorder.close(outcome)
    print("\nmode  heading  move       time  corr  then: max speed  max dist  final dist")
    for mode, h, st, el, c, vmax, dmax, dlast in rows:
        print(f"{mode:<5} {h:7g}  {st:<9} {el:5.1f}s  {c:4d}        {vmax:5.2f}     {dmax:5.2f}"
              f"     {dlast:5.2f}")
    print(f"Run log: {recorder.path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_basic_drone.jsonc")
    ap.add_argument("--headings", default="315,45,0")
    ap.add_argument("--modes", default="rate,velocity")
    sys.exit(asyncio.run(main(ap.parse_args())))
