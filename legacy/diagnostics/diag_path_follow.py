"""
Diagnostic: continuous corridor flight with the velocity path follower (Phase 0).

    python -m legacy.diagnostics.diag_path_follow [--speed 3] [--no-maneuvers]

No LLM. One drone (scene_basic_drone.jsonc, spawned at heading 315: the heading where
Simple Flight's position hold used to orbit). After takeoff and a climb to launch + 5 m,
the follower flies a 15 m x 10 m rectangle (50 m) back to the start at --speed and,
unless --no-maneuvers, interrupts it mid-flight:

    progress >= 20 m   hold 5 s, then resume            (stop distance, drift, restart)
    progress >= 30 m   change altitude +2 m while flying  (a layer change)
    progress >= 38 m   slow to 1.5 m/s                   (a speed change)

Measured from ground truth at 10 Hz: speed on the straight parts of each leg,
cross-track error, the hold (how far it travels after the hold command, drift while
holding), altitude-change time, arrival error, and 10 s of motion after arrival
(orbiting would show here). Everything is recorded in runs/.
"""
import argparse
import asyncio
import math
import statistics
import sys
import time
from dataclasses import asdict

from atal.core.telemetry import GroundState, ValidationStatus
from atal.flight.executor import ProjectAirSimExecutor
from atal.flight.path_follower import FollowerConfig, FollowMode, PathFollower
from atal.flight.shutdown import safe_shutdown
from atal.runtime.recorder import EvaluationRecorder
from atal.simulation.airsim import ProjectAirSimAdapter

VEHICLE = "Drone1"
SCENE = "scene_basic_drone.jsonc"
RECT = [(0.0, 0.0), (15.0, 0.0), (15.0, 10.0), (0.0, 10.0), (0.0, 0.0)]   # offsets from launch
CLIMB_M = 5.0
HOLD_AT_M, HOLD_S = 20.0, 5.0
ALT_AT_M, ALT_DELTA_M = 30.0, 2.0
SLOW_AT_M, SLOW_MPS = 38.0, 1.5
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


async def maneuvers(f: PathFollower, rec, notes: dict, adapter):
    """Interrupt the flight at fixed progress points (the mission runner's job later)."""
    async def wait_progress(m):
        # IDLE too: this task starts a moment before the follower does
        while f.mode in (FollowMode.IDLE, FollowMode.CRUISE) and (
                f.progress_m < m or f.mode == FollowMode.IDLE):
            await asyncio.sleep(0.05)
        return f.mode == FollowMode.CRUISE
    if await wait_progress(HOLD_AT_M):
        s = adapter.get_snapshot(VEHICLE)
        notes["hold_cmd"] = (time.monotonic(), s.position_ned_m.x, s.position_ned_m.y,
                             s.ground_speed_mps)
        f.hold()
        rec.record("diag_maneuver", what="hold", progress_m=f.progress_m)
        await asyncio.sleep(HOLD_S)
        notes["hold_end"] = time.monotonic()
        f.resume()
        rec.record("diag_maneuver", what="resume", progress_m=f.progress_m)
    if await wait_progress(ALT_AT_M):
        notes["alt_cmd"] = (time.monotonic(), f.altitude_target_m + ALT_DELTA_M)
        f.set_altitude(f.altitude_target_m + ALT_DELTA_M)
        rec.record("diag_maneuver", what="altitude", target=f.altitude_target_m)
    if await wait_progress(SLOW_AT_M):
        f.set_speed(SLOW_MPS)
        rec.record("diag_maneuver", what="speed", speed_mps=SLOW_MPS)


async def sample(adapter, f: PathFollower, rec, stop, samples):
    while not stop.is_set():
        try:
            s = adapter.get_snapshot(VEHICLE)
            _, xt = f.path.project(s.position_ned_m.x, s.position_ned_m.y)
            row = {"t": time.monotonic(), "n": s.position_ned_m.x, "e": s.position_ned_m.y,
                   "alt": s.altitude_local_m, "gs": s.ground_speed_mps, "xtrack": xt,
                   "progress": f.progress_m, "mode": f.mode.value, "hdg": s.heading_deg}
            samples.append(row)
            rec.record("diag_sample", **{k: (round(v, 3) if isinstance(v, float) else v)
                                         for k, v in row.items()})
        except Exception as err:
            rec.record("diag_sample_error", error=f"{type(err).__name__}: {err}")
        try:
            await asyncio.wait_for(stop.wait(), 0.1)
        except asyncio.TimeoutError:
            pass


def report(samples, f, result, notes, home, speed, watch):
    out = []
    cum = f.path.cum
    # straight parts: progress at least 3 m past a corner and 3 m before the next
    straight = [r for r in samples if r["mode"] == "cruise" and r["t"] > notes.get("start", 0)
                and any(cum[i] + 3 <= r["progress"] <= cum[i + 1] - 3 for i in range(len(cum) - 1))]
    fast = [r for r in straight if r["progress"] < SLOW_AT_M] if notes.get("maneuvers") else straight
    if fast:
        gs = [r["gs"] for r in fast]
        xt = [r["xtrack"] for r in fast]
        out.append(f"cruise (straight legs, commanded {speed:g} m/s): speed median "
                   f"{statistics.median(gs):.2f} m/s (min {min(gs):.2f}, max {max(gs):.2f}); "
                   f"cross-track median {statistics.median(xt):.2f} m, max {max(xt):.2f} m")
    allx = [r["xtrack"] for r in samples if r["mode"] in ("cruise", "hold")]
    if allx:
        out.append(f"cross-track over the whole flight (corners included): max {max(allx):.2f} m")
    if "hold_cmd" in notes:
        t0, n0, e0, v0 = notes["hold_cmd"]
        held = [r for r in samples if t0 <= r["t"] <= notes["hold_end"]]
        if held:
            far = max(math.hypot(r["n"] - n0, r["e"] - e0) for r in held)
            tail = [r for r in held if r["t"] >= notes["hold_end"] - 2.0]
            drift = max(math.hypot(r["n"] - tail[0]["n"], r["e"] - tail[0]["e"]) for r in tail)
            out.append(f"hold at {v0:.1f} m/s: travelled up to {far:.2f} m past the hold point; "
                       f"last 2 s drift {drift:.2f} m, speed at end {held[-1]['gs']:.2f} m/s")
    if "alt_cmd" in notes:
        t0, target = notes["alt_cmd"]
        hit = next((r for r in samples if r["t"] >= t0 and abs(r["alt"] - target) <= 0.3), None)
        out.append(f"altitude +{ALT_DELTA_M:g} m while cruising: within 0.3 m after "
                   + (f"{hit['t'] - t0:.1f} s" if hit else "NEVER"))
    end_n, end_e = home[0] + RECT[-1][0], home[1] + RECT[-1][1]
    if watch:
        vmax = max(r["gs"] for r in watch)
        dmax = max(math.hypot(r["n"] - end_n, r["e"] - end_e) for r in watch)
        out.append(f"after arrival ({WATCH_S:g} s): max speed {vmax:.2f} m/s, max distance from the "
                   f"end point {dmax:.2f} m")
    out.append(f"follower: {result.mode.value} ({result.reason}) in {result.elapsed_s:.1f} s, "
               f"waypoints passed {result.waypoints_passed}, max cross-track {result.max_cross_track_m:.2f} m")
    return out


async def main(args) -> int:
    rec = EvaluationRecorder(run_config={
        "script": "diag_path_follow.py", "scene": SCENE, "rect_offsets_m": RECT,
        "climb_m": CLIMB_M, "speed_mps": args.speed, "maneuvers": not args.no_maneuvers,
        "follower": asdict(FollowerConfig())})
    print(f"Recording to {rec.path}")
    samples, notes, watch = [], {"maneuvers": not args.no_maneuvers}, []
    result, outcome = None, "aborted"
    try:
        with ProjectAirSimAdapter(vehicle_ids=[VEHICLE], scene=SCENE) as adapter:
            await asyncio.sleep(0.5)
            try:
                ground = adapter.get_snapshot(VEHICLE)            # on the pad, before takeoff
                home = (ground.position_ned_m.x, ground.position_ned_m.y)
                await takeoff(adapter)
                ex = ProjectAirSimExecutor(adapter, on_event=rec.recorder_for(source="executor"))
                alt = round(ground.altitude_local_m + CLIMB_M, 2)
                r = await ex.change_altitude(VEHICLE, alt, reason="diag: climb")
                print(f"climb to {alt}: {r.status.value}")
                path = [(home[0] + dn, home[1] + de) for dn, de in RECT]
                f = PathFollower(adapter, VEHICLE, path, alt, args.speed,
                                 on_event=rec.recorder_for(source="follower"))
                stop = asyncio.Event()
                sampler = asyncio.create_task(sample(adapter, f, rec, stop, samples))
                man = None if args.no_maneuvers else asyncio.create_task(
                    maneuvers(f, rec, notes, adapter))
                notes["start"] = time.monotonic()
                print(f"flying {f.path.length:.0f} m at {args.speed:g} m/s ...")
                result = await f.run(timeout_s=180)
                if man is not None and not man.done():
                    man.cancel()
                print(f"follower: {result.mode.value} ({result.reason})")
                t_end = time.monotonic() + WATCH_S
                while time.monotonic() < t_end:          # arrival: does it stay put?
                    await asyncio.sleep(0.1)
                    if samples:
                        watch.append(samples[-1])
                stop.set()
                await sampler
                outcome = "done" if result.mode == FollowMode.DONE else "fail"
            finally:
                await safe_shutdown(adapter.drone(VEHICLE), adapter=adapter, vehicle_id=VEHICLE)
    finally:
        rec.close(outcome)
    print("\n================ PATH FOLLOW DIAGNOSTIC ================")
    if result is not None:
        for line in report(samples, f, result, notes, home, args.speed, watch):
            print("  " + line)
    print(f"Run log: {rec.path}")
    return 0 if outcome == "done" else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--no-maneuvers", action="store_true")
    sys.exit(asyncio.run(main(ap.parse_args())))
