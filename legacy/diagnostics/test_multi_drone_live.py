"""
Three drones flying at the same time, each driven by its own agent through
the observation channel and its action port, with every event recorded.

    python -m legacy.diagnostics.test_multi_drone_live

Scene: sim_config/scene_three_drones.jsonc (Drone1/2/3 spawned 3 m apart).

Plumbing (Pass 5.6):
    adapter (ground truth) -> ObservationFeed per drone, sampled at OBSERVATION_HZ
    agent -> ActionPort per drone (by observation_id) -> BypassGate (no gate yet)
          -> executor -> drone;  the agent gets ActionFeedback, never the full result
    EvaluationRecorder writes runs/run_<id>.jsonl DURING the run

Each agent learns its home and state only from its own observations and can
only command its own drone. No attack hooks are installed and no trust
assessor or risk gate exists yet: observations are NOT_ASSESSED and actions are
BYPASSED, and the run log says so.

Flight plan (each drone keeps its own altitude layer):
    setup    take off all three together (direct calls, recorded as setup)
    climb    Drone1 -> 6 m, Drone2 -> 8 m, Drone3 -> 10 m        (change_altitude)
    fan out  Drone1 north-west, Drone2 north, Drone3 north-east  (move_to_position)
    turn     each faces a different heading                      (rotate_to_heading)
    return   each flies back above its own launch point          (move_to_position)
    -- sync: nobody descends until all three have returned --
    descend  each drops to 1.5 m above its launch height         (change_altitude)
    land     safe_shutdown for all three, concurrently

Why the sync: in the Pass 5.4 run, Drone2 finished first and descended through
Drone1's 6.5 m layer while Drone1 was still correcting a return overshoot toward
Drone2's column (vertical gap 0.34 m, 3D 2.58 m). Altitude layers only protect
while every drone is still in its layer.

Separation (ground truth, evaluation only) is sampled at 20 Hz; a sample counts
only when both drones' sim timestamps advanced and are within 50 ms. Too little
coverage makes the separation verdict INCONCLUSIVE, which is not a PASS.

Exit code 0 only if every step for every drone SUCCEEDED, every drone landed and
disarmed, and the separation verdict is PASS.
"""
import asyncio
import sys
import time
from dataclasses import asdict

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.core.action import ActionType, CommandStatus, ProposedAction
from atal.core.telemetry import GroundState, ValidationStatus
from atal.evaluation.separation import SeparationTracker, assess, format_report
from atal.evaluation.summarize_run import summarize
from atal.flight.executor import ExecutorConfig, ProjectAirSimExecutor
from atal.flight.shutdown import safe_shutdown
from atal.runtime.recorder import EvaluationRecorder
from atal.simulation.airsim import ProjectAirSimAdapter
from atal.trust_layer.action_gate import BypassGate
from atal.trust_layer.assessor import NoTrustAssessor
from legacy.fleet import PhaseBarrier, run_all_or_cancel
from legacy.llm_pilot.scripted_agent import MissionStep, ScriptedMissionAgent

SCENE = "scene_three_drones.jsonc"
VEHICLES = ["Drone1", "Drone2", "Drone3"]
ALTITUDE = {"Drone1": 6.0, "Drone2": 8.0, "Drone3": 10.0}
# fan-out waypoints as offsets (north, east) from each drone's launch point
FAN_OUT = {"Drone1": (12.0, -10.0), "Drone2": (15.0, 0.0), "Drone3": (12.0, 10.0)}
TURN_TO = {"Drone1": 90.0, "Drone2": 180.0, "Drone3": 270.0}
SPEED = 3.0
CLIMB_SPEED = 2.0
MIN_SEPARATION_M = 2.0       # 3D; the 1.0 m position tolerance leaves margin at 3 m spacing
MIN_COVERAGE = 0.9
MAX_GAP_S = 0.5
SEPARATION_SAMPLE_HZ = 20
OBSERVATION_HZ = 5.0         # agent/assessor observation stream (not the evaluator's 20 Hz truth)
MAX_OBSERVATION_AGE_S = 10.0 # port rejects proposals based on older observations
DESCEND_ABOVE_LAUNCH = 1.5   # descend to launch altitude + this before landing
# (the spawn area in Blocks is ~2.7 m local, not 1.2, so this must be relative)
STEP_NAMES = ["climb", "fan out", "turn", "return", "descend"]


def row(s) -> str:
    return (f"  {s.vehicle_id:<7} {s.validation_status.value:<7} N={s.position_ned_m.x:6.1f} "
            f"E={s.position_ned_m.y:6.1f} alt={s.altitude_local_m:5.1f} hdg={s.heading_deg:5.1f} "
            f"gs={s.ground_speed_mps:4.1f} vs={s.vertical_speed_mps:+4.1f} {s.ground_state.value}")


def mission_steps(vid: str, descend_sync: PhaseBarrier) -> list[MissionStep]:
    """The plan for one drone. Targets come from the agent's OWN observations."""
    dn, de = FAN_OUT[vid]
    alt = ALTITUDE[vid]

    def act(action_type, reason, **params):
        return ProposedAction(vehicle_id=vid, action_type=action_type, reason=reason, **params)

    return [
        MissionStep("climb", lambda obs, home: act(
            ActionType.CHANGE_ALTITUDE, "climb to layer", altitude_m=alt, speed_mps=CLIMB_SPEED)),
        MissionStep("fan out", lambda obs, home: act(
            ActionType.MOVE_TO_POSITION, "fan out", north_m=home.north_m + dn,
            east_m=home.east_m + de, altitude_m=alt, speed_mps=SPEED)),
        MissionStep("turn", lambda obs, home: act(
            ActionType.ROTATE_TO_HEADING, "turn", heading_deg=TURN_TO[vid])),
        MissionStep("return", lambda obs, home: act(
            ActionType.MOVE_TO_POSITION, "return", north_m=home.north_m, east_m=home.east_m,
            altitude_m=alt, speed_mps=SPEED)),
        # land_async descends at ~0.2 m/s; drop to a low hover first so landing is quick
        MissionStep("descend", lambda obs, home: act(
            ActionType.CHANGE_ALTITUDE, "descend", speed_mps=CLIMB_SPEED,
            altitude_m=round(home.altitude_m + DESCEND_ABOVE_LAUNCH, 2)), sync=descend_sync),
    ]


async def monitor(adapter, stop: asyncio.Event, phase: dict):
    t0 = time.monotonic()
    while not stop.is_set():
        lines = []
        for vid in VEHICLES:
            try:
                lines.append(row(adapter.get_snapshot(vid)))
            except Exception as err:
                lines.append(f"  {vid:<7} snapshot error: {err}")
        steps = phase.get("steps") or {}
        label = phase["name"] + ("" if not steps else "  [" + ", ".join(
            f"{v}:{steps[v]}" for v in VEHICLES if v in steps) + "]")
        print(f"--- t+{time.monotonic() - t0:5.1f}s  phase: {label}")
        print("\n".join(lines))
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
        except asyncio.TimeoutError:
            pass


async def sample_truth(adapter, tracker, recorder, stop: asyncio.Event, phase: dict):
    """Evaluation side: separation samples, ground-truth poses and collisions, live."""
    seen_collision_ts = {v: None for v in VEHICLES}
    while not stop.is_set():
        steps = phase.get("steps") or {}
        label = phase["name"] if not steps else ", ".join(f"{v}:{steps[v]}" for v in VEHICLES if v in steps)
        poses = adapter.latest_poses()
        tracker.update(poses, phase=label)
        recorder.record("truth", phase=label, poses={
            v: {"sim_time_ns": p[0], "north_m": round(p[1], 3), "east_m": round(p[2], 3),
                "altitude_m": round(-p[3], 3)} for v, p in poses.items()})
        for v in VEHICLES:
            for c in adapter.collision_log(v):
                last = seen_collision_ts[v]
                if c.sim_time_ns is not None and (last is None or c.sim_time_ns > last):
                    recorder.record("collision", vehicle_id=v, collision=c)
                    seen_collision_ts[v] = c.sim_time_ns
        try:
            await asyncio.wait_for(stop.wait(), 1.0 / SEPARATION_SAMPLE_HZ)
        except asyncio.TimeoutError:
            pass


async def takeoff_all(adapter, recorder):
    """SETUP ONLY: direct Project AirSim calls, all three at once (not through agents)."""
    async def one(vid):
        d = adapter.drone(vid)
        d.enable_api_control()
        d.arm()
        await (await d.takeoff_async())
    recorder.record("setup", step="takeoff", status="started",
                    note="direct simulator calls, not through an agent or port")
    await run_all_or_cancel(one(v) for v in VEHICLES)
    deadline = time.monotonic() + 10.0
    pending = set(VEHICLES)
    while pending and time.monotonic() < deadline:
        for vid in list(pending):
            s = adapter.get_snapshot(vid)
            if s.validation_status == ValidationStatus.VALID and s.ground_state == GroundState.AIRBORNE:
                pending.discard(vid)
        await asyncio.sleep(0.1)
    if pending:
        recorder.record("setup", step="takeoff", status="failed", not_airborne=sorted(pending))
        raise RuntimeError(f"not airborne after takeoff: {sorted(pending)}")
    recorder.record("setup", step="takeoff", status="airborne")


def run_config() -> dict:
    return {
        "script": "test_multi_drone_live.py", "scene": SCENE, "vehicles": VEHICLES,
        "plan": {"altitude_m": ALTITUDE, "fan_out_offset_m": FAN_OUT, "turn_to_deg": TURN_TO,
                 "speed_mps": SPEED, "climb_speed_mps": CLIMB_SPEED,
                 "descend_above_launch_m": DESCEND_ABOVE_LAUNCH,
                 "sync_before": ["descend"], "steps": STEP_NAMES},
        "thresholds": {"min_separation_m": MIN_SEPARATION_M, "min_coverage": MIN_COVERAGE,
                       "max_gap_s": MAX_GAP_S, "separation_sample_hz": SEPARATION_SAMPLE_HZ},
        "executor": asdict(ExecutorConfig()),
        "pipeline": {"observation_source": "projectairsim_ground_truth", "attack_hooks": [],
                     "trust_assessor": NoTrustAssessor.name, "action_gate": BypassGate.name,
                     "agents": "ScriptedMissionAgent",
                     "observation_sample_hz": OBSERVATION_HZ,
                     "max_observation_age_s": MAX_OBSERVATION_AGE_S,
                     "agent_feedback": "ActionFeedback (no ground truth)"},
    }


async def main() -> int:
    phase = {"name": "connect"}
    recorder = EvaluationRecorder(run_config=run_config())
    print(f"Recording to {recorder.path}")
    outcome = "aborted"
    agents, reports, separation = {}, [], None
    try:
        with ProjectAirSimAdapter(vehicle_ids=VEHICLES, scene=SCENE) as adapter:
            await asyncio.sleep(0.5)

            def sim_clock():
                poses = adapter.latest_poses()
                return max(p[0] for p in poses.values()) if poses else None
            recorder.set_sim_clock(sim_clock)
            recorder.record("connected", validation_limits=getattr(adapter, "limits", None))

            executor = ProjectAirSimExecutor(adapter, on_event=recorder.recorder_for(source="executor"))
            descend_sync = PhaseBarrier(VEHICLES, name="all returned")

            def on_step(vid, label):
                phase.setdefault("steps", {})[vid] = label
            feeds = {}
            for vid in VEHICLES:
                feed = feeds[vid] = ObservationFeed(adapter, vid, hooks=[],
                                                    assessor=NoTrustAssessor(), recorder=recorder)
                port = ActionPort(vid, executor, feed, gate=BypassGate(), recorder=recorder,
                                  max_observation_age_s=MAX_OBSERVATION_AGE_S)
                agents[vid] = ScriptedMissionAgent(vid, feed, port, mission_steps(vid, descend_sync),
                                                   recorder=recorder, on_step=on_step)

            stop = asyncio.Event()
            tracker = SeparationTracker(VEHICLES)
            mon = asyncio.create_task(monitor(adapter, stop, phase))
            sep = asyncio.create_task(sample_truth(adapter, tracker, recorder, stop, phase))
            samplers = []
            try:
                # Homes come from each agent's own observation ON THE GROUND. (Pass 5.5
                # first live run captured them after takeoff, at ~4 m, so "descend to
                # launch + 1.5 m" became 5.5 m and Drone1's tiny descent never happened.)
                phase["name"] = "home observations"
                for vid in VEHICLES:
                    if agents[vid].capture_home() is None:
                        raise RuntimeError(f"{vid}: no home observation")
                # the regular observation stream: on from before takeoff to after landing
                samplers = [asyncio.create_task(feeds[v].sample(OBSERVATION_HZ, stop))
                            for v in VEHICLES]
                phase["name"] = "takeoff (all)"
                await takeoff_all(adapter, recorder)
                phase["name"] = "missions"
                # If one mission crashes, the others are cancelled (each executor command
                # hovers on cancel) and awaited BEFORE shutdown starts commanding the drones.
                await run_all_or_cancel(agents[v].run() for v in VEHICLES)
            finally:
                phase["name"] = "safe_shutdown (all)"
                phase["steps"] = {}
                reports = await asyncio.gather(
                    *(safe_shutdown(adapter.drone(v), adapter=adapter, vehicle_id=v,
                                    log=lambda m, v=v: print(f"[{v}] {m}"))
                      for v in VEHICLES),
                    return_exceptions=True)
                for v, rep in zip(VEHICLES, reports):
                    recorder.record("shutdown", vehicle_id=v,
                                    report=repr(rep) if isinstance(rep, Exception) else rep)
                stop.set()
                await mon
                await sep
                for t in samplers:
                    await t
                tracker.finish()      # counts a gap at the end (pose stream died before shutdown)
                collisions = {v: adapter.collision_log(v) for v in VEHICLES}
                separation = assess(tracker, collisions, VEHICLES, MIN_SEPARATION_M,
                                    min_coverage=MIN_COVERAGE, max_gap_s=MAX_GAP_S)
                recorder.record("separation", status=separation.status,
                                violations=separation.violations,
                                coverage_problems=separation.coverage_problems,
                                pairs=[{k: v for k, v in asdict(p).items()
                                        if k not in ("last_ts", "last_counted_host")}
                                       | {"coverage": p.coverage} for p in separation.pairs])

        print("\n================ RESULTS ================")
        ok = True
        for vid in VEHICLES:
            a = agents[vid]
            print(f"{vid}:  ({a.ended})")
            for name, r in a.results:      # what the agent was told (ActionFeedback)
                mark = "OK  " if r.status == CommandStatus.SUCCEEDED else "FAIL"
                print(f"  {mark} {name:<8} {r.status.value:<9} {r.elapsed_s:6.2f}s  "
                      f"{r.code}: {r.message}")
            done = ([n for n, _ in a.results] == STEP_NAMES
                    and all(r.status == CommandStatus.SUCCEEDED for _, r in a.results))
            ok = ok and done
        for vid, rep in zip(VEHICLES, reports):
            state = rep if isinstance(rep, Exception) else ("landed+disarmed" if rep.disarmed else "UNRESOLVED")
            print(f"  shutdown {vid}: {state}")
            ok = ok and not isinstance(rep, Exception) and rep.disarmed
        print()
        print(format_report(separation))
        ok = ok and separation.passed
        outcome = "pass" if ok else "fail"
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    except BaseException as err:
        recorder.record("error", error=f"{type(err).__name__}: {err}")
        raise
    finally:
        recorder.close(outcome, separation=separation.status if separation else None,
                       agents={v: a.ended for v, a in agents.items()})
        print(f"Run log: {recorder.path}  ({sum(recorder.counts.values())} events, outcome {outcome})")
        try:   # evaluation view: corrections, ground-truth end points, separation
            print("\n================ RUN LOG SUMMARY ================")
            print(summarize(recorder.path))
        except Exception as err:
            print(f"(summary failed: {type(err).__name__}: {err})")
        print(f"Summary:  python -m atal.evaluation.summarize_run \"{recorder.path}\"")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
