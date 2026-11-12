"""
One drone flown by an LLM pilot agent in Project AirSim.

    python -m legacy.llm_pilot.llm_pilot_live [--harness pydantic|native] [--model SPEC] [--task "..."]

    --harness pydantic (default)  Pydantic AI, any model: --model / LLM_MODEL, e.g.
                                  "ollama:qwen3.5:9b" or "google:gemini-3.5-flash-lite"
    --harness native              this repo's own Gemini loop (GEMINI_MODEL, bare names)

Run legacy.llm_pilot.check_llm first (same agent loop, fake drone, no simulator). Keys and model
settings go in the repo's .env (see README). Logfire tracing turns on if
LOGFIRE_TOKEN is set.

Pipeline (same interfaces as the scripted agents):
    adapter (ground truth) -> ObservationFeed (5 Hz) -> pilot tools -> model
    model -> submit_action -> ActionPort -> BypassGate (no gate) -> executor -> drone
    the model gets ActionFeedback only; everything is recorded in runs/run_<id>.jsonl

Setup and shutdown are NOT the agent's: takeoff is a recorded setup step (direct
calls), and safe_shutdown lands and disarms the drone afterwards whatever the
agent did. The model is told flight limits but nothing enforces them in this run
(no gate): this is an integration milestone, not a protected run.

PASS = the agent finished with a comment, at least one action succeeded, and the
drone landed and disarmed. The final comment and every action are printed, and
the run summary follows.
"""
import argparse
import asyncio
import sys
import time
from dataclasses import asdict

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.core.telemetry import GroundState, ValidationStatus
from atal.evaluation.recorder import EvaluationRecorder
from atal.evaluation.summarize_run import summarize
from atal.flight.executor import ExecutorConfig, ProjectAirSimExecutor
from atal.flight.shutdown import safe_shutdown
from atal.observability import configure_observability
from atal.simulation.airsim import ProjectAirSimAdapter
from atal.trust_layer.action_gate import BypassGate
from atal.trust_layer.assessor import NoTrustAssessor
from legacy.llm_pilot.llm_pilot import FlightRules, PilotLimits
from legacy.llm_pilot.pilot_setup import HARNESSES, prepare_pilot

VEHICLE = "Drone1"
SCENE = "scene_basic_drone.jsonc"
OBSERVATION_HZ = 5.0
# 30 s, not the port's 10 s default: model replies took 10-19 s in the first live run. The
# port only accepts observations taken after the previous command ended (the drone is holding
# position), so an observation this old still describes where the drone is.
MAX_OBSERVATION_AGE_S = 30.0
TRUTH_HZ = 10.0
TASK = ("Climb to 5 m above the launch point's altitude. Fly to a point 10 m north of the launch "
        "point at that altitude. Turn to face east (heading 90). Then fly back to directly above "
        "the launch point at the same altitude, and give your final comment.")
LIMITS = PilotLimits(max_model_turns=40, max_submissions=12, model_timeout_s=60.0,
                     time_budget_s=600.0)
RULES = FlightRules()


async def monitor(adapter, stop, phase):
    t0 = time.monotonic()
    while not stop.is_set():
        try:
            s = adapter.get_snapshot(VEHICLE)
            line = (f"N={s.position_ned_m.x:6.1f} E={s.position_ned_m.y:6.1f} "
                    f"alt={s.altitude_local_m:5.1f} hdg={s.heading_deg:5.1f} "
                    f"gs={s.ground_speed_mps:4.1f} {s.ground_state.value}")
        except Exception as err:
            line = f"snapshot error: {err}"
        print(f"--- t+{time.monotonic() - t0:5.1f}s  [{phase['name']}]  {line}")
        try:
            await asyncio.wait_for(stop.wait(), 2.0)
        except asyncio.TimeoutError:
            pass


async def record_truth(adapter, recorder, stop, phase):
    """Evaluation side only: ground-truth pose and collisions while the agent flies."""
    last_collision = None
    while not stop.is_set():
        p = adapter.latest_poses().get(VEHICLE)
        if p:
            recorder.record("truth", phase=phase["name"], poses={VEHICLE: {
                "sim_time_ns": p[0], "north_m": round(p[1], 3), "east_m": round(p[2], 3),
                "altitude_m": round(-p[3], 3)}})
        for c in adapter.collision_log(VEHICLE):
            if c.sim_time_ns is not None and (last_collision is None or c.sim_time_ns > last_collision):
                recorder.record("collision", vehicle_id=VEHICLE, collision=c)
                last_collision = c.sim_time_ns
        try:
            await asyncio.wait_for(stop.wait(), 1.0 / TRUTH_HZ)
        except asyncio.TimeoutError:
            pass


async def takeoff(adapter, recorder):
    """SETUP ONLY: direct Project AirSim calls, not through the agent or its port."""
    recorder.record("setup", step="takeoff", status="started",
                    note="direct simulator calls, not through an agent or port")
    d = adapter.drone(VEHICLE)
    d.enable_api_control()
    d.arm()
    await (await d.takeoff_async())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        s = adapter.get_snapshot(VEHICLE)
        if s.validation_status == ValidationStatus.VALID and s.ground_state == GroundState.AIRBORNE:
            recorder.record("setup", step="takeoff", status="airborne")
            return
        await asyncio.sleep(0.1)
    recorder.record("setup", step="takeoff", status="failed")
    raise RuntimeError("not airborne after takeoff")


async def main(args) -> int:
    traced = configure_observability()
    # Preflight BEFORE connecting to the simulator: nothing takes off unless a model answers.
    print(f"harness {args.harness}; checking model availability...")
    try:
        setup = await prepare_pilot(args.harness, args.model)
    except RuntimeError as err:
        print(f"\nNOT FLYING: {err}")
        return 2
    task = args.task or TASK
    print(f"model {setup.label}" + ("  (Logfire tracing on)" if traced else ""))
    recorder = EvaluationRecorder(run_config={
        "script": "test_llm_pilot_live.py", "scene": SCENE, "vehicles": [VEHICLE], "task": task,
        "model": {**setup.config, "preflight": setup.preflight, "logfire": traced},
        "limits": asdict(LIMITS), "rules_told_to_model": asdict(RULES),
        "executor": asdict(ExecutorConfig()),
        "pipeline": {"observation_source": "projectairsim_ground_truth", "attack_hooks": [],
                     "trust_assessor": NoTrustAssessor.name, "action_gate": BypassGate.name,
                     "agents": f"{setup.harness} pilot ({setup.label})",
                     "observation_sample_hz": OBSERVATION_HZ,
                     "max_observation_age_s": MAX_OBSERVATION_AGE_S,
                     "agent_feedback": "ActionFeedback (no ground truth)"}})
    print(f"Recording to {recorder.path}")
    setup.attach_recorder(recorder)
    phase = {"name": "connect"}
    outcome, out, report = "aborted", None, None
    try:
        with ProjectAirSimAdapter(vehicle_ids=[VEHICLE], scene=SCENE) as adapter:
            await asyncio.sleep(0.5)
            recorder.set_sim_clock(lambda: (adapter.latest_poses().get(VEHICLE) or (None,))[0])
            recorder.record("connected", validation_limits=getattr(adapter, "limits", None))
            executor = ProjectAirSimExecutor(adapter, on_event=recorder.recorder_for(source="executor"))
            feed = ObservationFeed(adapter, VEHICLE, hooks=[], assessor=NoTrustAssessor(),
                                   recorder=recorder)
            port = ActionPort(VEHICLE, executor, feed, gate=BypassGate(), recorder=recorder,
                              max_observation_age_s=MAX_OBSERVATION_AGE_S)

            def on_step(vid, label):
                phase["name"] = f"agent: {label}"
                print(f"  [{time.strftime('%H:%M:%S')}] agent: {label}")
            pilot = setup.make(VEHICLE, feed, port, task, recorder=recorder, limits=LIMITS,
                               rules=RULES, on_step=on_step)

            stop = asyncio.Event()
            mon = asyncio.create_task(monitor(adapter, stop, phase))
            truth = asyncio.create_task(record_truth(adapter, recorder, stop, phase))
            sampler = None
            try:
                phase["name"] = "home observation"
                if pilot.capture_home() is None:
                    raise RuntimeError("no home observation")
                sampler = asyncio.create_task(feed.sample(OBSERVATION_HZ, stop))
                phase["name"] = "takeoff"
                await takeoff(adapter, recorder)
                out = await pilot.run()
            finally:
                phase["name"] = "safe_shutdown"
                report = await safe_shutdown(adapter.drone(VEHICLE), adapter=adapter,
                                             vehicle_id=VEHICLE, log=lambda m: print(f"[{VEHICLE}] {m}"))
                recorder.record("shutdown", vehicle_id=VEHICLE, report=report)
                stop.set()
                for t in (mon, truth, sampler):
                    if t is not None:
                        await t

        print("\n================ RESULTS ================")
        print(f"agent: {out.ended}; {out.model_turns} model turns; "
              f"{out.succeeded_actions}/{len(out.submissions)} actions succeeded")
        for f in out.submissions:
            print(f"  {f['action_type'] or '?':<18} {f['code']:<24} {f['elapsed_s']:6.2f}s")
        print(f"\nfinal comment:\n{out.comment}\n")
        print(f"shutdown: {'landed+disarmed' if report.disarmed else 'UNRESOLVED'}")
        ok = out.ended == "completed" and out.succeeded_actions >= 1 and report.disarmed
        outcome = "pass" if ok else "fail"
        print("\nRESULT:", outcome.upper())
        return 0 if ok else 1
    except BaseException as err:
        recorder.record("error", error=f"{type(err).__name__}: {err}")
        raise
    finally:
        recorder.close(outcome, agent=out.ended if out else None)
        print(f"Run log: {recorder.path}  ({sum(recorder.counts.values())} events, outcome {outcome})")
        try:
            print("\n================ RUN LOG SUMMARY ================")
            print(summarize(recorder.path))
        except Exception as err:
            print(f"(summary failed: {type(err).__name__}: {err})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=HARNESSES, default="pydantic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default=None)
    sys.exit(asyncio.run(main(ap.parse_args())))
