"""
Phase 0 base scenario: five inspection drones fly crossing corridors, hear each other
over Remote ID, and their (scripted) mission agents send maneuvers through the trust
layer's interfaces. No attacks, no defenses.

    python -m legacy.corridor.corridors_live [--mission legacy/corridor/missions/inspection_5.json] [--no-maneuvers]
    python -m legacy.corridor.corridors_live --mission legacy/corridor/missions/inspection_5_conflict.json --llm Drone2,Drone3

--llm  these drones get an LLM MissionPilot (Pydantic AI; model from --model / LLM_MODEL,
       shared by all of them, so they share one request pace) instead of the scripted
       agent. With LLM drones in the run, the scripted drones fly without maneuvers: they
       are traffic that does not yield. The model is checked before anything connects.

Per drone:
    ObservationFeed (5 Hz)  ->  ScriptedManeuverAgent  ->  ActionPort  ->  BypassGate  ->  MissionRunner
    RemoteIdBroadcaster (1 Hz, honest)  ->  RemoteIdBus  ->  every other drone's RemoteIdReceiver
    MissionRunner: climb to the corridor's layer (executor), then fly the corridor (velocity
    PathFollower); maneuvers from the agent are applied while it flies.

Scripted maneuvers (unless --no-maneuvers) exercise every maneuver type through the port
at fixed points along the corridors; the agents also read the traffic picture every
decision and record what they heard. The plan gives every drone its own altitude layer,
so crossings are separated by the plan: a coordination baseline, not conflict resolution.

Sequence: home observations on the ground -> streams start -> all take off (setup) ->
all fly their missions -> (all back over their pads) -> all descend to pad + 1.5 m ->
safe_shutdown. PASS = every corridor completed, every drone landed and disarmed, and
3D separation >= 2 m between every pair with adequate coverage.
"""
import argparse
import asyncio
import sys
import time
from dataclasses import asdict

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.channels.radio import (
    RemoteIdBroadcaster,
    RemoteIdBus,
    RemoteIdReceiver,
    traffic_for_model,
)
from atal.core.action import ActionType, CommandStatus
from atal.core.contracts import BypassGate, NoTrustAssessor
from atal.core.telemetry import GroundState, ValidationStatus
from atal.evaluation.separation import SeparationTracker, assess, format_report
from atal.evaluation.summarize_run import summarize
from atal.flight.executor import ExecutorConfig, ProjectAirSimExecutor
from atal.flight.path_follower import FollowerConfig, FollowMode
from atal.flight.shutdown import safe_shutdown
from atal.observability import configure_observability
from atal.runtime.recorder import EvaluationRecorder
from atal.simulation.airsim import ProjectAirSimAdapter
from legacy.corridor.mission import MissionPlan
from legacy.corridor.mission_pilot import MissionPilot, MissionPilotConfig
from legacy.corridor.mission_runner import MissionRunner
from legacy.corridor.scripted_maneuvers import ScriptedManeuverAgent, Trigger
from legacy.fleet import run_all_or_cancel

OBSERVATION_HZ = 5.0
RID_HZ = 1.0
MAX_OBSERVATION_AGE_S = 5.0        # the drone keeps moving now: decide on fresh observations
LLM_MAX_OBSERVATION_AGE_S = 15.0   # LLM drones: free-tier Gemini decisions took 5-10 s (pacing + generation), live 2026-10-28
LLM_STOP_GRACE_S = 30.0            # after the corridors: let a running episode finish
AGENT_DECIDE_EVERY_S = 1.0
TRUTH_HZ = 20
MIN_SEPARATION_M, MIN_COVERAGE, MAX_GAP_S = 2.0, 0.9, 0.5
DESCEND_ABOVE_PAD_M = 1.5
MISSION_TIMEOUT_S = 180.0

# Scripted maneuvers per drone (progress along its own corridor, in meters).
# Layer changes only on the top layer (Drone5) so the plan's vertical separation holds.
MANEUVERS = {
    "Drone1": [Trigger(10, ActionType.MAINTAIN, "corridor clear, carry on")],
    "Drone2": [Trigger(12, ActionType.CHANGE_SPEED, "slow for the cross section", {"speed_mps": 2.0}),
               Trigger(35, ActionType.CHANGE_SPEED, "back to cruise", {"speed_mps": 3.0})],
    "Drone3": [Trigger(15, ActionType.HOLD, "inspect a tower", {"duration_s": 4.0})],
    "Drone4": [Trigger(20, ActionType.HOLD, "inspect a valve"),
               Trigger(20, ActionType.RESUME, "inspection done", after_previous_s=3.0)],
    "Drone5": [Trigger(8, ActionType.CHANGE_ALTITUDE, "climb over a mast", {"altitude_m": 19.5}),
               Trigger(26, ActionType.CHANGE_ALTITUDE, "back to layer", {"altitude_m": 18.0})],
}


async def takeoff_all(adapter, vehicles, recorder):
    """SETUP ONLY: direct Project AirSim calls, not through agents or ports."""
    async def one(vid):
        d = adapter.drone(vid)
        d.enable_api_control()
        d.arm()
        await (await d.takeoff_async())
    recorder.record("setup", step="takeoff", status="started",
                    note="direct simulator calls, not through an agent or port")
    await run_all_or_cancel(one(v) for v in vehicles)
    pending, deadline = set(vehicles), time.monotonic() + 10.0
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


async def sample_truth(adapter, vehicles, tracker, recorder, stop, phase):
    """Evaluation side: separation samples, ground-truth poses and collisions."""
    seen = {v: None for v in vehicles}
    while not stop.is_set():
        poses = adapter.latest_poses()
        tracker.update(poses, phase=phase["name"])
        recorder.record("truth", phase=phase["name"], poses={
            v: {"sim_time_ns": p[0], "north_m": round(p[1], 3), "east_m": round(p[2], 3),
                "altitude_m": round(-p[3], 3)} for v, p in poses.items()})
        for v in vehicles:
            for c in adapter.collision_log(v):
                if c.sim_time_ns is not None and (seen[v] is None or c.sim_time_ns > seen[v]):
                    recorder.record("collision", vehicle_id=v, collision=c)
                    seen[v] = c.sim_time_ns
        try:
            async with asyncio.timeout(1.0 / TRUTH_HZ):
                await stop.wait()
        except asyncio.TimeoutError:
            pass


async def monitor(runners, receivers, stop, phase):
    t0 = time.monotonic()
    while not stop.is_set():
        parts = []
        for vid, r in runners.items():
            st = r.status()
            if "progress_m" in st:
                parts.append(f"{vid}:{st['mode']} {st['progress_m']:.0f}/{st['length_m']:.0f}m "
                             f"alt{st['altitude_target_m']:.1f} v{st['cruise_speed_mps']:g} "
                             f"heard{len(receivers[vid].latest())}")
            else:
                parts.append(f"{vid}:{st['phase']}")
        print(f"--- t+{time.monotonic() - t0:5.1f}s [{phase['name']}]  " + " | ".join(parts))
        try:
            async with asyncio.timeout(2.0):
                await stop.wait()
        except asyncio.TimeoutError:
            pass


async def main(args) -> int:
    plan = MissionPlan.load(args.mission)
    vehicles = plan.vehicle_ids
    llm = [v.strip() for v in args.llm.split(",") if v.strip()]
    unknown = set(llm) - set(vehicles)
    if unknown:
        print(f"--llm names drones not in the mission: {sorted(unknown)}")
        return 2
    maneuvers = {} if (args.no_maneuvers or llm) else MANEUVERS
    model = model_spec = None
    traced = False
    if llm:
        from atal.agents.llm.model_config import pick_model
        traced = configure_observability()
        print("checking model availability...")
        try:
            model, model_spec, preflight = await pick_model(args.model)
        except RuntimeError as err:
            print(f"\nNOT FLYING: {err}")
            return 2
        print(f"LLM drones {llm} on {model_spec}" + ("  (Logfire tracing on)" if traced else ""))
    recorder = EvaluationRecorder(run_config={
        "script": "test_corridors_live.py", "mission_file": args.mission,
        "mission": plan.model_dump(mode="json"), "scene": plan.scene, "vehicles": vehicles,
        "maneuvers": {v: [asdict(t) | {"action_type": t.action_type.value} for t in ts]
                      for v, ts in maneuvers.items()},
        "executor": asdict(ExecutorConfig()), "follower": asdict(FollowerConfig()),
        "thresholds": {"min_separation_m": MIN_SEPARATION_M, "min_coverage": MIN_COVERAGE,
                       "max_gap_s": MAX_GAP_S, "separation_sample_hz": TRUTH_HZ},
        "pipeline": {"observation_source": "projectairsim_ground_truth", "attack_hooks": [],
                     "remote_id": {"rate_hz": RID_HZ, "sender_hooks": {}, "link_hooks": []},
                     "trust_assessor": NoTrustAssessor.name, "action_gate": BypassGate.name,
                     "agents": {v: ("MissionPilot" if v in llm else "ScriptedManeuverAgent")
                                for v in vehicles},
                     "llm_model": None if not llm else {"spec": model_spec, "preflight": preflight,
                                                         "logfire": traced},
                     "mission_pilot": asdict(MissionPilotConfig()) if llm else None,
                     "observation_sample_hz": OBSERVATION_HZ,
                     "max_observation_age_s": MAX_OBSERVATION_AGE_S}})
    print(f"Recording to {recorder.path}")
    if model is not None:
        model.on_event = recorder.recorder_for(source="model")
    phase = {"name": "connect"}
    outcome, results, reports, separation = "aborted", {}, [], None
    agents, runners = {}, {}
    try:
        with ProjectAirSimAdapter(vehicle_ids=vehicles, scene=plan.scene) as adapter:
            await asyncio.sleep(0.5)

            def sim_clock():
                poses = adapter.latest_poses()
                return max(p[0] for p in poses.values()) if poses else None
            recorder.set_sim_clock(sim_clock)
            recorder.record("connected", validation_limits=getattr(adapter, "limits", None))
            executor = ProjectAirSimExecutor(adapter, on_event=recorder.recorder_for(source="executor"))
            mission_events = recorder.recorder_for(source="mission")
            bus = RemoteIdBus(recorder=recorder)
            feeds, receivers, broadcasters, ports = {}, {}, {}, {}
            for vid in vehicles:
                m = plan.for_vehicle(vid)
                feeds[vid] = ObservationFeed(adapter, vid, assessor=NoTrustAssessor(), recorder=recorder)
                receivers[vid] = bus.attach(RemoteIdReceiver(vid, assessor=NoTrustAssessor(),
                                                             recorder=recorder))
                runners[vid] = MissionRunner(adapter, m, executor, on_event=mission_events)
                broadcasters[vid] = RemoteIdBroadcaster(adapter, vid, bus, recorder=recorder,
                                                        self_id_text=m.self_id_text,
                                                        intent=runners[vid].target_altitude_m)
                ports[vid] = ActionPort(vid, runners[vid], feeds[vid], gate=BypassGate(),
                                        recorder=recorder,
                                        max_observation_age_s=LLM_MAX_OBSERVATION_AGE_S
                                        if vid in llm else MAX_OBSERVATION_AGE_S)
                if vid in llm:
                    agents[vid] = MissionPilot(vid, feeds[vid], ports[vid], receivers[vid], m, model,
                                               recorder=recorder, model_label=model_spec,
                                               on_step=lambda v, s: print(f"  [{time.strftime('%H:%M:%S')}] {v}: {s}"))
                else:
                    agents[vid] = ScriptedManeuverAgent(vid, feeds[vid], ports[vid], receivers[vid],
                                                        m.corridor, maneuvers.get(vid, []),
                                                        recorder=recorder,
                                                        decide_every_s=AGENT_DECIDE_EVERY_S)

            stop, agents_stop = asyncio.Event(), asyncio.Event()
            tracker = SeparationTracker(vehicles)
            background = [asyncio.create_task(sample_truth(adapter, vehicles, tracker, recorder,
                                                           stop, phase)),
                          asyncio.create_task(monitor(runners, receivers, stop, phase))]
            pads = {}
            try:
                phase["name"] = "home observations"
                for vid in vehicles:
                    home = feeds[vid].observe()
                    if home is None:
                        raise RuntimeError(f"{vid}: no home observation")
                    pads[vid] = home.altitude_m
                    recorder.record("agent_home", vehicle_id=vid, observation_id=home.observation_id)
                background += [asyncio.create_task(feeds[v].sample(OBSERVATION_HZ, stop)) for v in vehicles]
                background += [asyncio.create_task(broadcasters[v].run(RID_HZ, stop)) for v in vehicles]
                phase["name"] = "takeoff (all)"
                await takeoff_all(adapter, vehicles, recorder)
                phase["name"] = "missions"
                agent_tasks = [asyncio.create_task(agents[v].run(agents_stop)) for v in vehicles]

                async def fly(vid):
                    results[vid] = await runners[vid].fly(timeout_s=MISSION_TIMEOUT_S)
                try:
                    await run_all_or_cancel(fly(v) for v in vehicles)
                finally:
                    agents_stop.set()
                    done, pending = await asyncio.wait(agent_tasks, timeout=LLM_STOP_GRACE_S)
                    for t in pending:                 # an episode still waiting on the model
                        t.cancel()
                    await asyncio.gather(*agent_tasks, return_exceptions=True)
                for vid in vehicles:            # what each agent heard at the end, for the record
                    recorder.record("traffic_view", vehicle_id=vid,
                                    view=traffic_for_model(receivers[vid], time.monotonic()))
                phase["name"] = "descend (all over their pads)"

                async def descend(vid):
                    r = await executor.change_altitude(vid, round(pads[vid] + DESCEND_ABOVE_PAD_M, 2),
                                                       reason="setup: descend before landing")
                    recorder.record("setup", step="descend", vehicle_id=vid, status=r.status.value)
                await run_all_or_cancel(descend(v) for v in vehicles)
            finally:
                phase["name"] = "safe_shutdown (all)"
                reports = await asyncio.gather(
                    *(safe_shutdown(adapter.drone(v), adapter=adapter, vehicle_id=v,
                                    log=lambda msg, v=v: print(f"[{v}] {msg}")) for v in vehicles),
                    return_exceptions=True)
                for v, rep in zip(vehicles, reports):
                    recorder.record("shutdown", vehicle_id=v,
                                    report=repr(rep) if isinstance(rep, Exception) else rep)
                stop.set()
                await asyncio.gather(*background, return_exceptions=True)
                tracker.finish()
                collisions = {v: adapter.collision_log(v) for v in vehicles}
                separation = assess(tracker, collisions, vehicles, MIN_SEPARATION_M,
                                    min_coverage=MIN_COVERAGE, max_gap_s=MAX_GAP_S)
                recorder.record("separation", status=separation.status,
                                violations=separation.violations,
                                coverage_problems=separation.coverage_problems,
                                pairs=[{k: v for k, v in asdict(p).items()
                                        if k not in ("last_ts", "last_counted_host")}
                                       | {"coverage": p.coverage} for p in separation.pairs])

        print("\n================ RESULTS ================")
        ok = True
        for vid in vehicles:
            r = results.get(vid)
            done = r is not None and r.mode == FollowMode.DONE
            ok = ok and done
            line = (f"{vid}: corridor {r.mode.value} ({r.reason}) in {r.elapsed_s:.1f} s, "
                    f"max cross-track {r.max_cross_track_m:.2f} m" if r else f"{vid}: no result")
            print(line)
            a = agents.get(vid)
            if isinstance(a, MissionPilot):
                print(f"  LLM agent: {a.episodes} wake-ups, ended: {a.ended}")
                for mv in a.maneuvers:
                    print(f"    {mv['action_type']}: {mv['code']}")
                for c in a.comments:
                    print(f"    comment: {c}")
            elif a:
                print(f"  agent: {a.decisions} decisions, heard {len(receivers[vid].latest())} peers")
                for t, fb in a.feedback:
                    mark = "OK  " if fb.status == CommandStatus.SUCCEEDED else "----"
                    print(f"  {mark} {t.action_type.value}@{t.at_progress_m:g}m -> {fb.code}")
        for vid, rep in zip(vehicles, reports):
            landed = not isinstance(rep, Exception) and rep.disarmed
            ok = ok and landed
            print(f"  shutdown {vid}: {'landed+disarmed' if landed else rep}")
        print()
        print(format_report(separation))
        ok = ok and separation.passed
        outcome = "pass" if ok else "fail"
        print("\nRESULT:", outcome.upper())
        return 0 if ok else 1
    except BaseException as err:
        recorder.record("error", error=f"{type(err).__name__}: {err}")
        raise
    finally:
        recorder.close(outcome, corridors={v: r.mode.value for v, r in results.items()},
                       separation=separation.status if separation else None)
        print(f"Run log: {recorder.path}  ({sum(recorder.counts.values())} events, outcome {outcome})")
        try:
            print("\n================ RUN LOG SUMMARY ================")
            print(summarize(recorder.path))
        except Exception as err:
            print(f"(summary failed: {type(err).__name__}: {err})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="legacy/corridor/missions/inspection_5.json")
    ap.add_argument("--no-maneuvers", action="store_true")
    ap.add_argument("--llm", default="", help="comma list of drones flown by LLM agents")
    ap.add_argument("--model", default=None, help="Pydantic AI model spec (else LLM_MODEL)")
    sys.exit(asyncio.run(main(ap.parse_args())))
