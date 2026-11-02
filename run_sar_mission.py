"""
Search and rescue runner: one drone, N seeded runs, one command.
Phase 1 (default): the fixed-order planner, no LLM. Phase 2 (--planner llm): an LLM picks
each next sector (agent/sar_llm_planner.py), checked by the invariant guard.

    python run_sar_mission.py                                   # deck mission, live sim
    python run_sar_mission.py --mission missions/sar/mission_dev.yaml \\
                              --scenario missions/sar/scenario_dev.yaml   # small grid, live
    python run_sar_mission.py --offline --speedup 50            # kinematic sim, no AirSim
    python run_sar_mission.py --offline --runs 1 --full-sweep   # keep sweeping after the find

    # Phase 2: LLM planner (LLM_MODEL in .env, or --model); guard in the gate slot
    python run_sar_mission.py --planner llm --mission missions/sar/mission_dev.yaml \
                              --scenario missions/sar/scenario_dev.yaml --runs 1
    python run_sar_mission.py --planner llm --model ollama:qwen3.5:9b --thinking off ...
    python run_sar_mission.py --planner llm --model scripted --offline   # no model server

Logfire: with LOGFIRE_TOKEN in .env, each run, each decision and each model request is
traced (observability.py). The JSONL run log is written either way.

Per drone (deck slide 12; every block is plain code):
    mission.yaml -> Mission (Pydantic) -> FixedSectorPlanner (Voronoi cells, fixed order)
      -> SweepSector -> ActionPort -> BypassGate -> SectorExecutor (boustrophedon lanes,
         velocity PathFollower) -> simulator
    VictimSensor (footprint geometry, ground truth) -> Detection -> memory + planner
      (+ stops the sweep when termination.stop_on_detection)
    LocalBelief: swept sectors and detections, saved as JSON after every change

Sequence per run: home observation on the ground -> takeoff (setup) -> planner: climb to
mission altitude, sweep sectors in order until every sector is swept, the victim is
detected or max_mission_time_s (sim) is used up -> return home -> descend -> safe_shutdown.

Output, per batch: runs/sar/<batch>/
    run_NN.jsonl          the run log (EvaluationRecorder): config incl. the scenario,
                          plan, every observation / proposal / gate decision / result /
                          feedback, sector events, belief updates, detection, truth track
    run_NN.memory.json    the drone's memory at the end of the run
    run_NN.summary.json   metrics (evaluation/sar_metrics.py)
    run_NN.svg            top-down plot (evaluation/plot_tracks.py)
    batch_summary.json    all runs + the Phase 1 verdict
PASS (deck slide 11, tightened after the 11/02 audit; run_failures() lists the reasons) =
the victim was detected, the search did not end in a failure, the return home reached home,
the drone landed inside the geofence with no impact, the run log is complete and no
background task died; with --full-sweep every sector must also be swept and the measured
footprint coverage must reach --min-coverage (default 95%).

Live: the scene (--scene, default sim_config/scene_sar_one_drone.jsonc) must spawn the
drone at the scenario's start and the area must be clear of buildings at the mission
altitude. The runner checks the spawn against the scenario start before flying.
"""
import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agent.sar_llm_planner import LLMSectorPlanner
from agent.sector_planner import FixedSectorPlanner
from channel.action_port import ActionPort
from channel.observation_feed import ObservationFeed
from evaluation.plot_tracks import plot_run
from evaluation.recorder import REPO_ROOT, EvaluationRecorder, read_run
from evaluation.sar_metrics import sar_metrics
from executor.path_follower import FollowerConfig
from executor.projectairsim_executor import ExecutorConfig, ProjectAirSimExecutor
from executor.sector_executor import SectorExecutor
from gate.action_gate import BypassGate
from gate.sar_guard import SarGuard, SarInvariantGate
from models.belief import LocalBelief
from models.sar import PlanFileError, load_mission, load_scenario
from models.telemetry import GroundState, ValidationStatus
from search.detection import VictimSensor
from trust.assessor import NoTrustAssessor
from utils.flight_safety import safe_shutdown
from observability import configure_observability, span

OBSERVATION_HZ = 1.0          # the planner decides at sector boundaries, not every second
SENSOR_HZ = 20.0
TRUTH_HZ = 2.0
MAX_OBSERVATION_AGE_S = 5.0
DESCEND_ABOVE_LAUNCH_M = 1.5
SPAWN_TOLERANCE_M = 5.0
MAX_HOST_HZ = 250.0           # offline: cap host-side polling rates


class Timing:
    """Host-side rates and polls for live (speedup 1) or the kinematic sim (speedup k)."""

    def __init__(self, speedup: float = 1.0):
        self.k = speedup

    def hz(self, sim_hz):
        return min(sim_hz * self.k, MAX_HOST_HZ) if self.k > 1 else sim_hz

    def poll(self, sim_s):
        return max(sim_s / self.k, 1.0 / MAX_HOST_HZ) if self.k > 1 else sim_s

    def follower(self) -> FollowerConfig:
        return FollowerConfig() if self.k == 1 else FollowerConfig(
            poll_interval_s=self.poll(0.1), hover_timeout_s=5.0)

    def executor(self) -> ExecutorConfig:
        return ExecutorConfig() if self.k == 1 else ExecutorConfig(poll_interval_s=self.poll(0.1))


async def _takeoff(adapter, vid, recorder, timeout_s):
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


async def _sample_truth(adapter, vid, mission, recorder, stop, period_s, on_breach):
    breached = False
    seen = None                                   # last collision sim time recorded
    while not stop.is_set():
        try:
            for c in adapter.collision_log(vid):
                if c.sim_time_ns is not None and (seen is None or c.sim_time_ns > seen):
                    recorder.record("collision", vehicle_id=vid, collision=c,
                                    sim_time_ns=c.sim_time_ns)
                    seen = c.sim_time_ns
        except Exception:
            pass
        poses = adapter.latest_poses()
        p = poses.get(vid)
        if p is not None:
            recorder.record("truth", sim_time_ns=p[0], poses={vid: {
                "sim_time_ns": p[0], "north_m": round(p[1], 3), "east_m": round(p[2], 3),
                "altitude_m": round(-p[3], 3)}})
            out = mission.geofence_excursion_m(p[1], p[2])
            if out > mission.limits.geofence_margin_m and not breached:
                breached = True
                recorder.record("geofence_breach", vehicle_id=vid, excursion_m=round(out, 2),
                                sim_time_ns=p[0])
                on_breach()
        try:
            async with asyncio.timeout(period_s):
                await stop.wait()
        except asyncio.TimeoutError:
            pass


async def _watch_time(adapter, vid, mission, planner, sector_ex, stop, period_s):
    """Mission time is SIM time from the start of the plan (termination.max_mission_time_s)."""
    while not stop.is_set():
        if planner.start_sim_s is not None:
            now = adapter.get_snapshot(vid).sim_time_s
            if now - planner.start_sim_s > mission.termination.max_mission_time_s:
                planner.on_time_up()
                sector_ex.interrupt("mission time limit")
                return
        try:
            async with asyncio.timeout(period_s):
                await stop.wait()
        except asyncio.TimeoutError:
            pass


async def run_one(args, mission, scenario, run_index, out_dir: Path, timing: Timing,
                  log=print, llm=None) -> dict:
    with span("sar run {run_index}", run_index=run_index, mission=mission.mission_id,
              planner=args.planner, model=llm["label"] if llm else None):
        return await _run_one(args, mission, scenario, run_index, out_dir, timing, log, llm)


async def _run_one(args, mission, scenario, run_index, out_dir, timing, log, llm) -> dict:
    drone = scenario.drones[0]
    vid = drone.vehicle
    victim = scenario.victim_for_run(mission, run_index)
    victim_sector = mission.grid.sector_at(victim.x_m, victim.y_m)
    stem = out_dir / f"run_{run_index:02d}"
    recorder = EvaluationRecorder(path=stem.with_suffix(".jsonl"), run_config={
        "script": "run_sar_mission.py", "phase": 2 if llm else 1, "mode": "offline" if args.offline else "live",
        "speedup": timing.k, "mission_file": args.mission, "scenario_file": args.scenario,
        "mission": mission.model_dump(mode="json"), "derived": mission.derived(),
        "scenario_id": scenario.scenario_id, "run_index": run_index,
        "seed": scenario.run_seed(run_index), "vehicle": vid, "team_id": drone.id,
        "start": drone.start.model_dump(), "victim": victim.model_dump(),
        "victim_sector": victim_sector, "full_sweep_override": args.full_sweep,
        "scene": None if args.offline else args.scene,
        "pipeline": {"planner": (LLMSectorPlanner if llm else FixedSectorPlanner).name,
                     "allocation": "voronoi", "trust_assessor": NoTrustAssessor.name,
                     "action_gate": (SarInvariantGate.name if gate_mode(args) == "invariants"
                                     else BypassGate.name),
                     "model": llm["config"] if llm else None,
                     "observation_hz": OBSERVATION_HZ, "sensor_hz": SENSOR_HZ,
                     "truth_hz": TRUTH_HZ, "detection": "footprint_geometry"},
        "executor": asdict(timing.executor()), "follower": asdict(timing.follower())})
    log(f"\n=== run {run_index} (seed {scenario.run_seed(run_index)}): victim at "
        f"({victim.x_m:g}, {victim.y_m:g}) in {victim_sector}; log {recorder.path}")
    outcome, planner_result, shutdown = "aborted", None, None
    bg_errors: list[str] = []
    run_error: str | None = None
    descend_status: str | None = None

    if args.offline:
        from adapters.kinematic_sim import KinematicSim
        adapter_cm = KinematicSim({vid: (drone.start.x_m, drone.start.y_m)}, speedup=timing.k)
    else:
        from adapters.projectairsim_adapter import ProjectAirSimAdapter
        adapter_cm = ProjectAirSimAdapter(vehicle_ids=[vid], scene=args.scene)
    try:
        with adapter_cm as adapter:
            await asyncio.sleep(0.5 / timing.k)
            def sim_clock():
                p = adapter.latest_poses().get(vid)
                return p[0] if p else None
            recorder.set_sim_clock(sim_clock)
            recorder.record("connected", adapter=type(adapter).__name__)
            flight = ProjectAirSimExecutor(adapter, config=timing.executor(),
                                           on_event=recorder.recorder_for(source="executor"))
            feed = ObservationFeed(adapter, vid, assessor=NoTrustAssessor(), recorder=recorder)
            home = feed.observe()
            if home is None:
                raise RuntimeError(f"{vid}: no home observation")
            home_xy = (home.north_m, home.east_m)
            off = ((home_xy[0] - drone.start.x_m) ** 2 + (home_xy[1] - drone.start.y_m) ** 2) ** 0.5
            recorder.record("agent_home", vehicle_id=vid, observation_id=home.observation_id,
                            home={"x_m": home_xy[0], "y_m": home_xy[1]},
                            offset_from_scenario_start_m=round(off, 2))
            if off > SPAWN_TOLERANCE_M:
                raise RuntimeError(f"{vid} spawned at ({home_xy[0]:.1f}, {home_xy[1]:.1f}), "
                                   f"{off:.1f} m from the scenario start; fix the scene or "
                                   f"scenario before flying")
            launch_alt = home.altitude_m
            belief = LocalBelief(vid, mission, path=stem.with_suffix(".memory.json"),
                                 on_change=recorder.recorder_for(source="belief"))
            sector_ex = SectorExecutor(adapter, vid, mission, home_xy, flight,
                                       follower_config=timing.follower(),
                                       on_event=recorder.recorder_for(source="mission"))
            # The planner-side guard (the LLM's output validator; the fixed planner's choice
            # under the guard) and the execution-side gate are separate switches: --gate
            # none bypasses only the gate. The LLM planner always validates its own output.
            guard = SarGuard(mission, belief, home_xy) if (
                llm or gate_mode(args) == "invariants") else None
            gate = SarInvariantGate(guard) if gate_mode(args) == "invariants" else BypassGate()
            port = ActionPort(vid, sector_ex, feed, gate=gate, recorder=recorder,
                              max_observation_age_s=MAX_OBSERVATION_AGE_S)
            if llm:
                if hasattr(llm["model"], "on_event"):
                    llm["model"].on_event = recorder.recorder_for(source="model")
                planner = LLMSectorPlanner(vid, mission, feed, port, belief, model=llm["model"],
                                           guard=guard, model_label=llm["label"],
                                           model_settings=llm["settings"],
                                           decision_timeout_s=args.decision_timeout,
                                           recorder=recorder,
                                           sweep_progress=sector_ex.last_progress, log=log)
            else:
                planner = FixedSectorPlanner(vid, mission, feed, port, belief, recorder=recorder,
                                             sweep_progress=sector_ex.last_progress, log=log,
                                             guard=guard)

            def detected(d):
                belief.add_detection(d)
                planner.on_detection(d)
                log(f"  victim detected by {vid} in {d.sector_id} at sim {d.sim_time_s:.1f} s")
                if mission.termination.stop_on_detection:
                    sector_ex.interrupt("victim detected")
            sensor = VictimSensor(adapter, vid, (victim.x_m, victim.y_m), mission.sensor.footprint_m,
                                  mission.flight.altitude_m, grid=mission.grid,
                                  on_detection=detected, recorder=recorder)

            stop = asyncio.Event()
            bg = [asyncio.create_task(feed.sample(timing.hz(OBSERVATION_HZ), stop)),
                  asyncio.create_task(sensor.run(timing.hz(SENSOR_HZ), stop)),
                  asyncio.create_task(_sample_truth(adapter, vid, mission, recorder, stop,
                                                    1.0 / timing.hz(TRUTH_HZ),
                                                    lambda: sector_ex.interrupt("geofence"))),
                  asyncio.create_task(_watch_time(adapter, vid, mission, planner, sector_ex,
                                                  stop, timing.poll(1.0)))]
            try:
                await _takeoff(adapter, vid, recorder, timeout_s=30.0)
                planner_result = await planner.run()
                r = await flight.change_altitude(vid, round(launch_alt + DESCEND_ABOVE_LAUNCH_M, 2),
                                                 reason="setup: descend before landing")
                recorder.record("setup", step="descend", vehicle_id=vid, status=r.status.value)
                descend_status = r.status.value
            finally:
                shutdown = await safe_shutdown(adapter.drone(vid), adapter=adapter, vehicle_id=vid,
                                               log=lambda m: log(f"[{vid}] {m}"),
                                               confirm_poll_s=timing.poll(0.1))
                recorder.record("shutdown", vehicle_id=vid, report=shutdown)
                stop.set()
                for name, res in zip(("feed", "sensor", "truth", "time_watch"),
                                     await asyncio.gather(*bg, return_exceptions=True)):
                    if isinstance(res, BaseException) and not isinstance(
                            res, asyncio.CancelledError):
                        bg_errors.append(f"{name}: {type(res).__name__}: {res}")
                        recorder.record("background_error", task=name,
                                        error=f"{type(res).__name__}: {res}")
        outcome = planner_result.outcome if planner_result else "aborted"
    except BaseException as err:
        run_error = f"{type(err).__name__}: {err}"
        recorder.record("error", error=run_error)
        if not isinstance(err, Exception):
            raise
        log(f"  run {run_index} ERROR: {type(err).__name__}: {err}")
    finally:
        recorder.close(outcome)
    metrics = sar_metrics(read_run(recorder.path), mission)
    metrics["landed"] = bool(shutdown and shutdown.disarmed)
    metrics["planner_reason"] = planner_result.reason if planner_result else None
    metrics["background_errors"] = bg_errors
    metrics["run_error"] = run_error
    metrics["descend"] = descend_status
    stem.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    try:
        metrics["plot"] = str(plot_run(recorder.path, stem.with_suffix(".svg")))
    except Exception as err:
        log(f"  (plot failed: {type(err).__name__}: {err})")
    return metrics


FULL_SWEEP_MIN_COVERAGE = 0.95   # measured footprint coverage a full sweep must reach


def run_failures(m: dict, full_sweep: bool,
                 min_coverage: float = FULL_SWEEP_MIN_COVERAGE) -> list[str]:
    """Why a run does not pass (empty = PASS). Finding the victim does not excuse a failure
    later in the run: the search, the return home, the landing and the evidence must all be
    sound."""
    why = []
    if not m["victim_found"]:
        why.append("victim not found")
    if m.get("ended_by") in ("failed", "aborted", None):
        why.append(f"search ended by {m.get('ended_by')}")
    if m.get("return_home") != "home_reached":
        why.append(f"return home: {m.get('return_home')}")
    if not m["landed"]:
        why.append("not landed")
    if m["geofence_breaches"]:
        why.append(f"{m['geofence_breaches']} geofence breaches")
    if m["impacts"]:
        why.append(f"{m['impacts']} impacts")
    if not m.get("run_complete"):
        why.append("run log incomplete")
    if "descend" in m and m["descend"] != "succeeded":
        why.append(f"descend before landing: {m['descend']}")
    if m.get("run_error"):
        why.append(f"run error: {m['run_error']}")
    if m.get("run_log_outcome") in (None, "aborted"):
        why.append(f"run log ended {m.get('run_log_outcome')}")
    if m.get("background_errors"):
        why.append("background task failed")
    if full_sweep:
        if m["sectors_swept"] != m["sector_count"]:
            why.append(f"{m['sectors_swept']}/{m['sector_count']} sectors swept")
        if m["area_coverage"] < min_coverage:
            why.append(f"measured coverage {m['area_coverage']:.1%} < {min_coverage:.0%}")
    return why


def run_passes(m: dict, full_sweep: bool, min_coverage: float = FULL_SWEEP_MIN_COVERAGE) -> bool:
    return not run_failures(m, full_sweep, min_coverage)


def gate_mode(args) -> str:
    """--gate auto: the invariant guard with the LLM planner, none (Phase 1 baseline) with
    the fixed planner. Comparisons of the two planners should pass --gate invariants."""
    if args.gate != "auto":
        return args.gate
    return "invariants" if args.planner == "llm" else "none"


async def main(args) -> int:
    try:
        mission = load_mission(args.mission)
        scenario = load_scenario(args.scenario, mission)
    except PlanFileError as err:
        print(err)
        return 2
    if len(scenario.drones) != 1:
        print(f"Phase 1 flies one drone; {args.scenario} lists {len(scenario.drones)}")
        return 2
    if args.full_sweep:
        mission = mission.model_copy(update={"termination": mission.termination.model_copy(
            update={"stop_on_detection": False})})
    runs = args.runs or scenario.runs
    timing = Timing(args.speedup if args.offline else 1.0)
    llm = None
    if args.planner == "llm":
        try:
            llm = await prepare_llm(args)
        except RuntimeError as err:
            print(f"no model available, nothing flown: {err}")
            return 2
    batch = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{mission.mission_id}"
    out_dir = Path(args.out) / batch
    out_dir.mkdir(parents=True, exist_ok=True)
    d = mission.derived()
    print(f"mission {mission.mission_id}: {d['sector_count']} sectors of {d['sector_size_m'][0]:g} x "
          f"{d['sector_size_m'][1]:g} m, lanes {d['lane_spacing_m']:g} m apart, "
          f"{d['passes_per_sector']} passes per sector; {runs} run(s), "
          f"{'offline x' + format(timing.k, 'g') if args.offline else 'live'}; output {out_dir}")
    results = []
    for i in range(runs):
        results.append(await run_one(args, mission, scenario, i, out_dir, timing, llm=llm))
        m = results[-1]
        print(f"  -> {m['outcome']}: found={m['victim_found']} time_to_find={m['time_to_find_s']} s, "
              f"{m['sectors_swept']}/{m['sector_count']} sectors, coverage {m['area_coverage']:.1%}, "
              f"landed={m['landed']}"
              + (f", IMPACTS {m['impacts']} ({', '.join(m['impact_objects'])})" if m["impacts"] else "")
              + (f"; {m['llm_decisions']} decisions (mean {m['decision_latency_mean_s']} s), "
                 f"{m['guard_rejections']} refused, {m['fallbacks']} fallbacks"
                 if m.get("llm_decisions") else ""))
    for m in results:
        m["fail_reasons"] = run_failures(m, args.full_sweep, args.min_coverage)
    passed = [not m["fail_reasons"] for m in results]
    streak = best = 0
    for p in passed:
        streak = streak + 1 if p else 0
        best = max(best, streak)
    found = [m["time_to_find_s"] for m in results if m["time_to_find_s"] is not None]
    summary = {"batch": batch, "mission_file": args.mission, "scenario_file": args.scenario,
               "planner": args.planner, "model": llm["label"] if llm else None,
               "mode": "offline" if args.offline else "live", "speedup": timing.k,
               "full_sweep": args.full_sweep, "min_coverage": args.min_coverage,
               "gate": gate_mode(args), "runs": results, "passed": passed,
               "longest_pass_streak": best, "all_passed": all(passed),
               "time_to_find_s": {"mean": round(sum(found) / len(found), 2) if found else None,
                                  "min": min(found, default=None), "max": max(found, default=None)}}
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n================ BATCH ================")
    print(f"{'run':>3} {'outcome':<13} {'found':<5} {'victim':<6} {'t_find_s':>9} {'sectors':>7} "
          f"{'coverage':>8} {'landed':<6} pass")
    for m, p in zip(results, passed):
        print(f"{m['run_index']:>3} {m['outcome']:<13} {str(m['victim_found']):<5} "
              f"{str(m['victim_sector']):<6} {str(m['time_to_find_s']):>9} "
              f"{m['sectors_swept']:>3}/{m['sector_count']:<3} {m['area_coverage']:>8.1%} "
              f"{str(m['landed']):<6} {'PASS' if p else 'FAIL: ' + '; '.join(m['fail_reasons'])}")
    print(f"\nlongest streak of passing runs: {best} of {runs}"
          f"\nRESULT: {'PASS' if all(passed) else 'FAIL'}   ({out_dir / 'batch_summary.json'})")
    return 0 if all(passed) else 1


async def prepare_llm(args) -> dict:
    """Pick and ping the model BEFORE anything connects to the simulator."""
    observed = configure_observability()
    print(f"Logfire tracing {'on' if observed else 'off (no LOGFIRE_TOKEN)'}")
    settings = {}
    spec = args.model or ""
    if args.thinking != "default":
        settings["thinking"] = args.thinking == "on"
    from agent.model_config import model_specs
    if args.thinking == "off" and model_specs(args.model)[0].startswith("ollama:"):
        # Pydantic AI does not translate `thinking` for Ollama; Ollama's OpenAI-compatible
        # endpoint takes reasoning_effort "none" (and its native API `think: false`).
        settings["extra_body"] = {"reasoning_effort": "none", "think": False}
    if spec == "test":
        # Pydantic AI's TestModel: no LLM, answers with schema-valid but meaningless actions
        # (sector "a"), so every decision exercises the guard's refusal and the fallback.
        from pydantic_ai.models.test import TestModel
        model = TestModel()
        return {"model": model, "label": "test", "settings": {},
                "config": {"model": "test", "harness": "pydantic-ai"}}
    if spec == "scripted" or spec.startswith("scripted:"):
        from agent.sar_scripted_model import scripted_search_model
        policy = spec.split(":", 1)[1] if ":" in spec else "nearest"
        model = scripted_search_model(policy)
        return {"model": model, "label": model.model_name, "settings": {},
                "config": {"model": model.model_name, "harness": "pydantic-ai"}}
    from agent.model_config import pick_model
    model, label, tried = await pick_model(args.model)
    return {"model": model, "label": label, "settings": settings,
            "config": {"model": label, "harness": "pydantic-ai", "preflight": tried,
                       "model_settings": settings, "logfire": observed,
                       "rpm": getattr(getattr(model, "limiter", None), "rpm", None)}}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SAR runner: one drone; fixed planner (Phase 1) "
                                             "or LLM planner (Phase 2)")
    ap.add_argument("--planner", choices=("fixed", "llm"), default="fixed")
    ap.add_argument("--gate", choices=("auto", "none", "invariants"), default="auto",
                    help="auto: invariants with --planner llm, none with fixed; use invariants "
                         "for both when comparing planners")
    ap.add_argument("--min-coverage", type=float, default=FULL_SWEEP_MIN_COVERAGE,
                    help="measured footprint coverage a --full-sweep run must reach")
    ap.add_argument("--model", default=None, help="LLM (Pydantic AI model string, default "
                    "LLM_MODEL from .env); 'scripted[:nearest|prior|stubborn|broken]' or 'test' "
                    "(Pydantic AI TestModel) need no model server")
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default",
                    help="model thinking/reasoning (off is much faster on qwen3.5)")
    ap.add_argument("--decision-timeout", type=float, default=60.0,
                    help="seconds per LLM decision before the fallback is used")
    ap.add_argument("--mission", default="missions/sar/mission.yaml")
    ap.add_argument("--scenario", default="missions/sar/scenario.yaml")
    ap.add_argument("--runs", type=int, default=None, help="override scenario.runs")
    ap.add_argument("--offline", action="store_true", help="kinematic sim instead of AirSim")
    ap.add_argument("--speedup", type=float, default=50.0, help="offline sim time / host time")
    ap.add_argument("--full-sweep", action="store_true",
                    help="keep sweeping after the detection (checks 'sweeps every sector')")
    ap.add_argument("--scene", default="scene_sar_one_drone.jsonc", help="sim_config scene (live)")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "sar"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
