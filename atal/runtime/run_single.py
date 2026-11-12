"""
Search and rescue runner: one drone, N seeded runs, one command.
Phase 1 (default): the fixed-order planner, no LLM. Phase 2 (--planner llm): an LLM picks
each next sector (atal/agents/llm_chooser.py), checked by the invariant guard.

    python run_sar_mission.py                                   # deck mission, live sim
    python run_sar_mission.py --mission bundled:missions/sar/mission_dev.yaml \\
                              --scenario bundled:missions/sar/scenario_dev.yaml   # small grid, live
    python run_sar_mission.py --offline --speedup 50            # kinematic sim, no AirSim
    python run_sar_mission.py --offline --runs 1 --full-sweep   # keep sweeping after the find

    # Phase 2: LLM planner (LLM_MODEL in .env, or --model); guard in the gate slot
    python run_sar_mission.py --planner llm --mission bundled:missions/sar/mission_dev.yaml \
                              --scenario bundled:missions/sar/scenario_dev.yaml --runs 1
    python run_sar_mission.py --planner llm --model ollama:qwen3.5:9b --thinking off ...
    python run_sar_mission.py --planner llm --model scripted --offline   # no model server

Logfire: with LOGFIRE_TOKEN in .env, each run, each decision and each model request is
traced (atal/observability.py). The JSONL run log is written either way.

Per drone (deck slide 12; every block is plain code):
    mission.yaml -> Mission (Pydantic) -> SearchLoop + FixedChooser (Voronoi cells, fixed order)
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
    run_NN.summary.json   metrics (atal/evaluation/sar_metrics.py)
    run_NN.svg            top-down plot (atal/evaluation/plot_tracks.py)
    batch_summary.json    all runs + the Phase 1 verdict
PASS (deck slide 11, tightened after the 11/02 audit; run_failures() lists the reasons) =
the victim was detected, the search did not end in a failure, the return home reached home,
the drone landed inside the geofence with no impact, the run log is complete and no
background task died; with --full-sweep every sector must also be swept and the measured
footprint coverage must reach --min-coverage (default 95%).

Live: the scene (--scene, default atal/configs/scenes/scene_sar_one_drone.jsonc) must spawn the
drone at the scenario's start and the area must be clear of buildings at the mission
altitude. The runner checks the spawn against the scenario start before flying.
"""
import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from atal.agents.search_loop import planner_name
from atal.core.contracts import BypassGate, NoTrustAssessor
from atal.core.mission import PlanFileError, load_mission
from atal.core.paths import ConfigNotFound, runs_dir
from atal.core.scenario import load_scenario
from atal.evaluation.evidence import write_manifest
from atal.evaluation.plot_tracks import plot_run
from atal.evaluation.run_log import read_run
from atal.evaluation.sar_metrics import sar_metrics
from atal.flight.executor import ProjectAirSimExecutor
from atal.observability import span
from atal.runtime import rates
from atal.runtime.acceptance import FULL_SWEEP_MIN_COVERAGE, single_run_failures
from atal.runtime.build_drone import build_single_drone
from atal.runtime.context import RunContext
from atal.runtime.landing import descend_before_landing, shut_down, takeoff
from atal.runtime.llm_setup import prepare_llm
from atal.runtime.rates import Timing
from atal.runtime.recorder import EvaluationRecorder
from atal.runtime.simulator import find_scene, open_simulator
from atal.runtime.watchers import sample_truth, stop_background, watch_mission_time
from atal.trust_layer.invariant_guard import SarInvariantGate


async def main(args) -> int:
    """A batch: load the plan, fly N runs, judge each one, write the summary."""
    try:
        mission = load_mission(args.mission)
        scenario = load_scenario(args.scenario, mission)
        if not args.offline:
            find_scene(args.scene)                 # fail before connecting, not after
    except (PlanFileError, ConfigNotFound) as err:
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
        print_run_line(results[-1])
    for m in results:
        m["fail_reasons"] = single_run_failures(m, args.full_sweep, args.min_coverage)
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
    manifest = write_manifest(
        out_dir, argv=sys.argv, args=vars(args),
        files={"mission": args.mission, "scenario": args.scenario,
               **({} if args.offline else {"scene": args.scene})},
        model=llm["config"] if llm else None, results=results, passed=passed, runs_planned=runs)
    print_batch(results, passed, best, runs, manifest, out_dir)
    return 0 if all(passed) else 1


async def run_one(args, mission, scenario, run_index, out_dir: Path, timing: Timing,
                  log=print, llm=None) -> dict:
    with span("sar run {run_index}", run_index=run_index, mission=mission.mission_id,
              planner=args.planner, model=llm["label"] if llm else None):
        return await _run_one(args, mission, scenario, run_index, out_dir, timing, log, llm)


async def _run_one(args, mission, scenario, run_index, out_dir, timing, log, llm) -> dict:
    """One run, start to finish. Returns the run's metrics."""
    spec = scenario.drones[0]
    vid = spec.vehicle
    victim = scenario.victim_for_run(mission, run_index)
    stem = out_dir / f"run_{run_index:02d}"
    recorder = EvaluationRecorder(path=stem.with_suffix(".jsonl"), run_config=run_config(
        args, mission, scenario, run_index, timing, llm))
    log(f"\n=== run {run_index} (seed {scenario.run_seed(run_index)}): victim at "
        f"({victim.x_m:g}, {victim.y_m:g}) in {mission.grid.sector_at(victim.x_m, victim.y_m)}; "
        f"log {recorder.path}")
    outcome, planner_result, shutdown = "aborted", None, None
    bg_errors: list[str] = []
    run_error: str | None = None
    descend_status: str | None = None

    try:
        # 1. Connect to the simulator.
        with open_simulator(args, {vid: (spec.start.x_m, spec.start.y_m)}, timing) as adapter:
            await asyncio.sleep(0.5 / timing.k)

            def sim_clock():
                p = adapter.latest_poses().get(vid)
                return p[0] if p else None
            recorder.set_sim_clock(sim_clock)
            recorder.record("connected", adapter=type(adapter).__name__)
            flight = ProjectAirSimExecutor(adapter, config=timing.executor(),
                                           on_event=recorder.recorder_for(source="executor"))
            ctx = RunContext(args, adapter, recorder, flight, timing, log, llm, stem)

            # 2. Build the drone.
            drone = build_single_drone(ctx, spec, mission, victim,
                                       gate_on=gate_mode(args) == "invariants")

            # 3. Start the background tasks: observations, the victim sensor, the truth
            #    record, the mission time limit.
            def impact(_vid, info):
                if drone.planner.on_impact(info):
                    drone.sector_executor.interrupt("impact")
            background = [
                ("feed", asyncio.create_task(
                    drone.feed.sample(timing.hz(rates.OBSERVATION_HZ), ctx.stop))),
                ("sensor", asyncio.create_task(
                    drone.sensor.run(timing.hz(rates.SENSOR_HZ), ctx.stop))),
                ("truth", asyncio.create_task(sample_truth(
                    adapter, {vid: mission}, recorder, ctx.stop, 1.0 / timing.hz(rates.TRUTH_HZ),
                    lambda _vid: drone.sector_executor.interrupt("geofence"), impact))),
                ("time_watch", asyncio.create_task(watch_mission_time(
                    adapter, vid, mission, drone.planner, drone.sector_executor, ctx.stop,
                    timing.poll(1.0))))]

            # 4. Fly: take off, search until the planner is done and home, descend.
            # 5. Always: land safely, then stop the background tasks.
            try:
                await takeoff(adapter, vid, recorder, timeout_s=30.0)
                planner_result = await drone.planner.run()
                descend_status = await descend_before_landing(ctx, drone)
            finally:
                shutdown = await shut_down(ctx, drone)
                bg_errors += await stop_background(ctx, background)
        outcome = planner_result.outcome if planner_result else "aborted"
    except BaseException as err:
        run_error = f"{type(err).__name__}: {err}"
        recorder.record("error", error=run_error)
        if not isinstance(err, Exception):
            raise
        log(f"  run {run_index} ERROR: {type(err).__name__}: {err}")
    finally:
        recorder.close(outcome)

    # 6. Compute the metrics from the run log and save them.
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


def run_config(args, mission, scenario, run_index, timing, llm) -> dict:
    """The first record of the run log: everything needed to know what was run."""
    spec = scenario.drones[0]
    victim = scenario.victim_for_run(mission, run_index)
    return {
        "script": "run_sar_mission.py", "phase": 2 if llm else 1, "mode": "offline" if args.offline else "live",
        "speedup": timing.k, "mission_file": args.mission, "scenario_file": args.scenario,
        "mission": mission.model_dump(mode="json"), "derived": mission.derived(),
        "scenario_id": scenario.scenario_id, "run_index": run_index,
        "seed": scenario.run_seed(run_index), "vehicle": spec.vehicle, "team_id": spec.id,
        "start": spec.start.model_dump(), "victim": victim.model_dump(),
        "victim_sector": mission.grid.sector_at(victim.x_m, victim.y_m),
        "full_sweep_override": args.full_sweep,
        "scene": None if args.offline else args.scene,
        "obstacles": (getattr(args, "obstacle", None) or []) if args.offline else None,
        "pipeline": {"planner": planner_name("llm" if llm else "fixed", team=False),
                     "allocation": "voronoi", "trust_assessor": NoTrustAssessor.name,
                     "action_gate": (SarInvariantGate.name if gate_mode(args) == "invariants"
                                     else BypassGate.name),
                     "model": llm["config"] if llm else None,
                     "observation_hz": rates.OBSERVATION_HZ, "sensor_hz": rates.SENSOR_HZ,
                     "truth_hz": rates.TRUTH_HZ, "detection": "footprint_geometry"},
        "executor": asdict(timing.executor()), "follower": asdict(timing.follower())}


def gate_mode(args) -> str:
    """--gate auto: the invariant guard with the LLM planner, none (Phase 1 baseline) with
    the fixed planner. Comparisons of the two planners should pass --gate invariants."""
    if args.gate != "auto":
        return args.gate
    return "invariants" if args.planner == "llm" else "none"


def print_run_line(m: dict) -> None:
    print(f"  -> {m['outcome']}: found={m['victim_found']} time_to_find={m['time_to_find_s']} s, "
          f"{m['sectors_swept']}/{m['sector_count']} sectors, coverage {m['area_coverage']:.1%}, "
          f"landed={m['landed']}"
          + (f", IMPACTS {m['impacts']} ({', '.join(m['impact_objects'])})" if m["impacts"] else "")
          + (f"; {m['llm_decisions']} decisions (mean {m['decision_latency_mean_s']} s), "
             f"{m['guard_rejections']} refused, {m['fallbacks']} fallbacks"
             if m.get("llm_decisions") else ""))


def print_batch(results, passed, best_streak, runs, manifest, out_dir) -> None:
    print("\n================ BATCH ================")
    print(f"{'run':>3} {'outcome':<13} {'found':<5} {'victim':<6} {'t_find_s':>9} {'sectors':>7} "
          f"{'coverage':>8} {'landed':<6} pass")
    for m, p in zip(results, passed):
        print(f"{m['run_index']:>3} {m['outcome']:<13} {str(m['victim_found']):<5} "
              f"{str(m['victim_sector']):<6} {str(m['time_to_find_s']):>9} "
              f"{m['sectors_swept']:>3}/{m['sector_count']:<3} {m['area_coverage']:>8.1%} "
              f"{str(m['landed']):<6} {'PASS' if p else 'FAIL: ' + '; '.join(m['fail_reasons'])}")
    print(f"evidence: {out_dir / 'manifest.json'} (commit {manifest['git']['commit_short']}"
          f"{' DIRTY' if manifest['git']['dirty'] else ''})")
    print(f"\nlongest streak of passing runs: {best_streak} of {runs}"
          f"\nRESULT: {'PASS' if all(passed) else 'FAIL'}   ({out_dir / 'batch_summary.json'})")


def parse_obstacle(text: str) -> dict:
    """--obstacle NAME:N_MIN,N_MAX,E_MIN,E_MAX,TOP_M (offline kinematic sim only): an
    axis-aligned box the drone cannot fly through below TOP_M, for testing obstacle
    recovery."""
    try:
        name, nums = text.split(":", 1)
        n0, n1, e0, e1, top = (float(x) for x in nums.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--obstacle wants NAME:N_MIN,N_MAX,E_MIN,E_MAX,TOP_M, "
                                         f"got {text!r}")
    if n0 >= n1 or e0 >= e1:
        raise argparse.ArgumentTypeError(f"--obstacle {text!r}: min must be below max")
    return {"name": name, "n_min": n0, "n_max": n1, "e_min": e0, "e_max": e1, "top_m": top}


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
    ap.add_argument("--mission", help="bundled:missions/... or a path to your own file",
                    default="bundled:missions/sar/mission.yaml")
    ap.add_argument("--scenario", help="bundled:missions/... or a path to your own file",
                    default="bundled:missions/sar/scenario.yaml")
    ap.add_argument("--runs", type=int, default=None, help="override scenario.runs")
    ap.add_argument("--offline", action="store_true", help="kinematic sim instead of AirSim")
    ap.add_argument("--speedup", type=float, default=50.0, help="offline sim time / host time")
    ap.add_argument("--full-sweep", action="store_true",
                    help="keep sweeping after the detection (checks 'sweeps every sector')")
    ap.add_argument("--scene", default="bundled:sim_config/scene_sar_one_drone.jsonc", 
                    help="live scene: bundled:sim_config/<file> or a path to your own .jsonc")
    ap.add_argument("--obstacle", type=parse_obstacle, action="append", default=None,
                    help="offline only: NAME:N_MIN,N_MAX,E_MIN,E_MAX,TOP_M box the drone "
                         "cannot fly through (repeatable), to exercise obstacle recovery")
    ap.add_argument("--out", default=str(runs_dir() / "sar"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
