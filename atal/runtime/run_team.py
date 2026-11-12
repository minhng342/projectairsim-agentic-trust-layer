"""
Search and rescue swarm runner (Phase 3): three honest drones, N seeded runs, one command.
Every drone runs the same design as Phases 1-2 (feed, planner, guard, port, sector
executor, sensor, memory) plus its team role and coordinator
(atal/agents/team_coordinator.py); they talk on one bus.

    python run_sar_swarm.py --offline                     # dev mission, 3 drones, 20 runs
    python run_sar_swarm.py --offline --runs 1 --full-sweep
    python run_sar_swarm.py --offline --planner llm --model scripted     # Pydantic AI agents
    python run_sar_swarm.py --planner llm --model ollama:qwen3.5:9b --thinking off --runs 1
    python run_sar_swarm.py --offline --radio-range 60    # range sensitivity
    python run_sar_swarm.py --offline --peer-clear-rule self_only        # no cooperation

Defaults: missions/sar/mission_dev.yaml + missions/sar/scenario_swarm_dev.yaml (the same
mission and the same per-run victims as the Phase 1/2 dev scenario, for paired comparison).

Per run:
    bus           RemoteIdBus with a radio-range filter (true positions, mission.swarm.
                  radio_range_m): 1 Hz Remote ID position broadcasts (with mission_state)
                  and typed peer messages SectorClaimed / SectorCleared / VictimDetected
    each drone    climbs on its pad, flies to its staging point if the mission has one
                  (swarm.staging), then flies its own layer (altitude_m + i * layer_spacing_m), re-partitions the
                  open sectors (Voronoi + claims) before every sweep, claims a sector before
                  sweeping it, reports it cleared after, reports its own find (repeated);
                  a peer's clear / find is applied under the peer-claim rules
    end           a drone stops when it or (rule accept) a peer found the victim, nothing is
                  left for it, or mission time is up; then it flies home on its layer,
                  descends and lands

Output, per batch: runs/sar_swarm/<batch>/run_NN.jsonl (one log for the whole team),
run_NN.<drone>.memory.json, run_NN.summary.json (atal/evaluation/swarm_metrics.py), run_NN.svg,
batch_summary.json (per-run table and the baseline numbers: found rate, time to find,
coverage, duplicate sweeps, message delivery).

PASS for a run = the victim was found by some drone, every drone's search ended without a
failure, every drone got home and landed inside the geofence with no impact, the log is
complete and no background task died (--full-sweep: also every sector swept by someone
and the team's measured coverage >= --min-coverage).
"""
import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from atal.agents.search_loop import planner_name
from atal.channels.radio import RemoteIdBus
from atal.core.sar import PlanFileError, load_mission, load_scenario
from atal.evaluation.compare import load_batch, paired_comparison
from atal.evaluation.evidence import write_manifest
from atal.evaluation.plot_tracks import plot_run
from atal.evaluation.recorder import REPO_ROOT, EvaluationRecorder, read_run
from atal.evaluation.swarm_metrics import batch_baseline, swarm_metrics
from atal.flight.executor import ProjectAirSimExecutor
from atal.flight.pad_zone import PadZone
from atal.observability import span
from atal.runtime import rates
from atal.runtime.acceptance import FULL_SWEEP_MIN_COVERAGE, team_run_failures
from atal.runtime.airspace import TeamAirspace
from atal.runtime.build_drone import TeamParts, build_team_drone
from atal.runtime.context import RunContext
from atal.runtime.landing import fly_team_drone
from atal.runtime.llm_setup import prepare_llm
from atal.runtime.rates import Timing
from atal.runtime.run_single import parse_obstacle
from atal.runtime.simulator import open_simulator
from atal.runtime.watchers import (
    inject_dropout,
    sample_truth,
    stop_background,
    watch_mission_time,
    watch_sim_clock,
)
from atal.search.voronoi import voronoi_partition
from atal.trust_layer.action_gate import BypassGate
from atal.trust_layer.assessor import NoTrustAssessor
from atal.trust_layer.invariant_guard import SarInvariantGate
from atal.trust_layer.peer_rules import PeerClaimRules


async def main(args) -> int:
    """A batch: load the plan, fly N runs, judge each one, write the summary."""
    try:
        mission = load_mission(args.mission)
        scenario = load_scenario(args.scenario, mission)
    except PlanFileError as err:
        print(err)
        return 2
    mission, scenario = apply_overrides(args, mission, scenario)
    if args.obstacle and not args.offline:
        print("--obstacle is for the offline kinematic sim only (live, the buildings are real)")
        return 2
    reference = None
    if args.reference:
        try:
            reference = load_batch(args.reference)
        except (OSError, ValueError) as err:
            print(f"--reference {args.reference}: {err}")
            return 2
    runs = args.runs or scenario.runs
    timing = Timing(args.speedup if args.offline else 1.0)
    llm = None
    if args.planner == "llm":
        try:
            llm = await prepare_llm(args)
        except RuntimeError as err:
            print(f"no model available, nothing flown: {err}")
            return 2
    batch = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{mission.mission_id}"
             f"-{len(scenario.drones)}drones")
    out_dir = Path(args.out) / batch
    out_dir.mkdir(parents=True, exist_ok=True)
    s = mission.swarm
    print(f"mission {mission.mission_id}: {len(scenario.drones)} drones "
          f"({', '.join(d.vehicle for d in scenario.drones)}), radio range "
          f"{s.radio_range_m if s.radio_range_m else 'unlimited'} m, layers "
          f"{mission.flight.altitude_m:g} m + {s.layer_spacing_m:g} m, peer clears: "
          f"{s.peer_clear_rule}, allocation {s.allocation}; {runs} run(s), "
          f"{'offline x' + format(timing.k, 'g') if args.offline else 'live'}; output {out_dir}")
    results = []
    for i in range(runs):
        m = await run_one(args, mission, scenario, i, out_dir, timing, llm=llm)
        m["fail_reasons"] = team_run_failures(m, args.full_sweep, args.min_coverage)
        results.append(m)
        print_run_line(m)
        if m.get("sim_stalled"):
            print("  the simulator stalled: stopping the batch. Restart CityEnviron, then rerun "
                  "(stalled runs are left out of the baseline).")
            break
    passed = [not m["fail_reasons"] for m in results]
    summary = {"batch": batch, "mission_file": args.mission, "scenario_file": args.scenario,
               "planner": args.planner, "model": llm["label"] if llm else None,
               "drones": [d.vehicle for d in scenario.drones], "swarm": s.model_dump(mode="json"),
               "mode": "offline" if args.offline else "live", "speedup": timing.k,
               "full_sweep": args.full_sweep, "gate": args.gate,
               "baseline": batch_baseline([m for m in results if not m.get("sim_stalled")]),
               "stalled_runs": [m["run_index"] for m in results if m.get("sim_stalled")], "passed": passed,
               "all_passed": all(passed), "runs": results}
    valid = [m for m in results if not m.get("sim_stalled")]
    if reference is not None:
        summary["paired_vs_reference"] = paired_comparison(reference, valid,
                                                           reference_label=args.reference)

    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    # last, so it hashes every file of the batch, batch_summary.json included
    mf = write_manifest(out_dir, argv=sys.argv, args=vars(args),
                        files={"mission": args.mission, "scenario": args.scenario,
                               **({} if args.offline else {"scene": f"sim_config/{args.scene}"})},
                        model=llm["config"] if llm else None, results=results, passed=passed,
                        runs_planned=runs)
    print_batch(args, summary, results, passed, mf, out_dir)
    return 0 if all(passed) else 1


async def run_one(args, mission, scenario, run_index, out_dir, timing, log=print, llm=None):
    with span("sar swarm run {run_index}", run_index=run_index, mission=mission.mission_id,
              planner=args.planner, model=llm["label"] if llm else None):
        return await _run_one(args, mission, scenario, run_index, out_dir, timing, log, llm)


async def _run_one(args, mission, scenario, run_index, out_dir: Path, timing: Timing, log, llm):
    """One run, start to finish. Returns the run's metrics."""
    specs = scenario.drones
    vids = [d.vehicle for d in specs]
    missions = {d.vehicle: mission.for_layer(i) for i, d in enumerate(specs)}   # one layer each
    victim = scenario.victim_for_run(mission, run_index)
    stem = out_dir / f"run_{run_index:02d}"
    recorder = EvaluationRecorder(path=stem.with_suffix(".jsonl"), run_config=run_config(
        args, mission, missions, scenario, run_index, timing, llm))
    log(f"\n=== run {run_index} (seed {scenario.run_seed(run_index)}): victim at "
        f"({victim.x_m:g}, {victim.y_m:g}) in {mission.grid.sector_at(victim.x_m, victim.y_m)}; "
        f"{len(vids)} drones; log {recorder.path}")
    results: dict[str, dict] = {v: {"planner": None, "shutdown": None, "descend": None,
                                    "error": None} for v in vids}
    bg_errors: list[str] = []
    airspace_summary = None
    team_parts: dict = {}
    run_error = None
    outcome = "aborted"

    # 1. Connect to the simulator.
    simulator = open_simulator(args, {d.vehicle: (d.start.x_m, d.start.y_m) for d in specs},
                               timing)
    try:
        with simulator as adapter:
            await asyncio.sleep(0.5 / timing.k)
            recorder.set_sim_clock(lambda: next((p[0] for p in adapter.latest_poses().values()),
                                                None))
            recorder.record("connected", adapter=type(adapter).__name__)
            flight = ProjectAirSimExecutor(adapter, config=timing.executor(),
                                           on_event=recorder.recorder_for(source="executor"))
            ctx = RunContext(args, adapter, recorder, flight, timing, log, llm, stem)

            # 2. Build what the team shares: the radio bus, the pad zone, and (static
            #    allocation only) the pre-mission split of the grid.
            bus = RemoteIdBus(recorder=recorder,
                              in_range=range_filter(adapter, mission.swarm.radio_range_m))
            pads = PadZone.around([(d.start.x_m, d.start.y_m) for d in specs],
                                  mission.swarm.pad_zone_m, recorder=recorder,
                                  sim_clock=lambda: (lambda p: None if p is None else p[0] / 1e9)(
                                      next(iter(adapter.latest_poses().values()), None)))
            recorder.record("pad_zone", center=pads.center, radius_m=round(pads.radius_m, 2))
            static = static_split(mission, specs) if mission.swarm.allocation == "static" else None
            if static is not None:
                recorder.record("static_split", cells=static,
                                sizes={v: len(c) for v, c in static.items()})
            # the trusted transition manager (managed airspace): every vertical move of
            # every drone is a reservation (atal/airspace/, atal/runtime/airspace.py)
            airspace = TeamAirspace.build(mission, vids, recorder, timing)
            shared = TeamParts(bus, pads, static, mission.swarm.staging or [], len(specs),
                               airspace=airspace)

            # 3. Build each drone.
            team = {d.vehicle: build_team_drone(ctx, d, i, missions[d.vehicle], victim,
                                                gate_on=args.gate == "invariants", team=shared)
                    for i, d in enumerate(specs)}
            team_parts = team

            # 4. Start the background tasks.
            background = start_background(ctx, team, mission)
            if airspace is not None:
                airspace.attach(team)
                background.append(("airspace", airspace.survey(adapter, ctx.stop)))

            # 5. Fly every drone at once: take off, search, land one at a time.
            #    Always: stop the background tasks.
            try:
                await asyncio.gather(*(fly_team_drone(ctx, pads, len(team), team[v], results[v],
                                                      airspace=airspace)
                                       for v in vids))
            finally:
                bg_errors += await stop_background(ctx, background)
                recorder.record("bus_stats", **bus.stats)
                if airspace is not None:
                    airspace_summary = airspace.summary()
                    recorder.record("airspace_summary", source="transition_manager",
                                    **airspace_summary)
        ps = [r["planner"] for r in results.values()]
        outcome = ("victim_found" if any(p and p.detection for p in ps)
                   else "searched" if all(ps) else "aborted")
    except BaseException as err:
        run_error = f"{type(err).__name__}: {err}"
        recorder.record("error", error=run_error)
        if not isinstance(err, Exception):
            raise
        log(f"  run {run_index} ERROR: {run_error}")
    finally:
        recorder.close(outcome)

    # 6. Compute the metrics from the run log and save them.
    run_log = read_run(recorder.path)
    metrics = swarm_metrics(run_log, mission)
    for vid, r in results.items():
        d = metrics["drones"].setdefault(vid, {})
        d["landed"] = bool(r["shutdown"] and r["shutdown"].disarmed)
        d["descend"] = r["descend"]
        d["error"] = r["error"]
        d["planner_reason"] = r["planner"].reason if r["planner"] else None
        d["recoveries"] = r["planner"].recoveries if r["planner"] else None
        d["blocked"] = r["planner"].blocked if r["planner"] else []
        d["landing_cleared"] = r.get("landing_cleared")
        d["landing_not_cleared"] = bool(r["shutdown"] and r["shutdown"].landing_not_cleared)
        rec = getattr(team_parts.get(vid), "recovery", None) if team_parts else None
        d["recovery_outcomes"] = rec.summary() if rec is not None else None
    metrics["airspace"] = airspace_summary            # None: unmanaged (labelled)
    metrics["airspace_managed"] = mission.swarm.airspace == "managed"
    metrics["background_errors"] = bg_errors
    metrics["sim_stalled"] = getattr(simulator, "stalled", None)
    metrics["run_error"] = run_error
    metrics["dropout"] = getattr(args, "dropout", None)
    metrics["pad_zone_waits"] = sum(1 for e in run_log.of_kind("pad_zone_enter")
                                    if e.get("was_held_by"))
    metrics["pad_zone_timeouts"] = len(run_log.of_kind("pad_zone_timeout"))
    metrics["contingency_landings"] = {e["vehicle_id"]: e.get("reason")
                                       for e in run_log.of_kind("contingency_landing")}
    metrics["pad_zone_held_unconfirmed"] = [e["vehicle_id"] for e in
                                            run_log.of_kind("pad_zone_held_unconfirmed")]
    metrics["all_landed"] = all(d.get("landed") for d in metrics["drones"].values())
    stem.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    try:
        metrics["plot"] = str(plot_run(recorder.path, stem.with_suffix(".svg")))
    except Exception as err:
        log(f"  (plot failed: {type(err).__name__}: {err})")
    return metrics


def start_background(ctx, team: dict, mission) -> list:
    """Start everything that runs beside the drones. Returns (name, task) pairs."""
    args, adapter, recorder, timing, stop, log = (ctx.args, ctx.adapter, ctx.recorder,
                                                  ctx.timing, ctx.stop, ctx.log)
    tasks = []

    # the dropout condition: one drone is told it has failed, part way through
    drop = getattr(args, "dropout", None)
    if drop:
        if drop["vehicle"] not in team:
            raise RuntimeError(f"--dropout {drop['vehicle']}: not a drone of this run")
        tasks.append(("dropout", asyncio.create_task(
            inject_dropout(ctx, team[drop["vehicle"]], drop["at_s"]))))

    # per drone: observations, victim sensor, Remote ID broadcast, repeating its own find,
    # the mission time limit
    for vid, drone in team.items():
        tasks += [
            (f"{vid}.feed", asyncio.create_task(
                drone.feed.sample(timing.hz(rates.OBSERVATION_HZ), stop))),
            (f"{vid}.sensor", asyncio.create_task(
                drone.sensor.run(timing.hz(rates.SENSOR_HZ), stop))),
            (f"{vid}.remote_id", asyncio.create_task(
                drone.broadcaster.run(timing.hz(rates.REMOTE_ID_HZ), stop))),
            (f"{vid}.victim_repeat", asyncio.create_task(drone.coordinator.repeat_victim(
                timing.poll(mission.swarm.victim_repeat_s), stop))),
            (f"{vid}.time_watch", asyncio.create_task(watch_mission_time(
                adapter, vid, drone.mission, drone.planner, drone.sector_executor, stop,
                timing.poll(1.0))))]

    # live only: end the run if the simulator's clock stops
    if not args.offline:
        def stalled(why):
            log(f"  [{time.strftime('%H:%M:%S')}] SIM STALLED: {why}; ending the run")
            recorder.record("sim_stalled", reason=why)
            for drone in team.values():
                drone.sector_executor.interrupt("sim stalled")
        tasks.append(("sim_clock", asyncio.create_task(watch_sim_clock(
            adapter, stop, rates.SIM_STALL_S, 1.0, stalled))))

    # the truth record: true poses, collisions, geofence
    def impact(vid, info):
        drone = team[vid]
        if drone.planner.on_impact(info):
            log(f"  [{time.strftime('%H:%M:%S')}] {vid}: IMPACT with "
                f"{info.get('object_name')}; stopping to recover")
            drone.sector_executor.interrupt("impact")
    tasks.append(("truth", asyncio.create_task(sample_truth(
        adapter, {vid: drone.mission for vid, drone in team.items()}, recorder, stop,
        1.0 / timing.hz(rates.TRUTH_HZ),
        lambda vid: team[vid].sector_executor.interrupt("geofence"), impact))))
    return tasks


def run_config(args, mission, missions, scenario, run_index, timing, llm) -> dict:
    """The first record of the run log: everything needed to know what was run."""
    victim = scenario.victim_for_run(mission, run_index)
    gate_on = args.gate == "invariants"
    return {
        "script": "run_sar_swarm.py", "phase": 3, "mode": "offline" if args.offline else "live",
        "speedup": timing.k, "mission_file": args.mission, "scenario_file": args.scenario,
        "mission": mission.model_dump(mode="json"), "derived": mission.derived(),
        "scenario_id": scenario.scenario_id, "run_index": run_index,
        "seed": scenario.run_seed(run_index), "victim": victim.model_dump(),
        "victim_sector": mission.grid.sector_at(victim.x_m, victim.y_m),
        "full_sweep_override": args.full_sweep,
        "scene": None if args.offline else args.scene,
        "obstacles": (getattr(args, "obstacle", None) or []) if args.offline else None,
        "dropout": getattr(args, "dropout", None),
        "drones": [{"vehicle": d.vehicle, "team_id": d.id, "start": d.start.model_dump(),
                    "altitude_m": missions[d.vehicle].flight.altitude_m} for d in scenario.drones],
        "swarm": mission.swarm.model_dump(mode="json"),
        "insider_enabled": scenario.insider.enabled,
        "pipeline": {"planner": planner_name("llm" if llm else "fixed", team=True),
                     "allocation": ("static" if mission.swarm.allocation == "static"
                                    else "voronoi_swarm"), "peer_rules": PeerClaimRules.name,
                     "trust_assessor": NoTrustAssessor.name,
                     "action_gate": SarInvariantGate.name if gate_on else BypassGate.name,
                     "model": llm["config"] if llm else None,
                     "observation_hz": rates.OBSERVATION_HZ, "sensor_hz": rates.SENSOR_HZ,
                     "truth_hz": rates.TRUTH_HZ, "remote_id_hz": rates.REMOTE_ID_HZ,
                     "detection": "footprint_geometry"},
        "executor": asdict(timing.executor()), "follower": asdict(timing.follower())}


def range_filter(adapter, radio_range_m):
    """in_range for the bus: TRUE 3D distance between sender and receiver (evaluation side)."""
    if radio_range_m is None:
        return None

    def in_range(sender, receiver):
        poses = adapter.latest_poses()
        a, b = poses.get(sender), poses.get(receiver)
        if a is None or b is None:
            return False, None
        d = math.dist(a[1:], b[1:])
        return d <= radio_range_m, d
    return in_range


def static_split(mission, drones) -> dict[str, list[str]]:
    """The static-allocation condition's pre-mission split: Voronoi of the whole grid over
    the staging points (else the pads), fixed for the run."""
    stage = mission.swarm.staging or []
    gens = {d.vehicle: ((stage[i].x_m, stage[i].y_m) if i < len(stage)
                        else (d.start.x_m, d.start.y_m)) for i, d in enumerate(drones)}
    return voronoi_partition(mission.grid, gens)


def apply_overrides(args, mission, scenario):
    """Command-line options that change the mission or the scenario for this batch."""
    sw = {}
    if args.radio_range is not None:
        sw["radio_range_m"] = None if args.radio_range <= 0 else args.radio_range
    if args.peer_clear_rule:
        sw["peer_clear_rule"] = args.peer_clear_rule
    if args.layer_spacing is not None:
        sw["layer_spacing_m"] = args.layer_spacing
    if args.allocation:
        sw["allocation"] = args.allocation
    if sw:
        mission = mission.model_copy(update={"swarm": mission.swarm.model_copy(update=sw)})
    if args.full_sweep:
        mission = mission.model_copy(update={"termination": mission.termination.model_copy(
            update={"stop_on_detection": False})})
    if args.drones:
        scenario = scenario.model_copy(update={"drones": scenario.drones[:args.drones]})
    return mission, scenario


def print_run_line(m: dict) -> None:
    print(f"  -> found={m['victim_found']} by {m['finder']} t_find={m['time_to_find_s']} s, "
          f"mission {m['mission_time_s']} s, {m['sectors_swept']}/{m['sector_count']} sectors "
          f"(dup {m['duplicate_sweeps']}), coverage {m['area_coverage']:.1%}, msgs "
          f"{m['messages']['peer_sent']} sent / {m['messages']['out_of_range']} out of range"
          + "".join(f"; {v}: {d['impacts']} impact(s) ({', '.join(d['impact_objects'])}), "
                    f"{d['obstacles_recovered']}/{d['obstacle_incidents']} recovered"
                    + (", LOST" if d.get("lost") else "")
                    for v, d in sorted(m["drones"].items())
                    if d.get("impacts") or d.get("obstacle_incidents"))
          + ("" if not m["fail_reasons"] else f"; FAIL: {'; '.join(m['fail_reasons'])}"))


def print_batch(args, summary, results, passed, mf, out_dir) -> None:
    print("\n================ BATCH ================")
    print(f"{'run':>3} {'victim':<6} {'finder':<8} {'t_find_s':>9} {'mission_s':>9} "
          f"{'sectors':>7} {'dup':>3} {'coverage':>8} pass")
    for m, p in zip(results, passed):
        print(f"{m['run_index']:>3} {str(m['victim_sector']):<6} {str(m['finder']):<8} "
              f"{str(m['time_to_find_s']):>9} {str(m['mission_time_s']):>9} "
              f"{m['sectors_swept']:>3}/{m['sector_count']:<3} {m['duplicate_sweeps']:>3} "
              f"{m['area_coverage']:>8.1%} {'PASS' if p else 'FAIL: ' + '; '.join(m['fail_reasons'])}")
    b = summary["baseline"]
    print(f"\nbaseline over {b['runs']} runs: found {b['found']}/{b['runs']}, time to find "
          f"mean {b['time_to_find_s']['mean']} s / median {b['time_to_find_s']['median']} s, "
          f"coverage at the end mean {b['area_coverage']['mean']:.1%}, duplicate sweeps "
          f"{b['duplicate_sweeps_total']}, message delivery {b['delivery_ratio']:.1%}")
    pc = summary.get("paired_vs_reference")
    if pc:
        r = pc["speedup"]
        ci = f" (95% CI {r['ci95'][0]}-{r['ci95'][1]}x)" if r["ci95"] else ""
        print(f"paired vs {args.reference}: {pc['pairs']} pairs, speedup mean {r['mean']}x"
              f"{ci}, median {r['median']}x, faster in "
              f"{pc['faster']}/{pc['pairs']}"
              + (f"; {pc['unpaired']} unpaired" if pc["unpaired"] else ""))
    print(f"evidence: {out_dir / 'manifest.json'} (commit {mf['git']['commit_short']}"
          f"{' DIRTY' if mf['git']['dirty'] else ''}; {mf['counts']['attempted']} attempted, "
          f"{mf['counts']['valid']} valid, {mf['counts']['stalled']} stalled, "
          f"{mf['counts']['passed']} passed)")
    print(f"RESULT: {'PASS' if all(passed) else 'FAIL'}   ({out_dir / 'batch_summary.json'})")


def parse_dropout(text: str) -> dict:
    try:
        vid, at = text.split("@")
        at_s = float(at)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--dropout wants DRONE@SECONDS, got {text!r}")
    if not vid or at_s < 0:
        raise argparse.ArgumentTypeError(f"--dropout {text!r}: needs a drone and seconds >= 0")
    return {"vehicle": vid, "at_s": at_s}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SAR swarm runner (Phase 3): honest drones on a "
                                             "shared bus")
    ap.add_argument("--planner", choices=("fixed", "llm"), default="fixed")
    ap.add_argument("--gate", choices=("none", "invariants"), default="invariants",
                    help="the invariant guard in each drone's port (default on)")
    ap.add_argument("--model", default=None, help="as the one-drone run: SPEC, scripted[:...] "
                    "or test")
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    ap.add_argument("--decision-timeout", type=float, default=60.0)
    ap.add_argument("--mission", default="missions/sar/mission_dev.yaml")
    ap.add_argument("--scenario", default="missions/sar/scenario_swarm_dev.yaml")
    ap.add_argument("--runs", type=int, default=None, help="override scenario.runs")
    ap.add_argument("--drones", type=int, default=None, help="use only the first N drones")
    ap.add_argument("--radio-range", type=float, default=None,
                    help="override swarm.radio_range_m (0 = unlimited)")
    ap.add_argument("--layer-spacing", type=float, default=None,
                    help="override swarm.layer_spacing_m")
    ap.add_argument("--peer-clear-rule", choices=("single", "corroborated", "self_only"),
                    default=None, help="override swarm.peer_clear_rule")
    ap.add_argument("--allocation", choices=("dynamic", "static"), default=None,
                    help="override swarm.allocation (static: fixed split, no claims or clears)")
    ap.add_argument("--reference", default=None,
                    help="batch_summary.json (or its folder) of a reference batch on the same "
                         "victims, e.g. one drone: adds the paired speedup with a 95%% "
                         "bootstrap interval")
    ap.add_argument("--obstacle", type=parse_obstacle, action="append", default=None,
                    help="offline only: NAME:N_MIN,N_MAX,E_MIN,E_MAX,TOP_M box (repeatable)")
    ap.add_argument("--dropout", type=parse_dropout, default=None, metavar="DRONE@SECONDS",
                    help="experiment: DRONE fails SECONDS into the search (drops out and lands "
                         "where it is), e.g. Drone2@120")
    ap.add_argument("--offline", action="store_true", help="kinematic sim instead of AirSim")
    ap.add_argument("--speedup", type=float, default=50.0)
    ap.add_argument("--full-sweep", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=FULL_SWEEP_MIN_COVERAGE)
    ap.add_argument("--scene", default="scene_sar_three_drones.jsonc")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "sar_swarm"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
