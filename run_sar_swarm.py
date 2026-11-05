"""
Search and rescue swarm runner (Phase 3): three honest drones, N seeded runs, one command.
Every drone runs the same design as Phases 1-2 (feed, planner, guard, port, sector
executor, sensor, memory) plus its swarm member (agent/swarm.py); they talk on one bus.

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
run_NN.<drone>.memory.json, run_NN.summary.json (evaluation/swarm_metrics.py), run_NN.svg,
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

from agent.swarm import (PeerMessenger, SwarmCoordinator, SwarmFixedPlanner, SwarmLLMPlanner,
                         SwarmReceiver)
from channel.action_port import ActionPort
from channel.observation_feed import ObservationFeed
from channel.remote_id import RemoteIdBroadcaster, RemoteIdBus, RemoteIdReceiver
from evaluation.plot_tracks import plot_run
from evaluation.recorder import REPO_ROOT, EvaluationRecorder, read_run
from evaluation.swarm_metrics import batch_baseline, swarm_metrics
from executor.projectairsim_executor import ProjectAirSimExecutor
from executor.sector_executor import SectorExecutor
from gate.action_gate import BypassGate
from gate.peer_rules import PeerClaimRules
from gate.sar_guard import SarGuard, SarInvariantGate
from models.belief import LocalBelief
from models.sar import PlanFileError, load_mission, load_scenario
from observability import span
from run_sar_mission import (DESCEND_ABOVE_LAUNCH_M, FULL_SWEEP_MIN_COVERAGE, MAX_OBSERVATION_AGE_S,
                             OBSERVATION_HZ, SENSOR_HZ, SPAWN_TOLERANCE_M, TRUTH_HZ, Timing,
                             _takeoff, prepare_llm)
from search.detection import VictimSensor
from trust.assessor import NoTrustAssessor
from utils.flight_safety import safe_shutdown

REMOTE_ID_HZ = 1.0
SIM_STALL_S = 30.0          # live: host seconds with no sim-clock progress before a run is ended


async def _sample_truth(adapter, vids, missions, recorder, stop, period_s, on_breach):
    breached, seen = set(), {}
    while not stop.is_set():
        for vid in vids:
            try:
                for c in adapter.collision_log(vid):
                    if c.sim_time_ns is not None and (vid not in seen or c.sim_time_ns > seen[vid]):
                        recorder.record("collision", vehicle_id=vid, collision=c,
                                        sim_time_ns=c.sim_time_ns)
                        seen[vid] = c.sim_time_ns
            except Exception:
                pass
        poses = adapter.latest_poses()
        snap = {}
        for vid in vids:
            p = poses.get(vid)
            if p is None:
                continue
            snap[vid] = {"sim_time_ns": p[0], "north_m": round(p[1], 3),
                         "east_m": round(p[2], 3), "altitude_m": round(-p[3], 3)}
            m = missions[vid]
            out = m.geofence_excursion_m(p[1], p[2])
            if out > m.limits.geofence_margin_m and vid not in breached:
                breached.add(vid)
                recorder.record("geofence_breach", vehicle_id=vid, excursion_m=round(out, 2),
                                sim_time_ns=p[0])
                on_breach(vid)
        if snap:
            recorder.record("truth", sim_time_ns=max(v["sim_time_ns"] for v in snap.values()),
                            poses=snap)
        try:
            async with asyncio.timeout(period_s):
                await stop.wait()
        except asyncio.TimeoutError:
            pass


async def _watch_sim_clock(adapter, stop, stall_s, period_s, on_stall, clock=time.monotonic):
    """Live only: if the sim clock (pose push topic) has not moved for stall_s of host
    time, mark the adapter stalled (every later snapshot raises SimStalled) and stop the
    sweeps, so the run winds down instead of waiting on a frozen sim. A pull request that
    never answers is caught by the adapter itself (request_timeout_s)."""
    last_ns, since = None, clock()
    while not stop.is_set():
        ns = max((p[0] for p in adapter.latest_poses().values()), default=None)
        if ns is not None and ns != last_ns:
            last_ns, since = ns, clock()
        elif clock() - since > stall_s and not getattr(adapter, "stalled", None):
            adapter.stalled = (f"sim clock stopped at {(last_ns or 0) / 1e9:.1f} s for "
                               f"{clock() - since:.0f} s")
            on_stall(adapter.stalled)
            return
        if getattr(adapter, "stalled", None):
            on_stall(adapter.stalled)
            return
        try:
            async with asyncio.timeout(period_s):
                await stop.wait()
        except asyncio.TimeoutError:
            pass


async def _watch_time(adapter, vid, mission, planner, sector_ex, stop, period_s):
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


async def run_one(args, mission, scenario, run_index, out_dir, timing, log=print, llm=None):
    with span("sar swarm run {run_index}", run_index=run_index, mission=mission.mission_id,
              planner=args.planner, model=llm["label"] if llm else None):
        return await _run_one(args, mission, scenario, run_index, out_dir, timing, log, llm)


async def _run_one(args, mission, scenario, run_index, out_dir: Path, timing: Timing, log, llm):
    drones = scenario.drones
    vids = [d.vehicle for d in drones]
    missions = {d.vehicle: mission.for_layer(i) for i, d in enumerate(drones)}
    victim = scenario.victim_for_run(mission, run_index)
    victim_sector = mission.grid.sector_at(victim.x_m, victim.y_m)
    stem = out_dir / f"run_{run_index:02d}"
    gate_on = args.gate == "invariants"
    recorder = EvaluationRecorder(path=stem.with_suffix(".jsonl"), run_config={
        "script": "run_sar_swarm.py", "phase": 3, "mode": "offline" if args.offline else "live",
        "speedup": timing.k, "mission_file": args.mission, "scenario_file": args.scenario,
        "mission": mission.model_dump(mode="json"), "derived": mission.derived(),
        "scenario_id": scenario.scenario_id, "run_index": run_index,
        "seed": scenario.run_seed(run_index), "victim": victim.model_dump(),
        "victim_sector": victim_sector, "full_sweep_override": args.full_sweep,
        "scene": None if args.offline else args.scene,
        "drones": [{"vehicle": d.vehicle, "team_id": d.id, "start": d.start.model_dump(),
                    "altitude_m": missions[d.vehicle].flight.altitude_m} for d in drones],
        "swarm": mission.swarm.model_dump(mode="json"),
        "pipeline": {"planner": (SwarmLLMPlanner if llm else SwarmFixedPlanner).name,
                     "allocation": "voronoi_swarm", "peer_rules": PeerClaimRules.name,
                     "trust_assessor": NoTrustAssessor.name,
                     "action_gate": SarInvariantGate.name if gate_on else BypassGate.name,
                     "model": llm["config"] if llm else None, "observation_hz": OBSERVATION_HZ,
                     "sensor_hz": SENSOR_HZ, "truth_hz": TRUTH_HZ, "remote_id_hz": REMOTE_ID_HZ,
                     "detection": "footprint_geometry"},
        "executor": asdict(timing.executor()), "follower": asdict(timing.follower())})
    log(f"\n=== run {run_index} (seed {scenario.run_seed(run_index)}): victim at "
        f"({victim.x_m:g}, {victim.y_m:g}) in {victim_sector}; {len(vids)} drones; "
        f"log {recorder.path}")
    results: dict[str, dict] = {v: {"planner": None, "shutdown": None, "descend": None,
                                    "error": None} for v in vids}
    bg_errors: list[str] = []
    run_error = None
    outcome = "aborted"

    if args.offline:
        from adapters.kinematic_sim import KinematicSim
        adapter_cm = KinematicSim({d.vehicle: (d.start.x_m, d.start.y_m) for d in drones},
                                  speedup=timing.k)
    else:
        from adapters.projectairsim_adapter import ProjectAirSimAdapter
        adapter_cm = ProjectAirSimAdapter(vehicle_ids=vids, scene=args.scene)
    try:
        with adapter_cm as adapter:
            await asyncio.sleep(0.5 / timing.k)

            def sim_clock_of(vid):
                def clock():
                    p = adapter.latest_poses().get(vid)
                    return None if p is None else round(p[0] / 1e9, 3)
                return clock
            recorder.set_sim_clock(lambda: next((p[0] for p in adapter.latest_poses().values()),
                                                None))
            recorder.record("connected", adapter=type(adapter).__name__)
            flight = ProjectAirSimExecutor(adapter, config=timing.executor(),
                                           on_event=recorder.recorder_for(source="executor"))
            bus = RemoteIdBus(recorder=recorder, in_range=range_filter(adapter,
                                                                       mission.swarm.radio_range_m))
            team = {}
            stop = asyncio.Event()
            for d in drones:
                vid, m = d.vehicle, missions[d.vehicle]
                feed = ObservationFeed(adapter, vid, assessor=NoTrustAssessor(), recorder=recorder)
                home = feed.observe()
                if home is None:
                    raise RuntimeError(f"{vid}: no home observation")
                home_xy = (home.north_m, home.east_m)
                off = math.dist(home_xy, (d.start.x_m, d.start.y_m))
                recorder.record("agent_home", vehicle_id=vid, observation_id=home.observation_id,
                                home={"x_m": home_xy[0], "y_m": home_xy[1]},
                                offset_from_scenario_start_m=round(off, 2))
                if off > SPAWN_TOLERANCE_M:
                    raise RuntimeError(f"{vid} spawned at ({home_xy[0]:.1f}, {home_xy[1]:.1f}), "
                                       f"{off:.1f} m from the scenario start; fix the scene or "
                                       f"scenario before flying")
                belief = LocalBelief(vid, m, path=stem.with_suffix(f".{vid}.memory.json"),
                                     on_change=recorder.recorder_for(source="belief"))
                sector_ex = SectorExecutor(adapter, vid, m, home_xy, flight,
                                           follower_config=timing.follower(),
                                           on_event=recorder.recorder_for(source="mission"))
                guard = SarGuard(m, belief, home_xy)
                gate = SarInvariantGate(guard) if gate_on else BypassGate()
                port = ActionPort(vid, sector_ex, feed, gate=gate, recorder=recorder,
                                  max_observation_age_s=MAX_OBSERVATION_AGE_S)
                rid_rx = RemoteIdReceiver(vid, recorder=None, clock=sim_clock_of(vid))
                messenger = PeerMessenger(vid, bus, recorder=recorder)
                coord = SwarmCoordinator(vid, m, belief, rid_rx, messenger, sim_clock_of(vid),
                                         guard=guard, recorder=recorder, log=log)
                bus.attach(SwarmReceiver(vid, rid_rx, coord.on_message, sim_clock_of(vid),
                                         recorder=recorder))
                stage = (mission.swarm.staging or [])
                i = [x.vehicle for x in drones].index(vid)
                common = dict(recorder=recorder, sweep_progress=sector_ex.last_progress,
                              log=log, swarm=coord,
                              staging_xy=(stage[i].x_m, stage[i].y_m) if i < len(stage) else None)
                if llm:
                    planner = SwarmLLMPlanner(vid, m, feed, port, belief, model=llm["model"],
                                              guard=guard, model_label=llm["label"],
                                              model_settings=llm["settings"],
                                              decision_timeout_s=args.decision_timeout, **common)
                else:
                    planner = SwarmFixedPlanner(vid, m, feed, port, belief, guard=guard, **common)

                # Stop only a SEARCH move: a drone already heading home (it found nothing
                # left, or an earlier report ended its search) keeps flying home even if it
                # passes over the victim on the way.
                def peer_victim(claim, sector_ex=sector_ex, m=m, coord=coord):
                    if m.termination.stop_on_detection and coord.state == "searching":
                        sector_ex.interrupt(f"{claim.sender_id} reported the victim")
                coord.on_peer_victim = peer_victim

                def detected(det, vid=vid, belief=belief, planner=planner, coord=coord,
                             sector_ex=sector_ex, m=m):
                    log(f"  victim detected by {vid} in {det.sector_id} at sim "
                        f"{det.sim_time_s:.1f} s")
                    belief.add_detection(det)
                    planner.on_detection(det)
                    coord.own_detection(det)
                    if m.termination.stop_on_detection and coord.state == "searching":
                        sector_ex.interrupt("victim detected")
                sensor = VictimSensor(adapter, vid, (victim.x_m, victim.y_m),
                                      m.sensor.footprint_m, m.flight.altitude_m, grid=m.grid,
                                      on_detection=detected, recorder=recorder)
                bcast = RemoteIdBroadcaster(adapter, vid, bus, recorder=recorder,
                                            status=lambda coord=coord: coord.state)
                team[vid] = dict(feed=feed, belief=belief, sector_ex=sector_ex, planner=planner,
                                 coord=coord, sensor=sensor, bcast=bcast, mission=m,
                                 launch_alt=home.altitude_m)

            bg = []
            for vid, t in team.items():
                bg += [(f"{vid}.feed", asyncio.create_task(t["feed"].sample(
                            timing.hz(OBSERVATION_HZ), stop))),
                       (f"{vid}.sensor", asyncio.create_task(t["sensor"].run(
                           timing.hz(SENSOR_HZ), stop))),
                       (f"{vid}.remote_id", asyncio.create_task(t["bcast"].run(
                           timing.hz(REMOTE_ID_HZ), stop))),
                       (f"{vid}.victim_repeat", asyncio.create_task(t["coord"].repeat_victim(
                           timing.poll(mission.swarm.victim_repeat_s), stop))),
                       (f"{vid}.time_watch", asyncio.create_task(_watch_time(
                           adapter, vid, t["mission"], t["planner"], t["sector_ex"], stop,
                           timing.poll(1.0))))]
            if not args.offline:
                def stalled(why):
                    log(f"  [{time.strftime('%H:%M:%S')}] SIM STALLED: {why}; ending the run")
                    recorder.record("sim_stalled", reason=why)
                    for t in team.values():
                        t["sector_ex"].interrupt("sim stalled")
                bg.append(("sim_clock", asyncio.create_task(_watch_sim_clock(
                    adapter, stop, SIM_STALL_S, 1.0, stalled))))
            bg.append(("truth", asyncio.create_task(_sample_truth(
                adapter, vids, missions, recorder, stop, 1.0 / timing.hz(TRUTH_HZ),
                lambda vid: team[vid]["sector_ex"].interrupt("geofence")))))

            async def fly(vid):
                t, res = team[vid], results[vid]
                try:
                    await _takeoff(adapter, vid, recorder, timeout_s=30.0)
                    res["planner"] = pr = await t["planner"].run()
                    log(f"  [{time.strftime('%H:%M:%S')}] {vid}: search ended: {pr.outcome} "
                        f"({pr.reason}); return home: {pr.return_home}")
                    r = await flight.change_altitude(
                        vid, round(t["launch_alt"] + DESCEND_ABOVE_LAUNCH_M, 2),
                        reason="setup: descend before landing")
                    recorder.record("setup", step="descend", vehicle_id=vid, status=r.status.value)
                    res["descend"] = r.status.value
                except Exception as err:
                    res["error"] = f"{type(err).__name__}: {err}"
                    recorder.record("error", vehicle_id=vid, error=res["error"])
                    log(f"  {vid} ERROR: {res['error']}")
                finally:
                    res["shutdown"] = await safe_shutdown(
                        adapter.drone(vid), adapter=adapter, vehicle_id=vid,
                        log=lambda s: log(f"[{vid}] {s}"), confirm_poll_s=timing.poll(0.1))
                    recorder.record("shutdown", vehicle_id=vid, report=res["shutdown"])
                    t["coord"].state = "landed"

            try:
                await asyncio.gather(*(fly(v) for v in vids))
            finally:
                stop.set()
                for (name, _), res in zip(bg, await asyncio.gather(*(task for _, task in bg),
                                                                   return_exceptions=True)):
                    if isinstance(res, BaseException) and not isinstance(
                            res, asyncio.CancelledError):
                        bg_errors.append(f"{name}: {type(res).__name__}: {res}")
                        recorder.record("background_error", task=name,
                                        error=f"{type(res).__name__}: {res}")
                recorder.record("bus_stats", **bus.stats)
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
    metrics = swarm_metrics(read_run(recorder.path), mission)
    for vid, r in results.items():
        d = metrics["drones"].setdefault(vid, {})
        d["landed"] = bool(r["shutdown"] and r["shutdown"].disarmed)
        d["descend"] = r["descend"]
        d["error"] = r["error"]
        d["planner_reason"] = r["planner"].reason if r["planner"] else None
    metrics["background_errors"] = bg_errors
    metrics["sim_stalled"] = getattr(adapter_cm, "stalled", None)
    metrics["run_error"] = run_error
    metrics["all_landed"] = all(d.get("landed") for d in metrics["drones"].values())
    stem.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    try:
        metrics["plot"] = str(plot_run(recorder.path, stem.with_suffix(".svg")))
    except Exception as err:
        log(f"  (plot failed: {type(err).__name__}: {err})")
    return metrics


def run_failures(m: dict, full_sweep: bool, min_coverage: float = FULL_SWEEP_MIN_COVERAGE):
    why = []
    if not m["victim_found"]:
        why.append("victim not found")
    for vid, d in sorted(m["drones"].items()):
        if d.get("ended_by") in ("failed", "aborted", None):
            why.append(f"{vid}: search ended by {d.get('ended_by')}")
        if d.get("return_home") != "home_reached":
            why.append(f"{vid}: return home {d.get('return_home')}")
        if not d.get("landed"):
            why.append(f"{vid}: not landed")
        if d.get("descend") != "succeeded":
            why.append(f"{vid}: descend {d.get('descend')}")
        if d.get("error"):
            why.append(f"{vid}: {d['error']}")
    if m["geofence_breaches"]:
        why.append(f"{m['geofence_breaches']} geofence breaches")
    if m["impacts"]:
        why.append(f"{m['impacts']} impacts")
    if not m.get("run_complete"):
        why.append("run log incomplete")
    if m.get("run_log_outcome") in (None, "aborted"):
        why.append(f"run log ended {m.get('run_log_outcome')}")
    if m.get("sim_stalled"):
        why.insert(0, f"SIM STALLED ({m['sim_stalled']})")
    if m.get("run_error"):
        why.append(f"run error: {m['run_error']}")
    if m.get("background_errors"):
        why.append("background task failed")
    if full_sweep:
        if m["sectors_swept"] != m["sector_count"]:
            why.append(f"{m['sectors_swept']}/{m['sector_count']} sectors swept")
        if m["area_coverage"] < min_coverage:
            why.append(f"measured coverage {m['area_coverage']:.1%} < {min_coverage:.0%}")
    return why


async def main(args) -> int:
    try:
        mission = load_mission(args.mission)
        scenario = load_scenario(args.scenario, mission)
    except PlanFileError as err:
        print(err)
        return 2
    sw = {}
    if args.radio_range is not None:
        sw["radio_range_m"] = None if args.radio_range <= 0 else args.radio_range
    if args.peer_clear_rule:
        sw["peer_clear_rule"] = args.peer_clear_rule
    if args.layer_spacing is not None:
        sw["layer_spacing_m"] = args.layer_spacing
    if sw:
        mission = mission.model_copy(update={"swarm": mission.swarm.model_copy(update=sw)})
    if args.full_sweep:
        mission = mission.model_copy(update={"termination": mission.termination.model_copy(
            update={"stop_on_detection": False})})
    if args.drones:
        scenario = scenario.model_copy(update={"drones": scenario.drones[:args.drones]})
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
          f"{s.peer_clear_rule}; {runs} run(s), "
          f"{'offline x' + format(timing.k, 'g') if args.offline else 'live'}; output {out_dir}")
    results = []
    for i in range(runs):
        m = await run_one(args, mission, scenario, i, out_dir, timing, llm=llm)
        m["fail_reasons"] = run_failures(m, args.full_sweep, args.min_coverage)
        results.append(m)
        print(f"  -> found={m['victim_found']} by {m['finder']} t_find={m['time_to_find_s']} s, "
              f"mission {m['mission_time_s']} s, {m['sectors_swept']}/{m['sector_count']} sectors "
              f"(dup {m['duplicate_sweeps']}), coverage {m['area_coverage']:.1%}, msgs "
              f"{m['messages']['peer_sent']} sent / {m['messages']['out_of_range']} out of range"
              + ("" if not m["fail_reasons"] else f"; FAIL: {'; '.join(m['fail_reasons'])}"))
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
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
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
    print(f"RESULT: {'PASS' if all(passed) else 'FAIL'}   ({out_dir / 'batch_summary.json'})")
    return 0 if all(passed) else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SAR swarm runner (Phase 3): honest drones on a "
                                             "shared bus")
    ap.add_argument("--planner", choices=("fixed", "llm"), default="fixed")
    ap.add_argument("--gate", choices=("none", "invariants"), default="invariants",
                    help="the invariant guard in each drone's port (default on)")
    ap.add_argument("--model", default=None, help="as run_sar_mission.py: SPEC, scripted[:...] "
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
    ap.add_argument("--offline", action="store_true", help="kinematic sim instead of AirSim")
    ap.add_argument("--speedup", type=float, default=50.0)
    ap.add_argument("--full-sweep", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=FULL_SWEEP_MIN_COVERAGE)
    ap.add_argument("--scene", default="scene_sar_three_drones.jsonc")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "sar_swarm"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
