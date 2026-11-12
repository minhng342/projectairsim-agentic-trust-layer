"""Phase 3 completion: obstacle handling (stuck detector, impact recovery, drop-out), the
static-split comparison condition, the paired speedup with a bootstrap interval, and the
evidence manifest / archive."""
import asyncio
import json
import tarfile

import pytest

from atal.core.action import ActionType, ProposedAction
from atal.core.belief import LocalBelief
from atal.core.mission import Mission
from atal.evaluation.archive import main as archive_main
from atal.evaluation.compare import bootstrap_ci, load_batch, paired_comparison, table
from atal.evaluation.evidence import MANIFEST, verify, write_manifest
from atal.evaluation.run_log import read_run
from atal.flight.executor import ProjectAirSimExecutor
from atal.flight.path_follower import FollowerConfig
from atal.flight.sector_sweep import SectorExecutor
from atal.runtime.run_single import parse_obstacle
from atal.simulation.kinematic_sim import KinematicSim
from atal.trust_layer.invariant_guard import SarGuard
from tests.runtime.test_sar_pipeline import MISSION
from tests.runtime.test_sar_pipeline import run as run_one_drone
from tests.runtime.test_sar_swarm import run as run_swarm


def _mission():
    return Mission.model_validate(MISSION)       # 40 m, 4 x 4 sectors of 10 m, altitude 10 m


# ------------------------------------------------------------------ the kinematic sim
def test_kinematic_sim_obstacle_stops_the_drone_and_logs_a_wall_impact():
    sim = KinematicSim({"D": (0.0, 0.0)}, speedup=1000,
                       obstacles=[parse_obstacle("wall:5,8,-10,10,20")])

    async def fly():
        d = sim.drone("D")
        d.arm()
        await (await d.takeoff_async())
        for _ in range(200):
            await d.move_by_velocity_async(3.0, 0.0, 0.0, duration=0.3)
            await asyncio.sleep(0.0005)
    asyncio.run(fly())
    snap = sim.get_snapshot("D")
    assert snap.position_ned_m.x < 5.0                    # never inside the wall
    (first, *_) = sim.collision_log("D")
    assert first.object_name == "wall" and first.is_supporting_surface is False
    assert first.is_resting_contact is False


def test_parse_obstacle_rejects_bad_boxes():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_obstacle("x:1,2,3")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_obstacle("x:5,1,0,1,10")


def test_sector_executor_reports_a_blocked_path_without_any_impact_callback():
    """No runner, no impact hook: the follower's own stuck detector ends the sweep as
    path_blocked within ~stuck_after_s of sim time, not after the 3x + 60 s timeout."""
    m = _mission()
    sim = KinematicSim({"Drone1": (0.0, 0.0)}, speedup=200,
                       obstacles=[parse_obstacle("wall:2,40,-5,45,30")])
    flight = ProjectAirSimExecutor(sim)

    async def go():
        d = sim.drone("Drone1")
        d.enable_api_control()
        d.arm()
        await (await d.takeoff_async())
        await (await d.move_to_position_async(0.0, 0.0, -10.0, 2.0))
        ex = SectorExecutor(sim, "Drone1", m, (0.0, 0.0), flight,
                            follower_config=FollowerConfig(poll_interval_s=0.0005,
                                                           hover_timeout_s=5.0))
        t0 = sim.now_s
        r = await ex.execute(ProposedAction(vehicle_id="Drone1", sector_id="A3",
                                            action_type=ActionType.SWEEP_SECTOR, reason="t"))
        return r, sim.now_s - t0
    r, took = asyncio.run(go())
    assert r.reason_code == "path_blocked" and "blocked" in r.reason
    assert took < 40                                      # the timeout would be > 60 s


# ------------------------------------------------------------------ guard
def test_guard_refuses_blocked_sectors_and_allows_only_the_recovery_climb():
    m = _mission()
    g = SarGuard(m, LocalBelief("Drone1", m), (0.0, 0.0))
    g.start_sim_s = 0.0
    g.blocked.add("B2")
    sweep = ProposedAction(vehicle_id="Drone1", action_type=ActionType.SWEEP_SECTOR,
                           sector_id="B2", reason="t")
    assert g.check(sweep, 0, 0, 0.0).code == "blocked_by_obstacle"
    assert "B2" not in g.open_sectors()
    climb = ProposedAction(vehicle_id="Drone1", action_type=ActionType.CHANGE_ALTITUDE,
                           altitude_m=20.0, reason="t")
    home_high = ProposedAction(vehicle_id="Drone1", action_type=ActionType.RETURN_HOME,
                               altitude_m=20.0, reason="t")
    assert g.check(climb, 0, 0, 0.0).code == "wrong_altitude"
    assert g.check(home_high, 0, 0, 0.0, planner_choice=False).code == "wrong_altitude"
    g.recovery_alt_m = 20.0
    assert g.check(climb, 0, 0, 0.0).ok
    assert g.check(home_high, 0, 0, 0.0, planner_choice=False).ok


# ------------------------------------------------------------------ runs with obstacles
def test_one_drone_recovers_from_a_hit_and_keeps_searching(tmp_path):
    """Phase 1's complaint: a hit no longer grinds and ends the search. The drone climbs,
    blocks that sector and carries on; the run still FAILS acceptance on the impact."""
    code, batch, s = run_one_drone(tmp_path, "--gate", "invariants", "--full-sweep",
                                   "--obstacle", "box:12,18,12,18,15")
    m = s["runs"][0]
    assert code == 1 and any("impact" in r for r in m["fail_reasons"])
    assert m["ended_by"] == "blocked_sectors" and m["landed"]
    assert m["return_home"] == "home_reached"
    log = read_run(batch / "run_00.jsonl")
    obstacles = log.of_kind("obstacle")
    assert obstacles and obstacles[0]["cause"] == "impact"
    assert log.of_kind("obstacle_recovered")
    end = log.of_kind("planner_end")[0]
    assert end["recoveries"] >= 1 and end["blocked"]
    assert m["sectors_swept"] >= 14                       # the rest of the grid was searched
    assert m["obstacle_incidents"] >= 1 and m["obstacles_recovered"] >= 1 and not m["lost"]


def test_one_drone_drops_out_after_repeated_hits_and_lands(tmp_path):
    code, batch, s = run_one_drone(tmp_path, "--gate", "invariants",
                                   "--obstacle", "wall:8,40,0,40,100")
    m = s["runs"][0]
    assert code == 1 and m["ended_by"] == "lost" and m["landed"]
    assert "search ended by lost" in m["fail_reasons"]
    log = read_run(batch / "run_00.jsonl")
    (lost,) = log.of_kind("drone_lost")
    assert lost["recoveries"] == 3


def test_swarm_carries_on_when_one_drone_drops_out(tmp_path):
    """A tall wall along the west column: Drone1 (nearest, so it owns that column) hits it
    flying to A1, A2 and A3, drops out and says so on Remote ID ("lost"), and lands; the
    others stop counting it and finish the search."""
    code, batch, s = run_swarm(tmp_path, "--obstacle", "wall:8,40,0,9,100")
    m = s["runs"][0]
    d1 = m["drones"]["Drone1"]
    assert d1["ended_by"] == "lost" and d1["landed"] and d1["recoveries"] == 3
    assert d1["return_home"] == "not_attempted"
    assert d1["lost"] and d1["impacts"] >= 3 and d1["impact_objects"] == ["wall"]
    assert d1["obstacle_incidents"] == 3 and d1["blocked_sectors"]
    assert s["baseline"]["drones_lost_total"] >= 1
    assert m["victim_found"]
    assert all(d["landed"] for d in m["drones"].values())
    assert any(m["drones"][v]["return_home"] == "home_reached" for v in ("Drone2", "Drone3"))
    assert any("Drone1: search ended by lost" in r for r in m["fail_reasons"])
    log = read_run(batch / "run_00.jsonl")
    states = {e["payload"].get("mission_state") for e in log.of_kind("rid_transmit")
              if isinstance(e.get("payload"), dict) and e["payload"].get("uas_id") == "Drone1"}
    assert "lost" in states


def _incidents(log):
    out = {}
    for e in log.of_kind("obstacle"):
        if e.get("sector_id"):
            out.setdefault(e["sector_id"], []).append(e["vehicle_id"])
    return out


def test_a_tall_obstacle_costs_the_team_one_incident_per_sector(tmp_path):
    """A wall too tall to climb over in the west column: however many sectors it stops,
    no second drone enters a sector a teammate reported, and the victim is still found."""
    code, batch, s = run_swarm(tmp_path, "--obstacle", "wall:8,40,0,9,100")
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    incidents = _incidents(log)
    assert incidents and all(len(v) == 1 for v in incidents.values()), incidents
    assert m["victim_found"] and sum(d["lost"] for d in m["drones"].values()) <= 1
    assert m["blocked_shared"] == len(incidents)
    assert set(m["unreachable_sectors"]) <= set(incidents)
    assert any(d["blocked_heard"] for d in m["drones"].values())
    sent = [e["msg"] for e in log.of_kind("peer_message_sent")
            if e["msg"]["msg_type"] == "sector_blocked"]
    assert {x["cause"] for x in sent} <= {"impact", "path_blocked"}


def test_ignoring_peer_reports_lets_a_teammate_enter(tmp_path):
    code, batch, s = run_swarm(tmp_path, "--obstacle", "wall:8,40,0,9,100",
                               peer_blocked_rule="ignore")
    m = s["runs"][0]
    assert not any(d["blocked_heard"] for d in m["drones"].values())


def test_a_victim_in_a_blocked_sector_fails_as_unreachable():
    from atal.runtime.acceptance import team_run_failures as run_failures
    base = {"victim_found": False, "victim_unreachable": True, "victim_sector": "A1", "drones": {},
            "geofence_breaches": 0, "impacts": 0, "run_complete": True,
            "run_log_outcome": "searched", "messages": {}}
    assert any("A1 is blocked" in w for w in run_failures(base, False))
    assert "victim not found" in run_failures(dict(base, victim_unreachable=False), False)


def test_receiver_stays_out_of_a_sector_a_peer_reported_and_keeps_it_unsearched():
    from atal.agents.team_coordinator import SwarmCoordinator
    from atal.channels.peer_link import PeerMessenger
    from atal.channels.radio import RemoteIdBus, RemoteIdReceiver
    from atal.core.peer_messages import SectorBlocked
    m = _mission()
    clock = lambda: 100.0
    bus = RemoteIdBus()
    rid = RemoteIdReceiver("Drone1", clock=clock)
    guard = SarGuard(m, LocalBelief("Drone1", m), (0.0, 0.0))
    belief = LocalBelief("Drone1", m)
    coord = SwarmCoordinator("Drone1", m, belief, rid, PeerMessenger("Drone1", bus), clock,
                             guard=guard, log=lambda s: None)
    msg = SectorBlocked(sender_id="Drone2", sim_time_s=1, sector_id="B2", altitude_m=10,
                        cause="impact")
    coord.on_message(msg, 100.0)
    coord.on_message(msg, 101.0)                          # a repeat: same msg_id, no effect
    assert coord.peer_blocked == {"B2": "Drone2"} and "B2" in guard.blocked
    assert belief.status("B2").value == "unsearched"      # blocked is not cleared
    assert len([c for c in belief.peer_claims if c.msg_type == "sector_blocked"]) == 1
    assert "B2" not in coord.my_sectors((0.0, 0.0), 100.0, exclude=set(coord.peer_blocked))


# ------------------------------------------------------------------ static split
def test_static_split_sends_no_claims_or_clears_and_keeps_to_fixed_parts(tmp_path):
    code, batch, s = run_swarm(tmp_path, "--allocation", "static", "--full-sweep")
    m = s["runs"][0]
    assert code == 0, m["fail_reasons"]
    by_type = m["messages"]["peer_sent_by_type"]
    assert set(by_type) <= {"victim_detected"}
    log = read_run(batch / "run_00.jsonl")
    (split,) = log.of_kind("static_split")
    cells = split["cells"]
    assert sorted(s for c in cells.values() for s in c) == sorted(_mission().grid.ids())
    for v, d in m["drones"].items():
        own = set()
        for e in log.of_kind("belief_update"):
            sec = e.get("sector")
            if e.get("vehicle_id") == v and sec and sec.get("status") == "swept" \
                    and sec.get("verified_by_self"):
                own.add(sec["sector_id"])
        assert own == set(cells[v])
    assert m["duplicate_sweeps"] == 0 and s["swarm"]["allocation"] == "static"
    assert {e["allocation"] for e in log.of_kind("sar_plan")} == {"static"}


# ------------------------------------------------------------------ paired comparison
def _runs(times, offset=0.0):
    return [{"run_index": i, "victim": {"x_m": float(i), "y_m": 1.0 + offset},
             "victim_sector": "A1", "time_to_find_s": t, "victim_found": t is not None,
             "area_coverage": 0.5} for i, t in enumerate(times)]


def test_paired_speedup_matches_victims_and_bootstraps_a_reproducible_interval():
    ref = {"batch": "one", "runs": _runs([100, 200, 300, 400, None])}
    batch = _runs([50, 100, 100, 400, 80]) + _runs([10], offset=99)   # last one unpaired
    pc = paired_comparison(ref, batch)
    assert pc["pairs"] == 4 and pc["unpaired"] == 1 and pc["not_found"]["reference"] == 1
    assert pc["speedup"]["mean"] == pytest.approx((2 + 2 + 3 + 1) / 4)
    assert pc["faster"] == 3
    lo, hi = pc["speedup"]["ci95"]
    assert lo <= pc["speedup"]["mean"] <= hi
    assert paired_comparison(ref, batch)["speedup"] == pc["speedup"]      # fixed seed
    assert bootstrap_ci([1.0]) is None


def test_comparison_table_has_a_row_per_batch():
    ref = {"runs": _runs([100, 200]), "passed": [True, True]}
    b = {"runs": _runs([50, 100]), "passed": [True, False]}
    md = table(ref, [b], ["1 drone", "3 drones"])
    rows = md.splitlines()
    assert len(rows) == 4 and "2.0x" in rows[3] and "1/2" in rows[3]


def test_swarm_batch_records_the_paired_comparison_against_a_reference(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _, ref_batch, _ = run_swarm(a, "--drones", "1")
    b = tmp_path / "b"
    b.mkdir()
    code, batch, s = run_swarm(b, "--reference", str(ref_batch))
    pc = s["paired_vs_reference"]
    assert code == 0 and pc["pairs"] == 1 and pc["speedup"]["mean"] > 0


# ------------------------------------------------------------------ evidence
def test_every_batch_writes_a_manifest_that_detects_later_changes(tmp_path):
    code, batch, s = run_swarm(tmp_path)
    mf = json.loads((batch / MANIFEST).read_text())
    assert mf["counts"] == {"planned": 1, "attempted": 1, "valid": 1, "stalled": 0,
                            "stalled_runs": [], "passed": 1, "failed": 0}
    assert mf["git"]["commit"] and len(mf["config"]["config_sha256"]) == 64
    assert mf["config"]["inputs"]["mission"]["sha256"]
    assert mf["seeds"][0]["seed"] == s["runs"][0]["seed"]
    assert "batch_summary.json" in mf["files"] and "run_00.jsonl" in mf["files"]
    assert verify(batch) == []
    (batch / "run_00.summary.json").write_text("{}")
    assert verify(batch) == ["changed: run_00.summary.json"]


def test_manifest_counts_stalled_runs_apart(tmp_path):
    results = [{"run_index": 0, "seed": 1}, {"run_index": 1, "seed": 2, "sim_stalled": "x"}]
    out = write_manifest(tmp_path, ["x"], {"out": "o", "runs": 2}, {}, None, results,
                         [True, False], 3)
    assert out["counts"]["valid"] == 1 and out["counts"]["stalled_runs"] == [1]
    assert out["counts"]["planned"] == 3


def test_archive_packs_intact_batches_and_refuses_changed_ones(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _, ref_batch, _ = run_swarm(a, "--drones", "1")
    b = tmp_path / "b"
    b.mkdir()
    _, batch, _ = run_swarm(b)
    out = tmp_path / "evidence"
    assert archive_main([str(batch), "--reference", str(ref_batch), "--out", str(out)]) == 0
    (tar_path,) = out.glob("*.tar.gz")
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
    assert any(n.endswith("EVIDENCE.md") for n in names)
    assert any(n.endswith(f"{batch.name}/manifest.json") for n in names)
    (md,) = out.glob("*.EVIDENCE.md")
    assert "Paired comparison" in md.read_text()
    (batch / "run_00.jsonl").write_text("tampered")
    assert archive_main([str(batch), "--out", str(out)]) == 1
    assert load_batch(batch)["runs"]


# ------------------------------------------------------------------ pad zone
def test_pad_zone_lets_one_drone_in_at_a_time():
    from atal.flight.pad_zone import PadZone
    z = PadZone.around([(0, -3), (0, 0), (0, 3)], pad_zone_m=10)
    assert z.center == (0.0, 0.0) and z.radius_m == pytest.approx(18.0)
    assert z.inside((0, 17)) and not z.inside((0, 19))
    assert z.approach_point((0, 100)) == pytest.approx((0.0, 18.0))
    order = []

    async def go():
        assert await z.acquire("A")
        assert await z.acquire("A")                       # re-entrant for the holder

        async def b():
            await z.acquire("B")
            order.append("B in")
        task = asyncio.create_task(b())
        await asyncio.sleep(0.01)
        order.append("A out")
        z.release("A")
        await task
        assert not await z.acquire("C", timeout_s=0.01)   # B still holds it
    asyncio.run(go())
    assert order == ["A out", "B in"] and z.holder == "B"


def test_swarm_lands_one_drone_at_a_time(tmp_path):
    code, batch, s = run_swarm(tmp_path, "--full-sweep")
    assert code == 0, s["runs"][0]["fail_reasons"]
    log = read_run(batch / "run_00.jsonl")
    enters = log.of_kind("pad_zone_enter")
    leaves = log.of_kind("pad_zone_leave")
    assert sorted(e["vehicle_id"] for e in enters) == ["Drone1", "Drone2", "Drone3"]
    assert len(leaves) == 3
    # never two drones inside at once: enter / leave alternate
    seq = sorted([(e["event_id"], "in", e["vehicle_id"]) for e in enters]
                 + [(e["event_id"], "out", e["vehicle_id"]) for e in leaves])
    inside = None
    for _, what, v in seq:
        if what == "in":
            assert inside is None
            inside = v
        else:
            assert inside == v
            inside = None


# ------------------------------------------------------------------ dropout condition
def test_dropout_static_split_leaves_the_part_unswept_but_cooperation_takes_it_over(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    _, sb, s_static = run_swarm(a, "--full-sweep", "--allocation", "static",
                                "--dropout", "Drone2@20")
    b = tmp_path / "b"
    b.mkdir()
    code, db, s_dyn = run_swarm(b, "--full-sweep", "--dropout", "Drone2@20")
    st, dy = s_static["runs"][0], s_dyn["runs"][0]
    for m in (st, dy):
        d2 = m["drones"]["Drone2"]
        assert d2["ended_by"] == "lost" and d2["landed"]
        assert not any(r.startswith("Drone2: search ended") for r in m["fail_reasons"])
    assert st["sectors_swept"] < st["sector_count"]       # nobody takes Drone2's part
    assert dy["sectors_swept"] == dy["sector_count"], dy["fail_reasons"]
    assert code == 0
    assert read_run(db / "run_00.jsonl").of_kind("dropout_injected")


def test_finder_stops_repeating_the_find_once_no_peer_is_searching(tmp_path):
    code, batch, s = run_swarm(tmp_path)
    assert code == 0, s["runs"][0]["fail_reasons"]
    log = read_run(batch / "run_00.jsonl")
    finder = s["runs"][0]["finder"]
    (stopped,) = [e for e in log.of_kind("victim_repeat_stopped") if e["vehicle_id"] == finder]
    last_sent = max(e["event_id"] for e in log.of_kind("peer_message_sent")
                    if e["vehicle_id"] == finder and e["msg"]["msg_type"] == "victim_detected")
    assert last_sent < stopped["event_id"]
    assert s["runs"][0]["messages"]["delivery_ratio"] == 1.0


# ------------------------------------------------------------------ pad zone timeout
def _airborne_at(sim, vid, north, east, alt=10.0):
    async def go():
        d = sim.drone(vid)
        d.enable_api_control()
        d.arm()
        await (await d.takeoff_async())
        await (await d.move_to_position_async(north, east, -alt, 5.0))
    return go()


def _return_home_with_zone_held(start_xy):
    """Drone1 flies home while Drone2 keeps the pad zone past the wait (review 359eb49 P1
    reproduction): the return must end pad_zone_timeout outside the zone, never go in."""
    from atal.flight.pad_zone import PadZone
    m = _mission()
    sim = KinematicSim({"Drone1": (0.0, 0.0)}, speedup=200)
    flight = ProjectAirSimExecutor(sim)
    zone = PadZone((0.0, 0.0), 10.0)

    async def go():
        await _airborne_at(sim, "Drone1", *start_xy, alt=m.flight.altitude_m)
        assert await zone.acquire("Drone2")
        ex = SectorExecutor(sim, "Drone1", m, (0.0, 0.0), flight, pad_zone=zone,
                            pad_wait_s=0.001,
                            follower_config=FollowerConfig(poll_interval_s=0.0005,
                                                           hover_timeout_s=5.0))
        events = []
        ex._emit = lambda kind, **f: events.append((kind, f))
        r = await ex.execute(ProposedAction(vehicle_id="Drone1", reason="t",
                                            action_type=ActionType.RETURN_HOME))
        p = sim.get_snapshot("Drone1").position_ned_m
        return r, (p.x, p.y), events, zone
    return asyncio.run(go())


def test_pad_zone_timeout_never_flies_in_from_outside():
    r, xy, events, zone = _return_home_with_zone_held((40.0, 0.0))
    assert r.reason_code == "pad_zone_timeout" and r.status.value == "failed"
    assert zone.holder == "Drone2"
    assert not zone.inside(xy) and xy[0] == pytest.approx(12.0, abs=1.0)   # held at the edge
    kinds = [k for k, _ in events]
    assert "return_home_started" not in kinds                  # no inward leg was flown


def test_pad_zone_timeout_from_inside_leaves_the_zone_first():
    r, xy, events, zone = _return_home_with_zone_held((3.0, 0.0))
    assert r.reason_code == "pad_zone_timeout"
    assert not zone.inside(xy)                                 # out to the edge, not over the pads
    assert "return_home_started" not in [k for k, _ in events]


def test_exit_point_stays_allowed():
    from atal.flight.pad_zone import PadZone
    z = PadZone((0.0, 0.0), 18.0)
    quadrant = lambda p: p[0] >= 0 and p[1] >= 0
    assert z.exit_point((5.0, 0.0), 2.0, quadrant) == pytest.approx((20.0, 0.0))   # straight out
    x, y = z.exit_point((0.0, -3.0), 2.0, quadrant)        # straight out would leave the area
    assert quadrant((x, y)) and not z.inside((x, y))
    assert z.exit_point((0.0, -3.0), 2.0, lambda p: False) is None


def test_leaving_the_zone_stays_inside_the_geofence():
    """A drone inside a held zone whose straight way out crosses the geofence (found as a
    flaky geofence breach in the three-drone run): it waits inside the area."""
    from atal.flight.pad_zone import PadZone
    m = _mission()
    sim = KinematicSim({"Drone1": (0.0, 0.0)}, speedup=200)
    zone = PadZone((0.0, 20.0), 30.0)                      # straight out from (0, 25) is (0, 52)

    async def go():
        await _airborne_at(sim, "Drone1", 0.0, 25.0, alt=m.flight.altitude_m)
        assert await zone.acquire("Drone2")
        ex = SectorExecutor(sim, "Drone1", m, (0.0, 0.0), ProjectAirSimExecutor(sim),
                            pad_zone=zone, pad_wait_s=0.001,
                            follower_config=FollowerConfig(poll_interval_s=0.0005,
                                                           hover_timeout_s=5.0))
        r = await ex.execute(ProposedAction(vehicle_id="Drone1", reason="t",
                                            action_type=ActionType.RETURN_HOME))
        p = sim.get_snapshot("Drone1").position_ned_m
        return r, (p.x, p.y)
    r, xy = asyncio.run(go())
    assert r.reason_code == "pad_zone_timeout"
    assert m.geofence_excursion_m(*xy) <= 0.5 and not zone.inside(xy)


def test_approach_point_from_the_centre_is_on_the_edge():
    from atal.flight.pad_zone import PadZone
    z = PadZone((1.0, 2.0), 10.0)
    assert not z.inside(z.approach_point((1.0, 2.0)))


def _zone_of(log):
    (pz,) = log.of_kind("pad_zone")
    return tuple(pz["center"]), pz["radius_m"]


def _final_xy(log, vid):
    last = [e["poses"][vid] for e in log.of_kind("truth") if vid in e["poses"]][-1]
    return last["north_m"], last["east_m"]


def test_runner_lands_outside_the_zone_when_the_wait_times_out(tmp_path, monkeypatch):
    from atal.runtime import rates
    monkeypatch.setattr(rates, "PAD_WAIT_S", 0.5)
    code, batch, s = run_swarm(tmp_path)
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    assert m["pad_zone_timeouts"] >= 1 and m["contingency_landings"], m["fail_reasons"]
    assert code != 0 and any("pad zone timeout" in w for w in m["fail_reasons"])
    center, radius = _zone_of(log)
    holders = {e["vehicle_id"] for e in log.of_kind("pad_zone_enter")}
    for vid in m["contingency_landings"]:
        assert vid not in holders
        x, y = _final_xy(log, vid)
        assert ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 >= radius - 0.5
    for e in log.of_kind("contingency_landing"):
        assert e["inside_zone"] is False
    assert all(d["landed"] for d in m["drones"].values())


def test_unconfirmed_landing_keeps_the_zone_held(tmp_path, monkeypatch):
    from atal.runtime import landing, rates
    real = landing.safe_shutdown

    async def unconfirmed(*a, **kw):
        rep = await real(*a, **kw)
        rep.grounded_confirmed = False
        return rep
    monkeypatch.setattr(landing, "safe_shutdown", unconfirmed)
    monkeypatch.setattr(rates, "PAD_WAIT_S", 2.0)
    code, batch, s = run_swarm(tmp_path)
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    assert not log.of_kind("pad_zone_leave")
    assert len(log.of_kind("pad_zone_enter")) == 1            # the first lander keeps it
    assert m["pad_zone_held_unconfirmed"] and len(m["contingency_landings"]) == 2
    assert code != 0
