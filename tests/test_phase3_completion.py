"""Phase 3 completion: obstacle handling (stuck detector, impact recovery, drop-out), the
static-split comparison condition, the paired speedup with a bootstrap interval, and the
evidence manifest / archive."""
import asyncio
import json
import tarfile

import pytest

from adapters.kinematic_sim import KinematicSim
from evaluation.archive import main as archive_main
from evaluation.compare import bootstrap_ci, load_batch, paired_comparison, table
from evaluation.evidence import MANIFEST, verify, write_manifest
from evaluation.recorder import read_run
from executor.path_follower import FollowerConfig
from executor.projectairsim_executor import ProjectAirSimExecutor
from executor.sector_executor import SectorExecutor
from gate.sar_guard import SarGuard
from models.action import ActionType, ProposedAction
from models.belief import LocalBelief
from models.sar import Mission
from run_sar_mission import parse_obstacle
from tests.test_sar_pipeline import MISSION
from tests.test_sar_pipeline import run as run_one_drone
from tests.test_sar_swarm import run as run_swarm


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
    from executor.pad_zone import PadZone
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
