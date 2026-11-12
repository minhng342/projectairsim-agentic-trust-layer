"""Phase 1 end to end on the kinematic sim: mission file -> planner -> port -> sector
executor -> path follower -> sensor -> memory -> run log -> metrics -> plot."""
import asyncio
import json

import pytest
import yaml

from atal.core.belief import Detection, LocalBelief, SectorStatus
from atal.core.mission import Mission
from atal.evaluation.run_log import read_run
from atal.evaluation.sar_metrics import coverage_raster
from atal.runtime.run_single import main, parse_args
from atal.simulation.kinematic_sim import KinematicSim
from tests.channel_fakes import ListRecorder

MISSION = {
    "mission_id": "sar-test", "area": {"origin": {"x_m": 0, "y_m": 0},
                                       "size_m": {"width": 40, "height": 40},
                                       "sectors": {"x": 4, "y": 4}},
    "flight": {"altitude_m": 10, "speed_m_s": 5.0}, "sensor": {"footprint_m": 6, "overlap": 0.2},
    "search": {"pattern": "boustrophedon", "sector_order": "serpentine", "start_sector": "A1"},
    "termination": {"stop_on_detection": True, "max_mission_time_s": 900},
    "limits": {"min_battery_pct": 25, "geofence_margin_m": 5, "altitude_ceiling_m": 100},
}


def files(tmp_path, victim=(25.0, 33.0), runs=1, **mission_over):
    m = json.loads(json.dumps(MISSION))
    for k, v in mission_over.items():
        m[k].update(v)
    sc = {"scenario_id": "sc-test", "seed": 7, "runs": runs,
          "drones": [{"id": "A", "vehicle": "Drone1", "start": {"x_m": 0, "y_m": 0}}]}
    sc |= {"victim": {"x_m": victim[0], "y_m": victim[1]}} if victim else {"randomize_victim": True}
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(sc), encoding="utf-8")
    return str(tmp_path / "m.yaml"), str(tmp_path / "s.yaml")


def run(tmp_path, *extra, **kw):
    m, s = files(tmp_path, **kw)
    out = tmp_path / "out"
    code = asyncio.run(main(parse_args(["--offline", "--speedup", "100", "--mission", m,
                                        "--scenario", s, "--out", str(out), *extra])))
    batch = next(out.iterdir())
    return code, batch, json.loads((batch / "batch_summary.json").read_text())


def test_finds_the_victim_stops_returns_home_and_lands(tmp_path):
    code, batch, summary = run(tmp_path)
    assert code == 0 and summary["all_passed"]
    m = summary["runs"][0]
    assert m["outcome"] == "victim_found" and m["victim_sector"] == "D3"
    assert m["time_to_find_s"] > 0 and m["landed"] and m["geofence_breaches"] == 0
    assert m["sectors_swept"] < 16                         # stopped on detection
    log = read_run(batch / "run_00.jsonl")
    assert log.complete
    plan = log.of_kind("sar_plan")[0]
    assert plan["order"][:5] == ["A1", "B1", "C1", "D1", "D2"]
    # every action went through the port: proposal -> gate decision -> result -> feedback
    kinds = [e["action"]["action_type"] for e in log.of_kind("proposal")]
    assert kinds[0] == "change_altitude" and kinds[-1] == "return_home"
    assert len(log.of_kind("gate_decision")) == len(kinds) == len(log.of_kind("feedback"))
    codes = [e["feedback"]["code"] for e in log.of_kind("feedback")]
    assert codes[-2] == "sweep_stopped" and codes[-1] == "home_reached"
    # memory: saved to disk, partial sector where it stopped, detection with own position only
    mem = json.loads((batch / "run_00.memory.json").read_text())
    assert mem["detections"] and "victim" not in json.dumps(mem["detections"])
    statuses = {s: v["status"] for s, v in mem["sectors"].items()}
    assert list(statuses.values()).count("partial") == 1
    assert sum(1 for v in statuses.values() if v == "swept") == m["sectors_swept"]
    assert (batch / "run_00.svg").read_text().startswith("<svg")


def test_full_sweep_covers_every_sector_and_the_whole_area(tmp_path):
    code, batch, summary = run(tmp_path, "--full-sweep")
    m = summary["runs"][0]
    assert code == 0 and m["victim_found"]
    assert m["sectors_swept"] == m["sector_count"] == 16
    assert m["area_coverage"] >= 0.99


def test_random_victims_over_several_runs_all_pass(tmp_path):
    code, batch, summary = run(tmp_path, victim=None, runs=3)
    assert code == 0 and summary["longest_pass_streak"] == 3
    assert len({json.dumps(r["victim"]) for r in summary["runs"]}) == 3


def test_time_limit_stops_the_search_and_still_lands(tmp_path):
    code, batch, summary = run(tmp_path, victim=(39.0, 1.0),
                               termination={"max_mission_time_s": 30})
    m = summary["runs"][0]
    assert code == 1 and m["outcome"] == "time_limit" and not m["victim_found"] and m["landed"]


def test_bad_mission_file_exits_with_the_field_named(tmp_path, capsys):
    m, s = files(tmp_path, sensor={"footprint_m": 50})
    assert asyncio.run(main(parse_args(["--offline", "--mission", m, "--scenario", s,
                                        "--out", str(tmp_path)]))) == 2
    assert "footprint_m" in capsys.readouterr().out


def test_belief_persists_and_reloads(tmp_path):
    mission = Mission.model_validate(MISSION)
    rec = ListRecorder()
    path = tmp_path / "mem.json"
    b = LocalBelief("Drone1", mission, path=path, on_change=lambda k, f: rec.record(k, **f))
    b.mark_started("A1", 1.0, "r1")
    b.mark_swept("A1", 5.0, "r1")
    b.mark_partial("B1", 9.0, 0.4, "r2")
    b.add_detection(Detection(vehicle_id="Drone1", sim_time_s=9.0, x_m=12, y_m=3, altitude_m=10,
                              sector_id="B1", footprint_m=6))
    again = LocalBelief.load(path, mission)
    assert again.swept() == ["A1"] and again.status("B1") == SectorStatus.PARTIAL
    assert again.sectors["A1"].source == "self" and again.sectors["A1"].verified_by_self
    assert again.sectors["A1"].evidence == ["r1"] and again.victim_detected
    assert again.coverage_fraction() == pytest.approx(1 / 16)
    assert [e["what"] for e in rec.of("belief_update")] == ["sector"] * 3 + ["detection"]
    with pytest.raises(KeyError):
        b.mark_swept("Z9", 1.0)


def test_coverage_raster_counts_only_search_altitude():
    mission = Mission.model_validate(MISSION)
    low = [(0, x, 20.0, 2.0) for x in range(0, 41)]
    assert coverage_raster(mission, low)[0] == set()
    high = [(0, x, 20.0, 10.0) for x in range(0, 41)]
    covered, total, _ = coverage_raster(mission, high)
    assert 0.14 < len(covered) / total < 0.17                  # a 6 m strip of a 40 m square


def test_kinematic_sim_flies_velocity_and_position_commands():
    async def go():
        sim = KinematicSim({"D": (0.0, 0.0)}, speedup=200)
        d = sim.drone("D")
        d.enable_api_control()
        d.arm()
        await (await d.takeoff_async())
        assert sim.get_snapshot("D").ground_state.value == "airborne"
        await (await d.move_to_position_async(10.0, 5.0, -8.0, 3.0))
        s = sim.get_snapshot("D")
        assert abs(s.position_ned_m.x - 10) < 0.1 and abs(s.altitude_local_m - 8) < 0.1
        await d.move_by_velocity_async(2.0, 0.0, 0.0, duration=1.0)
        await asyncio.sleep(3.0 / 200)
        s = sim.get_snapshot("D")
        assert 11.0 < s.position_ned_m.x < 13.0 and s.ground_speed_mps < 0.5   # expired: braked
        await (await d.land_async())
        assert sim.get_snapshot("D").ground_state.value == "grounded"
        d.disarm()
    asyncio.run(go())


def test_unknown_sector_and_foreign_actions_are_refused_without_flying():
    from atal.core.action import ActionType, ProposedAction
    from atal.flight.executor import ProjectAirSimExecutor
    from atal.flight.sector_sweep import SectorExecutor
    mission = Mission.model_validate(MISSION)

    async def go():
        sim = KinematicSim({"Drone1": (0.0, 0.0)}, speedup=100)
        ex = SectorExecutor(sim, "Drone1", mission, (0.0, 0.0), ProjectAirSimExecutor(sim),
                            log=lambda *_: None)
        r1 = await ex.execute(ProposedAction(vehicle_id="Drone1", action_type=ActionType.SWEEP_SECTOR,
                                             sector_id="H8", reason="outside a 4 x 4 grid"))
        r2 = await ex.execute(ProposedAction(vehicle_id="Drone1", action_type=ActionType.HOLD,
                                             reason="not a search action"))
        return r1, r2
    r1, r2 = asyncio.run(go())
    assert r1.reason_code == "unknown_sector" and not r1.sent_to_simulator
    assert r2.reason_code == "unsupported_action"


def test_impact_collisions_are_logged_and_fail_the_run(tmp_path, monkeypatch):
    from atal.core.telemetry import CollisionState
    spawn = CollisionState(has_collided=True, object_name="Landscape_1", sim_time_ns=3_000_000_000,
                           impact_speed_mps=None, is_resting_contact=None, normal_z=-1.0,
                           is_supporting_surface=True, count=1)
    hit = CollisionState(has_collided=True, object_name="BG_Building_B10", sim_time_ns=10_500_000_000,
                         impact_speed_mps=4.0, is_resting_contact=False, normal_z=0.0,
                         is_supporting_surface=False, count=2, impact_count=1)
    monkeypatch.setattr(KinematicSim, "collision_log", lambda self, vid: [spawn, hit])
    code, batch, summary = run(tmp_path)
    m = summary["runs"][0]
    assert code == 1 and m["impacts"] == 1 and m["impact_objects"] == ["BG_Building_B10"]
    assert m["collisions"] == 2
    assert len(read_run(batch / "run_00.jsonl").of_kind("collision")) == 2


def test_spawn_ground_contact_is_not_an_impact(tmp_path, monkeypatch):
    from atal.core.telemetry import CollisionState
    spawn = CollisionState(has_collided=True, object_name="Landscape_1", sim_time_ns=3_000_000_000,
                           impact_speed_mps=None, is_resting_contact=None, normal_z=-1.0,
                           is_supporting_surface=True, count=1)
    monkeypatch.setattr(KinematicSim, "collision_log", lambda self, vid: [spawn])
    code, batch, summary = run(tmp_path)
    m = summary["runs"][0]
    assert m["collisions"] == 1 and m["impacts"] == 0
