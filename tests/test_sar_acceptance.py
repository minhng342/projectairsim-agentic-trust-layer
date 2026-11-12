"""Regressions from the 2026-11-02 audit: a find must not hide a later failure, a detection
during a decision must end the search, a full sweep must be measured, and both planners
can run under the same guard."""
import asyncio
import json

from atal.agents.sector_planner import FixedSectorPlanner
from atal.core.action import ActionType, CommandResult, CommandStatus
from atal.core.belief import Detection
from atal.core.sar import Mission
from atal.evaluation.recorder import read_run
from atal.evaluation.sar_metrics import coverage_raster
from atal.flight.sector_sweep import SectorExecutor
from atal.runtime.run_single import run_failures, run_passes
from tests.test_sar_pipeline import MISSION, run


def test_failed_return_home_fails_the_run_even_after_a_find(tmp_path, monkeypatch):
    real = SectorExecutor.execute

    async def execute(self, action):
        if action.action_type == ActionType.RETURN_HOME:
            return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                                 action_type=action.action_type, status=CommandStatus.FAILED,
                                 reason_code="path_failed", sent_to_simulator=True,
                                 reason="injected")
        return await real(self, action)
    monkeypatch.setattr(SectorExecutor, "execute", execute)
    code, batch, summary = run(tmp_path)
    m = summary["runs"][0]
    assert m["victim_found"] and m["outcome"] == "victim_found"
    assert m["return_home"] == "failed" and code == 1 and not summary["all_passed"]
    assert any("return home" in r for r in m["fail_reasons"])


def test_detection_during_a_decision_ends_the_search(tmp_path, monkeypatch):
    real = FixedSectorPlanner.choose
    calls = {"n": 0}

    async def choose(self, remaining):
        a = await real(self, remaining)
        calls["n"] += 1
        if calls["n"] == 2:              # the sensor fires while the planner is "thinking"
            d = Detection(vehicle_id=self.vehicle_id, sim_time_s=1.0, x_m=1, y_m=1,
                          altitude_m=10, sector_id="A1", footprint_m=6)
            self.belief.add_detection(d)
            self.on_detection(d)
        return a
    monkeypatch.setattr(FixedSectorPlanner, "choose", choose)
    code, batch, summary = run(tmp_path, victim=(39.0, 1.0))
    actions = [e["feedback"]["action_type"] for e in read_run(batch / "run_00.jsonl")
               .of_kind("feedback")]
    # climb, the first sweep, then straight home: no second sweep after the detection
    assert actions == ["change_altitude", "sweep_sector", "return_home"]
    assert summary["runs"][0]["ended_by"] == "victim_found"


def test_the_guard_refuses_sweeps_once_the_victim_is_found(tmp_path):
    from atal.core.action import ProposedAction
    from atal.core.belief import LocalBelief
    from atal.trust_layer.invariant_guard import SarGuard
    m = Mission.model_validate(MISSION)
    b = LocalBelief("Drone1", m)
    g = SarGuard(m, b, (0.0, 0.0))
    b.add_detection(Detection(vehicle_id="Drone1", sim_time_s=1.0, x_m=1, y_m=1,
                              altitude_m=10, sector_id="A1", footprint_m=6))
    a = ProposedAction(vehicle_id="Drone1", action_type=ActionType.SWEEP_SECTOR,
                       sector_id="B2", reason="t")
    assert g.check(a, 0, 0, 5.0).code == "search_over"


def good_metrics(**over):
    m = {"victim_found": True, "ended_by": "all_swept", "return_home": "home_reached",
         "landed": True, "geofence_breaches": 0, "impacts": 0, "run_complete": True,
         "background_errors": [], "sectors_swept": 16, "sector_count": 16,
         "area_coverage": 0.99, "run_log_outcome": "all_swept", "run_error": None,
         "descend": "succeeded"}
    m.update(over)
    return m


def test_full_sweep_pass_needs_measured_coverage_and_complete_evidence():
    assert run_passes(good_metrics(), full_sweep=True)
    low = good_metrics(area_coverage=0.01)
    assert not run_passes(low, full_sweep=True)
    assert run_passes(low, full_sweep=False)          # early find: coverage is not the goal
    assert not run_passes(good_metrics(run_complete=False), full_sweep=False)
    assert not run_passes(good_metrics(ended_by="failed"), full_sweep=False)
    assert not run_passes(good_metrics(background_errors=["sensor: boom"]), full_sweep=False)
    assert "measured coverage" in " ".join(run_failures(low, True))


def test_coverage_does_not_interpolate_across_long_truth_gaps():
    m = Mission.model_validate(MISSION)
    alt = m.flight.altitude_m
    joined = [(0.0, 5.0, 2.0, alt), (1.0, 5.0, 38.0, alt)]
    gapped = [(0.0, 5.0, 2.0, alt), (10.0, 5.0, 38.0, alt)]
    assert len(coverage_raster(m, joined)[0]) > 3 * len(coverage_raster(m, gapped)[0])


def test_fixed_planner_runs_under_the_same_guard(tmp_path):
    code, batch, summary = run(tmp_path, "--gate", "invariants")
    m = summary["runs"][0]
    assert code == 0 and m["victim_found"] and summary["gate"] == "invariants"
    log = read_run(batch / "run_00.jsonl")
    assert log.events[0]["config"]["pipeline"]["action_gate"] == "sar_invariants"
    assert all(g["decision"]["assessed"] for g in log.of_kind("gate_decision"))


def test_fixed_planner_under_the_guard_goes_home_when_time_runs_short(tmp_path):
    code, batch, summary = run(tmp_path, "--gate", "invariants", victim=(39.0, 1.0),
                               termination={"max_mission_time_s": 60})
    m = summary["runs"][0]
    assert m["outcome"] == "time_limit" and m["return_home"] == "home_reached"
    assert m["gate_rejections"] == 0 and m["landed"]


def test_an_exception_after_a_successful_search_fails_the_run(tmp_path, monkeypatch):
    from atal.flight.executor import ProjectAirSimExecutor

    async def boom(self, *a, **kw):
        raise RuntimeError("injected descent failure")
    monkeypatch.setattr(ProjectAirSimExecutor, "change_altitude", boom)
    code, batch, summary = run(tmp_path)
    m = summary["runs"][0]
    assert m["ended_by"] == "victim_found" and m["return_home"] == "home_reached"
    assert m["run_log_outcome"] == "aborted" and "injected" in m["run_error"]
    assert code == 1 and not summary["all_passed"]
    assert any("run error" in r for r in m["fail_reasons"])


def test_predicate_rejects_an_aborted_log_or_failed_descend():
    assert not run_passes(good_metrics(run_log_outcome="aborted"), full_sweep=False)
    assert not run_passes(good_metrics(run_error="RuntimeError: x"), full_sweep=False)
    assert not run_passes(good_metrics(descend="failed"), full_sweep=False)


import pytest  # noqa: E402


@pytest.mark.parametrize("planner,gate,expected_gate", [
    ("fixed", "none", "none"), ("fixed", "invariants", "sar_invariants"),
    ("llm", "none", "none"), ("llm", "invariants", "sar_invariants")])
def test_every_planner_gate_combination_runs(tmp_path, planner, gate, expected_gate):
    extra = ["--planner", planner, "--gate", gate] + (["--model", "scripted"] if planner == "llm"
                                                      else [])
    code, batch, summary = run(tmp_path, *extra)
    m = summary["runs"][0]
    assert code == 0 and m["victim_found"] and not m["fail_reasons"]
    log = read_run(batch / "run_00.jsonl")
    assert log.events[0]["config"]["pipeline"]["action_gate"] == expected_gate
    if planner == "llm":
        assert m["llm_decisions"] >= 1 and m["fallbacks"] == 0
