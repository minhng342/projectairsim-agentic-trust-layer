"""Phase 2: the invariant guard, the LLM planner's prompt pieces, and the LLM planner end to
end on the kinematic sim with the scripted stand-in model (no model server)."""
import json

import pytest

from atal.agents.llm.scripted_model import scripted_search_model
from atal.agents.llm_chooser import decision_prompt, memory_map, prior_mass, system_prompt
from atal.core.action import ActionType, DroneState, ProposedAction, ReturnHome, SweepSector
from atal.core.belief import Detection, LocalBelief
from atal.core.mission import Mission, load_mission
from atal.core.observation import GateOutcome
from atal.evaluation.run_log import read_run
from atal.observability import span
from atal.trust_layer.invariant_guard import SarGuard, SarInvariantGate
from tests.test_sar_pipeline import MISSION, run


@pytest.fixture
def mission():
    return Mission.model_validate(MISSION)       # 40 m, 4 x 4 sectors of 10 m, 900 s


@pytest.fixture
def guard(mission):
    b = LocalBelief("Drone1", mission)
    g = SarGuard(mission, b, (0.0, 0.0))
    g.start_sim_s = 0.0
    return g


def sweep(sid):
    return ProposedAction(vehicle_id="Drone1", action_type=ActionType.SWEEP_SECTOR,
                          sector_id=sid, reason="t")


def home():
    return ProposedAction(vehicle_id="Drone1", action_type=ActionType.RETURN_HOME, reason="t")


# ------------------------------------------------------------------ guard
def test_guard_allows_an_open_sector(guard):
    assert guard.check(sweep("B2"), 0, 0, 10.0).ok


def test_guard_refuses_unknown_unassigned_and_swept_sectors(guard):
    assert guard.check(sweep("Z9"), 0, 0, 10.0).code == "unknown_sector"
    guard.cells = {"A1", "B1"}
    assert guard.check(sweep("C3"), 0, 0, 10.0).code == "not_assigned"
    guard.belief.mark_swept("A1", 5.0)
    v = guard.check(sweep("A1"), 0, 0, 10.0)
    assert not v.ok and v.code == "already_swept" and "A1" in v.message


def test_guard_refuses_a_sector_that_overruns_the_mission_time(guard):
    need = guard.sector_time_s("D4", (0, 0))
    assert guard.check(sweep("D4"), 0, 0, 900 - need + 1).code == "not_enough_time"
    assert guard.check(sweep("D4"), 0, 0, 900 - need - 1).ok


def test_return_home_is_refused_while_sectors_still_fit(guard):
    v = guard.check(home(), 0, 0, 10.0)
    assert v.code == "premature_return" and "keep searching" in v.message
    assert guard.check(home(), 0, 0, 899.0).ok                  # nothing fits any more
    assert guard.check(home(), 0, 0, 10.0, planner_choice=False).ok   # gate: always allowed
    guard.belief.add_detection(Detection(vehicle_id="Drone1", x_m=1, y_m=1, altitude_m=10,
                                         sector_id="A1", sim_time_s=9.0, footprint_m=6))
    assert guard.check(home(), 0, 0, 10.0).ok                   # victim found


def test_guard_limits_altitude_and_action_types(guard):
    up = ProposedAction(vehicle_id="Drone1", action_type=ActionType.CHANGE_ALTITUDE,
                        altitude_m=10.0, reason="t")
    assert guard.check(up, 0, 0, None).ok
    wrong = up.model_copy(update={"altitude_m": 40.0})
    assert guard.check(wrong, 0, 0, None).code == "wrong_altitude"
    hold = ProposedAction(vehicle_id="Drone1", action_type=ActionType.HOLD, reason="t")
    assert guard.check(hold, 0, 0, None).code == "not_allowed"


def test_gate_is_assessed_and_rejects_with_the_rule(guard):
    from types import SimpleNamespace
    obs = SimpleNamespace(observation_id="obs-1", north_m=0.0, east_m=0.0,
                          source_sim_time_ns=10_000_000_000)
    gate = SarInvariantGate(guard)
    guard.belief.mark_swept("A1", 1.0)
    d = gate.decide(sweep("A1"), obs)
    assert d.assessed and d.outcome == GateOutcome.REJECTED and "already swept" in d.reason
    assert gate.decide(sweep("B1"), obs).outcome == GateOutcome.APPROVED


# ------------------------------------------------------------------ prompt pieces
def test_prior_mass_is_normalized_and_peaks_at_the_center():
    m = load_mission("missions/sar/mission.yaml")
    assert prior_mass(m) == {}                                  # disabled in the file
    m = m.model_copy(update={"prior": m.prior.model_copy(update={"enabled": True})})
    p = prior_mass(m)
    assert abs(sum(p.values()) - 1) < 1e-9
    assert max(p, key=p.get) == m.grid.sector_at(m.prior.center.x_m, m.prior.center.y_m)


def test_memory_map_and_prompts_use_the_grid_convention(mission, guard):
    guard.belief.mark_swept("A1", 1.0)
    guard.belief.mark_partial("B1", 2.0, 0.5)
    lines = memory_map(mission, guard.belief, "B1").splitlines()
    assert lines[0].split() == ["A", "B", "C", "D"]
    assert lines[-2].split() == ["1", "#", "~", ".", "."]       # row 1 last, A1 west
    assert "A1 is the south-west corner" in system_prompt("Drone1", mission)

    class P:
        pass
    p = P()
    p.mission, p.belief, p.guard, p.last_sector, p.prior = mission, guard.belief, guard, "A1", {}
    st = DroneState(vehicle_id="Drone1", sim_time_s=100.0, x_m=5, y_m=5, altitude_m=10,
                    sector_id="A1", airborne=True)
    text = decision_prompt(p, st, [s for s in mission.grid.ids() if s != "A1"])
    sit = json.loads(text.split("```json\n")[1].split("\n```")[0])
    assert sit["time"]["left_s"] == 800 and sit["memory"]["unsearched"] == 15
    assert sit["nearest_unsearched"][0]["sector_id"] in ("A2", "B1")
    assert "victim" not in text.lower() and "prior_likely_unsearched" not in sit


# ------------------------------------------------------------------ end to end
def llm_run(tmp_path, model="scripted", *extra, **kw):
    return run(tmp_path, "--planner", "llm", "--model", model, *extra, **kw)


def test_llm_planner_finds_the_victim_through_the_guarded_port(tmp_path):
    code, batch, summary = llm_run(tmp_path)
    m = summary["runs"][0]
    assert code == 0 and m["outcome"] == "victim_found" and m["landed"]
    assert m["llm_decisions"] >= 1 and m["fallbacks"] == 0 and m["gate_rejections"] == 0
    log = read_run(batch / "run_00.jsonl")
    cfg = log.events[0]["config"]
    assert cfg["phase"] == 2 and cfg["pipeline"]["planner"] == "LLMSectorPlanner"
    assert cfg["pipeline"]["action_gate"] == "sar_invariants"
    gates = log.of_kind("gate_decision")
    assert gates and all(g["decision"]["assessed"] for g in gates)
    d = log.of_kind("llm_decision")[0]
    assert d["choice"].startswith("sweep_sector") and d["model"] == "scripted:nearest"
    assert d["reason"] == "scripted nearest choice" and "Memory map" in d["prompt"]
    assert summary["planner"] == "llm" and summary["model"] == "scripted:nearest"


def test_refused_choices_go_back_to_the_model(tmp_path):
    code, batch, summary = llm_run(tmp_path, "scripted:stubborn")
    m = summary["runs"][0]
    assert code == 0 and m["guard_rejections"] >= 1 and m["fallbacks"] == 0
    rej = [r for e in read_run(batch / "run_00.jsonl").of_kind("llm_decision")
           for r in e.get("guard_rejections") or []]
    assert rej[0]["code"] == "already_swept"


def test_a_broken_model_falls_back_to_the_fixed_order_and_still_finds(tmp_path):
    code, batch, summary = llm_run(tmp_path, "scripted:broken")
    m = summary["runs"][0]
    assert code == 0 and m["victim_found"]
    assert m["fallbacks"] == m["llm_decisions"] >= 1
    plan = read_run(batch / "run_00.jsonl").of_kind("sar_plan")[0]["order"]
    chosen = [e["choice"].split()[-1] for e in read_run(batch / "run_00.jsonl")
              .of_kind("llm_decision")]
    assert chosen == plan[:len(chosen)]                         # Phase 1's order


def test_llm_planner_stops_at_the_time_limit_and_goes_home(tmp_path):
    code, batch, summary = llm_run(tmp_path, "scripted", victim=(39.0, 1.0),
                                   termination={"max_mission_time_s": 60})
    m = summary["runs"][0]
    assert code == 1 and m["outcome"] == "time_limit" and not m["victim_found"] and m["landed"]
    assert m["gate_rejections"] == 0


def test_span_is_a_no_op_without_logfire():
    with span("x {a}", a=1):
        pass


def test_scripted_model_rejects_unknown_policy():
    with pytest.raises(ValueError):
        scripted_search_model("clever")


def test_typed_actions_are_the_agent_output(mission):
    assert SweepSector(sector_id="A1", reason="r").to_proposed("D").sector_id == "A1"
    assert ReturnHome(reason="r").to_proposed("D").action_type == ActionType.RETURN_HOME


# ------------------------------------------------------------------ TestModel (deck slide 9)
def test_testmodel_plumbing_no_invalid_action_reaches_the_simulator(tmp_path):
    """TestModel answers with schema-shaped nonsense (sector "a"): the action schema refuses
    every answer (sector id pattern), the planner falls back each time, and the mission still
    completes. Nothing the model proposed reaches the executor."""
    code, batch, summary = llm_run(tmp_path, "test")
    m = summary["runs"][0]
    assert code == 0 and m["victim_found"] and m["landed"] and not m["fail_reasons"]
    assert m["fallbacks"] == m["llm_decisions"] >= 1 and m["gate_rejections"] == 0
    log = read_run(batch / "run_00.jsonl")
    sent = [e["feedback"] for e in log.of_kind("feedback")
            if e["feedback"]["action_type"] == "sweep_sector"]
    assert sent and all(f["code"] == "sector_swept" or f["stage"] not in ("port", "gate")
                        for f in sent)
    decisions = log.of_kind("llm_decision")
    assert all("output retries" in (e["error"] or "") for e in decisions)
    assert all(e["choice"].startswith("sweep_sector") and "fallback" in e["reason"]
               for e in decisions)


def test_model_requests_are_blocked_in_tests():
    from pydantic_ai import models
    assert models.ALLOW_MODEL_REQUESTS is False
