"""The restructuring reference checker (tools/reference_cases.py): what it must report."""
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "reference_cases", Path(__file__).resolve().parents[2] / "tools" / "reference_cases.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

HEALTHY = {"landed": {"Drone1": True}, "descend": {"Drone1": "succeeded"}, "drone_errors": {},
           "run_complete": True, "run_log_outcome": "victim_found", "run_error": None,
           "background_errors": [], "sim_stalled": None, "processing_errors": 0,
           "pad_zone_held_unconfirmed": []}


def case(fail_reasons=((), (), ()), exit_code=None):
    runs = [{"run_index": i, "seed": 42 + i, "victim_sector": "B2", "victim_found": True,
             "passed": not why, "fail_reasons": list(why), "health": copy.deepcopy(HEALTHY),
             "time_to_find_s": 100.0} for i, why in enumerate(fail_reasons)]
    if exit_code is None:
        exit_code = 0 if all(r["passed"] for r in runs) else 1
    return {"exit_code": exit_code, "runs": runs}


def check(name, ref, new):
    return rc.compare(name, ref, new, 0.15, 30.0)


def test_identical_executions_have_no_differences():
    assert check("team_fixed", case(), case()) == []
    assert check("team_obstacle", case((["1 impacts"], (), ())), case(((), ["2 impacts"], ()))) == []


def test_obstacle_case_still_reports_failures_that_are_not_impacts():
    ref = case((["1 impacts"], (), ()))
    new = case((["1 impacts", "Drone2: not landed", "background task failed",
                 "run log incomplete"], (), ()))
    new["runs"][0]["health"].update(landed={"Drone1": True, "Drone2": False},
                                    run_complete=False, background_errors=["truth: boom"])
    problems = check("team_obstacle", ref, new)
    assert any("not expected in this case: Drone2: not landed" in p for p in problems)
    assert any("Drone2 not landed" in p for p in problems)
    assert any("run log incomplete" in p for p in problems)
    assert any("background task failed" in p for p in problems)


def test_health_is_checked_even_when_the_reference_also_failed():
    bad = case((["not landed"], (), ()))
    bad["runs"][0]["health"]["landed"] = {"Drone1": False}
    assert any("Drone1 not landed" in p for p in check("one_fixed", bad, copy.deepcopy(bad)))


def test_a_changed_verdict_seed_or_victim_is_reported():
    assert any("passed False" in p for p in check("team_fixed", case(), case((["x"], (), ()))))
    new = case()
    new["runs"][1]["seed"] = 7
    new["runs"][2]["victim_sector"] = "A1"
    problems = check("team_fixed", case(), new)
    assert any("seed 7" in p for p in problems) and any("victim_sector A1" in p for p in problems)


def test_an_exit_status_that_does_not_match_the_verdicts_is_reported():
    assert any("exit status 2" in p for p in check("team_fixed", case(), case(exit_code=2)))
    assert any("exit status 0" in p
               for p in check("team_obstacle", case(), case((["1 impacts"], (), ()), exit_code=0)))


def test_time_is_compared_only_where_the_sweep_order_is_fixed():
    slow = case()
    slow["runs"][0]["time_to_find_s"] = 400.0
    assert any("time to find" in p for p in check("one_fixed", case(), slow))
    assert check("team_fixed", case(), slow) == []
