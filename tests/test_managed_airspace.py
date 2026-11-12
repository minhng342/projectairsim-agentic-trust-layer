"""The managed airspace in whole team runs (offline): startups top-down, every landing
under a landing reservation, recoveries through the transition manager with traffic
pauses, and the accounting of every outcome. Team runs are not repeatable per seed, so
these tests check what must hold in every run (separation, no manager flags, every pause
ended, every recovery episode closed, an unresolved drone failing the run), not one
particular outcome. The manager is trusted flight-safety infrastructure and runs the same
way in every comparison condition."""
import pytest

from atal.evaluation.run_log import read_run
from tests.test_sar_swarm import run

TOWER = "tower:12,28,12,28,20"        # 16 m square in the middle, 20 m tall: above the 10 m
#                                      and 17 m layers, below the 24 m one


def _per_vehicle(log, kind):
    out: dict[str, int] = {}
    for e in log.of_kind(kind):
        out[e["vehicle_id"]] = out.get(e["vehicle_id"], 0) + 1
    return out


def _always(m, log):
    """What every managed team run must show, whatever its outcome."""
    air = m["airspace"]
    assert m["airspace_managed"] is True and air is not None
    assert air["flags"] == {}, air["flags"]                # no unauthorised motion etc.
    assert m["separation"]["min_cruise_m"] >= 4.9           # the 4.5 m floor, as before
    # every pause the executor held ended in a resume, and the manager counted each
    assert _per_vehicle(log, "traffic_pause_hold") == _per_vehicle(log, "traffic_pause_resume")
    assert sum(air["pauses"].values()) == sum(1 for e in log.of_kind("traffic_pause")
                                              if len(e["owners"]) == 1)
    # an intentional pause never ends a sweep or counts as an obstacle
    for e in log.of_kind("sector_stopped") + log.of_kind("obstacle"):
        assert "traffic" not in str(e.get("reason", "")) + str(e.get("cause", ""))
    for vid, d in m["drones"].items():
        rec = d["recovery_outcomes"]
        assert rec["open"] == 0                             # every episode has an outcome
        if d["ended_by"] == "unresolved" or d["landing_not_cleared"]:
            assert any(w.startswith(vid) for w in m["fail_reasons"])
        if d["landed"]:                                     # under a landing reservation
            assert any(e["vehicle_id"] == vid and e["transition"] == "landing"
                       and e["reason"] == "touchdown confirmed"
                       for e in log.of_kind("transition_completed"))


def test_startups_go_top_down_and_every_drone_lands_under_a_reservation(tmp_path):
    code, batch, s = run(tmp_path)
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    _always(m, log)
    asked = [e["vehicle_id"] for e in log.of_kind("transition_requested")
             if e["transition"] == "startup"]
    assert asked == ["Drone3", "Drone2", "Drone1"]          # highest level first
    assert code == 0, m["fail_reasons"]
    assert all(d["landed"] for d in m["drones"].values())
    assert m["airspace"]["on_roster"] == []


def test_a_tower_two_layers_hit_keeps_separation_and_accounts_for_every_recovery(tmp_path):
    code, batch, s = run(tmp_path, "--obstacle", TOWER, "--full-sweep")
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    _always(m, log)
    hit = {e["vehicle_id"] for e in log.of_kind("obstacle")}
    assert hit and hit <= {"Drone1", "Drone2"}              # the 24 m layer is above it
    assert log.of_kind("recovery_started")
    grants = [e for e in log.of_kind("transition_granted") if e["transition"] == "climb"]
    assert grants                                           # climbs only through the manager
    # the result is whatever it was, and the run says so: a recovery that ended
    # unresolved or lost fails it
    outcomes = [e["outcome"] for e in log.of_kind("recovery_outcome")]
    if any(o in ("unresolved", "lost") for o in outcomes):
        assert code != 0


def test_an_obstacle_above_every_transit_level_ends_the_drone_lost_not_through_it(tmp_path):
    code, batch, s = run(tmp_path, "--obstacle", "tower:12,28,12,28,100", "--full-sweep")
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    _always(m, log)
    outcomes = [(e["vehicle_id"], e["outcome"]) for e in log.of_kind("recovery_outcome")]
    assert outcomes
    assert code != 0                                        # impacts, lost or unresolved


@pytest.mark.parametrize("allocation", ["dynamic", "static"])
def test_the_same_manager_runs_in_every_condition(tmp_path, allocation):
    _, batch, s = run(tmp_path, "--allocation", allocation)
    m = s["runs"][0]
    log = read_run(batch / "run_00.jsonl")
    (rec,) = log.of_kind("airspace")
    assert rec["managed"] is True and rec["source"] == "transition_manager"
    _always(m, log)


def test_an_unmanaged_run_is_labelled(tmp_path):
    _, batch, s = run(tmp_path, airspace="unmanaged", layer_spacing_m=5)
    m = s["runs"][0]
    assert m["airspace_managed"] is False and m["airspace"] is None
    assert not read_run(batch / "run_00.jsonl").of_kind("transition_requested")
