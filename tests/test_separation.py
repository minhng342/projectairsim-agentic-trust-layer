"""Separation tracker and collision assessment (no simulator)."""
import math

import pytest

from models.telemetry import CollisionState
from utils.separation import SeparationTracker, assess, format_report

IDS = ["Drone1", "Drone2", "Drone3"]
T = 10_000_000_000


def poses(**xyz):
    """poses(Drone1=(n, e, alt), ...) -> adapter.latest_poses() format (NED down = -alt)."""
    return {v: (T, n, e, -alt) for v, (n, e, alt) in xyz.items()}


def test_three_drones_make_three_pairs():
    t = SeparationTracker(IDS)
    assert set(t.pairs) == {("Drone1", "Drone2"), ("Drone1", "Drone3"), ("Drone2", "Drone3")}


def test_minimum_3d_horizontal_and_vertical_parts():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, -3, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10)), phase="climb")
    t.update(poses(Drone1=(0, 0, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10)), phase="crossing")
    r = t.pairs[("Drone1", "Drone2")]
    assert r.min_3d_m == pytest.approx(2.0)              # directly below, 2 m apart vertically
    assert r.horizontal_at_min_m == pytest.approx(0.0)
    assert r.vertical_at_min_m == pytest.approx(2.0)
    assert r.phase_at_min == "crossing"
    assert r.min_horizontal_m == pytest.approx(0.0)
    assert r.samples == 2
    assert t.overall_min.pair == ("Drone1", "Drone2")


def test_minimum_is_kept_not_overwritten_by_later_larger_distances():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, 0, 5), Drone2=(0, 1.5, 5)), phase="close")
    t.update(poses(Drone1=(0, 0, 5), Drone2=(0, 20, 5)), phase="far")
    r = t.pairs[("Drone1", "Drone2")]
    assert r.min_3d_m == pytest.approx(1.5) and r.phase_at_min == "close"


def test_samples_from_different_moments_are_not_compared():
    t = SeparationTracker(IDS)
    t.update({"Drone1": (T, 0, 0, -5), "Drone2": (T + 200_000_000, 0, 0.1, -5)})
    assert t.pairs[("Drone1", "Drone2")].samples == 0 and t.skipped_samples == 1


def test_missing_vehicle_is_skipped():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, 0, 5), Drone2=(0, 3, 5)))
    assert t.pairs[("Drone1", "Drone3")].samples == 0


# ------------------------------------------------------------ assessment
def ground_contact(t_s):
    return CollisionState(has_collided=True, object_name="Ground", sim_time_ns=int(t_s * 1e9),
                          impact_speed_mps=0.2, is_resting_contact=True, normal_z=-1.0,
                          is_supporting_surface=True, count=1)


def test_pass_with_safe_separation_and_only_landing_contacts():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, -3, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10)))
    rep = assess(t, {v: [ground_contact(50)] for v in IDS}, IDS, min_separation_m=2.0)
    assert rep.passed, rep.violations
    assert "none" in format_report(rep)


def test_too_close_fails_and_names_the_pair_and_phase():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, -3, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10)), phase="ok")
    t.update(poses(Drone1=(0, 0, 6.0), Drone2=(0, 0.5, 7.0), Drone3=(0, 3, 10)), phase="D1:return")
    rep = assess(t, {}, IDS, min_separation_m=2.0)
    assert not rep.passed
    assert any("Drone1-Drone2" in v and "D1:return" in v for v in rep.violations)
    assert "FAIL Drone1-Drone2" in format_report(rep)


def test_drone_to_drone_collision_fails_even_if_separation_samples_missed_it():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, -3, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10)))
    hit = CollisionState(has_collided=True, object_name="Drone2", sim_time_ns=int(30e9),
                         impact_speed_mps=0.4, is_resting_contact=True, count=1)
    rep = assess(t, {"Drone1": [hit]}, IDS)
    assert not rep.passed and rep.vehicle_collisions == ["Drone1 collided with Drone2 at 30.00s"]


def test_impact_with_scenery_fails_but_resting_contact_does_not():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, -3, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10)))
    wall = CollisionState(has_collided=True, object_name="TemplateCube_3", sim_time_ns=int(12e9),
                          impact_speed_mps=4.2, is_resting_contact=False, count=1)
    rep = assess(t, {"Drone3": [ground_contact(50), wall]}, IDS)
    assert rep.impacts == ["Drone3 impact with TemplateCube_3 at 12.00s (~4.2 m/s)"]
    assert not rep.passed


def test_unsampled_pair_fails_rather_than_passing_silently():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, 0, 5), Drone2=(0, 3, 5)))      # Drone3 never reported
    rep = assess(t, {}, IDS)
    assert not rep.passed and any("never sampled" in v for v in rep.violations)


def test_live_pass_5_2_final_positions_are_safe():
    """Final hover positions from the live run: ~3.3 m apart at landing height."""
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0.4, -3.3, 4.3), Drone2=(0.0, 0.0, 4.3), Drone3=(0.3, 3.3, 4.3)))
    rep = assess(t, {}, IDS, 2.0)
    assert rep.passed
    closest = rep.pairs[0]                                    # Drone2-Drone3
    assert closest.pair == ("Drone2", "Drone3")
    assert closest.min_3d_m == pytest.approx(math.hypot(0.3, 3.3), abs=0.01)
