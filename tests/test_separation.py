"""Separation tracker and collision assessment (no simulator)."""
import math

import pytest

from models.telemetry import CollisionState
from utils.separation import SeparationTracker, assess, format_report

IDS = ["Drone1", "Drone2", "Drone3"]
T = 10_000_000_000
_clock = {"t": T}


def poses(ts=None, **xyz):
    """poses(Drone1=(n, e, alt), ...) -> adapter.latest_poses() format (NED down = -alt).

    Each call advances the sim timestamp by 50 ms (like a live stream) unless ts is given.
    """
    if ts is None:
        _clock["t"] += 50_000_000
        ts = _clock["t"]
    return {v: (ts, n, e, -alt) for v, (n, e, alt) in xyz.items()}


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
    r = t.pairs[("Drone1", "Drone2")]
    assert r.samples == 0 and r.skewed == 1 and r.attempts == 1


def test_missing_vehicle_is_skipped():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, 0, 5), Drone2=(0, 3, 5)))
    assert t.pairs[("Drone1", "Drone3")].samples == 0
    assert t.pairs[("Drone1", "Drone3")].missing == 1


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


def test_unsampled_pair_is_inconclusive_not_pass():
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0, 0, 5), Drone2=(0, 3, 5)))      # Drone3 never reported
    rep = assess(t, {}, IDS)
    assert not rep.passed and rep.status == "inconclusive"
    assert any("never sampled" in c for c in rep.coverage_problems)
    assert rep.violations == []                               # nothing unsafe was SEEN


def test_live_pass_5_2_final_positions_are_safe():
    """Final hover positions from the live run: ~3.3 m apart at landing height."""
    t = SeparationTracker(IDS)
    t.update(poses(Drone1=(0.4, -3.3, 4.3), Drone2=(0.0, 0.0, 4.3), Drone3=(0.3, 3.3, 4.3)))
    rep = assess(t, {}, IDS, 2.0)
    assert rep.passed
    closest = rep.pairs[0]                                    # Drone2-Drone3
    assert closest.pair == ("Drone2", "Drone3")
    assert closest.min_3d_m == pytest.approx(math.hypot(0.3, 3.3), abs=0.01)


# ------------------------------------------------------------ coverage (Pass 5.4)
SAFE = dict(Drone1=(0, -3, 6), Drone2=(0, 0, 8), Drone3=(0, 3, 10))


def run(tracker, frames):
    """frames: [(host_time, poses_dict), ...]"""
    for host, p in frames:
        tracker.update(p, phase="x", host_time=host)


def test_healthy_20hz_stream_passes_with_full_coverage():
    t = SeparationTracker(IDS)
    run(t, [(i * 0.05, poses(**SAFE)) for i in range(200)])
    t.finish(host_time=200 * 0.05)
    rep = assess(t, {}, IDS)
    assert rep.status == "pass", rep.coverage_problems
    assert all(r.samples == 200 and r.coverage == 1.0 for r in rep.pairs)
    assert all(r.max_gap_s <= 0.051 for r in rep.pairs)


def test_frozen_stream_does_not_inflate_samples_and_is_inconclusive():
    """Pose callbacks stop: latest_poses() keeps returning the same timestamp."""
    t = SeparationTracker(IDS)
    run(t, [(i * 0.05, poses(**SAFE)) for i in range(10)])
    stuck = poses(**SAFE)
    run(t, [(0.5 + i * 0.05, stuck) for i in range(100)])     # same ts every time
    t.finish(host_time=5.5)
    rep = assess(t, {}, IDS)
    r = rep.pairs[0]
    assert r.samples == 11 and r.frozen == 99 and r.attempts == 110
    assert rep.status == "inconclusive"
    assert any("usable" in c for c in rep.coverage_problems)
    assert any("without a usable sample" in c for c in rep.coverage_problems)


def test_replayed_older_timestamp_is_rejected():
    t = SeparationTracker(IDS)
    new = poses(**SAFE)
    old = poses(ts=T, **SAFE)                                 # earlier than `new`
    run(t, [(0.0, new), (0.05, old)])
    assert t.pairs[("Drone1", "Drone2")].samples == 1
    assert t.pairs[("Drone1", "Drone2")].frozen == 1


def test_mostly_out_of_sync_samples_are_inconclusive():
    t = SeparationTracker(IDS)
    frames = []
    for i in range(100):
        p = poses(**SAFE)
        if i % 2:                                             # half the time Drone3 lags 200 ms
            ts, n, e, d = p["Drone3"]
            p["Drone3"] = (ts - 200_000_000, n, e, d)
        frames.append((i * 0.05, p))
    run(t, frames)
    t.finish(host_time=5.0)
    rep = assess(t, {}, IDS)
    assert rep.status == "inconclusive"
    assert any("Drone1-Drone3" in c and "out of sync" in c for c in rep.coverage_problems)
    assert not any(c.startswith("Drone1-Drone2") for c in rep.coverage_problems)


def test_gap_in_the_middle_is_inconclusive_even_with_high_coverage():
    """Sampler stalled for 2 s (e.g. event loop blocked): no attempts, so % looks fine."""
    t = SeparationTracker(IDS)
    run(t, [(i * 0.05, poses(**SAFE)) for i in range(100)])
    run(t, [(7.0 + i * 0.05, poses(**SAFE)) for i in range(100)])   # 4.95 -> 7.0 gap
    t.finish(host_time=12.0)
    rep = assess(t, {}, IDS)
    assert all(r.coverage == 1.0 for r in rep.pairs)
    assert rep.status == "inconclusive"
    assert all(r.max_gap_s == pytest.approx(2.05) for r in rep.pairs)


def test_gap_at_the_end_is_caught_by_finish():
    """Stream dies near the end; without finish() the tail would be invisible."""
    t = SeparationTracker(IDS)
    run(t, [(i * 0.05, poses(**SAFE)) for i in range(100)])          # last counted at 4.95
    rep_before = assess(t, {}, IDS)
    assert rep_before.status == "pass"
    t.finish(host_time=8.0)
    rep = assess(t, {}, IDS)
    assert rep.status == "inconclusive"
    assert "INSUFFICIENT" in format_report(rep) and "INCONCLUSIVE" in format_report(rep)


def test_fail_outranks_inconclusive():
    t = SeparationTracker(IDS)
    run(t, [(0.0, poses(Drone1=(0, 0, 5), Drone2=(0, 0.5, 5)))])     # too close, Drone3 missing
    rep = assess(t, {}, IDS)
    assert rep.coverage_problems and rep.violations
    assert rep.status == "fail"
    assert "Separation verdict: FAIL" in format_report(rep)


def test_counted_sample_after_frozen_period_uses_new_timestamps():
    t = SeparationTracker(IDS)
    a = poses(**SAFE)
    run(t, [(0.0, a), (0.05, a), (0.10, poses(**SAFE))])
    r = t.pairs[("Drone1", "Drone2")]
    assert r.samples == 2 and r.frozen == 1 and r.max_gap_s == pytest.approx(0.10)
