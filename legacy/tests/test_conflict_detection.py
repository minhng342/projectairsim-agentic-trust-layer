"""Conflict detection math: head-on, crossing, parallel, vertical separation, diverging,
already in conflict, stale reports, garbage input."""
import math

import pytest

from legacy.corridor.conflict_detection import (
    Minima,
    Track,
    detect,
    predict,
    prediction_view,
    track_from_peer,
)

M = Minima(horizontal_m=5.0, vertical_m=2.0)


def T(id, n, e, alt=10.0, vn=0.0, ve=0.0, vup=0.0):
    return Track(id, n, e, alt, vn, ve, vup)


def test_head_on_same_altitude():
    own = T("me", 0, 0, ve=1.0)
    other = T("x", 0, 26, ve=-1.0)                     # 26 m apart, closing at 2 m/s
    p = predict(own, other, M, 30)
    assert p.tcpa_s == pytest.approx(13.0) and p.dcpa_m == pytest.approx(0.0)
    assert p.t_los_s == pytest.approx((26 - 5) / 2) and p.conflict and not p.in_los_now
    assert p.los_end_s == pytest.approx((26 + 5) / 2)


def test_perpendicular_crossing_arriving_together():
    own = T("me", 0, -10, ve=1.0)                      # reaches (0,0) at t=10
    other = T("x", -10, 0, vn=1.0)                     # also at t=10
    p = predict(own, other, M, 30)
    assert p.tcpa_s == pytest.approx(10.0) and p.dcpa_m == pytest.approx(0.0, abs=1e-9)
    assert p.t_los_s == pytest.approx(10 - 5 / math.sqrt(2))


def test_crossing_with_a_time_gap_is_no_conflict():
    own = T("me", 0, -10, ve=1.0)                      # at the crossing at t=10
    other = T("x", -20, 0, vn=1.0)                     # at the crossing at t=20
    assert not predict(own, other, M, 30).conflict


def test_vertical_separation_prevents_conflict_and_climbing_into_it_does_not():
    own = T("me", 0, 0, alt=10, ve=1.0)
    level_apart = T("x", 0, 26, alt=13, ve=-1.0)       # 3 m above: fine
    assert not predict(own, level_apart, M, 30).conflict
    # descending at 0.2 m/s it is within 2 m vertically from t=5 to t=25: overlaps the
    # horizontal LOS window (10.5..15.5), so a conflict starting when horizontal LOS does
    slow = T("x", 0, 26, alt=13, ve=-1.0, vup=-0.2)
    p = predict(own, slow, M, 30)
    assert p.conflict and p.t_los_s == pytest.approx((26 - 5) / 2)
    # at 0.5 m/s it passes through the layer (t=2..10) before they meet: no conflict
    fast = T("x", 0, 26, alt=13, ve=-1.0, vup=-0.5)
    assert not predict(own, fast, M, 30).conflict


def test_parallel_same_direction_far_enough_apart():
    own = T("me", 0, 0, ve=2.0)
    other = T("x", 6, 0, ve=2.0)
    assert not predict(own, other, M, 30).conflict


def test_diverging_has_tcpa_zero_and_no_conflict():
    own = T("me", 0, 0, ve=-1.0)
    other = T("x", 0, 8, ve=1.0)
    p = predict(own, other, M, 30)
    assert p.tcpa_s == 0.0 and p.dcpa_m == pytest.approx(8.0) and not p.conflict


def test_already_in_loss_of_separation():
    own = T("me", 0, 0)
    other = T("x", 1, 1, alt=10.5)
    p = predict(own, other, M, 30)
    assert p.in_los_now and p.t_los_s == 0.0 and p.los_end_s == 30.0


def test_conflict_beyond_the_lookahead_is_not_reported():
    own = T("me", 0, 0, ve=1.0)
    other = T("x", 0, 100, ve=-1.0)                    # LOS at t=47.5
    assert not predict(own, other, M, 30).conflict
    assert predict(own, other, M, 60).conflict


def test_detect_sorts_soonest_first_and_skips_self_and_garbage():
    own = T("me", 0, 0, ve=1.0)
    far = T("far", 0, 40, ve=-1.0)
    near = T("near", 0, 20, ve=-1.0)
    garbage = T("bad", math.nan, 0)
    out = detect(own, [far, near, garbage, own], M, 30)
    assert [p.other_id for p in out] == ["near", "far"]


def test_stale_report_is_extrapolated_along_its_reported_motion():
    peer = {"uas_id": "x", "age_s": 2.0, "position": {"north_m": 0, "east_m": 10, "altitude_m": 10},
            "ground_speed_mps": 1.0, "track_deg": 270.0, "vertical_speed_mps": 0.0}
    t = track_from_peer(peer)
    assert t.east_m == pytest.approx(8.0) and t.v_east_mps == pytest.approx(-1.0)
    hover = dict(peer, track_deg=None, ground_speed_mps=0.1)
    assert track_from_peer(hover).east_m == pytest.approx(10.0)


def test_missing_reported_values_give_a_non_finite_track_that_detect_skips():
    peer = {"uas_id": "x", "age_s": 0.0, "position": {"north_m": None, "east_m": 1, "altitude_m": 10}}
    assert not track_from_peer(peer).finite
    assert detect(T("me", 0, 0), [track_from_peer(peer)], M) == []


def test_prediction_view_is_rounded_and_json_safe():
    import json
    v = prediction_view(predict(T("me", 0, 0, ve=1.0), T("x", 0, 26, ve=-1.0), M, 30))
    json.dumps(v)
    assert v["with"] == "x" and v["time_to_loss_of_separation_s"] == 10.5


def test_intent_track_puts_the_peer_level_at_its_announced_target():
    from legacy.corridor.conflict_detection import intent_track_from_peer
    peer = {"uas_id": "x", "age_s": 0.0, "position": {"north_m": 0, "east_m": 10, "altitude_m": 11},
            "ground_speed_mps": 1.0, "track_deg": 270.0, "vertical_speed_mps": 1.5,
            "target_altitude_m": 17.0}
    t = intent_track_from_peer(peer)
    assert (t.id, t.altitude_m, t.v_up_mps) == ("x", 17.0, 0.0)
    assert t.v_east_mps == pytest.approx(-1.0)                       # same horizontal motion
    assert intent_track_from_peer(dict(peer, target_altitude_m=None)) is None
    assert intent_track_from_peer(dict(peer, target_altitude_m=11.2)) is None   # already there
    assert intent_track_from_peer({k: v for k, v in peer.items() if k != "target_altitude_m"}) is None
