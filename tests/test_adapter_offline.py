"""
Offline tests: no simulator needed. tests/fake_projectairsim.py replays values
recorded in Pass 3 so the adapter's math, caching, collision logic, callback
robustness and validation can be checked anywhere.

    python -m pytest tests                      (from the repo root)
    python -m tests.test_adapter_offline        (no pytest needed)
"""
import math
import sys
import threading
import time

from tests import fake_projectairsim as fake

fake.install()

from adapters.projectairsim_adapter import ProjectAirSimAdapter, yaw_deg_from_quaternion  # noqa: E402
from models.telemetry import GroundState, LandedState, Quaternion, ValidationStatus  # noqa: E402
from validation.telemetry_validator import ValidationLimits  # noqa: E402

T0 = 9_711_000_000  # kinematics time_stamp in GOOD_KINEMATICS


# ------------------------------------------------------------ helpers
def make_adapter(**kw):
    a = ProjectAirSimAdapter(vehicle_ids=["Drone1"], **kw)
    a.connect()
    return a


def pose_msg(ts_ns, x, y, z):
    return {"time_stamp": ts_ns, "position": {"x": x, "y": y, "z": z},
            "orientation": {"w": 0.92388, "x": 0, "y": 0, "z": -0.38268}}


def publish_pose(a, *msgs):
    topic = a.drone("Drone1").robot_info["actual_pose"]
    for m in msgs:
        a._client.publish(topic, m)


def publish_collision(a, msg):
    a._client.publish(a.drone("Drone1").robot_info["collision_info"], msg)


def fresh_pose(a, ts=T0):
    publish_pose(a, pose_msg(ts - 3_000_000, 10.30, 16.51, -9.05), pose_msg(ts, 10.30, 16.51, -9.05))


def straight_line(t_end_ns, speed_mps, n=40, dt_ns=3_000_000, z=-1.19):
    """n poses ending at t_end_ns, moving north at speed_mps."""
    out = []
    for i in range(n):
        t = t_end_ns - (n - 1 - i) * dt_ns
        out.append(pose_msg(t, 10.0 + speed_mps * (t - t_end_ns) / 1e9, 16.5, z))
    return out


# ------------------------------------------------------------ derived values
def test_heading_from_spawn_quaternion():
    h = yaw_deg_from_quaternion(Quaternion(w=0.9238795, x=0, y=0, z=-0.3826834))
    assert abs(h - 315.0) < 0.01, h


def test_valid_snapshot_and_derived_fields():
    a = make_adapter()
    fresh_pose(a)
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.VALID, s.validation_errors
    assert abs(s.ground_speed_mps - math.hypot(0.98658, 2.64307)) < 1e-6
    assert abs(s.vertical_speed_mps - 0.14708) < 1e-6          # NED z<0 = climbing
    assert abs(s.altitude_local_m - 9.0487) < 1e-6
    assert abs(s.track_deg - math.degrees(math.atan2(2.64307, 0.98658))) < 1e-6
    assert s.landed_state == LandedState.FLYING
    assert s.sim_time_s == 9.711


# ------------------------------------------------------------ freshness / timestamps
def test_no_pose_messages_is_invalid():
    a = make_adapter()
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.INVALID
    assert s.telemetry_age_ms is None


def test_stale_pose_is_stale():
    a = make_adapter(limits=ValidationLimits(max_telemetry_age_ms=50))
    fresh_pose(a)
    time.sleep(0.1)
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.STALE, s.validation_errors


def test_pose_kinematics_skew_is_invalid_not_warning():
    a = make_adapter()
    fresh_pose(a, ts=T0 - 250_000_000)  # pose topic 250 ms behind kinematics
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.INVALID
    assert any("differ by 250 ms" in e for e in s.validation_errors), s.validation_errors


# ------------------------------------------------------------ bad data
def test_disabled_sensor_style_values_are_invalid():
    """Values modeled on the disabled GPS / magnetometer output from Pass 3."""
    a = make_adapter()
    fresh_pose(a)
    d = a.drone("Drone1")
    d.kinematics = {"time_stamp": 0,
                    "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                             "orientation": {"w": 1, "x": 0, "y": 0, "z": 0}},
                    "twist": {"linear": {"x": 1.875, "y": -6.691897e29, "z": 1.875},
                              "angular": {"x": 0, "y": 0, "z": 0}},
                    "accels": {"linear": {"x": float("nan"), "y": float("nan"), "z": 0.0},
                               "angular": {"x": 0, "y": 0, "z": 0}}}
    d.geo = {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0}
    s = a.get_snapshot("Drone1")
    errs = " | ".join(s.validation_errors)
    assert s.validation_status == ValidationStatus.INVALID
    for expected in ("sim_time_ns=0", "not finite", "0,0", "ground_speed_mps implausible"):
        assert expected in errs, (expected, errs)


def test_missing_and_non_dict_responses_do_not_crash():
    a = make_adapter()
    fresh_pose(a)
    d = a.drone("Drone1")
    for kin, geo in (({}, {}), (None, None), ("garbage", 42), ({"pose": "x", "twist": 3}, {"latitude": "n/a"})):
        d.kinematics, d.geo = kin, geo
        s = a.get_snapshot("Drone1")
        assert s.validation_status == ValidationStatus.INVALID, (kin, geo)


def test_malformed_pose_messages_never_raise_and_make_snapshot_invalid():
    bad_messages = [
        None,
        "not a dict",
        {"position": {"x": 1, "y": 2, "z": 3}},                      # missing time_stamp
        {"time_stamp": T0},                                          # missing position
        {"time_stamp": T0, "position": {"x": "a", "y": 2, "z": 3}},  # non-numeric
        {"time_stamp": T0, "position": {"x": 1, "y": None, "z": 3}},
        {"time_stamp": 0, "position": {"x": 1, "y": 2, "z": 3}},     # zero timestamp
        {"time_stamp": "soon", "position": {"x": 1, "y": 2, "z": 3}},
    ]
    for bad in bad_messages:
        a = make_adapter()
        fresh_pose(a)
        publish_pose(a, bad)  # would raise into the receive thread if unguarded
        s = a.get_snapshot("Drone1")
        assert s.validation_status == ValidationStatus.INVALID, bad
        assert any("bad push message" in e for e in s.validation_errors), (bad, s.validation_errors)
        assert s.pose_topic_sim_time_ns == T0, "last valid sample must be preserved"
        assert a.topic_stats()["Drone1"]["bad_msgs"] == 1


def test_bad_message_error_expires_to_warning():
    a = make_adapter(bad_msg_error_window_s=0.05)
    fresh_pose(a, ts=T0 - 6_000_000)
    publish_pose(a, None)
    time.sleep(0.1)
    fresh_pose(a)
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.VALID, s.validation_errors
    assert any("bad push message" in w for w in s.validation_warnings)


def test_reversed_pose_timestamp_is_flagged_and_window_reset():
    a = make_adapter()
    publish_pose(a, *straight_line(T0, 5.0))
    publish_pose(a, pose_msg(1_000_000_000, 0, 0, -1))  # clock went backwards
    s = a.get_snapshot("Drone1")
    assert a.topic_stats()["Drone1"]["clock_resets"] == 1
    assert any("went backwards" in e for e in s.validation_errors)
    assert len(a._cache["Drone1"].pose_window) == 1


def test_malformed_collision_message_never_raises():
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, None)
    publish_collision(a, {"object_name": None, "time_stamp": "x"})
    s = a.get_snapshot("Drone1")
    assert s.collision.count == 1 and s.collision.recent_collision is False


# ------------------------------------------------------------ collisions
def test_resting_contact_vs_impact_using_pose_window():
    a = make_adapter()
    publish_pose(a, *straight_line(T0 - 200_000_000, 0.2))
    publish_collision(a, {"time_stamp": T0 - 200_000_000, "object_name": "Ground"})
    s = a.get_snapshot("Drone1")
    assert s.collision.recent_collision and s.collision.is_resting_contact is True
    assert abs(s.collision.impact_speed_mps - 0.2) < 0.01
    assert all("impact" not in w for w in s.validation_warnings)

    publish_pose(a, *straight_line(T0, 10.0))
    publish_collision(a, {"time_stamp": T0, "object_name": "TemplateCube_Rounded_1"})
    s = a.get_snapshot("Drone1")
    assert s.collision.is_resting_contact is False and s.collision.impact_count == 1
    assert abs(s.collision.impact_speed_mps - 10.0) < 0.01
    assert any("collision impact" in w for w in s.validation_warnings)
    assert s.validation_status == ValidationStatus.VALID  # safety event, not bad data


def test_collision_before_any_pose_is_unknown_not_impact():
    a = make_adapter()
    publish_collision(a, {"time_stamp": 528_000_000, "object_name": "TemplateCube_Rounded_1"})
    fresh_pose(a)
    s = a.get_snapshot("Drone1")
    assert s.collision.is_resting_contact is None and s.collision.impact_count == 0
    assert s.collision.recent_collision is False  # 9.2 s ago in sim time


def test_collision_from_the_future_is_not_recent():
    """Sim clock reset: an old collision timestamp can exceed the new clock."""
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, {"time_stamp": T0 + 500_000_000, "object_name": "Ground"})
    s = a.get_snapshot("Drone1")
    assert s.collision.recent_collision is False


# ------------------------------------------------------------ lifecycle
def test_reconnect_discards_previous_session_state():
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, {"time_stamp": T0, "object_name": "Ground"})
    publish_pose(a, None)
    a.connect()  # reconnect
    stats = a.topic_stats()["Drone1"]
    assert stats == {"actual_pose_msgs": 0, "bad_msgs": 0, "clock_resets": 0, "collisions": 0}
    fresh_pose(a)
    assert a.get_snapshot("Drone1").collision.has_collided is False


def test_failed_connect_cleans_up():
    fake.FakeWorld.fail_next = True
    a = ProjectAirSimAdapter(vehicle_ids=["Drone1"])
    try:
        a.connect()
        raise AssertionError("connect should have raised")
    except RuntimeError as err:
        assert "scene load failed" in str(err)
    assert not a.connected
    assert fake.FakeClient.instances[-1].disconnect_calls == 1
    try:
        a.get_snapshot("Drone1")
        raise AssertionError("snapshot on a disconnected adapter should raise")
    except RuntimeError:
        pass


# ------------------------------------------------------------ landed state
def test_landed_state_mapping_and_consistency_warning():
    a = make_adapter()
    fresh_pose(a)
    d = a.drone("Drone1")
    d.landed = 0
    s = a.get_snapshot("Drone1")
    assert s.landed_state == LandedState.LANDED
    assert any("LANDED but vehicle is moving" in w for w in s.validation_warnings)  # gs ~2.8 m/s

    d.landed_raises = True
    s = a.get_snapshot("Drone1")
    assert s.landed_state == LandedState.UNKNOWN
    assert s.validation_status == ValidationStatus.INVALID


# ------------------------------------------------------------ threading
def test_concurrent_snapshots_do_not_overlap_requests():
    a = make_adapter()
    fresh_pose(a)
    errors = []

    def worker():
        try:
            for _ in range(5):
                a.get_snapshot("Drone1")
        except Exception as err:  # collected so the main thread can assert on it
            errors.append(err)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], errors
    assert a.drone("Drone1").overlapping_requests == 0


# ------------------------------------------------------------ callback failures
def test_nan_collision_timestamp_is_counted():
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, {"time_stamp": float("nan"), "object_name": "Ground"})
    assert a.topic_stats()["Drone1"]["bad_msgs"] == 1
    s = a.get_snapshot("Drone1")
    assert s.collision.count == 1 and s.collision.recent_collision is False
    assert s.validation_status == ValidationStatus.INVALID


def test_unexpected_callback_exception_is_counted_not_raised():
    a = make_adapter()

    def boom(*_args):
        raise ZeroDivisionError("simulated bug")

    a._update_ground_detector = boom
    fresh_pose(a)  # must not raise into the receive thread
    stats = a.topic_stats()["Drone1"]
    assert stats["bad_msgs"] >= 1
    assert "ZeroDivisionError" in a._cache["Drone1"].last_bad_msg


# ------------------------------------------------------------ sim progress
def test_duplicate_timestamps_keep_transport_fresh_but_go_stale():
    a = make_adapter(limits=ValidationLimits(max_telemetry_age_ms=80))
    fresh_pose(a)
    for _ in range(6):                 # same time_stamp replayed for ~150 ms
        time.sleep(0.025)
        publish_pose(a, pose_msg(T0, 10.30, 16.51, -9.05))
    s = a.get_snapshot("Drone1")
    assert s.telemetry_age_ms < 80
    assert s.sim_progress_age_ms > 80
    assert s.validation_status == ValidationStatus.STALE, s.validation_errors
    assert any("has not advanced" in e for e in s.validation_errors)


# ------------------------------------------------------------ ground detector
STILL_KINEMATICS = dict(fake.GOOD_KINEMATICS,
                        twist={"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                               "angular": {"x": 0.0, "y": 0.0, "z": 0.0}})
UP = {"x": 0.0, "y": 0.0, "z": -1.0}        # NED normal of level ground (observed live)
WALL = {"x": -1.0, "y": 0.0, "z": 0.0}      # vertical surface


def descend_then_rest(a, t_end, rest_s, z_ground=-1.19, descend_mps=0.2, dt=3_000_000,
                      object_name="Ground", normal=UP):
    """Descend at 0.2 m/s for 0.3 s, touch (resting collision), then sit still."""
    rest_n = int(rest_s * 1e9 / dt)
    t_touch = t_end - rest_n * dt
    for i in range(100, 0, -1):
        t = t_touch - i * dt
        publish_pose(a, pose_msg(t, 10.4, 16.2, z_ground - descend_mps * (t_touch - t) / 1e9))
    publish_pose(a, pose_msg(t_touch, 10.4, 16.2, z_ground))
    publish_collision(a, {"time_stamp": t_touch, "object_name": object_name, "normal": normal})
    for i in range(1, rest_n + 1):
        publish_pose(a, pose_msg(t_touch + i * dt, 10.4, 16.2, z_ground))


def test_grounded_by_contact_while_raw_state_still_flying():
    """Reproduces the Pass 4.1 live run: touchdown, raw landed_state stays FLYING."""
    a = make_adapter()
    a.drone("Drone1").landed = 1  # FLYING (lagging)
    descend_then_rest(a, T0, rest_s=1.2)
    s = a.get_snapshot("Drone1")
    assert s.landed_state == LandedState.FLYING           # raw value untouched
    assert s.ground_state == GroundState.GROUNDED
    assert s.ground_state_basis == "resting_contact+still"


def test_not_grounded_before_stillness_dwell():
    a = make_adapter()
    descend_then_rest(a, T0, rest_s=0.6)
    s = a.get_snapshot("Drone1")
    assert s.ground_state == GroundState.AIRBORNE


def test_ground_latch_released_after_climbing():
    a = make_adapter()
    descend_then_rest(a, T0 - 600_000_000, rest_s=1.2)
    # take off: climb 1 m over 0.5 s, then hover still for 1.2 s at altitude
    t = T0 - 600_000_000
    for i in range(1, 168):
        t += 3_000_000
        publish_pose(a, pose_msg(t, 10.4, 16.2, -1.19 - 2.0 * i * 0.003))
    z_hover = -1.19 - 2.0 * 167 * 0.003
    for i in range(1, 400):
        t += 3_000_000
        publish_pose(a, pose_msg(t, 10.4, 16.2, z_hover))
    a.drone("Drone1").kinematics = dict(fake.GOOD_KINEMATICS, time_stamp=t)
    s = a.get_snapshot("Drone1")
    assert s.ground_state == GroundState.AIRBORNE, s.ground_state_basis


def test_impact_collision_does_not_latch_ground():
    a = make_adapter()
    publish_pose(a, *straight_line(T0 - 1_300_000_000, 10.0))
    publish_collision(a, {"time_stamp": T0 - 1_300_000_000, "object_name": "Wall"})
    for i in range(1, 434):
        publish_pose(a, pose_msg(T0 - 1_300_000_000 + i * 3_000_000, 10.0, 16.5, -1.19))
    s = a.get_snapshot("Drone1")
    assert s.collision.is_resting_contact is False
    assert s.ground_state == GroundState.AIRBORNE


def test_raw_landed_state_grounds_immediately_when_still():
    a = make_adapter()
    fresh_pose(a)
    d = a.drone("Drone1")
    d.landed, d.kinematics = 0, STILL_KINEMATICS
    s = a.get_snapshot("Drone1")
    assert s.ground_state == GroundState.GROUNDED and s.ground_state_basis == "landed_state"


def test_raw_landed_while_moving_is_not_grounded():
    """Review finding 1: raw LANDED at 2.8 m/s must not make the drone 'grounded'."""
    a = make_adapter()
    fresh_pose(a)
    a.drone("Drone1").landed = 0            # GOOD_KINEMATICS moves at ~2.8 m/s
    s = a.get_snapshot("Drone1")
    assert s.landed_state == LandedState.LANDED
    assert s.ground_state == GroundState.UNKNOWN
    assert "moving" in s.ground_state_basis


def test_low_speed_wall_contact_does_not_ground():
    """Review finding 2: a slow touch on a wall, then hovering still, is NOT grounded."""
    a = make_adapter()
    descend_then_rest(a, T0, rest_s=1.2, object_name="Wall", normal=WALL)
    s = a.get_snapshot("Drone1")
    assert s.collision.is_resting_contact is True
    assert s.collision.is_supporting_surface is False
    assert s.ground_state == GroundState.AIRBORNE, s.ground_state_basis


def test_contact_without_normal_does_not_ground():
    a = make_adapter()
    descend_then_rest(a, T0, rest_s=1.2, normal=None)
    assert a.get_snapshot("Drone1").ground_state == GroundState.AIRBORNE


def test_upward_facing_platform_contact_can_ground():
    """Supporting surfaces are recognized by the normal, not the object name."""
    a = make_adapter()
    descend_then_rest(a, T0, rest_s=1.2, z_ground=-6.0, object_name="Roof_Platform_3",
                      normal={"x": 0.1, "y": 0.2, "z": -0.97})
    s = a.get_snapshot("Drone1")
    assert s.collision.is_supporting_surface is True
    assert s.ground_state == GroundState.GROUNDED and s.ground_state_basis == "resting_contact+still"


def test_moving_away_from_contact_releases_latch():
    """Slide 1 m sideways at the same height after touching down, then stop."""
    a = make_adapter()
    descend_then_rest(a, T0 - 1_800_000_000, rest_s=0.3)
    t = T0 - 1_800_000_000
    for i in range(1, 168):                         # 1 m east over 0.5 s
        t += 3_000_000
        publish_pose(a, pose_msg(t, 10.4, 16.2 + 2.0 * i * 0.003, -1.19))
    while t < T0:                                   # then still for ~1.3 s
        t += 3_000_000
        publish_pose(a, pose_msg(t, 10.4, 16.2 + 2.0 * 167 * 0.003, -1.19))
    s = a.get_snapshot("Drone1")
    assert s.ground_state == GroundState.AIRBORNE, s.ground_state_basis


def test_clock_reset_clears_ground_latch_and_collision():
    """Review finding 3: an old-timeline collision can't ground the new timeline."""
    a = make_adapter()
    descend_then_rest(a, T0 + 5_000_000_000, rest_s=1.2)   # old timeline, later timestamps
    assert a._cache["Drone1"].contact_latch is not None
    t = 1_000_000_000                                       # clock restarts
    for i in range(500):                                    # still for 1.5 s on new timeline
        publish_pose(a, pose_msg(t + i * 3_000_000, 10.4, 16.2, -1.19))
    a.drone("Drone1").kinematics = dict(STILL_KINEMATICS, time_stamp=t + 499 * 3_000_000)
    s = a.get_snapshot("Drone1")
    assert a.topic_stats()["Drone1"]["clock_resets"] == 1
    assert a._cache["Drone1"].contact_latch is None
    assert s.collision.has_collided is False
    assert s.ground_state == GroundState.AIRBORNE, s.ground_state_basis



# ------------------------------------------------------------ evaluation read-outs
def test_latest_poses_returns_newest_cached_pose_per_vehicle():
    a = make_adapter()
    assert a.latest_poses() == {}
    publish_pose(a, pose_msg(T0 - 3_000_000, 1.0, 2.0, -3.0), pose_msg(T0, 1.5, 2.5, -3.5))
    assert a.latest_poses() == {"Drone1": (T0, 1.5, 2.5, -3.5)}


def test_collision_log_keeps_every_collision_not_just_the_latest():
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, {"time_stamp": T0 - 2_000_000_000, "object_name": "TemplateCube_1"})
    publish_collision(a, {"time_stamp": T0, "object_name": "Ground", "normal": UP})
    log = a.collision_log("Drone1")
    assert [c.object_name for c in log] == ["TemplateCube_1", "Ground"]
    assert a.get_snapshot("Drone1").collision.object_name == "Ground"   # snapshot: latest only


def test_clock_reset_clears_collision_log():
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, {"time_stamp": T0, "object_name": "Ground", "normal": UP})
    publish_pose(a, pose_msg(1_000_000_000, 0, 0, -1))      # clock went backwards
    assert a.collision_log("Drone1") == []


def test_reconnect_clears_collision_log():
    a = make_adapter()
    fresh_pose(a)
    publish_collision(a, {"time_stamp": T0, "object_name": "Ground", "normal": UP})
    a.connect()
    assert a.collision_log("Drone1") == []


if __name__ == "__main__":
    tests = [(n, f) for n, f in dict(globals()).items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as err:
            failed += 1
            print(f"FAIL {name}: {type(err).__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
