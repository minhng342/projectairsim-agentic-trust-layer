"""
Offline tests: no simulator needed. A fake `projectairsim` module replays the
values recorded in Pass 3 (telemetry_dump.json) so the adapter's math,
caching, collision logic and validation can be checked anywhere.

    python -m tests.test_adapter_offline        (from the repo root)
    python -m pytest tests                      (if pytest is installed)
"""
import math
import sys
import time
import types

# ------------------------------------------------------------ fake simulator
GOOD_KINEMATICS = {  # get_ground_truth_kinematics() sample from Pass 3
    "time_stamp": 9711000000,
    "pose": {"position": {"x": 10.3007, "y": 16.5129, "z": -9.0487},
             "orientation": {"w": 0.92265, "x": -0.00293, "y": 0.07677, "z": -0.37790}},
    "twist": {"linear": {"x": 0.98658, "y": 2.64307, "z": -0.14708},
              "angular": {"x": 0.08641, "y": -0.19053, "z": 0.00215}},
    "accels": {"linear": {"x": -1.37573, "y": 0.47591, "z": 0.49863},
               "angular": {"x": -0.08733, "y": 0.21280, "z": -0.00286}},
}
GOOD_GEO = {"latitude": 47.641560559, "longitude": -122.139944731, "altitude": 131.049}


class FakeClient:
    def __init__(self, address="127.0.0.1", **_):
        self.subs = {}

    def connect(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, cb):
        self.subs[topic] = cb

    def publish(self, topic, msg):
        self.subs[topic](topic, msg)


class FakeWorld:
    def __init__(self, client, scene, delay_after_load_sec=0, sim_config_path=""):
        self.scene = scene


class FakeDrone:
    kinematics = GOOD_KINEMATICS
    geo = GOOD_GEO

    def __init__(self, client, world, name):
        base = f"/Sim/Scene/robots/{name}"
        self.robot_info = {"actual_pose": f"{base}/actual_pose",
                           "collision_info": f"{base}/collision_info"}

    def get_ground_truth_kinematics(self):
        return self.kinematics

    def get_ground_truth_geo_location(self):
        return self.geo


fake = types.ModuleType("projectairsim")
fake.ProjectAirSimClient, fake.World, fake.Drone = FakeClient, FakeWorld, FakeDrone
sys.modules["projectairsim"] = fake

from adapters.projectairsim_adapter import ProjectAirSimAdapter, yaw_deg_from_quaternion  # noqa: E402
from models.telemetry import Quaternion, ValidationStatus  # noqa: E402


def make_adapter():
    a = ProjectAirSimAdapter(vehicle_ids=["Drone1"])
    a.connect()
    return a


def pose_msg(ts_ns, x, y, z):
    return {"time_stamp": ts_ns, "position": {"x": x, "y": y, "z": z},
            "orientation": {"w": 0.92388, "x": 0, "y": 0, "z": -0.38268}}


def feed_pose(a, *msgs):
    topic = a.drone("Drone1").robot_info["actual_pose"]
    for m in msgs:
        a._client.publish(topic, m)


# ------------------------------------------------------------ tests
def test_heading_from_spawn_quaternion():
    # Pass 3 spawn orientation: w=0.9239, z=-0.3827 -> yaw -45 deg -> 315
    h = yaw_deg_from_quaternion(Quaternion(w=0.9238795, x=0, y=0, z=-0.3826834))
    assert abs(h - 315.0) < 0.01, h


def test_valid_snapshot_and_derived_fields():
    a = make_adapter()
    feed_pose(a, pose_msg(9_708_000_000, 10.29, 16.50, -9.05), pose_msg(9_711_000_000, 10.30, 16.51, -9.05))
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.VALID, s.validation_errors
    assert abs(s.ground_speed_mps - math.hypot(0.98658, 2.64307)) < 1e-6
    assert abs(s.vertical_speed_mps - 0.14708) < 1e-6          # NED z<0 = climbing
    assert abs(s.altitude_local_m - 9.0487) < 1e-6
    assert abs(s.track_deg - math.degrees(math.atan2(2.64307, 0.98658))) < 1e-6
    assert s.sim_time_s == 9.711
    assert s.telemetry_age_ms is not None and s.telemetry_age_ms < 100


def test_no_pose_messages_is_invalid():
    a = make_adapter()
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.INVALID
    assert any("no messages" in e for e in s.validation_errors)


def test_stale_pose_is_stale():
    a = make_adapter()
    a.limits = type(a.limits)(max_telemetry_age_ms=50)
    feed_pose(a, pose_msg(9_711_000_000, 10.30, 16.51, -9.05))
    time.sleep(0.1)
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.STALE, s.validation_errors


def test_disabled_sensor_style_values_are_invalid():
    """Values modeled on the disabled GPS / magnetometer output from Pass 3."""
    a = make_adapter()
    feed_pose(a, pose_msg(1, 0, 0, 0))
    bad = {"time_stamp": 0,
           "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1, "x": 0, "y": 0, "z": 0}},
           "twist": {"linear": {"x": 1.875, "y": -6.691897e29, "z": 1.875},
                     "angular": {"x": 0, "y": 0, "z": 0}},
           "accels": {"linear": {"x": float("nan"), "y": float("nan"), "z": 0.0},
                      "angular": {"x": 0, "y": 0, "z": 0}}}
    a.drone("Drone1").kinematics = bad
    a.drone("Drone1").geo = {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0}
    s = a.get_snapshot("Drone1")
    errs = " | ".join(s.validation_errors)
    assert s.validation_status == ValidationStatus.INVALID
    for expected in ("sim_time_ns=0", "not finite", "0,0", "ground_speed_mps implausible"):
        assert expected in errs, (expected, errs)


def test_missing_fields_do_not_crash():
    a = make_adapter()
    feed_pose(a, pose_msg(9_711_000_000, 0, 0, 0))
    a.drone("Drone1").kinematics = {}
    a.drone("Drone1").geo = {}
    s = a.get_snapshot("Drone1")
    assert s.validation_status == ValidationStatus.INVALID


def test_resting_contact_vs_impact():
    a = make_adapter()
    coll_topic = a.drone("Drone1").robot_info["collision_info"]
    # landing: 3 mm moved in 3 ms = 1 m/s? keep it slower: 0.3 mm in 3 ms = 0.1 m/s
    feed_pose(a, pose_msg(9_700_000_000, 10.3, 16.5, -1.1920), pose_msg(9_703_000_000, 10.3, 16.5, -1.1917))
    a._client.publish(coll_topic, {"time_stamp": 9_703_000_000, "object_name": "Ground"})
    s = a.get_snapshot("Drone1")
    assert s.collision.in_contact and s.collision.is_resting_contact is True
    assert s.collision.impact_count == 0 and not s.validation_warnings

    # impact: 30 mm in 3 ms = 10 m/s
    feed_pose(a, pose_msg(9_706_000_000, 10.3, 16.5, -1.19), pose_msg(9_709_000_000, 10.33, 16.5, -1.19))
    a._client.publish(coll_topic, {"time_stamp": 9_709_000_000, "object_name": "TemplateCube_Rounded_1"})
    s = a.get_snapshot("Drone1")
    assert s.collision.is_resting_contact is False and s.collision.impact_count == 1
    assert abs(s.collision.impact_speed_mps - 10.0) < 0.01
    assert any("collision impact" in w for w in s.validation_warnings)
    assert s.validation_status == ValidationStatus.VALID  # safety event, not bad data


def test_collision_before_any_pose_is_unknown_not_impact():
    a = make_adapter()
    coll_topic = a.drone("Drone1").robot_info["collision_info"]
    a._client.publish(coll_topic, {"time_stamp": 528_000_000, "object_name": "TemplateCube_Rounded_1"})
    feed_pose(a, pose_msg(9_711_000_000, 10.3, 16.5, -9.05))
    s = a.get_snapshot("Drone1")
    assert s.collision.is_resting_contact is None and s.collision.impact_count == 0
    assert s.collision.in_contact is False  # 9.2 s ago in sim time


if __name__ == "__main__":
    tests = [(n, f) for n, f in dict(globals()).items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL {name}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
