"""
Minimal stand-in for the `projectairsim` package so adapter logic can be tested
without the simulator. Values come from the Pass 3 recording.

install() registers it as `projectairsim` in sys.modules; call it before
importing adapters.projectairsim_adapter.
"""
import sys
import threading
import types

GOOD_KINEMATICS = {
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
    instances = []

    def __init__(self, address="127.0.0.1", **_):
        self.subs = {}
        self.connected = False
        self.disconnect_calls = 0
        FakeClient.instances.append(self)

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.disconnect_calls += 1

    def subscribe(self, topic, cb):
        self.subs[topic] = cb

    def publish(self, topic, msg):
        """Mimics the real receive loop: callback exceptions are NOT caught."""
        self.subs[topic](topic, msg)


class FakeWorld:
    fail_next = False

    def __init__(self, client, scene, delay_after_load_sec=0, sim_config_path=""):
        if FakeWorld.fail_next:
            FakeWorld.fail_next = False
            raise RuntimeError("scene load failed")
        self.scene = scene


class FakeDrone:
    def __init__(self, client, world, name):
        base = f"/Sim/Scene/robots/{name}"
        self.robot_info = {"actual_pose": f"{base}/actual_pose",
                           "collision_info": f"{base}/collision_info"}
        self.kinematics = GOOD_KINEMATICS
        self.geo = GOOD_GEO
        self.landed = 1  # FLYING
        self.landed_raises = False
        self._in_request = threading.Lock()
        self.overlapping_requests = 0

    def _request(self, value):
        # Detect concurrent use of the (non-thread-safe) request socket.
        if not self._in_request.acquire(blocking=False):
            self.overlapping_requests += 1
            return value
        try:
            import time
            time.sleep(0.002)
            return value
        finally:
            self._in_request.release()

    def get_ground_truth_kinematics(self):
        return self._request(self.kinematics)

    def get_ground_truth_geo_location(self):
        return self._request(self.geo)

    def get_landed_state(self):
        if self.landed_raises:
            raise RuntimeError("request timed out")
        return self._request(self.landed)


def install():
    mod = types.ModuleType("projectairsim")
    mod.ProjectAirSimClient, mod.World, mod.Drone = FakeClient, FakeWorld, FakeDrone
    sys.modules["projectairsim"] = mod
    return mod
