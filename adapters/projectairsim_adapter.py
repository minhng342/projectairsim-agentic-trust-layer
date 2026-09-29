"""
Project AirSim adapter (Pass 4): connect, load the world, cache telemetry,
validate it, and return one normalized TelemetrySnapshot per vehicle.

No drone commands yet; those come in a later pass.

Data sources (from Pass 3 exploration):
- actual_pose topic (~330 Hz, push): cached for freshness / liveness and
  to estimate speed at collision time. Only the latest two messages are kept.
- collision_info topic (event, push): cached for collision state.
- get_ground_truth_kinematics() (pull): position, orientation, velocity,
  acceleration, angular velocity, all at one consistent sim timestamp.
- get_ground_truth_geo_location() (pull): lat / lon / MSL altitude.
- GPS / barometer / magnetometer are disabled in the sample robot config and
  are deliberately not used.

Usage:
    with ProjectAirSimAdapter(vehicle_ids=["Drone1"]) as adapter:
        snap = adapter.get_snapshot("Drone1")
"""
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from projectairsim import Drone, ProjectAirSimClient, World

from models.telemetry import (CollisionState, Quaternion, TelemetrySnapshot,
                              ValidationStatus, Vector3)
from validation.telemetry_validator import ValidationLimits, validate_snapshot

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SIM_CONFIG = os.path.join(
    _REPO_ROOT, "..", "ProjectAirSim-v1.0.1", "client", "python",
    "example_user_scripts", "sim_config") + os.sep

NAN = float("nan")


# ---------------------------------------------------------------- helpers
def _vec(d: dict | None) -> Vector3:
    d = d or {}
    return Vector3(x=float(d.get("x", NAN)), y=float(d.get("y", NAN)), z=float(d.get("z", NAN)))


def _quat(d: dict | None) -> Quaternion:
    d = d or {}
    return Quaternion(w=float(d.get("w", NAN)), x=float(d.get("x", NAN)),
                      y=float(d.get("y", NAN)), z=float(d.get("z", NAN)))


def yaw_deg_from_quaternion(q: Quaternion) -> float:
    """Heading (0-360, 0 = north, clockwise) from an NED orientation quaternion."""
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
    return math.degrees(yaw) % 360.0


def track_deg_from_velocity(v: Vector3, min_speed_mps: float) -> float | None:
    gs = math.hypot(v.x, v.y)
    if not math.isfinite(gs) or gs < min_speed_mps:
        return None
    return math.degrees(math.atan2(v.y, v.x)) % 360.0


@dataclass
class _VehicleCache:
    pose: dict | None = None
    pose_received_mono: float | None = None
    prev_pose: dict | None = None
    pose_count: int = 0
    collision: CollisionState = field(default_factory=CollisionState)


# ---------------------------------------------------------------- adapter
class ProjectAirSimAdapter:
    def __init__(
        self,
        vehicle_ids=("Drone1",),
        scene: str = "scene_basic_drone.jsonc",
        sim_config_path: str = DEFAULT_SIM_CONFIG,
        address: str = "127.0.0.1",
        load_delay_s: int = 2,
        limits: ValidationLimits = ValidationLimits(),
        collision_window_s: float = 1.0,
        impact_speed_threshold_mps: float = 1.0,
        min_track_speed_mps: float = 0.5,
    ):
        self.vehicle_ids = list(vehicle_ids)
        self.scene = scene
        self.sim_config_path = os.path.abspath(sim_config_path) + os.sep
        self.address = address
        self.load_delay_s = load_delay_s
        self.limits = limits
        self.collision_window_ns = int(collision_window_s * 1e9)
        self.impact_speed_threshold_mps = impact_speed_threshold_mps
        self.min_track_speed_mps = min_track_speed_mps

        self._client: ProjectAirSimClient | None = None
        self._world: World | None = None
        self._drones: dict[str, Drone] = {}
        self._cache: dict[str, _VehicleCache] = {v: _VehicleCache() for v in self.vehicle_ids}
        self._lock = threading.Lock()

    # ---------------- lifecycle
    def connect(self) -> None:
        self._client = ProjectAirSimClient(address=self.address)
        self._client.connect()
        self._world = World(self._client, self.scene,
                            delay_after_load_sec=self.load_delay_s,
                            sim_config_path=self.sim_config_path)
        for vid in self.vehicle_ids:
            drone = Drone(self._client, self._world, vid)
            self._drones[vid] = drone
            self._client.subscribe(drone.robot_info["actual_pose"], self._on_pose(vid))
            self._client.subscribe(drone.robot_info["collision_info"], self._on_collision(vid))

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    def drone(self, vehicle_id: str) -> Drone:
        """Raw Project AirSim handle. For test scripts only until commands are added."""
        return self._drones[vehicle_id]

    # ---------------- push callbacks (run on the client's receive thread)
    def _on_pose(self, vid: str):
        def cb(_topic, msg):
            now = time.perf_counter()
            with self._lock:
                c = self._cache[vid]
                c.prev_pose, c.pose = c.pose, msg
                c.pose_received_mono = now
                c.pose_count += 1
        return cb

    def _on_collision(self, vid: str):
        def cb(_topic, msg):
            with self._lock:
                c = self._cache[vid]
                speed = self._speed_from_poses(c.prev_pose, c.pose)
                # None = unknown (no pose history yet, e.g. a collision latched at spawn)
                resting = None if speed is None else speed < self.impact_speed_threshold_mps
                old = c.collision
                c.collision = CollisionState(
                    has_collided=True,
                    object_name=msg.get("object_name"),
                    sim_time_ns=msg.get("time_stamp"),
                    impact_speed_mps=speed,
                    is_resting_contact=resting,
                    count=old.count + 1,
                    impact_count=old.impact_count + (1 if resting is False else 0),
                )
        return cb

    @staticmethod
    def _speed_from_poses(prev: dict | None, cur: dict | None) -> float | None:
        if not prev or not cur:
            return None
        dt = (cur["time_stamp"] - prev["time_stamp"]) / 1e9
        if dt <= 0:
            return None
        p0, p1 = prev["position"], cur["position"]
        return math.dist((p0["x"], p0["y"], p0["z"]), (p1["x"], p1["y"], p1["z"])) / dt

    # ---------------- snapshots
    def get_snapshot(self, vehicle_id: str) -> TelemetrySnapshot:
        drone = self._drones[vehicle_id]
        pull_errors: list[str] = []

        try:
            kin = drone.get_ground_truth_kinematics() or {}
        except Exception as err:
            kin = {}
            pull_errors.append(f"get_ground_truth_kinematics failed: {err}")
        try:
            geo = drone.get_ground_truth_geo_location() or {}
        except Exception as err:
            geo = {}
            pull_errors.append(f"get_ground_truth_geo_location failed: {err}")

        received_at = datetime.now(timezone.utc)
        with self._lock:
            c = self._cache[vehicle_id]
            pose_ts = c.pose.get("time_stamp") if c.pose else None
            age_ms = ((time.perf_counter() - c.pose_received_mono) * 1000.0
                      if c.pose_received_mono is not None else None)
            collision = c.collision.model_copy()

        sim_time_ns = int(kin.get("time_stamp") or 0)
        if collision.sim_time_ns is not None and sim_time_ns > 0:
            collision.in_contact = (sim_time_ns - collision.sim_time_ns) <= self.collision_window_ns

        pose = kin.get("pose", {})
        twist = kin.get("twist", {})
        accels = kin.get("accels", {})
        position = _vec(pose.get("position"))
        orientation = _quat(pose.get("orientation"))
        velocity = _vec(twist.get("linear"))

        snap = TelemetrySnapshot(
            vehicle_id=vehicle_id,
            sim_time_ns=sim_time_ns,
            received_at_utc=received_at,
            pose_topic_sim_time_ns=pose_ts,
            position_ned_m=position,
            orientation_quaternion=orientation,
            velocity_ned_mps=velocity,
            acceleration_ned_mps2=_vec(accels.get("linear")),
            angular_velocity_rad_s=_vec(twist.get("angular")),
            latitude_deg=float(geo.get("latitude", NAN)),
            longitude_deg=float(geo.get("longitude", NAN)),
            altitude_msl_m=float(geo.get("altitude", NAN)),
            altitude_local_m=-position.z,
            heading_deg=yaw_deg_from_quaternion(orientation),
            track_deg=track_deg_from_velocity(velocity, self.min_track_speed_mps),
            ground_speed_mps=math.hypot(velocity.x, velocity.y),
            vertical_speed_mps=-velocity.z,
            collision=collision,
            telemetry_age_ms=age_ms,
        )
        snap = validate_snapshot(snap, self.limits)
        if pull_errors:
            snap.validation_errors = pull_errors + snap.validation_errors
            snap.validation_status = ValidationStatus.INVALID
        return snap

    def get_all_snapshots(self) -> list[TelemetrySnapshot]:
        return [self.get_snapshot(v) for v in self.vehicle_ids]

    def topic_stats(self) -> dict:
        with self._lock:
            return {v: {"actual_pose_msgs": c.pose_count, "collisions": c.collision.count}
                    for v, c in self._cache.items()}
