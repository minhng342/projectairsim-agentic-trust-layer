"""
Project AirSim adapter: connect, load the world, cache telemetry, validate it,
and return one normalized TelemetrySnapshot per vehicle.

Telemetry only. Command primitives come in Pass 5; agent-level
execute_action() will live in a separate executor behind the risk gate.

Data sources (from the Pass 3 exploration):
- actual_pose topic (~330 Hz, push): liveness / freshness, and a ~100 ms
  window used to estimate speed at collision time.
- collision_info topic (event, push): collision history.
- get_ground_truth_kinematics() (pull): position, orientation, velocity,
  acceleration, angular velocity at one consistent sim timestamp.
- get_ground_truth_geo_location() (pull): lat / lon / MSL altitude.
- get_landed_state() (pull): LANDED / FLYING, the primary on-ground signal.
- GPS / barometer / magnetometer are disabled in the robot config and are
  deliberately not used.

Threading
- Push callbacks run on Project AirSim's receive thread. The client does not
  catch callback exceptions, so an exception here would silently stop ALL
  telemetry. Every callback is therefore fully guarded; bad messages are
  counted and surfaced as snapshot errors instead.
- The client's synchronous request socket is not safe for concurrent callers,
  so all pull requests are serialized with _request_lock.

Usage:
    with ProjectAirSimAdapter(vehicle_ids=["Drone1"]) as adapter:
        snap = adapter.get_snapshot("Drone1")
"""
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from projectairsim import Drone, ProjectAirSimClient, World

from models.telemetry import (CollisionState, GroundState, LandedState,
                              Quaternion, TelemetrySnapshot, Vector3)
from validation.telemetry_validator import ValidationLimits, validate_snapshot

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SIM_CONFIG = os.path.join(_REPO_ROOT, "sim_config") + os.sep

NAN = float("nan")


# ---------------------------------------------------------------- helpers
def _vec(d) -> Vector3:
    d = d if isinstance(d, dict) else {}
    return Vector3(x=_num(d.get("x")), y=_num(d.get("y")), z=_num(d.get("z")))


def _quat(d) -> Quaternion:
    d = d if isinstance(d, dict) else {}
    return Quaternion(w=_num(d.get("w")), x=_num(d.get("x")),
                      y=_num(d.get("y")), z=_num(d.get("z")))


def _num(v) -> float:
    """Float or NaN; never raises. NaN is then caught by validation."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def yaw_deg_from_quaternion(q: Quaternion) -> float:
    """Heading (0-360, 0 = north, clockwise) from an NED orientation quaternion."""
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
    return math.degrees(yaw) % 360.0


def track_deg_from_velocity(v: Vector3, min_speed_mps: float) -> float | None:
    gs = math.hypot(v.x, v.y)
    if not math.isfinite(gs) or gs < min_speed_mps:
        return None
    return math.degrees(math.atan2(v.y, v.x)) % 360.0


def parse_pose_message(msg) -> tuple[int, float, float, float]:
    """Return (time_stamp_ns, x, y, z) or raise ValueError describing the problem."""
    if not isinstance(msg, dict):
        raise ValueError(f"message is {type(msg).__name__}, not a dict")
    ts = msg.get("time_stamp")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise ValueError(f"time_stamp missing or non-numeric ({ts!r})")
    if ts <= 0:
        raise ValueError(f"time_stamp={ts} is not positive")
    pos = msg.get("position")
    if not isinstance(pos, dict):
        raise ValueError("position missing")
    try:
        x, y, z = float(pos["x"]), float(pos["y"]), float(pos["z"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"position not numeric ({err})") from None
    if not all(map(math.isfinite, (x, y, z))):
        raise ValueError("position not finite")
    return int(ts), x, y, z


@dataclass
class _VehicleCache:
    pose_ts: int | None = None
    pose_received: float | None = None          # host perf_counter
    pose_window: deque = field(default_factory=deque)   # (ts_ns, x, y, z)
    pose_count: int = 0
    bad_msg_count: int = 0
    last_bad_msg: str | None = None
    last_bad_msg_at: float | None = None       # host perf_counter
    clock_resets: int = 0
    pose_advanced: float | None = None          # host perf_counter when time_stamp last increased
    collision: CollisionState = field(default_factory=CollisionState)
    # Ground detector (updated on the pose thread so it runs at ~330 Hz, not at snapshot rate)
    contact_latch: tuple[float, float, float] | None = None  # NED xyz at the last supporting contact
    still_since_ts: int | None = None           # sim ts when the vehicle became still
    # Every collision this session (CollisionState only keeps the latest one).
    collision_log: deque = field(default_factory=lambda: deque(maxlen=1000))


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
        speed_window_s: float = 0.1,
        bad_msg_error_window_s: float = 1.0,
        ground_still_max_gs_mps: float = 0.25,
        ground_still_max_vs_mps: float = 0.15,
        ground_still_dwell_s: float = 1.0,
        ground_latch_release_climb_m: float = 0.3,
        ground_latch_release_horizontal_m: float = 0.5,
        support_normal_max_z: float = -0.7,
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
        self.speed_window_ns = int(speed_window_s * 1e9)
        self.bad_msg_error_window_s = bad_msg_error_window_s
        self.ground_still_max_gs_mps = ground_still_max_gs_mps
        self.ground_still_max_vs_mps = ground_still_max_vs_mps
        self.ground_still_dwell_ns = int(ground_still_dwell_s * 1e9)
        self.ground_latch_release_climb_m = ground_latch_release_climb_m
        self.ground_latch_release_horizontal_m = ground_latch_release_horizontal_m
        self.support_normal_max_z = support_normal_max_z

        self._client: ProjectAirSimClient | None = None
        self._world: World | None = None
        self._drones: dict[str, Drone] = {}
        self._cache: dict[str, _VehicleCache] = {}
        self._lock = threading.Lock()          # guards _cache (push thread vs. callers)
        self._request_lock = threading.Lock()  # serializes sync service requests
        self._reset_state()

    # ---------------- lifecycle
    def _reset_state(self) -> None:
        with self._lock:
            self._world = None
            self._drones = {}
            self._cache = {v: _VehicleCache() for v in self.vehicle_ids}

    @property
    def connected(self) -> bool:
        return self._client is not None and bool(self._drones)

    def connect(self) -> None:
        """Connect and load the scene. Any previous session state is discarded.
        On failure, the partial connection is torn down and the error re-raised."""
        if self._client is not None:
            self.disconnect()
        self._reset_state()
        client = ProjectAirSimClient(address=self.address)
        try:
            client.connect()
            self._client = client
            world = World(client, self.scene, delay_after_load_sec=self.load_delay_s,
                          sim_config_path=self.sim_config_path)
            drones = {}
            for vid in self.vehicle_ids:
                drone = Drone(client, world, vid)
                drones[vid] = drone
                client.subscribe(drone.robot_info["actual_pose"], self._on_pose(vid))
                client.subscribe(drone.robot_info["collision_info"], self._on_collision(vid))
            with self._lock:
                self._world, self._drones = world, drones
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
        with self._lock:
            self._world = None
            self._drones = {}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    def drone(self, vehicle_id: str) -> Drone:
        """Raw Project AirSim handle. For test scripts only until Pass 5 commands exist."""
        return self._drones[vehicle_id]

    # ---------------- push callbacks (Project AirSim receive thread; must never raise)
    def _record_bad_msg(self, c: _VehicleCache, what: str) -> None:
        c.bad_msg_count += 1
        c.last_bad_msg = what
        c.last_bad_msg_at = time.perf_counter()

    def _record_callback_failure(self, vid: str, topic: str, err: Exception) -> None:
        """Last-resort handler: an unexpected exception in a callback is still counted."""
        try:
            with self._lock:
                c = self._cache.get(vid)
                if c is not None:
                    self._record_bad_msg(c, f"{topic}: unexpected {type(err).__name__}: {err}")
        except Exception:
            pass

    def _on_pose(self, vid: str):
        def cb(_topic, msg):
            try:
                now = time.perf_counter()
                with self._lock:
                    c = self._cache.get(vid)
                    if c is None:
                        return
                    try:
                        ts, x, y, z = parse_pose_message(msg)
                    except ValueError as err:
                        self._record_bad_msg(c, f"actual_pose: {err}")
                        return
                    if c.pose_ts is not None and ts < c.pose_ts:
                        # Sim clock went backwards (scene reload / restart): start a new window.
                        c.clock_resets += 1
                        c.pose_window.clear()
                        # A new timeline: nothing observed on the old one may ground the vehicle.
                        c.contact_latch = None
                        c.still_since_ts = None
                        c.collision = CollisionState()
                        c.collision_log.clear()
                        self._record_bad_msg(
                            c, f"actual_pose: time_stamp went backwards ({c.pose_ts} -> {ts})")
                    if c.pose_ts is None or ts != c.pose_ts:
                        c.pose_window.append((ts, x, y, z))
                        while c.pose_window and ts - c.pose_window[0][0] > self.speed_window_ns:
                            c.pose_window.popleft()
                        c.pose_advanced = now
                        self._update_ground_detector(c, ts, z)
                    c.pose_ts = ts
                    c.pose_received = now
                    c.pose_count += 1
            except Exception as err:  # never let an exception reach the receive thread
                self._record_callback_failure(vid, "actual_pose", err)
        return cb

    def _update_ground_detector(self, c: _VehicleCache, ts: int, z: float) -> None:
        """Called with the lock held on every advancing pose."""
        if c.contact_latch is not None:
            lx, ly, lz = c.contact_latch
            x, y = c.pose_window[-1][1], c.pose_window[-1][2]
            if (abs(lz - z) > self.ground_latch_release_climb_m
                    or math.hypot(x - lx, y - ly) > self.ground_latch_release_horizontal_m):
                c.contact_latch = None  # moved away from the supporting contact: release
        gs, vs = self._window_velocity(c.pose_window)
        still = (gs is not None and gs <= self.ground_still_max_gs_mps
                 and abs(vs) <= self.ground_still_max_vs_mps)
        if still:
            if c.still_since_ts is None:
                c.still_since_ts = ts
        else:
            c.still_since_ts = None

    def _grounded_by_contact(self, c: _VehicleCache) -> bool:
        return (c.contact_latch is not None and c.still_since_ts is not None
                and c.pose_ts is not None
                and c.pose_ts - c.still_since_ts >= self.ground_still_dwell_ns)

    def _on_collision(self, vid: str):
        def cb(_topic, msg):
            try:
                with self._lock:
                    c = self._cache.get(vid)
                    if c is None:
                        return
                    if not isinstance(msg, dict):
                        self._record_bad_msg(c, f"collision_info: message is {type(msg).__name__}")
                        return
                    ts = msg.get("time_stamp")
                    if (isinstance(ts, bool) or not isinstance(ts, (int, float))
                            or not math.isfinite(ts) or ts <= 0):
                        self._record_bad_msg(c, f"collision_info: invalid time_stamp ({ts!r})")
                        ts = None
                    else:
                        ts = int(ts)
                    normal = msg.get("normal")
                    normal_z = None
                    if isinstance(normal, dict):
                        nz = normal.get("z")
                        if isinstance(nz, (int, float)) and not isinstance(nz, bool) and math.isfinite(nz):
                            normal_z = float(nz)
                    speed = self._window_speed(c.pose_window)
                    resting = None if speed is None else speed < self.impact_speed_threshold_mps
                    old = c.collision
                    c.collision = CollisionState(
                        has_collided=True,
                        object_name=str(msg.get("object_name")) if msg.get("object_name") is not None else None,
                        sim_time_ns=ts,
                        impact_speed_mps=speed,
                        is_resting_contact=resting,
                        normal_z=normal_z,
                        is_supporting_surface=(normal_z is not None
                                               and normal_z <= self.support_normal_max_z),
                        count=old.count + 1,
                        impact_count=old.impact_count + (1 if resting is False else 0),
                    )
                    c.collision_log.append(c.collision.model_copy())
                    # Only a slow touch on an upward-facing surface can support the vehicle.
                    # A wall (normal ~horizontal) or missing normal never latches.
                    if resting is True and c.collision.is_supporting_surface and c.pose_window:
                        _, lx, ly, lz = c.pose_window[-1]
                        c.contact_latch = (lx, ly, lz)
            except Exception as err:
                self._record_callback_failure(vid, "collision_info", err)
        return cb

    @staticmethod
    def _window_velocity(window: deque) -> tuple[float | None, float | None]:
        """(ground speed, vertical speed +up) across the pose window, or (None, None)."""
        if len(window) < 2:
            return None, None
        t0, x0, y0, z0 = window[0]
        t1, x1, y1, z1 = window[-1]
        dt = (t1 - t0) / 1e9
        if dt < 0.05:
            return None, None
        return math.hypot(x1 - x0, y1 - y0) / dt, -(z1 - z0) / dt

    @staticmethod
    def _window_speed(window: deque) -> float | None:
        """Mean speed across the pose window (~100 ms), or None if too short."""
        if len(window) < 2:
            return None
        t0, x0, y0, z0 = window[0]
        t1, x1, y1, z1 = window[-1]
        dt = (t1 - t0) / 1e9
        if dt < 0.01:
            return None
        return math.dist((x0, y0, z0), (x1, y1, z1)) / dt

    # ---------------- snapshots
    def _pull(self, drone: Drone, errors: list[str]):
        """All synchronous requests for one snapshot, serialized."""
        kin, geo, landed = {}, {}, LandedState.UNKNOWN
        with self._request_lock:
            try:
                kin = drone.get_ground_truth_kinematics() or {}
            except Exception as err:
                errors.append(f"get_ground_truth_kinematics failed: {err}")
            try:
                geo = drone.get_ground_truth_geo_location() or {}
            except Exception as err:
                errors.append(f"get_ground_truth_geo_location failed: {err}")
            try:
                raw = int(drone.get_landed_state())
                landed = {0: LandedState.LANDED, 1: LandedState.FLYING}.get(raw, LandedState.UNKNOWN)
            except Exception as err:
                errors.append(f"get_landed_state failed: {err}")
        if not isinstance(kin, dict):
            errors.append(f"kinematics response is {type(kin).__name__}")
            kin = {}
        if not isinstance(geo, dict):
            errors.append(f"geo response is {type(geo).__name__}")
            geo = {}
        return kin, geo, landed

    def get_snapshot(self, vehicle_id: str) -> TelemetrySnapshot:
        if not self.connected:
            raise RuntimeError("ProjectAirSimAdapter is not connected")
        drone = self._drones[vehicle_id]
        extra_errors: list[str] = []
        extra_warnings: list[str] = []

        kin, geo, landed = self._pull(drone, extra_errors)
        received_at = datetime.now(timezone.utc)

        now = time.perf_counter()
        with self._lock:
            c = self._cache[vehicle_id]
            pose_ts = c.pose_ts
            age_ms = (now - c.pose_received) * 1000.0 if c.pose_received is not None else None
            progress_ms = (now - c.pose_advanced) * 1000.0 if c.pose_advanced is not None else None
            by_contact = self._grounded_by_contact(c)
            collision = c.collision.model_copy()
            if c.bad_msg_count:
                recent = (c.last_bad_msg_at is not None
                          and now - c.last_bad_msg_at <= self.bad_msg_error_window_s)
                note = f"{c.bad_msg_count} bad push message(s); last: {c.last_bad_msg}"
                (extra_errors if recent else extra_warnings).append(note)

        ts_raw = kin.get("time_stamp")
        sim_time_ns = int(ts_raw) if isinstance(ts_raw, (int, float)) and math.isfinite(ts_raw) else 0
        if collision.sim_time_ns is not None and sim_time_ns > 0:
            delta = sim_time_ns - collision.sim_time_ns
            collision.recent_collision = 0 <= delta <= self.collision_window_ns

        vel = _vec(kin.get("twist", {}).get("linear") if isinstance(kin.get("twist"), dict) else None)
        kin_still = (math.isfinite(vel.x) and math.isfinite(vel.y) and math.isfinite(vel.z)
                     and math.hypot(vel.x, vel.y) <= self.ground_still_max_gs_mps
                     and abs(vel.z) <= self.ground_still_max_vs_mps)
        if landed == LandedState.LANDED and kin_still:
            ground, basis = GroundState.GROUNDED, "landed_state"
        elif landed == LandedState.LANDED:
            ground, basis = GroundState.UNKNOWN, "landed_state LANDED but vehicle moving"
        elif by_contact:
            ground, basis = GroundState.GROUNDED, "resting_contact+still"
        elif landed == LandedState.FLYING:
            ground, basis = GroundState.AIRBORNE, "landed_state"
        else:
            ground, basis = GroundState.UNKNOWN, "landed_state unavailable"

        pose = kin.get("pose") if isinstance(kin.get("pose"), dict) else {}
        twist = kin.get("twist") if isinstance(kin.get("twist"), dict) else {}
        accels = kin.get("accels") if isinstance(kin.get("accels"), dict) else {}
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
            latitude_deg=_num(geo.get("latitude")),
            longitude_deg=_num(geo.get("longitude")),
            altitude_msl_m=_num(geo.get("altitude")),
            altitude_local_m=-position.z,
            heading_deg=yaw_deg_from_quaternion(orientation),
            track_deg=track_deg_from_velocity(velocity, self.min_track_speed_mps),
            ground_speed_mps=math.hypot(velocity.x, velocity.y),
            vertical_speed_mps=-velocity.z,
            landed_state=landed,
            ground_state=ground,
            ground_state_basis=basis,
            collision=collision,
            telemetry_age_ms=age_ms,
            sim_progress_age_ms=progress_ms,
        )
        return validate_snapshot(snap, self.limits, extra_errors, extra_warnings)

    def get_all_snapshots(self) -> list[TelemetrySnapshot]:
        return [self.get_snapshot(v) for v in self.vehicle_ids]

    def latest_poses(self) -> dict[str, tuple[int, float, float, float]]:
        """{vehicle_id: (sim_time_ns, north, east, down)} from the push cache.

        No simulator requests, so it is cheap enough to sample at 20+ Hz for
        separation monitoring. Vehicles with no valid pose yet are omitted.
        This is simulator ground truth: evaluation only, never agent input.
        """
        with self._lock:
            return {v: c.pose_window[-1] for v, c in self._cache.items() if c.pose_window}

    def collision_log(self, vehicle_id: str) -> list[CollisionState]:
        """Every collision reported for this vehicle this session, oldest first."""
        with self._lock:
            return list(self._cache[vehicle_id].collision_log)

    def topic_stats(self) -> dict:
        with self._lock:
            return {v: {"actual_pose_msgs": c.pose_count, "bad_msgs": c.bad_msg_count,
                        "clock_resets": c.clock_resets, "collisions": c.collision.count}
                    for v, c in self._cache.items()}
