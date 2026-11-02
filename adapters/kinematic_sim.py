"""
Kinematic stand-in for Project AirSim: runs the whole stack offline, faster than real time.

    with KinematicSim({"Drone1": (0.0, 0.0)}, speedup=50) as adapter:
        ...same calls as ProjectAirSimAdapter: drone(vid), get_snapshot(vid),
           latest_poses(), collision_log(vid)

It is NOT a flight-dynamics model. Each vehicle is a point mass whose velocity follows the
commanded velocity with an acceleration limit; position commands fly straight to the
target at the requested speed; takeoff climbs to `takeoff_height_m`, land descends to the
ground. It exists so the Phase 1 pipeline (mission file -> planner -> port -> executor ->
path follower -> sensor -> memory -> run log) can be tested, and batches of runs checked,
on any machine. Measured results come from the real simulator.

Time: sim time = host time since start x `speedup`, integrated lazily (in steps of at
most `max_step_s` sim seconds) whenever anything reads or commands the world. Code that
polls on the host clock (path follower, executor, feed, sensor) should poll `speedup`
times faster than in a live run, so each poll still covers about the same sim time; the
runner's --offline mode does that.

Velocity commands last `duration` SIM seconds, then the vehicle brakes to a hover, like
Simple Flight's move_by_velocity. No collisions, no wind, no buildings.
"""
import asyncio
import math
import time
from datetime import datetime, timezone

from models.telemetry import (GroundState, LandedState, Quaternion, TelemetrySnapshot,
                              ValidationStatus, Vector3)


class _Vehicle:
    def __init__(self, vid, n, e, ground_alt):
        self.vid = vid
        self.n, self.e, self.alt = float(n), float(e), float(ground_alt)
        self.ground = float(ground_alt)
        self.vn = self.ve = self.vu = 0.0
        self.heading = 0.0
        self.mode = "idle"                  # idle | velocity | position | hover | land
        self.cmd = (0.0, 0.0, 0.0)          # velocity command (n, e, up)
        self.cmd_until = 0.0
        self.target = None                  # (n, e, alt)
        self.speed = 1.0
        self.armed = self.api = False
        self.grounded = True

    def at_target(self, tol=0.05):
        if self.target is None:
            return True
        n, e, a = self.target
        return math.hypot(n - self.n, e - self.e) <= tol and abs(a - self.alt) <= tol


class _Drone:
    def __init__(self, sim, v: _Vehicle):
        self._sim, self._v = sim, v

    # api / arming
    def enable_api_control(self):
        self._v.api = True

    def disable_api_control(self):
        self._v.api = False

    def arm(self):
        self._v.armed = True

    def disarm(self):
        if not self._v.grounded:
            raise RuntimeError("refusing to disarm in the air")
        self._v.armed = False

    def get_landed_state(self):
        return 0 if self._v.grounded else 1

    # commands: each returns an awaitable task, like the real client
    async def takeoff_async(self, **_):
        self._sim.integrate()
        v = self._v
        if not v.armed:
            raise RuntimeError("not armed")
        v.grounded = False
        v.mode, v.target, v.speed = "position", (v.n, v.e, v.ground + self._sim.takeoff_height_m), 1.0
        return self._wait(lambda: v.at_target(0.1))

    async def land_async(self, **_):
        self._sim.integrate()
        v = self._v
        v.mode, v.target = "land", None
        return self._wait(lambda: v.grounded)

    async def hover_async(self):
        self._sim.integrate()
        v = self._v
        if not v.grounded:
            v.mode = "hover"
        return self._done()

    async def move_by_velocity_async(self, v_north, v_east, v_down, duration=0.1, **_):
        sim = self._sim
        sim.integrate()
        v = self._v
        if not v.grounded:
            v.mode, v.cmd = "velocity", (float(v_north), float(v_east), -float(v_down))
            v.cmd_until = sim.now_s + float(duration)
        return self._done()

    async def move_to_position_async(self, north, east, down, velocity, timeout_sec=None, **_):
        self._sim.integrate()
        v = self._v
        if not v.grounded:
            v.mode, v.target, v.speed = "position", (float(north), float(east), -float(down)), float(velocity)
        return self._wait(lambda: v.mode != "position" or v.at_target())

    async def rotate_to_yaw_async(self, yaw, timeout_sec=None, margin=None):
        self._sim.integrate()
        self._v.heading = math.degrees(yaw) % 360.0
        return self._done()

    # tasks
    def _done(self):
        f = asyncio.get_running_loop().create_future()
        f.set_result(None)
        return f

    def _wait(self, cond):
        sim = self._sim

        async def wait():
            while True:
                sim.integrate()
                if cond():
                    return None
                await asyncio.sleep(sim.task_poll_s)
        return asyncio.ensure_future(wait())


class KinematicSim:
    source = "kinematic_sim"

    def __init__(self, starts: dict[str, tuple[float, float]], ground_alt_m: float = 0.0,
                 speedup: float = 20.0, max_step_s: float = 0.02, max_accel_mps2: float = 4.0,
                 max_vertical_mps: float = 3.0, land_speed_mps: float = 1.0,
                 takeoff_height_m: float = 3.0, clock=time.monotonic):
        if speedup <= 0:
            raise ValueError("speedup must be > 0")
        self.vehicle_ids = list(starts)
        self._v = {vid: _Vehicle(vid, n, e, ground_alt_m) for vid, (n, e) in starts.items()}
        self._drones = {vid: _Drone(self, v) for vid, v in self._v.items()}
        self.speedup, self.max_step_s = float(speedup), float(max_step_s)
        self.max_accel, self.max_vertical = max_accel_mps2, max_vertical_mps
        self.land_speed, self.takeoff_height_m = land_speed_mps, takeoff_height_m
        self._clock = clock
        self._t0 = clock()
        self.now_s = 0.0                       # sim seconds integrated so far
        self.t0_ns = 10_000_000_000            # sim clock starts at 10 s, like the real one
        self.task_poll_s = 0.002

    # ------------------------------------------------------------ adapter API
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def drone(self, vid):
        return self._drones[vid]

    def get_snapshot(self, vid) -> TelemetrySnapshot:
        self.integrate()
        v = self._v[vid]
        yaw = math.radians(v.heading)
        gs = math.hypot(v.vn, v.ve)
        return TelemetrySnapshot(
            vehicle_id=vid, source=self.source, sim_time_ns=self.sim_time_ns,
            received_at_utc=datetime.now(timezone.utc), pose_topic_sim_time_ns=self.sim_time_ns,
            position_ned_m=Vector3(x=v.n, y=v.e, z=-v.alt),
            orientation_quaternion=Quaternion(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)),
            velocity_ned_mps=Vector3(x=v.vn, y=v.ve, z=-v.vu),
            acceleration_ned_mps2=Vector3(x=0.0, y=0.0, z=0.0),
            angular_velocity_rad_s=Vector3(x=0.0, y=0.0, z=0.0),
            latitude_deg=47.641468 + v.n / 111_111.0,
            longitude_deg=-122.140165 + v.e / (111_111.0 * math.cos(math.radians(47.64))),
            altitude_msl_m=122.0 + v.alt, altitude_local_m=v.alt, heading_deg=v.heading,
            track_deg=(math.degrees(math.atan2(v.ve, v.vn)) % 360.0) if gs > 0.3 else None,
            ground_speed_mps=gs, vertical_speed_mps=v.vu,
            landed_state=LandedState.LANDED if v.grounded else LandedState.FLYING,
            ground_state=GroundState.GROUNDED if v.grounded else GroundState.AIRBORNE,
            ground_state_basis="kinematic_sim", telemetry_age_ms=0.0, sim_progress_age_ms=0.0,
            validation_status=ValidationStatus.VALID)

    def latest_poses(self) -> dict:
        self.integrate()
        return {vid: (self.sim_time_ns, v.n, v.e, -v.alt) for vid, v in self._v.items()}

    def collision_log(self, vid) -> list:
        return []

    @property
    def sim_time_ns(self) -> int:
        return self.t0_ns + int(round(self.now_s * 1e9))

    # ------------------------------------------------------------ physics
    def integrate(self) -> None:
        target = (self._clock() - self._t0) * self.speedup
        while self.now_s < target - 1e-9:
            dt = min(self.max_step_s, target - self.now_s)
            for v in self._v.values():
                self._step(v, dt)
            self.now_s += dt

    def _step(self, v: _Vehicle, dt: float) -> None:
        if v.grounded:
            v.vn = v.ve = v.vu = 0.0
            return
        want = (0.0, 0.0, 0.0)
        if v.mode == "velocity":
            if self.now_s < v.cmd_until:
                want = v.cmd
            else:
                v.mode = "hover"
        elif v.mode == "position" and v.target is not None:
            tn, te, ta = v.target
            dn, de, da = tn - v.n, te - v.e, ta - v.alt
            d = math.sqrt(dn * dn + de * de + da * da)
            if d > 1e-6:
                s = min(v.speed, 1.2 * d)              # slow down into the target
                want = (s * dn / d, s * de / d, s * da / d)
        elif v.mode == "land":
            want = (0.0, 0.0, -self.land_speed)
        # acceleration-limited velocity tracking
        dv = [w - c for w, c in zip(want, (v.vn, v.ve, v.vu))]
        mag = math.sqrt(sum(x * x for x in dv))
        lim = self.max_accel * dt
        if mag > lim:
            dv = [x * lim / mag for x in dv]
        v.vn += dv[0]
        v.ve += dv[1]
        v.vu = max(-self.max_vertical, min(self.max_vertical, v.vu + dv[2]))
        v.n += v.vn * dt
        v.e += v.ve * dt
        v.alt += v.vu * dt
        if v.alt <= v.ground and v.vu <= 0:
            v.alt, v.vn, v.ve, v.vu = v.ground, 0.0, 0.0, 0.0
            if v.mode in ("land", "hover", "velocity", "idle"):
                v.grounded, v.mode = True, "idle"
