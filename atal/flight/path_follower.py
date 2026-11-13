"""
Velocity path follower: flies a corridor polyline continuously, interruptible.

Why not move_to_position per waypoint: each one is a full stop-and-settle (~10 s),
so corridor flight would be stop-start and arrival times at crossings would drift.
Live diag 3 (2026-10-27) showed Simple Flight's velocity control settles cleanly at
any heading, so this closes the loop ourselves:

    every poll (10 Hz): snapshot -> guidance -> move_by_velocity (0.3 s duration)

The 0.3 s duration means a stalled loop (crash, blocked event loop) stops the drone
within 0.3 s instead of leaving it flying a stale velocity.

Guidance
    CRUISE   pure pursuit: aim at the point `lookahead_m` further along the path than
             the drone's projection on it; horizontal speed = cruise speed, reduced near
             the end (arrive_gain * distance to the end) so the drone arrives without
             overshoot. Progress along the path never goes backwards.
    HOLD     brake (zero velocity) until slow, then stay where it stopped (P-control,
             clamped); progress frozen. Live diag (2026-10-28): holding the point where the
             command arrived made the drone stop ~2 m past it at 3 m/s and fly back.
             resume() continues the path from wherever the drone is.
    altitude every mode tracks the altitude target (set_altitude: e.g. a layer change),
             vertical speed clamped.
    DONE     within tolerance of the final point, slow, for `done_dwell_s` of SIM time;
             then hover.
    BLOCKED  (a FAILED result, reason starting "blocked:") CRUISE progress stalled for
             `stuck_after_s` of SIM time away from the end of the path: hover, give up the
             path so the planner can recover instead of grinding into a wall.

Maneuver commands (hold, resume, set_altitude, set_speed, stop) are plain method calls,
safe to make from another task while run() is flying: they change setpoints, nothing is
cancelled mid-flight.

Fail-safe, like the executor: telemetry unavailable / not VALID, grounded, overall
timeout or an unexpected error -> hover, FAILED. Cancellation -> shielded hover, re-raise.
The heading is held at its value when run() starts.

Ground truth: like the executor, the follower reads the adapter's simulator snapshots
(it is the flight controller, not the agent).
"""
import asyncio
import math
import time
from dataclasses import dataclass, field
from enum import Enum

from atal.core.telemetry import GroundState, TelemetrySnapshot, ValidationStatus
from atal.flight.completion import Deadline, DwellTracker


class FollowMode(str, Enum):
    IDLE = "idle"
    CRUISE = "cruise"
    HOLD = "hold"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"        # stop() called: hovering where it is, path abandoned


# The live altitude-hold candidate (zeus-g 2026-11-12), switched on by --corner-slowdown.
CORNER_SLOWDOWN = {"max_cmd_accel_mps2": 3.0, "corner_speed_mps": 1.5}


@dataclass(frozen=True)
class FollowerConfig:
    poll_interval_s: float = 0.1
    cmd_duration_s: float = 0.3
    lookahead_m: float = 3.0
    max_speed_mps: float = 5.0
    arrive_gain: float = 0.6            # near the end: speed <= arrive_gain * distance to end
    min_approach_speed_mps: float = 0.3
    hold_gain: float = 0.8
    hold_max_speed_mps: float = 1.0
    hold_latch_speed_mps: float = 0.3   # pin the hold point once braked below this...
    hold_brake_max_s: float = 3.0       # ...or after this long braking (sim time), whichever first
    vertical_gain: float = 0.8
    max_vertical_mps: float = 1.5     # layer changes: 3 m in ~2.5 s (Simple Flight climbs 2 m/s fine)
    deadband_m: float = 0.15
    done_tolerance_m: float = 0.5       # horizontal, to the final point
    done_altitude_tolerance_m: float = 0.5
    done_speed_mps: float = 0.3
    done_dwell_s: float = 1.0
    hover_timeout_s: float = 5.0
    # Blocked: in CRUISE, progress along the path has not grown by stuck_min_progress_m for
    # stuck_after_s of SIM time (pressed against a wall, wedged on a roof edge). Without it
    # the follower ground on against the obstacle until the whole-path timeout (3x the
    # expected time + 60 s: many minutes in CityEnviron). None turns it off.
    stuck_after_s: float | None = 10.0
    stuck_min_progress_m: float = 1.0
    # A follower_trace event at most every trace_period_s of SIM time: what was commanded
    # next to what was measured (altitude, vertical and horizontal speed, gap between
    # commands). Live altitude hold is only as good as these show. None turns it off.
    trace_period_s: float | None = 0.5
    # The horizontal velocity command changes by at most this much per second of SIM time
    # (starting from the measured velocity). Live, Simple Flight climbs up to ~2 m against
    # a full descent command when the horizontal command swings at a corner or a new path
    # (zeus-g 2026-11-12). Off by default; --corner-slowdown turns both on with the values in
    # CORNER_SLOWDOWN (still being checked live). CRUISE only: braking for a hold or a stop is never slowed, so the
    # stopping distances behind the transition manager's stop-line are unchanged. None = off.
    max_cmd_accel_mps2: float | None = None
    # Slow down before a corner (a turn of at least corner_turn_deg) so the command swings at
    # corner_speed_mps, not at cruise speed: v <= sqrt(v_c^2 + 2 a d), d = distance to the
    # point where the carrot passes the corner. Without it the slew limit above would carry
    # the drone past the corner. None = off.
    corner_speed_mps: float | None = None
    corner_decel_mps2: float = 2.0
    corner_turn_deg: float = 30.0


@dataclass
class FollowResult:
    mode: FollowMode
    reason: str
    progress_m: float
    path_length_m: float
    elapsed_s: float
    max_cross_track_m: float = 0.0
    waypoints_passed: list[int] = field(default_factory=list)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Polyline:
    """Horizontal path through waypoints (north, east), parametrized by arc length."""

    def __init__(self, waypoints):
        pts = [(float(n), float(e)) for n, e in waypoints]
        if not pts:
            raise ValueError("path needs at least one waypoint")
        if not all(math.isfinite(c) for p in pts for c in p):
            raise ValueError("waypoints must be finite")
        self.points = pts
        self.cum = [0.0]
        for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(n1 - n0, e1 - e0))
        self.length = self.cum[-1]

    def point_at(self, s: float) -> tuple[float, float]:
        s = _clamp(s, 0.0, self.length)
        for i in range(len(self.points) - 1):
            seg = self.cum[i + 1] - self.cum[i]
            if s <= self.cum[i + 1] or i == len(self.points) - 2:
                t = 0.0 if seg <= 0 else (s - self.cum[i]) / seg
                (n0, e0), (n1, e1) = self.points[i], self.points[i + 1]
                return n0 + t * (n1 - n0), e0 + t * (e1 - e0)
        return self.points[-1]

    def next_corner(self, s: float, min_turn_deg: float) -> float | None:
        """Arc length of the first interior waypoint after s where the path turns by at
        least min_turn_deg, or None."""
        for i in range(1, len(self.points) - 1):
            if self.cum[i] <= s:
                continue
            (n0, e0), (n1, e1), (n2, e2) = self.points[i - 1], self.points[i], self.points[i + 1]
            a = math.atan2(e1 - e0, n1 - n0)
            b = math.atan2(e2 - e1, n2 - n1)
            turn = abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi))
            if turn >= min_turn_deg:
                return self.cum[i]
        return None

    def project(self, n: float, e: float, s_min: float = 0.0,
                window_m: float = math.inf) -> tuple[float, float]:
        """(arc length of the closest point on the path within [s_min, s_min + window_m],
        distance to it). The window keeps progress from jumping to a later leg that
        passes nearby (crossing corridors) or back to an earlier one."""
        if len(self.points) == 1:
            return 0.0, math.hypot(n - self.points[0][0], e - self.points[0][1])
        best = (s_min, math.inf)
        for i in range(len(self.points) - 1):
            a, b = self.cum[i], self.cum[i + 1]
            if b < s_min or a > s_min + window_m:
                continue
            (n0, e0), (n1, e1) = self.points[i], self.points[i + 1]
            dn, de = n1 - n0, e1 - e0
            seg2 = dn * dn + de * de
            t = 0.0 if seg2 == 0 else _clamp(((n - n0) * dn + (e - e0) * de) / seg2, 0.0, 1.0)
            s = _clamp(a + t * (b - a), s_min, s_min + window_m)
            pn, pe = self.point_at(s)
            d = math.hypot(n - pn, e - pe)
            if d < best[1]:
                best = (s, d)
        return best


class PathFollower:
    def __init__(self, adapter, vehicle_id: str, waypoints, altitude_m: float, speed_mps: float,
                 config: FollowerConfig = FollowerConfig(), clock=time.monotonic,
                 sleep=asyncio.sleep, on_event=None, log=print):
        self.adapter, self.vehicle_id = adapter, vehicle_id
        self.path = Polyline(waypoints)
        self.config = config
        self._clock, self._sleep, self._on_event, self._log = clock, sleep, on_event, log
        self.altitude_target_m = float(altitude_m)
        self.speed_mps = self._checked_speed(speed_mps)
        self.mode = FollowMode.IDLE
        self.progress_m = 0.0
        self._hold_point: tuple[float, float] | None = None
        self._hold_started_ns: int | None = None
        self._hold_requested = False
        self._stop_requested = False
        self._yaw_rad = 0.0
        self._passed: list[int] = []
        self._max_xtrack = 0.0
        self._stuck_ref: tuple[float, int] | None = None   # (progress, sim ns) last advance
        self.blocked = False
        self._trace_ns: int | None = None      # sim time of the last follower_trace
        self._cmd: tuple[float, float, int] | None = None   # last horizontal command, sim ns
        self._last_cmd_host: float | None = None

    # ------------------------------------------------------------ maneuvers
    def hold(self) -> None:
        """Stop and stay where the drone is (progress frozen) until resume()."""
        if self.mode in (FollowMode.CRUISE, FollowMode.IDLE):
            self._hold_requested = True

    def resume(self) -> None:
        if self.mode == FollowMode.HOLD or self._hold_requested:
            self._hold_requested = False
            self._hold_point = None
            self._hold_started_ns = None
            if self.mode == FollowMode.HOLD:
                self._set_mode(FollowMode.CRUISE, "resume")

    def set_altitude(self, altitude_m: float) -> None:
        if not math.isfinite(altitude_m):
            raise ValueError("altitude must be finite")
        self.altitude_target_m = float(altitude_m)
        self._emit("follower_altitude", altitude_m=self.altitude_target_m)

    def set_speed(self, speed_mps: float) -> None:
        self.speed_mps = self._checked_speed(speed_mps)
        self._emit("follower_speed", speed_mps=self.speed_mps)

    def _checked_speed(self, speed_mps: float) -> float:
        if not (math.isfinite(speed_mps) and 0 < speed_mps <= self.config.max_speed_mps):
            raise ValueError(f"speed must be in (0, {self.config.max_speed_mps}] m/s")
        return float(speed_mps)

    def stop(self) -> None:
        """Abandon the path: hover where it is and return from run()."""
        self._stop_requested = True

    @property
    def remaining_m(self) -> float:
        return max(0.0, self.path.length - self.progress_m)

    # ------------------------------------------------------------ run
    async def run(self, timeout_s: float | None = None) -> FollowResult:
        cfg = self.config
        started = self._clock()
        deadline = Deadline(timeout_s, clock=self._clock) if timeout_s else None
        drone = self.adapter.drone(self.vehicle_id)
        done = DwellTracker(cfg.done_dwell_s)
        try:
            snap = self._snapshot()
            self._yaw_rad = math.radians(((snap.heading_deg + 180.0) % 360.0) - 180.0)
            self.progress_m, _ = self.path.project(snap.position_ned_m.x, snap.position_ned_m.y,
                                                   0.0, cfg.lookahead_m)
            self._set_mode(FollowMode.CRUISE, "start", path_length_m=round(self.path.length, 2),
                           waypoints=self.path.points, altitude_m=self.altitude_target_m,
                           speed_mps=self.speed_mps)
            while True:
                if self._stop_requested:
                    await self._hover(drone)
                    return self._result(FollowMode.STOPPED, "stopped on request", started)
                if deadline is not None and deadline.expired:
                    await self._hover(drone)
                    return self._result(FollowMode.FAILED, f"did not finish within {timeout_s:g} s",
                                        started)
                snap = self._snapshot()
                if self._hold_requested and self.mode == FollowMode.CRUISE:
                    self._hold_requested = False
                    self._hold_point = None                    # braking first
                    self._hold_started_ns = snap.sim_time_ns
                    self._set_mode(FollowMode.HOLD, "hold",
                                   at=(snap.position_ned_m.x, snap.position_ned_m.y))
                vn, ve, vup, arrived = self._guidance(snap)
                vn, ve = self._slew(snap, vn, ve)
                self._trace(snap, vn, ve, vup)
                stuck = self._stuck_for(snap)
                if stuck is not None:
                    self.blocked = True
                    await self._hover(drone)
                    self._emit("follower_blocked", progress_m=round(self.progress_m, 2),
                               stalled_s=round(stuck, 1), at=(snap.position_ned_m.x,
                                                              snap.position_ned_m.y),
                               sim_time_ns=snap.sim_time_ns)
                    return self._result(FollowMode.FAILED,
                                        f"blocked: no progress along the path for "
                                        f"{stuck:.0f} s", started)
                if done.update(snap.sim_time_ns, arrived):
                    self.progress_m = self.path.length      # arrived within tolerance: finished
                    self._mark_passed(snap)
                    await self._hover(drone)
                    return self._result(FollowMode.DONE, "reached the end of the path", started)
                # asyncio.timeout, not wait_for: on Python 3.11 wait_for can swallow a
                # cancellation that arrives as the call completes, and a lost cancel here
                # would leave the drone flying the path (found by the cancellation test).
                async with asyncio.timeout(cfg.hover_timeout_s):
                    await drone.move_by_velocity_async(
                        v_north=vn, v_east=ve, v_down=-vup, duration=cfg.cmd_duration_s,
                        yaw_is_rate=False, yaw=self._yaw_rad)
                await self._sleep(cfg.poll_interval_s)
        except _TelemetryProblem as err:
            await self._hover(drone)
            return self._result(FollowMode.FAILED, str(err), started)
        except asyncio.CancelledError:
            hover = asyncio.ensure_future(self._hover(drone))
            try:
                await asyncio.shield(hover)
            except asyncio.CancelledError:
                pass
            self._emit("follower_cancelled", progress_m=round(self.progress_m, 2))
            raise
        except Exception as err:            # never leave the drone on its last velocity
            await self._hover(drone)
            return self._result(FollowMode.FAILED, f"unexpected {type(err).__name__}: {err}",
                                started)

    # ------------------------------------------------------------ guidance
    def _guidance(self, snap: TelemetrySnapshot) -> tuple[float, float, float, bool]:
        cfg = self.config
        n, e = snap.position_ned_m.x, snap.position_ned_m.y
        dz = self.altitude_target_m - snap.altitude_local_m
        vup = 0.0 if abs(dz) <= cfg.deadband_m else _clamp(cfg.vertical_gain * dz,
                                                         -cfg.max_vertical_mps, cfg.max_vertical_mps)
        if self.mode == FollowMode.HOLD:
            if self._hold_point is None:
                speed = math.hypot(snap.velocity_ned_mps.x, snap.velocity_ned_mps.y)
                braking_s = (snap.sim_time_ns - (self._hold_started_ns or snap.sim_time_ns)) / 1e9
                if speed > cfg.hold_latch_speed_mps and braking_s < cfg.hold_brake_max_s:
                    return 0.0, 0.0, vup, False          # brake: don't fly back to the call point
                self._hold_point = (n, e)
                self._emit("follower_hold_point", at=self._hold_point,
                           braking_s=round(braking_s, 2), sim_time_ns=snap.sim_time_ns)
            hn, he = self._hold_point
            vn, ve = self._toward(hn - n, he - e, cfg.hold_gain, cfg.hold_max_speed_mps)
            return vn, ve, vup, False

        s, xtrack = self.path.project(n, e, self.progress_m, 2 * cfg.lookahead_m)
        self.progress_m = max(self.progress_m, s)
        self._max_xtrack = max(self._max_xtrack, xtrack)
        self._mark_passed(snap)
        end_n, end_e = self.path.points[-1]
        to_end = math.hypot(end_n - n, end_e - e)
        speed = math.hypot(snap.velocity_ned_mps.x, snap.velocity_ned_mps.y)
        arrived = (to_end <= cfg.done_tolerance_m and abs(dz) <= cfg.done_altitude_tolerance_m
                   and speed <= cfg.done_speed_mps and self.remaining_m <= cfg.done_tolerance_m)
        # Only the last stretch counts as "near the end": on a closed loop the end point is
        # also the start, and straight-line distance alone would stop the drone there.
        final = self.remaining_m <= cfg.lookahead_m
        if final and to_end <= cfg.deadband_m:
            return 0.0, 0.0, vup, arrived
        cn, ce = self.path.point_at(self.progress_m + cfg.lookahead_m)
        dn, de = cn - n, ce - e
        if math.hypot(dn, de) < 1e-6:                  # carrot reached: aim at the end
            dn, de = end_n - n, end_e - e
        if math.hypot(dn, de) < 1e-6:
            return 0.0, 0.0, vup, arrived
        d_end = to_end if final else self.remaining_m
        v = min(self.speed_mps, max(cfg.min_approach_speed_mps, cfg.arrive_gain * d_end))
        if cfg.corner_speed_mps is not None:
            c = self.path.next_corner(self.progress_m, cfg.corner_turn_deg)
            if c is not None:
                d = max(0.0, c - self.progress_m - cfg.lookahead_m)
                v = min(v, math.sqrt(cfg.corner_speed_mps ** 2 + 2 * cfg.corner_decel_mps2 * d))
        d = math.hypot(dn, de)
        return v * dn / d, v * de / d, vup, arrived

    def _stuck_for(self, snap) -> float | None:
        """Sim seconds CRUISE progress has stalled, once past stuck_after_s; else None.
        Not in HOLD (progress is frozen on purpose) and not on the last stretch (the
        arrival slows down by design)."""
        cfg = self.config
        if (cfg.stuck_after_s is None or self.mode != FollowMode.CRUISE
                or self.remaining_m <= cfg.lookahead_m or snap.sim_time_ns is None):
            self._stuck_ref = None
            return None
        ref = self._stuck_ref
        if ref is None or self.progress_m >= ref[0] + cfg.stuck_min_progress_m:
            self._stuck_ref = (self.progress_m, snap.sim_time_ns)
            return None
        stalled = (snap.sim_time_ns - ref[1]) / 1e9
        return stalled if stalled >= cfg.stuck_after_s else None

    def _mark_passed(self, snap) -> None:
        for i, c in enumerate(self.path.cum):
            if i > 0 and i not in self._passed and self.progress_m >= c:
                self._passed.append(i)
                self._emit("follower_waypoint", index=i, waypoint=self.path.points[i],
                           sim_time_ns=snap.sim_time_ns)

    def _toward(self, dn, de, gain, vmax):
        d = math.hypot(dn, de)
        if d <= self.config.deadband_m:
            return 0.0, 0.0
        v = min(gain * d, vmax)
        return v * dn / d, v * de / d

    # ------------------------------------------------------------ plumbing
    def _snapshot(self) -> TelemetrySnapshot:
        try:
            snap = self.adapter.get_snapshot(self.vehicle_id)
        except Exception as err:
            raise _TelemetryProblem(f"telemetry unavailable ({type(err).__name__}: {err})")
        if snap.validation_status != ValidationStatus.VALID:
            raise _TelemetryProblem(f"telemetry {snap.validation_status.value}: "
                                    f"{'; '.join(snap.validation_errors)}")
        if snap.ground_state == GroundState.GROUNDED:
            raise _TelemetryProblem("vehicle is grounded")
        return snap

    async def _hover(self, drone) -> str:
        t = self.config.hover_timeout_s
        try:
            async with asyncio.timeout(t):
                task = await drone.hover_async()
            async with asyncio.timeout(t):
                await task
            return "hover"
        except Exception as err:
            self._log(f"[follower] hover failed: {type(err).__name__}: {err}")
            return f"hover failed: {type(err).__name__}: {err}"

    def _set_mode(self, mode: FollowMode, why: str, **fields) -> None:
        self.mode = mode
        self._emit("follower_mode", mode=mode.value, why=why,
                   progress_m=round(self.progress_m, 2), **fields)

    def _result(self, mode: FollowMode, reason: str, started: float) -> FollowResult:
        self.mode = mode
        r = FollowResult(mode=mode, reason=reason, progress_m=round(self.progress_m, 2),
                         path_length_m=round(self.path.length, 2),
                         elapsed_s=round(self._clock() - started, 3),
                         max_cross_track_m=round(self._max_xtrack, 3),
                         waypoints_passed=list(self._passed))
        self._emit("follower_end", mode=mode.value, reason=reason, progress_m=r.progress_m,
                   elapsed_s=r.elapsed_s, max_cross_track_m=r.max_cross_track_m)
        return r

    def _slew(self, snap: TelemetrySnapshot, vn: float, ve: float) -> tuple[float, float]:
        """Limit how fast the horizontal command vector changes (max_cmd_accel_mps2)."""
        a = self.config.max_cmd_accel_mps2
        t = snap.sim_time_ns
        if a is None or t is None or self.mode != FollowMode.CRUISE:
            self._cmd = None if t is None else (vn, ve, t)
            return vn, ve
        if self._cmd is None:                  # first command: start from what it is doing
            pn, pe, dt = snap.velocity_ned_mps.x, snap.velocity_ned_mps.y, self.config.poll_interval_s
        else:
            pn, pe, t_prev = self._cmd
            dt = max(0.0, (t - t_prev) / 1e9)
        dn, de = vn - pn, ve - pe
        step, lim = math.hypot(dn, de), a * dt
        if step > lim:
            vn, ve = pn + dn * lim / step, pe + de * lim / step
        self._cmd = (vn, ve, t)
        return vn, ve

    def _trace(self, snap: TelemetrySnapshot, vn: float, ve: float, vup: float) -> None:
        """Record the command about to be sent beside the measured state, rate-limited."""
        now_host = self._clock()
        gap = None if self._last_cmd_host is None else now_host - self._last_cmd_host
        self._last_cmd_host = now_host
        period = self.config.trace_period_s
        if period is None or snap.sim_time_ns is None:
            return
        if self._trace_ns is not None and snap.sim_time_ns - self._trace_ns < period * 1e9:
            return
        self._trace_ns = snap.sim_time_ns
        self._emit("follower_trace", sim_time_ns=snap.sim_time_ns, mode=self.mode.value,
                   alt_target_m=round(self.altitude_target_m, 2),
                   alt_m=round(snap.altitude_local_m, 2),
                   vz_cmd_mps=round(vup, 2), vz_mps=round(snap.vertical_speed_mps, 2),
                   vh_cmd_mps=round(math.hypot(vn, ve), 2),
                   vh_mps=round(math.hypot(snap.velocity_ned_mps.x, snap.velocity_ned_mps.y), 2),
                   cmd_gap_s=None if gap is None else round(gap, 3),
                   progress_m=round(self.progress_m, 2))

    def _emit(self, kind: str, **fields) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, {"vehicle_id": self.vehicle_id, **fields})
        except Exception as err:
            self._log(f"[follower] on_event({kind}) failed: {type(err).__name__}: {err}")


class _TelemetryProblem(Exception):
    pass
