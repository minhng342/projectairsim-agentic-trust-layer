"""
Pure completion predicates for command primitives. No simulator calls.

Project AirSim command futures can resolve while the drone is still moving
(Pass 4.1: move_to_position returned at 2.6 m/s). A command only SUCCEEDS
when telemetry shows the target condition holding continuously for a dwell
time. Each check returns (ok, detail) so failures are explainable.

Every check first requires a VALID snapshot: stale or invalid telemetry can
never confirm completion.
"""
import math
import time
from dataclasses import dataclass

from models.telemetry import GroundState, TelemetrySnapshot, ValidationStatus

Check = tuple[bool, str]


# ---------------------------------------------------------------- tolerances
@dataclass(frozen=True)
class HeadingTolerance:
    max_error_deg: float = 3.0
    max_drift_m: float = 0.5
    max_total_speed_mps: float = 0.3
    max_yaw_rate_dps: float = 5.0
    dwell_s: float = 0.5


@dataclass(frozen=True)
class AltitudeTolerance:
    max_error_m: float = 0.3
    max_vertical_speed_mps: float = 0.2
    max_drift_m: float = 0.5
    dwell_s: float = 0.75


@dataclass(frozen=True)
class PositionTolerance:
    # 1.0 m, not 0.5 m: Simple Flight's move_to_position is built to stop
    # within ~0.5-0.75 m of the target. In the Project AirSim source,
    # SimpleFlightApi::GetDistanceAccuracy() returns 0.5 m and the automatic
    # lookahead is >= 1.5x that (0.75 m); the move counts as done once the end
    # of the path is within lookahead, then the controller holds wherever it
    # is. Live Pass 5.2: three drones stopped 0.55-0.69 m off even after slow
    # corrections. A tighter tolerance would need our own final-approach control.
    max_error_m: float = 1.0
    max_total_speed_mps: float = 0.35
    dwell_s: float = 1.0


@dataclass(frozen=True)
class TrackTolerance:
    max_track_error_deg: float = 5.0
    max_speed_error_mps: float = 0.3
    max_vertical_speed_mps: float = 0.3
    max_heading_change_deg: float = 5.0
    dwell_s: float = 0.5


@dataclass(frozen=True)
class GroundTolerance:
    dwell_s: float = 0.0   # ground_state already includes its own 1 s stillness dwell


# ---------------------------------------------------------------- geometry
def heading_error_deg(target_deg: float, actual_deg: float) -> float:
    """Signed smallest angle from actual to target, in (-180, 180]."""
    err = (target_deg - actual_deg) % 360.0
    return err - 360.0 if err > 180.0 else err


def horizontal_distance_m(s: TelemetrySnapshot, north_m: float, east_m: float) -> float:
    return math.hypot(s.position_ned_m.x - north_m, s.position_ned_m.y - east_m)


def total_speed_mps(s: TelemetrySnapshot) -> float:
    v = s.velocity_ned_mps
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _valid(s: TelemetrySnapshot) -> Check:
    if s.validation_status != ValidationStatus.VALID:
        return False, f"telemetry {s.validation_status.value}: {'; '.join(s.validation_errors) or '-'}"
    return True, ""


# ---------------------------------------------------------------- settle checks
def heading_settled(s: TelemetrySnapshot, target_deg: float, hold_north_m: float,
                    hold_east_m: float, tol: HeadingTolerance = HeadingTolerance()) -> Check:
    ok, why = _valid(s)
    if not ok:
        return ok, why
    err = heading_error_deg(target_deg, s.heading_deg)
    drift = horizontal_distance_m(s, hold_north_m, hold_east_m)
    speed = total_speed_mps(s)
    yaw_rate = math.degrees(s.angular_velocity_rad_s.z)
    detail = (f"heading error {err:+.1f} deg, drift {drift:.2f} m, speed {speed:.2f} m/s, "
              f"yaw rate {yaw_rate:+.1f} deg/s")
    return (abs(err) <= tol.max_error_deg and drift <= tol.max_drift_m
            and speed <= tol.max_total_speed_mps
            and abs(yaw_rate) <= tol.max_yaw_rate_dps), detail


def altitude_settled(s: TelemetrySnapshot, target_alt_m: float, hold_north_m: float,
                     hold_east_m: float, tol: AltitudeTolerance = AltitudeTolerance()) -> Check:
    """CHANGE_ALTITUDE promises to hold N/E, so horizontal drift is checked too."""
    ok, why = _valid(s)
    if not ok:
        return ok, why
    err = s.altitude_local_m - target_alt_m
    drift = horizontal_distance_m(s, hold_north_m, hold_east_m)
    detail = f"altitude error {err:+.2f} m, vs {s.vertical_speed_mps:+.2f} m/s, drift {drift:.2f} m"
    return (abs(err) <= tol.max_error_m
            and abs(s.vertical_speed_mps) <= tol.max_vertical_speed_mps
            and drift <= tol.max_drift_m), detail


def position_settled(s: TelemetrySnapshot, north_m: float, east_m: float, alt_m: float,
                     tol: PositionTolerance = PositionTolerance()) -> Check:
    ok, why = _valid(s)
    if not ok:
        return ok, why
    err = math.sqrt(horizontal_distance_m(s, north_m, east_m) ** 2
                    + (s.altitude_local_m - alt_m) ** 2)
    speed = total_speed_mps(s)
    detail = f"position error {err:.2f} m, speed {speed:.2f} m/s"
    return err <= tol.max_error_m and speed <= tol.max_total_speed_mps, detail


def track_velocity_ok(s: TelemetrySnapshot, track_deg: float, speed_mps: float,
                      start_heading_deg: float, tol: TrackTolerance = TrackTolerance()) -> Check:
    """Steady-state check DURING move_along_track: right direction, right speed,
    level flight, and the nose has NOT been turned (heading independence)."""
    ok, why = _valid(s)
    if not ok:
        return ok, why
    if s.track_deg is None:
        return False, f"no track yet (gs {s.ground_speed_mps:.2f} m/s)"
    trk_err = heading_error_deg(track_deg, s.track_deg)
    spd_err = s.ground_speed_mps - speed_mps
    hdg_change = heading_error_deg(start_heading_deg, s.heading_deg)
    detail = (f"track error {trk_err:+.1f} deg, speed error {spd_err:+.2f} m/s, "
              f"vs {s.vertical_speed_mps:+.2f} m/s, heading change {hdg_change:+.1f} deg")
    return (abs(trk_err) <= tol.max_track_error_deg
            and abs(spd_err) <= tol.max_speed_error_mps
            and abs(s.vertical_speed_mps) <= tol.max_vertical_speed_mps
            and abs(hdg_change) <= tol.max_heading_change_deg), detail


def grounded(s: TelemetrySnapshot) -> Check:
    ok, why = _valid(s)
    if not ok:
        return ok, why
    return s.ground_state == GroundState.GROUNDED, \
        f"ground_state {s.ground_state.value} ({s.ground_state_basis})"


# ---------------------------------------------------------------- timing
class DwellTracker:
    """True once a condition has held continuously for `required_s` of SIM time.

    Resets whenever the condition is false or the sim clock goes backwards.
    Sim time is used so a paused simulator can't "dwell" its way to success.
    """

    def __init__(self, required_s: float):
        self.required_ns = int(required_s * 1e9)
        self._since_ns: int | None = None
        self._last_ns: int | None = None

    def update(self, sim_time_ns: int, ok: bool) -> bool:
        if self._last_ns is not None and sim_time_ns < self._last_ns:
            self._since_ns = None
        self._last_ns = sim_time_ns
        if not ok:
            self._since_ns = None
            return False
        if self._since_ns is None:
            self._since_ns = sim_time_ns
        return sim_time_ns - self._since_ns >= self.required_ns

    @property
    def held_s(self) -> float:
        if self._since_ns is None or self._last_ns is None:
            return 0.0
        return (self._last_ns - self._since_ns) / 1e9


class Deadline:
    """Wall-clock timeout. Host time, so a frozen simulator still times out."""

    def __init__(self, timeout_s: float, clock=time.perf_counter):
        self._clock = clock
        self._end = clock() + timeout_s

    @property
    def expired(self) -> bool:
        return self._clock() >= self._end

    @property
    def remaining_s(self) -> float:
        return max(0.0, self._end - self._clock())
