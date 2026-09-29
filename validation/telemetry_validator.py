"""
Deterministic telemetry checks. Adapter-agnostic: works on any TelemetrySnapshot.

Rules come from what Pass 3 actually observed: disabled sensors returned
timestamp 0, NaN fields, lat/lon 0,0 and a -6.7e29 velocity. A numeric field
existing is not the same as it being trustworthy.

Status policy
- INVALID: any data-quality error (bad timestamp, non-finite value, out of
  bounds, implausible physics, pose/kinematics skew, adapter errors).
- STALE: data is well-formed but the pose stream stopped updating.
- VALID: no errors. Only VALID telemetry should ever justify a command.
Warnings never change the status; they carry safety-relevant observations
(collision impacts, landed-state inconsistencies) for the risk gate to weigh.
"""
import math
from dataclasses import dataclass

from models.telemetry import LandedState, TelemetrySnapshot, ValidationStatus


@dataclass(frozen=True)
class ValidationLimits:
    max_telemetry_age_ms: float = 500.0
    max_ground_speed_mps: float = 40.0
    max_vertical_speed_mps: float = 20.0
    max_acceleration_mps2: float = 100.0
    max_quaternion_norm_error: float = 0.01
    max_pose_skew_ms: float = 100.0
    landed_max_speed_mps: float = 1.0


STALE_PREFIX = "actual_pose stale"


def _numeric_fields(s: TelemetrySnapshot):
    for name in ("position_ned_m", "velocity_ned_mps", "acceleration_ned_mps2",
                 "angular_velocity_rad_s", "orientation_quaternion"):
        for axis, value in getattr(s, name).model_dump().items():
            yield f"{name}.{axis}", value
    for name in ("latitude_deg", "longitude_deg", "altitude_msl_m", "altitude_local_m",
                 "heading_deg", "ground_speed_mps", "vertical_speed_mps"):
        yield name, getattr(s, name)


def validate_snapshot(s: TelemetrySnapshot,
                      limits: ValidationLimits = ValidationLimits(),
                      extra_errors: list[str] | None = None,
                      extra_warnings: list[str] | None = None) -> TelemetrySnapshot:
    """Fill in validation_status / errors / warnings on the snapshot and return it.

    extra_errors / extra_warnings let the adapter add source-specific problems
    (failed requests, malformed push messages) that force the same policy.
    """
    errors: list[str] = list(extra_errors or [])
    warnings: list[str] = list(extra_warnings or [])

    # --- timestamps / freshness ---
    if s.sim_time_ns <= 0:
        errors.append(f"sim_time_ns={s.sim_time_ns}: no valid simulation timestamp")
    if s.pose_topic_sim_time_ns is not None and s.pose_topic_sim_time_ns <= 0:
        errors.append(f"pose_topic_sim_time_ns={s.pose_topic_sim_time_ns}: invalid pose timestamp")
    if s.telemetry_age_ms is None:
        errors.append("actual_pose: no valid messages received")
    elif s.telemetry_age_ms > limits.max_telemetry_age_ms:
        errors.append(f"{STALE_PREFIX}: {s.telemetry_age_ms:.0f} ms old "
                      f"(limit {limits.max_telemetry_age_ms:.0f} ms)")
    if s.pose_topic_sim_time_ns and s.pose_topic_sim_time_ns > 0 and s.sim_time_ns > 0:
        skew_ms = abs(s.sim_time_ns - s.pose_topic_sim_time_ns) / 1e6
        if skew_ms > limits.max_pose_skew_ms:
            errors.append(f"pose topic and kinematics differ by {skew_ms:.0f} ms sim time "
                          f"(limit {limits.max_pose_skew_ms:.0f} ms)")

    # --- NaN / infinity ---
    for name, value in _numeric_fields(s):
        if not math.isfinite(value):
            errors.append(f"{name} is not finite ({value})")

    # --- geographic bounds ---
    if math.isfinite(s.latitude_deg) and not -90 <= s.latitude_deg <= 90:
        errors.append(f"latitude_deg out of range: {s.latitude_deg}")
    if math.isfinite(s.longitude_deg) and not -180 <= s.longitude_deg <= 180:
        errors.append(f"longitude_deg out of range: {s.longitude_deg}")
    if s.latitude_deg == 0 and s.longitude_deg == 0:
        errors.append("lat/lon is exactly 0,0 (unset or no fix)")

    # --- orientation ---
    q = s.orientation_quaternion
    norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    if math.isfinite(norm) and abs(norm - 1.0) > limits.max_quaternion_norm_error:
        errors.append(f"orientation quaternion not unit length (norm={norm:.4f})")

    # --- physical plausibility ---
    if math.isfinite(s.ground_speed_mps) and s.ground_speed_mps > limits.max_ground_speed_mps:
        errors.append(f"ground_speed_mps implausible: {s.ground_speed_mps:.3g}")
    if math.isfinite(s.vertical_speed_mps) and abs(s.vertical_speed_mps) > limits.max_vertical_speed_mps:
        errors.append(f"vertical_speed_mps implausible: {s.vertical_speed_mps:.3g}")
    a = s.acceleration_ned_mps2
    acc = math.sqrt(a.x**2 + a.y**2 + a.z**2)
    if math.isfinite(acc) and acc > limits.max_acceleration_mps2:
        errors.append(f"acceleration implausible: {acc:.3g} m/s^2")

    # --- safety-relevant observations (warnings, not data-quality failures) ---
    if s.collision.recent_collision and s.collision.is_resting_contact is False:
        speed = s.collision.impact_speed_mps
        warnings.append(f"collision impact with {s.collision.object_name}"
                        + (f" at ~{speed:.1f} m/s" if speed is not None else ""))
    if s.landed_state == LandedState.LANDED:
        moving = math.hypot(s.ground_speed_mps, s.vertical_speed_mps)
        if math.isfinite(moving) and moving > limits.landed_max_speed_mps:
            warnings.append(f"landed_state is LANDED but vehicle is moving at {moving:.1f} m/s")
    if s.landed_state == LandedState.UNKNOWN:
        warnings.append("landed_state unknown")

    non_stale = [e for e in errors if not e.startswith(STALE_PREFIX)]
    if non_stale:
        status = ValidationStatus.INVALID
    elif errors:
        status = ValidationStatus.STALE
    else:
        status = ValidationStatus.VALID

    s.validation_status = status
    s.validation_errors = errors
    s.validation_warnings = warnings
    return s
