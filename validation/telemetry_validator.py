"""
Deterministic telemetry checks. Adapter-agnostic: works on any TelemetrySnapshot.

Rules come from what Pass 3 actually observed: disabled sensors returned
timestamp 0, NaN fields, lat/lon 0,0 and a -6.7e29 velocity. A numeric field
existing is not the same as it being trustworthy.
"""
import math
from dataclasses import dataclass

from models.telemetry import TelemetrySnapshot, ValidationStatus


@dataclass(frozen=True)
class ValidationLimits:
    max_telemetry_age_ms: float = 500.0
    max_ground_speed_mps: float = 40.0
    max_vertical_speed_mps: float = 20.0
    max_acceleration_mps2: float = 100.0
    max_quaternion_norm_error: float = 0.01
    max_pose_skew_ms: float = 100.0


def _numeric_fields(s: TelemetrySnapshot):
    for name in ("position_ned_m", "velocity_ned_mps", "acceleration_ned_mps2",
                 "angular_velocity_rad_s", "orientation_quaternion"):
        for axis, value in getattr(s, name).model_dump().items():
            yield f"{name}.{axis}", value
    for name in ("latitude_deg", "longitude_deg", "altitude_msl_m", "altitude_local_m",
                 "heading_deg", "ground_speed_mps", "vertical_speed_mps"):
        yield name, getattr(s, name)


def validate_snapshot(s: TelemetrySnapshot,
                      limits: ValidationLimits = ValidationLimits()) -> TelemetrySnapshot:
    """Fill in validation_status / errors / warnings on the snapshot and return it."""
    errors: list[str] = []
    warnings: list[str] = []

    # --- timestamps / freshness ---
    if s.sim_time_ns <= 0:
        errors.append(f"sim_time_ns={s.sim_time_ns}: no valid simulation timestamp")
    stale = False
    if s.telemetry_age_ms is None:
        errors.append("actual_pose: no messages received")
    elif s.telemetry_age_ms > limits.max_telemetry_age_ms:
        stale = True
        errors.append(f"actual_pose stale: {s.telemetry_age_ms:.0f} ms old "
                      f"(limit {limits.max_telemetry_age_ms:.0f} ms)")
    if s.pose_topic_sim_time_ns and s.sim_time_ns > 0:
        skew_ms = abs(s.sim_time_ns - s.pose_topic_sim_time_ns) / 1e6
        if skew_ms > limits.max_pose_skew_ms:
            warnings.append(f"pose topic and kinematics differ by {skew_ms:.0f} ms sim time")

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

    # --- safety-relevant state (reported, not a data-quality failure) ---
    if s.collision.in_contact and s.collision.is_resting_contact is False:
        warnings.append(f"collision impact with {s.collision.object_name} "
                        f"at ~{s.collision.impact_speed_mps:.1f} m/s")

    non_stale_errors = [e for e in errors if not e.startswith("actual_pose stale")]
    if non_stale_errors:
        status = ValidationStatus.INVALID
    elif stale:
        status = ValidationStatus.STALE
    else:
        status = ValidationStatus.VALID

    s.validation_status = status
    s.validation_errors = errors
    s.validation_warnings = warnings
    return s
