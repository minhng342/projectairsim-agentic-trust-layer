"""
Normalized telemetry contract shared by all simulator adapters.

Fields carry no pydantic range constraints: a snapshot is always constructed,
even from bad data, and problems are reported in validation_status /
validation_errors instead of raising. The trust layer needs to *see* invalid
telemetry, not crash on it.

Frames and units
- NED: x = north, y = east, z = down (meters). Positive z / down-velocity
  means descending.
- SI units throughout (m, m/s, m/s^2, rad/s). Convert to ft/kt only at the
  presentation layer (e.g. agent/telemetry_formatter.py).

Heading vs. track
- heading_deg: where the nose points (yaw from the orientation quaternion).
- track_deg: where the vehicle is moving (from horizontal velocity).
  A multirotor can hold one heading while moving along any track, so the two
  must never be treated as the same quantity.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    w: float
    x: float
    y: float
    z: float


class LandedState(str, Enum):
    LANDED = "landed"
    FLYING = "flying"
    UNKNOWN = "unknown"


class GroundState(str, Enum):
    """Operational on-ground decision (see TelemetrySnapshot.ground_state)."""
    GROUNDED = "grounded"
    AIRBORNE = "airborne"
    UNKNOWN = "unknown"


class CollisionState(BaseModel):
    has_collided: bool = False
    """True once any collision_info message has been received this session."""
    recent_collision: bool = False
    """The latest collision message arrived within the adapter's recent window
    (sim time). This is temporal: it does NOT mean the vehicle is touching
    something now. Use TelemetrySnapshot.landed_state for that."""
    object_name: str | None = None
    sim_time_ns: int | None = None
    impact_speed_mps: float | None = None
    """Approximate speed just before the collision, from a ~100 ms window of
    actual_pose samples. Indicative only, not authoritative impact severity."""
    is_resting_contact: bool | None = None
    """Latest collision was a low-speed touch (landing, resting on a surface).
    None = speed unknown (e.g. collision reported before any pose arrived)."""
    normal_z: float | None = None
    """z of the contact normal (NED). Ground observed at -1 (upward-facing);
    walls are near 0. None if the message had no usable normal."""
    is_supporting_surface: bool = False
    """normal_z <= -0.7: an upward-facing surface that can hold the vehicle up.
    Only these contacts can make ground_state GROUNDED via contact+stillness."""
    count: int = 0
    impact_count: int = 0
    """Collisions above the impact speed threshold (not resting contact)."""


class ValidationStatus(str, Enum):
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"


class TelemetrySnapshot(BaseModel):
    vehicle_id: str
    source: str = "projectairsim_ground_truth"

    sim_time_ns: int
    received_at_utc: datetime
    pose_topic_sim_time_ns: int | None = None

    position_ned_m: Vector3
    orientation_quaternion: Quaternion
    velocity_ned_mps: Vector3
    acceleration_ned_mps2: Vector3
    angular_velocity_rad_s: Vector3

    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float
    altitude_local_m: float
    """-position_ned_m.z: height relative to the NED origin (not above ground)."""

    heading_deg: float
    """Yaw from the orientation quaternion, 0-360, 0 = north."""
    track_deg: float | None = None
    """Direction of horizontal motion; None when ground speed is too low to be meaningful."""
    ground_speed_mps: float
    vertical_speed_mps: float
    """Positive = climbing (i.e. -velocity_ned_mps.z)."""

    landed_state: LandedState = LandedState.UNKNOWN
    """Raw Project AirSim get_landed_state(), unmodified. Observed to stay FLYING
    for ~12 s after physical touchdown, so don't use it alone for decisions."""
    ground_state: GroundState = GroundState.UNKNOWN
    """Operational on-ground state. GROUNDED only if the vehicle is still
    (ground speed <= 0.25 m/s, |vertical speed| <= 0.15 m/s) AND either
      - raw landed_state is LANDED, or
      - a slow contact with an upward-facing surface (normal_z <= -0.7) was
        observed, the vehicle hasn't moved > 0.5 m horizontally or > 0.3 m
        vertically from it, and it has stayed still for >= 1 s.
    Raw LANDED while moving gives UNKNOWN. A sim clock reset clears everything.
    Use this, not landed_state, for commands and shutdown decisions, and only
    when validation_status is VALID."""
    ground_state_basis: str = ""
    """Why ground_state has its value (e.g. "landed_state", "resting_contact+still")."""
    collision: CollisionState = Field(default_factory=CollisionState)

    telemetry_age_ms: float | None = None
    """Transport age: host time since the last valid actual_pose message arrived
    (duplicates included). None if none received."""
    sim_progress_age_ms: float | None = None
    """Host time since the pose timestamp last ADVANCED. A frozen or replayed
    stream keeps telemetry_age_ms low but lets this grow. None if none received."""

    validation_status: ValidationStatus = ValidationStatus.VALID
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)

    @property
    def sim_time_s(self) -> float:
        return self.sim_time_ns / 1e9
