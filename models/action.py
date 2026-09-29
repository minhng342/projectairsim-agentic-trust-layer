"""
Action contract: what an agent (or a test) may ask the executor to do, and
what the executor reports back.

Design rules
- Every action type has an explicit, unambiguous parameter set. Supplying a
  parameter that doesn't belong to the action is an error, so "heading" can
  never silently mean "track" (Pass 4 review finding).
- Heading = where the nose points. Track = where the vehicle moves.
- Altitudes are local (meters above the NED origin, i.e. -z), not MSL.
- This model only checks structure and physical domain (finite, 0-360, > 0).
  Policy limits (max altitude, speed, duration, geofence, allowed actions)
  belong to the action-risk gate, not here.
"""
import math
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.telemetry import TelemetrySnapshot


class ActionType(str, Enum):
    HOLD = "hold"                              # hover in place for duration_s
    TAKEOFF = "takeoff"
    ROTATE_TO_HEADING = "rotate_to_heading"    # turn the nose; position held
    CHANGE_ALTITUDE = "change_altitude"        # climb/descend; N/E held
    MOVE_ALONG_TRACK = "move_along_track"      # move in a direction for a time; heading unchanged
    MOVE_TO_POSITION = "move_to_position"      # go to N/E/altitude and stop
    LAND = "land"


PARAMS = ("heading_deg", "track_deg", "speed_mps", "altitude_m",
          "north_m", "east_m", "duration_s")

REQUIRED: dict[ActionType, set[str]] = {
    ActionType.HOLD: {"duration_s"},
    ActionType.TAKEOFF: set(),
    ActionType.ROTATE_TO_HEADING: {"heading_deg"},
    ActionType.CHANGE_ALTITUDE: {"altitude_m"},
    ActionType.MOVE_ALONG_TRACK: {"track_deg", "speed_mps", "duration_s"},
    ActionType.MOVE_TO_POSITION: {"north_m", "east_m", "altitude_m"},
    ActionType.LAND: set(),
}
OPTIONAL: dict[ActionType, set[str]] = {
    ActionType.CHANGE_ALTITUDE: {"speed_mps"},     # vertical speed
    ActionType.MOVE_TO_POSITION: {"speed_mps"},    # cruise speed
}


class ProposedAction(BaseModel):
    # Trust boundary: unknown fields from an agent are rejected, never ignored.
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    vehicle_id: str = Field(min_length=1)
    action_type: ActionType
    reason: str = Field(min_length=1, max_length=500)

    heading_deg: float | None = None
    track_deg: float | None = None
    speed_mps: float | None = None
    altitude_m: float | None = None
    north_m: float | None = None
    east_m: float | None = None
    duration_s: float | None = None

    @model_validator(mode="after")
    def _check_parameters(self):
        given = {p for p in PARAMS if getattr(self, p) is not None}
        required = REQUIRED[self.action_type]
        allowed = required | OPTIONAL.get(self.action_type, set())
        missing = required - given
        extra = given - allowed
        problems = []
        if missing:
            problems.append(f"missing {sorted(missing)}")
        if extra:
            problems.append(f"not allowed for {self.action_type.value}: {sorted(extra)}")
        for p in given:
            if not math.isfinite(getattr(self, p)):
                problems.append(f"{p} must be finite")
        for p in ("heading_deg", "track_deg"):
            v = getattr(self, p)
            if v is not None and math.isfinite(v) and not 0.0 <= v < 360.0:
                problems.append(f"{p} must be in [0, 360)")
        for p in ("speed_mps", "duration_s"):
            v = getattr(self, p)
            if v is not None and math.isfinite(v) and v <= 0:
                problems.append(f"{p} must be > 0")
        if problems:
            raise ValueError(f"{self.action_type.value}: " + "; ".join(problems))
        return self


class CommandStatus(str, Enum):
    SUCCEEDED = "succeeded"    # command completed and telemetry settled within tolerance
    REFUSED = "refused"        # not sent: telemetry not VALID, disconnected, bad parameters
    TIMED_OUT = "timed_out"    # sent, but did not settle before the deadline; fallback applied
    FAILED = "failed"          # sent, but the simulator reported an error; fallback applied


class CommandResult(BaseModel):
    request_id: str
    vehicle_id: str
    action_type: ActionType
    status: CommandStatus
    reason: str = ""
    sent_to_simulator: bool = False
    fallback_applied: str | None = None     # e.g. "hover", "land"
    start_snapshot: TelemetrySnapshot | None = None
    final_snapshot: TelemetrySnapshot | None = None
    started_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_s: float = 0.0
