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
    # Flight-controller commands (executor/projectairsim_executor.py): each runs to completion.
    HOLD = "hold"                              # stop and stay; optional duration_s
    TAKEOFF = "takeoff"
    ROTATE_TO_HEADING = "rotate_to_heading"    # turn the nose; position held
    CHANGE_ALTITUDE = "change_altitude"        # climb/descend; N/E held
    MOVE_ALONG_TRACK = "move_along_track"      # move in a direction for a time; heading unchanged
    MOVE_TO_POSITION = "move_to_position"      # go to N/E/altitude and stop
    LAND = "land"
    # Mission-layer maneuvers (executor/mission_runner.py): change how the drone flies its
    # corridor; applied at once, the drone carries them out while the mission continues.
    #   HOLD and CHANGE_ALTITUDE (a layer change) above are maneuvers there too.
    MAINTAIN = "maintain"                      # no change: a recorded decision to carry on
    RESUME = "resume"                          # continue the corridor after a hold
    CHANGE_SPEED = "change_speed"              # new cruise speed along the corridor
    # Search and rescue actions (executor/sector_executor.py), Phase 1 onward. Each runs to
    # completion: a sector is swept (or the sweep is stopped by a detection), home is reached.
    SWEEP_SECTOR = "sweep_sector"              # boustrophedon sweep of one grid sector
    RETURN_HOME = "return_home"                # fly back over the launch point at mission altitude


PARAMS = ("heading_deg", "track_deg", "speed_mps", "altitude_m",
          "north_m", "east_m", "duration_s", "sector_id")
FLOAT_PARAMS = PARAMS[:-1]
# Grid sector ids: column letter (along y / east) + row number (along x / north), e.g. "F5".
SECTOR_ID_PATTERN = r"^[A-Z][1-9][0-9]?$"

REQUIRED: dict[ActionType, set[str]] = {
    ActionType.HOLD: set(),
    ActionType.TAKEOFF: set(),
    ActionType.ROTATE_TO_HEADING: {"heading_deg"},
    ActionType.CHANGE_ALTITUDE: {"altitude_m"},
    ActionType.MOVE_ALONG_TRACK: {"track_deg", "speed_mps", "duration_s"},
    ActionType.MOVE_TO_POSITION: {"north_m", "east_m", "altitude_m"},
    ActionType.LAND: set(),
    ActionType.MAINTAIN: set(),
    ActionType.RESUME: set(),
    ActionType.CHANGE_SPEED: {"speed_mps"},
    ActionType.SWEEP_SECTOR: {"sector_id"},
    ActionType.RETURN_HOME: set(),
}
OPTIONAL: dict[ActionType, set[str]] = {
    ActionType.HOLD: {"duration_s"},               # without it: hold until resume
    ActionType.CHANGE_ALTITUDE: {"speed_mps"},     # vertical speed
    ActionType.MOVE_TO_POSITION: {"speed_mps"},    # cruise speed
    ActionType.RETURN_HOME: {"altitude_m"},        # SAR obstacle recovery: home higher up
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
    sector_id: str | None = Field(default=None, pattern=SECTOR_ID_PATTERN)

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
        for p in given & set(FLOAT_PARAMS):
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
    """The executor's full report. EVALUATION SIDE: `reason` may quote measured
    errors and the snapshots are simulator ground truth. Agents get an
    ActionFeedback instead (see channel/action_port.py)."""
    request_id: str
    vehicle_id: str
    action_type: ActionType
    status: CommandStatus
    reason: str = ""
    reason_code: str = ""                   # machine-readable cause, see executor REASON_CODES
    sent_to_simulator: bool = False
    fallback_applied: str | None = None     # e.g. "hover", "land"
    corrections: int = 0                    # bounded re-sends of a position target
    start_snapshot: TelemetrySnapshot | None = None
    final_snapshot: TelemetrySnapshot | None = None
    started_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_s: float = 0.0


class FeedbackStage(str, Enum):
    PORT = "port"            # rejected by the action port; never reached gate or executor
    GATE = "gate"            # stopped by the action-risk gate; never sent
    EXECUTOR = "executor"    # reached the executor (sent or refused at preflight)


class ActionFeedback(BaseModel):
    """What the AGENT is told about its proposal. Everything here is either about the
    agent's own request or a coarse completion status a real autopilot would report.
    No positions, no measured errors, no telemetry-health detail, no exception text:
    those come from ground truth and stay in the evaluation record (CommandResult)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str | None                  # None only if the action could not be read at all
    vehicle_id: str                         # the port's vehicle
    action_type: ActionType | None
    observation_id: str | None
    stage: FeedbackStage
    status: CommandStatus
    code: str                               # stable, small vocabulary (see action_port)
    message: str                            # fixed text per code, or the port's / gate's own words
    sent_to_simulator: bool = False
    fallback: str | None = None             # "hover", "hover_failed", "none"
    gate_outcome: str | None = None
    gate_assessed: bool | None = None
    elapsed_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status == CommandStatus.SUCCEEDED
