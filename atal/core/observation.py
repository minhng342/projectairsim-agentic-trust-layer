"""
What an agent is allowed to see, and the decisions made about it.

Observation
    One message delivered to ONE vehicle's agent over its observation channel.
    It is NOT a TelemetrySnapshot: the adapter's validation verdict is removed
    before delivery, because that verdict was computed on the clean simulator
    data. If an attack hook modifies the content in transit, an old VALID would
    otherwise ride along as false proof that the modified data was checked.

    Fields set by the RECEIVER (the feed) and not changeable in transit:
        observation_id, vehicle_id (the channel it arrived on), sequence,
        channel (source identity of the feed), received_host_s, received_at_utc
    Content as DELIVERED (possibly modified in transit):
        payload: the raw message, kept whole for the trust assessor. It includes
        the sender's own claims (claimed vehicle_id, source, sim_time_ns).

    Immutable (frozen): fields cannot be reassigned. The payload dict itself is
    still a dict, so the feed never hands out its own record: it keeps the
    authoritative copy and gives the agent a deep copy. A proposal refers to an
    observation by observation_id; the port resolves that id to the feed's own
    copy, so whatever the agent did to its copy cannot reach the gate.

TrustAssessment
    The trust/provenance assessor's verdict on one delivered observation.
    NOT_ASSESSED (assessed=False) is what the placeholder assessor returns; it
    must never be read as "trusted".

GateDecision
    The action-risk gate's verdict on one proposal. BYPASSED (assessed=False) is
    what the placeholder gate returns: the action went through WITHOUT a risk
    assessment, and the log says so.
"""
import math
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> str:
    return uuid4().hex


# ------------------------------------------------------------ trust assessment
class TrustStatus(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    NOT_ASSESSED = "not_assessed"


class TrustAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(default_factory=new_id)
    observation_id: str
    assessor: str
    assessed: bool
    status: TrustStatus
    reasons: list[str] = Field(default_factory=list)


# ------------------------------------------------------------ observation
class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    observation_id: str = Field(default_factory=new_id)
    vehicle_id: str
    sequence: int
    channel: str
    received_host_s: float
    received_at_utc: datetime
    payload: dict[str, Any]
    assessment: TrustAssessment | None = None

    # Convenience readers. They read the DELIVERED payload, so tampering shows
    # up here exactly as an agent would experience it. Missing or malformed
    # fields read as NaN (an agent action built from NaN is rejected by
    # ProposedAction), never as a plausible default.
    def _num(self, *path) -> float:
        v: Any = self.payload
        try:
            for p in path:
                v = v[p]
            v = float(v)
        except (KeyError, TypeError, ValueError, IndexError):
            return math.nan
        return v

    @property
    def source_sim_time_ns(self) -> int | None:
        """The sender's claimed timestamp (part of the content, so it can be forged)."""
        v = self.payload.get("sim_time_ns")
        return int(v) if isinstance(v, (int, float)) and math.isfinite(v) else None

    @property
    def north_m(self) -> float:
        return self._num("position_ned_m", "x")

    @property
    def east_m(self) -> float:
        return self._num("position_ned_m", "y")

    @property
    def altitude_m(self) -> float:
        return self._num("altitude_local_m")

    @property
    def heading_deg(self) -> float:
        return self._num("heading_deg")

    @property
    def ground_speed_mps(self) -> float:
        return self._num("ground_speed_mps")

    @property
    def vertical_speed_mps(self) -> float:
        return self._num("vertical_speed_mps")

    @property
    def ground_state(self) -> str | None:
        v = self.payload.get("ground_state")
        return v if isinstance(v, str) else None


# ------------------------------------------------------------ gate decision
class GateOutcome(str, Enum):
    APPROVED = "approved"      # assessed and allowed
    REJECTED = "rejected"      # assessed and blocked (never sent)
    BYPASSED = "bypassed"      # NOT assessed; passed through because no gate is installed


class GateDecision(BaseModel):
    decision_id: str = Field(default_factory=new_id)
    request_id: str
    observation_id: str
    gate: str
    outcome: GateOutcome
    assessed: bool
    reason: str = ""

    @property
    def allows_execution(self) -> bool:
        return self.outcome in (GateOutcome.APPROVED, GateOutcome.BYPASSED)
