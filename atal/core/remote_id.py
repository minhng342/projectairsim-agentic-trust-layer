"""
Remote ID reports: what one drone hears about another on the shared broadcast channel.

Modeled on ASTM F3411 broadcast Remote ID: unauthenticated, anyone can transmit any
content. A message is a plain dict in transit (so attacks can change, add or drop any
field), with these fields when honest (see atal/channels/radio.py, message_from_snapshot):

    uas_id              claimed sender id (forgeable)
    sim_time_ns         sender's claimed timestamp
    north_m, east_m, altitude_m             claimed position (local NED frame, as observations)
    ground_speed_mps, track_deg, vertical_speed_mps   claimed motion
    self_id_text        free text. Real Self-ID is 23 characters; a longer "operator
                        message" is an explicit modeling assumption. It is the only free
                        text any agent reads, which makes it the prompt-injection carrier.

RemoteIdReport is one RECEIVED message, stamped by the receiver (fields it controls)
around the payload as delivered. Like Observation it is frozen and the receiver hands
out deep copies. The receiver never learns who really transmitted: streams are keyed
by the CLAIMED uas_id, exactly as a real receiver must.
"""
import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from atal.core.observation import TrustAssessment, new_id

NUMERIC_FIELDS = ("north_m", "east_m", "altitude_m", "ground_speed_mps", "track_deg",
                  "vertical_speed_mps")


class RemoteIdReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    # receiver-owned
    observation_id: str = Field(default_factory=new_id)    # "observation" of a peer; id for refs
    receiver_id: str                                        # the drone that heard it
    channel: str                                            # "remote_id"
    claimed_uas_id: str                                     # stream key (from the payload)
    sequence: int                                           # per (receiver, claimed id) stream
    received_host_s: float
    received_at_utc: datetime
    # as delivered
    payload: dict[str, Any]
    assessment: TrustAssessment | None = None

    def num(self, name: str) -> float:
        v = self.payload.get(name)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return math.nan
        return v

    @property
    def self_id_text(self) -> str | None:
        v = self.payload.get("self_id_text")
        return v if isinstance(v, str) else None

    @property
    def source_sim_time_ns(self) -> int | None:
        v = self.payload.get("sim_time_ns")
        return int(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
