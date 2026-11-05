"""
Peer messages (Phase 3): what one drone tells the others on the swarm bus.

    SectorClaimed    "I am sweeping this sector now" (until expires_sim_s)
    SectorCleared    "I swept this sector and did not see the victim"
    VictimDetected   "my sensor saw the victim here" (the finder's position, like Detection)

Every message is a Pydantic model with a sender_id (deck slide 16). A payload that does
not validate (unknown type, missing or extra field, malformed sector id) is dropped by the
receiver before anything else sees it, and the drop is recorded. Phase 4 shows that
validation alone does not stop a well-formed lie.

sender_id is a CLAIM, like Remote ID's uas_id: the receiver never learns who really
transmitted (the bus records that on the evaluation side only). sim_time_s is the sender's
own clock, also a claim; receivers stamp their own receipt time.

Messages travel on the same bus as the 1 Hz Remote ID position broadcast
(channel/remote_id.py), so they share its radio-range filter, sender hooks (the insider's
slot in Phase 4) and link hooks. A payload with a "msg_type" field is a peer message;
anything else is a Remote ID report.

Design rules (deck slide 25): peers inform beliefs, never goals (no message can change the
mission), and each drone emits only facts it verified itself (agent/swarm.py).
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from models.action import SECTOR_ID_PATTERN
from models.observation import new_id


class _PeerMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    msg_id: str = Field(default_factory=new_id, min_length=1, max_length=64)
    sender_id: str = Field(min_length=1, max_length=64)
    sim_time_s: float = Field(ge=0)


class SectorClaimed(_PeerMessage):
    msg_type: Literal["sector_claimed"] = "sector_claimed"
    sector_id: str = Field(pattern=SECTOR_ID_PATTERN)
    expires_sim_s: float = Field(ge=0)


class SectorCleared(_PeerMessage):
    msg_type: Literal["sector_cleared"] = "sector_cleared"
    sector_id: str = Field(pattern=SECTOR_ID_PATTERN)


class VictimDetected(_PeerMessage):
    msg_type: Literal["victim_detected"] = "victim_detected"
    sector_id: str | None = Field(default=None, pattern=SECTOR_ID_PATTERN)
    x_m: float          # where the FINDER was (the victim is inside its footprint)
    y_m: float
    footprint_m: float = Field(gt=0)


PeerMessage = Annotated[Union[SectorClaimed, SectorCleared, VictimDetected],
                        Field(discriminator="msg_type")]
PEER_MESSAGE = TypeAdapter(PeerMessage)
MESSAGE_TYPES = ("sector_claimed", "sector_cleared", "victim_detected")


def is_peer_message(payload) -> bool:
    """A bus payload that claims to be a peer message (validated separately)."""
    return isinstance(payload, dict) and "msg_type" in payload
