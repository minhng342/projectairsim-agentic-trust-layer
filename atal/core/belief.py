"""
Local belief: what one drone knows about the search, kept as its memory ("save searched
portions to memory", 10/28 meeting).

    belief = LocalBelief("Drone1", mission, path="runs/.../run_00.memory.json")
    belief.mark_started("A1", sim_s, evidence=request_id)
    belief.mark_swept("A1", sim_s, evidence=request_id)
    belief.add_detection(Detection(...))
    belief.swept() / belief.unsearched() / belief.coverage_fraction()

Every item carries its SOURCE (design rule 3, "a claim is not a fact"). In Phase 1 the
only source is "self": the drone's own sweep or its own sensor. Phase 3 adds peer claims:

    belief.add_peer_claim(PeerClaim(...))        every valid peer message, as heard: who
                                                 (claimed sender), what, when sent and
                                                 received, message id (the evidence link)
    belief.mark_peer_cleared("C4", sim_s, sender, msg_ids)
                                                 only when the peer-claim rule accepts it
                                                 (atal/trust_layer/peer_rules.py): status peer_cleared,
                                                 source = the peer, verified_by_self False
    belief.is_done(sid)                          swept by self, or peer_cleared
    belief.cleared_by(sid) / belief.peer_victim_reports()

The raw claims are kept whether or not a rule accepted them, so Phase 4 can count which
drone acted on which claim, and Phase 5 adds a trust value per item (an item inherits the
lowest trust of its sources). Phase 1 memory files stay readable.

The memory is written to `path` (JSON, atomically) after every change, so a crash keeps
everything the drone had learned, and `LocalBelief.load` reads it back.
"""
import json
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from atal.core.sar import Mission

SELF = "self"


class SectorStatus(str, Enum):
    UNSEARCHED = "unsearched"
    IN_PROGRESS = "in_progress"     # a sweep of it is running now
    SWEPT = "swept"                 # every lane flown (by this drone)
    PARTIAL = "partial"             # a sweep started but did not finish (detection, failure)
    PEER_CLEARED = "peer_cleared"   # a peer said it is clear and the peer-claim rule accepted it


DONE = (SectorStatus.SWEPT, SectorStatus.PEER_CLEARED)


class SectorBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sector_id: str
    status: SectorStatus = SectorStatus.UNSEARCHED
    source: str = SELF
    verified_by_self: bool = False
    updated_sim_s: float | None = None
    evidence: list[str] = Field(default_factory=list)    # request ids / message ids behind it
    progress: float = 0.0                                # share of the sweep path flown
    trust: float | None = None                           # Phase 5


class Detection(BaseModel):
    """The drone's own sensor saw the victim. Position = where the DRONE was (the victim is
    somewhere inside the footprint around it): the planner never learns the true victim
    position from the scenario."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    vehicle_id: str
    sim_time_s: float
    x_m: float
    y_m: float
    altitude_m: float
    sector_id: str | None
    footprint_m: float
    source: str = SELF


class PeerClaim(BaseModel):
    """One valid peer message as this drone heard it (Phase 3). The claim, its claimed
    sender, the sender's time and this drone's receipt time; never trusted by itself."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    msg_id: str
    msg_type: str                         # sector_claimed | sector_cleared | sector_blocked | victim_detected
    sender_id: str                        # CLAIMED sender
    sector_id: str | None = None
    sent_sim_s: float                     # the sender's claimed time
    received_sim_s: float | None          # this drone's clock
    expires_sim_s: float | None = None    # sector_claimed
    x_m: float | None = None              # victim_detected: where the finder was
    y_m: float | None = None
    accepted: bool | None = None          # did a peer-claim rule act on it (None = no rule)
    rule: str | None = None               # the rule's code
    trust: float | None = None            # Phase 5


class LocalBelief:
    def __init__(self, vehicle_id: str, mission: Mission, path: str | os.PathLike | None = None,
                 on_change=None):
        self.vehicle_id = vehicle_id
        self.mission_id = mission.mission_id
        self.sectors: dict[str, SectorBelief] = {
            s: SectorBelief(sector_id=s) for s in mission.grid.ids()}
        self.detections: list[Detection] = []
        self.peer_claims: list[PeerClaim] = []
        self.path = Path(path) if path else None
        self._on_change = on_change
        self._save()

    # ------------------------------------------------------------ updates
    def mark_started(self, sid: str, sim_s: float | None, evidence: str | None = None) -> None:
        self._set(sid, SectorStatus.IN_PROGRESS, sim_s, evidence)

    def mark_swept(self, sid: str, sim_s: float | None, evidence: str | None = None) -> None:
        self._set(sid, SectorStatus.SWEPT, sim_s, evidence, progress=1.0)

    def mark_partial(self, sid: str, sim_s: float | None, progress: float,
                     evidence: str | None = None) -> None:
        self._set(sid, SectorStatus.PARTIAL, sim_s, evidence, progress=progress)

    def mark_peer_cleared(self, sid: str, sim_s: float | None, sender_id: str,
                          evidence: list[str]) -> bool:
        """A peer's clear, accepted by the peer-claim rule. Never overrides this drone's own
        sweep or a sweep it is flying now. Returns True if the status changed."""
        if sid not in self.sectors:
            raise KeyError(f"no sector {sid!r} in mission {self.mission_id}")
        b = self.sectors[sid]
        if b.status in (SectorStatus.SWEPT, SectorStatus.IN_PROGRESS, SectorStatus.PEER_CLEARED):
            return False
        ev = b.evidence + [e for e in evidence if e not in b.evidence]
        self.sectors[sid] = b.model_copy(update={
            "status": SectorStatus.PEER_CLEARED, "source": sender_id, "verified_by_self": False,
            "updated_sim_s": sim_s, "evidence": ev})
        self._changed("sector", sector=self.sectors[sid].model_dump(mode="json"))
        return True

    def add_peer_claim(self, c: PeerClaim) -> None:
        self.peer_claims.append(c)
        self._changed("peer_claim", claim=c.model_dump(mode="json"))

    def add_detection(self, d: Detection) -> None:
        self.detections.append(d)
        self._changed("detection", detection=d.model_dump(mode="json"))

    # ------------------------------------------------------------ queries
    def status(self, sid: str) -> SectorStatus:
        return self.sectors[sid].status

    def swept(self) -> list[str]:
        """Sectors this drone swept itself."""
        return [s for s, b in self.sectors.items() if b.status == SectorStatus.SWEPT]

    def is_done(self, sid: str) -> bool:
        """Nothing left to search here, as far as this drone believes: swept by itself, or
        cleared by a peer under the peer-claim rule."""
        return self.sectors[sid].status in DONE

    def done(self) -> list[str]:
        return [s for s, b in self.sectors.items() if b.status in DONE]

    def unsearched(self) -> list[str]:
        return [s for s, b in self.sectors.items() if b.status not in DONE]

    def coverage_fraction(self) -> float:
        """Share of the grid this drone believes is done (own sweeps + accepted peer clears)."""
        return len(self.done()) / len(self.sectors)

    def cleared_by(self, sid: str) -> list[str]:
        """Distinct peers that claimed this sector clear (accepted or not)."""
        out = []
        for c in self.peer_claims:
            if c.msg_type == "sector_cleared" and c.sector_id == sid and c.sender_id not in out:
                out.append(c.sender_id)
        return out

    def peer_victim_reports(self) -> list[PeerClaim]:
        return [c for c in self.peer_claims if c.msg_type == "victim_detected"]

    @property
    def victim_detected(self) -> bool:
        return bool(self.detections)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"vehicle_id": self.vehicle_id, "mission_id": self.mission_id,
                "coverage_fraction": round(self.coverage_fraction(), 4),
                "sectors": {s: b.model_dump(mode="json") for s, b in self.sectors.items()},
                "detections": [d.model_dump(mode="json") for d in self.detections],
                "peer_claims": [c.model_dump(mode="json") for c in self.peer_claims]}

    @classmethod
    def load(cls, path, mission: Mission) -> "LocalBelief":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("mission_id") != mission.mission_id:
            raise ValueError(f"memory {path} is for mission {data.get('mission_id')!r}, "
                             f"not {mission.mission_id!r}")
        b = cls(data["vehicle_id"], mission)
        b.sectors.update({s: SectorBelief.model_validate(v) for s, v in data["sectors"].items()})
        b.detections = [Detection.model_validate(d) for d in data["detections"]]
        b.peer_claims = [PeerClaim.model_validate(c) for c in data.get("peer_claims", [])]
        b.path = Path(path)
        return b

    # ------------------------------------------------------------ internal
    def _set(self, sid, status, sim_s, evidence, progress=None) -> None:
        if sid not in self.sectors:
            raise KeyError(f"no sector {sid!r} in mission {self.mission_id}")
        b = self.sectors[sid]
        ev = b.evidence + ([evidence] if evidence and evidence not in b.evidence else [])
        upd = {"status": status, "source": SELF, "verified_by_self": True,
               "updated_sim_s": sim_s, "evidence": ev}
        if progress is not None:
            upd["progress"] = round(max(b.progress, progress), 4)
        self.sectors[sid] = b.model_copy(update=upd)
        self._changed("sector", sector=self.sectors[sid].model_dump(mode="json"))

    def _changed(self, what: str, **fields) -> None:
        self._save()
        if self._on_change is not None:
            try:
                self._on_change("belief_update", {"vehicle_id": self.vehicle_id, "what": what,
                                                  "coverage_fraction": round(self.coverage_fraction(), 4),
                                                  **fields})
            except Exception:
                pass

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
