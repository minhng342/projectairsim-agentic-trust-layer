"""
Precondition rules for peer claims (Phase 3, deck slide 16): when may a claim from a peer
change what this drone believes or does? Plain Python, written by the team, never the LLM.

    rule                 claim             accepted when
    clear: single        SectorCleared     any one peer says the sector is clear (the Phase 3
                                           cooperation baseline: honest peers, so one is enough)
    clear: corroborated  SectorCleared     corroboration_n DIFFERENT peers said it (a Phase 5
                                           defense; with 3 drones that is both other drones)
    clear: self_only     SectorCleared     never: the claim is recorded, the drone sweeps the
                                           sector itself if it is in its cells
    victim: accept       VictimDetected    the search is over for this drone (Phase 3 baseline)
    victim: ignore       VictimDetected    recorded only
    claim                SectorClaimed     always honoured while the claimant is alive (heard
                                           within peer_timeout_s) and the claim has not
                                           expired; expiry is capped at max_claim_s after
                                           receipt; a newer claim from the same sender
                                           replaces its older one (one claim per drone)

Decision for Phase 3 (roadmap open item): a single honest peer's SectorCleared IS accepted.
That is the baseline Phase 4 attacks ("E6 is clear") and Phase 5's corroboration rule
defends. Every verdict is recorded with the message ids behind it (design rule 4).
"""
from dataclasses import dataclass

from models.belief import LocalBelief, PeerClaim
from models.sar import Swarm


@dataclass(frozen=True)
class RuleVerdict:
    accepted: bool
    code: str
    message: str
    evidence: tuple[str, ...] = ()


class PeerClaimRules:
    name = "peer_claim_rules_v1"

    def __init__(self, swarm: Swarm):
        self.swarm = swarm

    def on_cleared(self, belief: LocalBelief, claim: PeerClaim) -> RuleVerdict:
        sid = claim.sector_id
        rule = self.swarm.peer_clear_rule
        ev = tuple(c.msg_id for c in belief.peer_claims
                   if c.msg_type == "sector_cleared" and c.sector_id == sid
                   and c.msg_id != claim.msg_id) + (claim.msg_id,)
        if sid not in belief.sectors:
            return RuleVerdict(False, "unknown_sector", f"{sid} is not in the mission grid", ev)
        if belief.sectors[sid].verified_by_self and belief.status(sid).value == "swept":
            return RuleVerdict(False, "already_swept_by_self", f"{sid} already swept by this "
                               f"drone", ev)
        if rule == "self_only":
            return RuleVerdict(False, "self_only", f"{sid}: peer clears are not accepted "
                               f"(rule self_only)", ev)
        senders = belief.cleared_by(sid)
        if claim.sender_id not in senders:          # the claim being decided on
            senders = senders + [claim.sender_id]
        if rule == "single":
            return RuleVerdict(True, "single_peer", f"{sid} cleared by {claim.sender_id}", ev)
        n = self.swarm.corroboration_n
        if len(senders) >= n:
            return RuleVerdict(True, "corroborated", f"{sid} cleared by {len(senders)} peers "
                               f"({', '.join(senders)})", ev)
        return RuleVerdict(False, "needs_corroboration", f"{sid}: {len(senders)} of {n} "
                           f"peers say clear", ev)

    def on_blocked(self, belief: LocalBelief, claim: PeerClaim) -> RuleVerdict:
        ev = (claim.msg_id,)
        if claim.sector_id not in belief.sectors:
            return RuleVerdict(False, "unknown_sector", f"{claim.sector_id} is not in the "
                               f"mission grid", ev)
        if self.swarm.peer_blocked_rule == "accept":
            return RuleVerdict(True, "accept", f"{claim.sender_id} was stopped by an obstacle "
                               f"in {claim.sector_id}", ev)
        return RuleVerdict(False, "ignore", "peer blocked reports are recorded only", ev)

    def on_victim(self, belief: LocalBelief, claim: PeerClaim) -> RuleVerdict:
        if self.swarm.peer_victim_rule == "accept":
            return RuleVerdict(True, "accept", f"{claim.sender_id} reports the victim in "
                               f"{claim.sector_id}", (claim.msg_id,))
        return RuleVerdict(False, "ignore", "peer victim reports are recorded only",
                           (claim.msg_id,))

    def claim_expiry(self, claim: PeerClaim) -> float | None:
        """When a SectorClaimed stops counting, on this drone's clock: the claimed expiry,
        capped at max_claim_s after receipt."""
        if claim.received_sim_s is None:
            return claim.expires_sim_s
        cap = claim.received_sim_s + self.swarm.max_claim_s
        exp = claim.expires_sim_s if claim.expires_sim_s is not None else cap
        # the sender's clock is a claim: measure the duration it asked for from receipt
        exp = claim.received_sim_s + max(0.0, exp - claim.sent_sim_s)
        return min(exp, cap)
