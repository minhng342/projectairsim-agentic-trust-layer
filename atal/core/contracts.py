"""The slots other folders plug into, in one place, so no folder has to import a
specific implementation to use one.

    TrustAssessor   judges one delivered observation (atal/channels/observation_feed.py,
                    radio.py call it). It sees ONLY what arrived on one vehicle's
                    channel: the delivered observation (raw payload included) and the
                    previous one from the same channel. Never the clean pre-attack
                    payload, the attack hooks, or ground truth.
    ActionGate      decides on one proposed action (atal/channels/action_port.py calls
                    it), from the proposal and the observation it was based on. It does
                    not see ground truth.
    HookContext     what an attack hook (atal/attacks/) is told about the message it
                    may change on the observation channel.
    PeerSender      what a drone's coordinator sends peer messages through
                    (atal/channels/peer_link.py: PeerMessenger).

Two placeholders are part of the contract, because they say plainly that nothing was
checked: NoTrustAssessor returns NOT_ASSESSED (never equivalent to TRUSTED), and
BypassGate returns BYPASSED with assessed=False. A run logged with BYPASSED decisions is
an UNPROTECTED run, never a run where every action "passed" a risk check. Real checks
live in atal/trust_layer/.
"""
from dataclasses import dataclass
from typing import Protocol

from atal.core.action import ProposedAction
from atal.core.observation import (
    GateDecision,
    GateOutcome,
    Observation,
    TrustAssessment,
    TrustStatus,
)


class TrustAssessor(Protocol):
    name: str

    def assess(self, observation: Observation, previous: Observation | None) -> TrustAssessment: ...


class NoTrustAssessor:
    """Explicitly does not assess. NOT_ASSESSED is never equivalent to TRUSTED."""
    name = "none"

    def assess(self, observation: Observation, previous: Observation | None) -> TrustAssessment:
        return TrustAssessment(observation_id=observation.observation_id, assessor=self.name,
                               assessed=False, status=TrustStatus.NOT_ASSESSED,
                               reasons=["no trust assessor installed"])


class ActionGate(Protocol):
    name: str

    def decide(self, action: ProposedAction, based_on: Observation) -> GateDecision: ...


class BypassGate:
    name = "none"

    def decide(self, action: ProposedAction, based_on: Observation) -> GateDecision:
        return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                            gate=self.name, outcome=GateOutcome.BYPASSED, assessed=False,
                            reason="no action-risk gate installed; passed through unassessed")


@dataclass(frozen=True)
class HookContext:
    vehicle_id: str
    sequence: int
    received_host_s: float


class PeerSender(Protocol):
    vehicle_id: str

    def send(self, msg) -> list[dict]: ...
