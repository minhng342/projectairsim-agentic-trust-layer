"""
Action-risk gate interface.

The gate sees the proposal and the observation it was based on (the same
delivered data, with its trust assessment). It does not see ground truth.

Placeholder for now: BypassGate. It does not assess anything and says so:
outcome BYPASSED, assessed=False. A run logged with BYPASSED decisions is an
UNPROTECTED run, never a run where every action "passed" a risk check.
"""
from typing import Protocol

from atal.core.action import ProposedAction
from atal.core.observation import GateDecision, GateOutcome, Observation


class ActionGate(Protocol):
    name: str

    def decide(self, action: ProposedAction, based_on: Observation) -> GateDecision: ...


class BypassGate:
    name = "none"

    def decide(self, action: ProposedAction, based_on: Observation) -> GateDecision:
        return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                            gate=self.name, outcome=GateOutcome.BYPASSED, assessed=False,
                            reason="no action-risk gate installed; passed through unassessed")
