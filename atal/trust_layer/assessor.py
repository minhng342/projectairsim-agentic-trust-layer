"""
Trust / provenance assessor interface.

The assessor sees ONLY what arrived on one vehicle's channel: the delivered
observation (raw payload included) and the previous one from the same channel.
It never sees the clean pre-attack payload, the attack hooks, or ground truth;
those exist only on the evaluation side (the recorder).

Placeholder for now: NoTrustAssessor returns NOT_ASSESSED. Building a real
assessor (freshness, consistency, physical plausibility, provenance checks)
is the research component that plugs in here.
"""
from typing import Protocol

from atal.core.observation import Observation, TrustAssessment, TrustStatus


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
