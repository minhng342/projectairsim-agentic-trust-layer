"""
Scripted mission agent for corridor flight: the Phase 0 stand-in for an LLM agent.

It has exactly the inputs and outputs a mission agent gets:
    in   its own observations (feed), its own mission (corridor), and the Remote ID
         traffic view (traffic_for_model on its receiver)
    out  maneuvers through its ActionPort, each tied to the observation it decided from

Every `decide_every_s` it observes, estimates its own progress along its corridor (by
projecting its OBSERVED position on the corridor, so an attack on its observations moves
this estimate too), reads the traffic picture, records the decision, and submits the
first trigger whose progress point it has passed. No conflict logic: that is the mission
track's job (Phase 1); this proves the plumbing end to end without a model.
"""
import asyncio
import math
import time
from dataclasses import dataclass, field

from atal.channels.radio import traffic_for_model
from atal.core.action import ActionType, ProposedAction
from atal.flight.path_follower import Polyline
from legacy.corridor.mission import Corridor


@dataclass(frozen=True)
class Trigger:
    """Fires at the first decision with progress >= at_progress_m and, if after_previous_s
    is set, at least that long after the previous maneuver (e.g. resume 3 s after a hold:
    a held drone's progress no longer advances, so a progress point alone could never fire)."""
    at_progress_m: float
    action_type: ActionType
    reason: str
    params: dict = field(default_factory=dict)
    after_previous_s: float | None = None


class ScriptedManeuverAgent:
    def __init__(self, vehicle_id: str, feed, port, receiver, corridor: Corridor,
                 triggers: list[Trigger] = (), recorder=None, decide_every_s: float = 1.0,
                 clock=time.monotonic):
        if feed.vehicle_id != vehicle_id or port.vehicle_id != vehicle_id \
                or receiver.owner_id != vehicle_id:
            raise ValueError("feed, port and receiver must belong to the agent's own vehicle")
        self.vehicle_id = vehicle_id
        self._feed, self._port, self._rx = feed, port, receiver
        self._path = Polyline(corridor.waypoints)
        self._pending = sorted(triggers, key=lambda t: t.at_progress_m)
        self._recorder, self._every, self._clock = recorder, decide_every_s, clock
        self.progress_m = 0.0
        self.decisions = 0
        self._last_fired: float | None = None
        self.feedback: list = []

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self._decide()
            try:
                async with asyncio.timeout(self._every):
                    await stop.wait()
            except asyncio.TimeoutError:
                pass

    async def _decide(self) -> None:
        obs = await self._feed.current(after_sequence=self._port.last_command_ended_sequence,
                                       timeout_s=2.0)
        if obs is None or not (math.isfinite(obs.north_m) and math.isfinite(obs.east_m)):
            self._event("agent_decision", decision="no usable observation")
            return
        s, xtrack = self._path.project(obs.north_m, obs.east_m, self.progress_m, 10.0)
        self.progress_m = max(self.progress_m, s)
        traffic = traffic_for_model(self._rx, self._clock())
        nearest = None
        for p in traffic["peers"]:
            pos = p["position"]
            if None in (pos["north_m"], pos["east_m"], pos["altitude_m"]):
                continue
            d = math.dist((obs.north_m, obs.east_m, obs.altitude_m),
                          (pos["north_m"], pos["east_m"], pos["altitude_m"]))
            if nearest is None or d < nearest[1]:
                nearest = (p["uas_id"], round(d, 2))
        due = self._pending[0] if self._pending and self._is_due(self._pending[0]) else None
        self.decisions += 1
        self._event("agent_decision", observation_id=obs.observation_id,
                    believed_progress_m=round(self.progress_m, 2), cross_track_m=round(xtrack, 2),
                    peers_heard=len(traffic["peers"]), lost_contact=len(traffic["lost_contact"]),
                    nearest_peer=nearest,
                    decision=due.action_type.value if due else "maintain (no action)")
        if due is None:
            return
        self._pending.pop(0)                  # one maneuver per decision, on this observation
        self._last_fired = self._clock()
        action = ProposedAction(vehicle_id=self.vehicle_id, action_type=due.action_type,
                                reason=due.reason, **due.params)
        self._event("agent_step", step=f"{due.action_type.value}@{due.at_progress_m:g}m",
                    request_id=action.request_id, observation_id=obs.observation_id)
        fb = await self._port.submit(action, based_on=obs.observation_id)
        self.feedback.append((due, fb))

    def _is_due(self, t: Trigger) -> bool:
        if self.progress_m < t.at_progress_m:
            return False
        if t.after_previous_s is None:
            return True
        return self._last_fired is not None and self._clock() - self._last_fired >= t.after_previous_s

    def _event(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)
