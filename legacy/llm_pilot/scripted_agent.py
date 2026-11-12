"""
Scripted mission agent: a fixed list of steps, no decision logic.

It exists to prove the plumbing with flights we already know work. It holds only
its own ObservationFeed and ActionPort, the same access a real (LLM) agent gets:
    - it learns its home and current state ONLY from its observations;
    - every action goes through its port, tied to the observation it was built from;
    - each step acts on an observation received AFTER its previous command ended
      (with a sampler running, the next sample; otherwise one taken now);
    - it is told only the port's ActionFeedback, never the executor's full result;
    - it never sees the adapter, a drone handle, ground truth or other drones.

A step builds a ProposedAction from (current observation, home observation).
A step may first wait at a PhaseBarrier shared with other agents. The mission
stops at the first step that does not SUCCEED, has no observation to act on,
cannot build a valid action, or finds its barrier broken.
"""
from dataclasses import dataclass
from typing import Callable

from atal.core.action import ActionFeedback, CommandStatus, ProposedAction
from atal.core.observation import Observation
from legacy.fleet import PhaseBarrier


@dataclass
class MissionStep:
    name: str
    build: Callable[[Observation, Observation], ProposedAction]
    sync: PhaseBarrier | None = None


class ScriptedMissionAgent:
    def __init__(self, vehicle_id: str, feed, port, steps: list[MissionStep],
                 recorder=None, on_step: Callable[[str, str], None] | None = None):
        if feed.vehicle_id != vehicle_id or port.vehicle_id != vehicle_id:
            raise ValueError("feed and port must belong to the agent's own vehicle")
        self.vehicle_id = vehicle_id
        self._feed = feed
        self._port = port
        self._steps = steps
        self._recorder = recorder
        self._on_step = on_step or (lambda vid, label: None)
        self.results: list[tuple[str, ActionFeedback]] = []
        self.ended: str = "not started"
        self.home: Observation | None = None

    def capture_home(self) -> Observation | None:
        """Take the home observation now (e.g. on the ground, BEFORE takeoff).

        run() reuses it; without this call, run() captures home when it starts,
        which after a separate takeoff step is the HOVER point, not the ground.
        """
        self.home = self._feed.observe()
        if self.home is not None:
            self._event("agent_home", observation_id=self.home.observation_id,
                        believed_home={"north_m": self.home.north_m, "east_m": self.home.east_m,
                                       "altitude_m": self.home.altitude_m})
        return self.home

    async def run(self) -> list[tuple[str, ActionFeedback]]:
        try:
            home = self.home or self.capture_home()
            if home is None:
                return self._end("aborted: no initial observation")
            for step in self._steps:
                if step.sync is not None:
                    self._on_step(self.vehicle_id, f"wait:{step.name}")
                    self._event("agent_wait", step=step.name, barrier=step.sync.name)
                    if not await step.sync.wait(self.vehicle_id):
                        return self._end(f"aborted before {step.name}: barrier {step.sync.name!r} "
                                         f"broken by {step.sync.broken_by}")
                self._on_step(self.vehicle_id, step.name)
                obs = await self._feed.current(
                    after_sequence=getattr(self._port, "last_command_ended_sequence", None))
                if obs is None:
                    return self._end(f"aborted at {step.name}: no observation")
                try:
                    action = step.build(obs, home)
                except Exception as err:
                    return self._end(f"aborted at {step.name}: could not build action "
                                     f"({type(err).__name__}: {err})")
                self._event("agent_step", step=step.name, request_id=action.request_id,
                            observation_id=obs.observation_id)
                result = await self._port.submit(action, based_on=obs.observation_id)
                self.results.append((step.name, result))
                if result.status != CommandStatus.SUCCEEDED:
                    self._on_step(self.vehicle_id, f"{step.name} FAILED")
                    return self._end(f"stopped at {step.name}: {result.status.value} ({result.code})")
            self._on_step(self.vehicle_id, "done")
            return self._end("completed")
        except BaseException as err:
            self.ended = f"interrupted: {type(err).__name__}"
            self._event("agent_end", ended=self.ended)
            raise
        finally:
            for step in self._steps:     # never leave peers waiting on us
                if step.sync is not None:
                    step.sync.depart(self.vehicle_id)

    def _end(self, why: str):
        self.ended = why
        self._event("agent_end", ended=why, steps_done=len(self.results))
        return self.results

    def _event(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)
