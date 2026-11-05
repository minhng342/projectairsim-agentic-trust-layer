"""
Per-vehicle action port: the ONLY way an agent can act.

    agent -> port.submit(action, based_on=observation_id)
          -> action, ownership, observation and freshness checks
                                            (port_rejected: never reaches gate or executor)
          -> action-risk gate               (gate_decision; BypassGate for now)
          -> executor                       (command_result: full, evaluation side)
          -> ActionFeedback to the agent    (feedback: coarse, no ground truth)

`based_on` is the observation_id the agent decided from (an Observation object is
accepted too; only its id is used, and its content must match the record). The
port resolves the id through ITS feed, so the gate checks the feed's protected
record, never the agent's copy, and a serialized agent can refer to observations.

Checks, in order (any failure -> REFUSED at stage "port", nothing sent):
    invalid_action          the action does not re-validate (e.g. edited after creation)
    wrong_vehicle           the action is not for THIS port's vehicle
    unknown_observation     the id was not delivered by this vehicle's feed (or is
                            forgotten); another vehicle's ids are unknown here
    altered_observation     an Observation object was passed whose content differs
                            from what the feed delivered
    busy                    another command from this port is still being handled
    duplicate_request       the request_id was submitted before (no replays)
    observation_expired     the observation is older than max_observation_age_s
    observation_superseded  the observation predates the end of this vehicle's
                            previous command (it can't reflect that command's outcome)
After the gate, the age is checked again (a slow review can expire it).

What the agent is told (ActionFeedback): its own request's ids, the stage, a status,
a small fixed vocabulary of codes with fixed messages for executor outcomes, whether
anything was sent and the fallback category. The executor's full CommandResult
(measured errors, ground-truth snapshots, exception text) goes only to the recorder.
Port and gate messages are passed through: they are computed from what the agent
sent and saw, not from ground truth.

If the caller is cancelled during execution, a command_cancelled event is recorded
(the executor hovers the drone) and the cancellation propagates.
"""
import asyncio

from pydantic import ValidationError

from gate.action_gate import BypassGate
from models.action import (ActionFeedback, CommandResult, CommandStatus, FeedbackStage,
                           ProposedAction)
from models.observation import Observation

DEFAULT_MAX_OBSERVATION_AGE_S = 10.0

# executor reason_code -> (agent code, agent message). Anything unlisted -> "failed".
_EXECUTOR_FEEDBACK = {
    "completed": ("completed", "Command completed: the flight controller reports the target "
                               "reached and is holding there."),
    "unsupported_action": ("unsupported_action", "This action type is not supported yet; "
                                                 "nothing was sent."),
    "preflight_not_ready": ("not_ready", "The flight controller refused the command because the "
                                         "vehicle is not ready; nothing was sent."),
    "task_timeout": ("timed_out", "The command did not complete in time."),
    "settle_timeout": ("timed_out", "The command did not complete in time."),
    # mission runner (executor/mission_runner.py): maneuvers are applied, not completed
    "applied": ("applied", "Maneuver applied: the drone is carrying it out while continuing "
                           "its mission."),
    "not_applicable": ("not_applicable", "That maneuver does not apply right now (for example "
                                         "resume while not holding); nothing changed."),
    "out_of_range": ("out_of_range", "The flight controller cannot fly that value; nothing "
                                     "changed."),
    "not_flying": ("not_flying", "The drone is not flying its corridor right now; nothing "
                                 "changed."),
    "mission_finished": ("mission_finished", "Your corridor is complete; no more maneuvers are "
                                             "accepted."),
    # sector executor (executor/sector_executor.py): search and rescue actions
    "sector_swept": ("sector_swept", "Sector swept: every lane was flown."),
    "sweep_stopped": ("sweep_stopped", "The sweep was stopped before it finished (victim "
                                       "detected or mission time limit); the drone is hovering."),
    "home_reached": ("home_reached", "The drone is over its launch point."),
    "position_reached": ("position_reached", "The drone is at the target position."),
    "unknown_sector": ("unknown_sector", "That sector is not part of the mission grid; nothing "
                                         "was sent."),
    "path_blocked": ("path_blocked", "The drone could not make progress along the path "
                                     "(blocked by an obstacle); it is hovering."),
    "pad_zone_timeout": ("pad_zone_timeout", "A teammate kept the airspace over the pads "
                                             "too long; the drone did not go in and is "
                                             "holding outside it."),
    "path_failed": ("failed", "The flight controller could not complete the path; the drone "
                              "was told to hover."),
}
_FAILED = ("failed", "The command failed in the flight controller.")


def agent_fallback(fallback: str | None) -> str | None:
    if fallback is None:
        return None
    if fallback == "hover":
        return "hover"
    if fallback.startswith("hover failed"):
        return "hover_failed"
    return "none"


_FALLBACK_TEXT = {"hover": " The vehicle was told to hover.",
                  "hover_failed": " A hover was commanded but could not be confirmed.",
                  "none": " No further command was sent to the vehicle."}


class ActionPort:
    def __init__(self, vehicle_id: str, executor, feed, gate=None, recorder=None,
                 clock=None, max_observation_age_s: float = DEFAULT_MAX_OBSERVATION_AGE_S):
        if feed.vehicle_id != vehicle_id:
            raise ValueError(f"feed for {feed.vehicle_id!r} cannot back the port for {vehicle_id!r}")
        self.vehicle_id = vehicle_id
        self.max_observation_age_s = max_observation_age_s
        self._executor = executor
        self._feed = feed
        self._gate = gate or BypassGate()
        self._recorder = recorder
        self._clock = clock or feed.clock      # same clock as the feed's receipt stamps
        self._seen_requests: set[str] = set()
        self._busy: str | None = None
        self._last_end: float | None = None
        self._last_end_seq: int | None = None

    @property
    def last_command_ended_host_s(self) -> float | None:
        """When this port's previous command left the executor (host clock), or None."""
        return self._last_end

    @property
    def last_command_ended_sequence(self) -> int | None:
        """The feed's sequence number when this port's previous command left the executor,
        or None. Decide from a later observation: feed.current(after_sequence=...)."""
        return self._last_end_seq

    @property
    def supported_actions(self) -> list[str]:
        return [a.value for a in getattr(self._executor, "SUPPORTED", ())]

    async def submit(self, action, based_on) -> ActionFeedback:
        obs_id = based_on.observation_id if isinstance(based_on, Observation) else based_on
        obs_id = obs_id if isinstance(obs_id, str) else None
        record = self._feed.resolve(obs_id)
        age = None if record is None else round(self._clock() - record.received_host_s, 3)

        # Re-validate BEFORE recording, so the proposal, decision, result and feedback all
        # carry the same request_id and the log holds the structured action. (A dict without
        # a request_id gets one here.) Re-validating also matters for ProposedAction objects:
        # pydantic does not validate on assignment, and the executor must get a copy the
        # agent can't touch.
        raw = action.model_dump(mode="json") if isinstance(action, ProposedAction) else action
        try:
            action = ProposedAction.model_validate(
                action.model_dump() if isinstance(action, ProposedAction) else action)
        except ValidationError as err:
            raw_rid = raw.get("request_id") if isinstance(raw, dict) else None
            raw_rid = raw_rid if isinstance(raw_rid, str) else None
            self._record("proposal", request_id=raw_rid, observation_id=obs_id,
                         observation_age_s=age, action=raw if isinstance(raw, dict) else repr(raw),
                         valid=False)
            msg = "; ".join(e["msg"] for e in err.errors())
            return self._reject(None, obs_id, "invalid_action", f"action is not valid: {msg}",
                                request_id=raw_rid)
        self._record("proposal", request_id=action.request_id, observation_id=obs_id,
                     observation_age_s=age, action=action.model_dump(mode="json"), valid=True)

        rid = action.request_id
        if action.vehicle_id != self.vehicle_id:
            return self._reject(action, obs_id, "wrong_vehicle",
                                f"port for {self.vehicle_id} cannot submit an action for "
                                f"{action.vehicle_id}")
        if record is None:
            return self._reject(action, obs_id, "unknown_observation",
                                f"observation {obs_id!r} was not delivered by {self.vehicle_id}'s "
                                f"feed (or is too old to be remembered)")
        if (isinstance(based_on, Observation)
                and based_on.model_dump_json() != record.model_dump_json()):
            return self._reject(action, obs_id, "altered_observation",
                                "the observation passed in differs from the one the feed "
                                "delivered under that id")
        if self._busy is not None:
            return self._reject(action, obs_id, "busy",
                                f"another command ({self._busy}) is still being handled for "
                                f"{self.vehicle_id}; wait for its result")
        if rid in self._seen_requests:
            return self._reject(action, obs_id, "duplicate_request",
                                f"request_id {rid} was already submitted")
        if age > self.max_observation_age_s:
            return self._reject(action, obs_id, "observation_expired",
                                f"observation is {age:.1f} s old (limit "
                                f"{self.max_observation_age_s:.1f} s); observe again")
        if self._last_end_seq is not None and record.sequence <= self._last_end_seq:
            return self._reject(action, obs_id, "observation_superseded",
                                "observation was received before your previous command ended; "
                                "observe again")

        self._seen_requests.add(rid)
        self._busy = rid
        try:
            return await self._gate_and_execute(action, record)
        finally:
            self._busy = None

    # ------------------------------------------------------------ internal
    async def _gate_and_execute(self, action: ProposedAction, record: Observation) -> ActionFeedback:
        rid, obs_id = action.request_id, record.observation_id
        try:
            decision = self._gate.decide(action, record)
        except Exception as err:   # a broken gate blocks; it never waves actions through
            self._record("gate_error", request_id=rid, observation_id=obs_id,
                         error=f"{type(err).__name__}: {err}")
            return self._feedback(action, obs_id, FeedbackStage.GATE, CommandStatus.REFUSED,
                                  "gate_error", "The action-risk gate failed; the action was "
                                  "blocked and nothing was sent.")
        self._record("gate_decision", decision=decision.model_dump(mode="json"))
        gate_info = {"gate_outcome": decision.outcome.value, "gate_assessed": decision.assessed}
        if not decision.allows_execution:
            return self._feedback(action, obs_id, FeedbackStage.GATE, CommandStatus.REFUSED,
                                  f"gate_{decision.outcome.value}",
                                  f"Blocked by the action-risk gate: {decision.reason}", **gate_info)

        age = self._clock() - record.received_host_s
        if age > self.max_observation_age_s:
            return self._reject(action, obs_id, "observation_expired",
                                f"observation expired while the action was being reviewed "
                                f"({age:.1f} s old, limit {self.max_observation_age_s:.1f} s); "
                                f"observe again", **gate_info)

        started = self._clock()
        try:
            result = await self._executor.execute(action)
        except BaseException as err:
            kind = "command_cancelled" if isinstance(err, asyncio.CancelledError) else "command_error"
            self._record(kind, request_id=rid, observation_id=obs_id,
                         elapsed_s=round(self._clock() - started, 3),
                         error=f"{type(err).__name__}: {err}")
            raise
        finally:
            self._last_end = self._clock()
            self._last_end_seq = self._feed.last_sequence
        self._record("command_result", observation_id=obs_id, decision_id=decision.decision_id,
                     observation_age_at_dispatch_s=round(started - record.received_host_s, 3),
                     result=summarize_result(result))
        code, message = _EXECUTOR_FEEDBACK.get(result.reason_code, _FAILED)
        if code == "unsupported_action" and self.supported_actions:
            message += f" Supported: {', '.join(self.supported_actions)}."
        fallback = agent_fallback(result.fallback_applied)
        if fallback is not None:
            message += _FALLBACK_TEXT[fallback]
        return self._feedback(action, obs_id, FeedbackStage.EXECUTOR, result.status, code, message,
                              sent_to_simulator=result.sent_to_simulator, fallback=fallback,
                              elapsed_s=result.elapsed_s, **gate_info)

    def _reject(self, action, obs_id, code, reason, request_id=None, **kw) -> ActionFeedback:
        rid = action.request_id if action is not None else request_id
        self._record("port_rejected", request_id=rid, observation_id=obs_id, code=code,
                     reason=reason)
        return self._feedback(action, obs_id, FeedbackStage.PORT, CommandStatus.REFUSED, code,
                              f"Rejected by the action port: {reason}", request_id=rid, **kw)

    def _feedback(self, action, obs_id, stage, status, code, message, request_id=None,
                  **kw) -> ActionFeedback:
        fb = ActionFeedback(
            request_id=action.request_id if action is not None else request_id,
            vehicle_id=self.vehicle_id,
            action_type=action.action_type if action is not None else None,
            observation_id=obs_id, stage=stage, status=status, code=code, message=message, **kw)
        self._record("feedback", feedback=fb.model_dump(mode="json"))
        return fb

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, **fields)


def summarize_result(r: CommandResult) -> dict:
    """CommandResult without the full snapshots, plus where the drone really ended up
    (ground truth from the executor's final snapshot; evaluation side)."""
    d = r.model_dump(mode="json", exclude={"start_snapshot", "final_snapshot"})
    for name in ("start_snapshot", "final_snapshot"):
        s = getattr(r, name)
        if s is not None:
            d[name.replace("snapshot", "truth")] = {
                "sim_time_ns": s.sim_time_ns,
                "north_m": round(s.position_ned_m.x, 3), "east_m": round(s.position_ned_m.y, 3),
                "altitude_m": round(s.altitude_local_m, 3), "heading_deg": round(s.heading_deg, 2),
                "validation_status": s.validation_status.value,
            }
    return d
