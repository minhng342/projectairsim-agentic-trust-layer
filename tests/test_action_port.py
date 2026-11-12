"""ActionPort: observation references, freshness, ownership, gate handling,
agent-facing feedback (no ground truth) and recording (no simulator)."""
import asyncio
import json

import pytest

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.core.action import (
    ActionFeedback,
    ActionType,
    CommandResult,
    CommandStatus,
    FeedbackStage,
    ProposedAction,
)
from atal.core.observation import GateDecision, GateOutcome
from tests.channel_fakes import ListRecorder, SnapAdapter, snapshot


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeExecutor:
    SUPPORTED = (ActionType.ROTATE_TO_HEADING, ActionType.CHANGE_ALTITUDE,
                 ActionType.MOVE_TO_POSITION)

    def __init__(self, block=False, result=None, clock=None, takes_s=0.0):
        self.calls = []
        self.block = block
        self.started = asyncio.Event() if block else None
        self.result = result or {}
        self.clock, self.takes_s = clock, takes_s

    async def execute(self, action):
        self.calls.append(action)
        if self.block:
            self.started.set()
            await asyncio.sleep(3600)
        if self.clock is not None:
            self.clock.t += self.takes_s
        kw = {"status": CommandStatus.SUCCEEDED, "reason": "settled within tolerance",
              "reason_code": "completed", "sent_to_simulator": True} | self.result
        return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                             action_type=action.action_type, **kw)


def make(gate=None, block=False, result=None, max_age=10.0, takes_s=0.0):
    adapter, rec, clock = SnapAdapter(), ListRecorder(), Clock()
    feed1 = ObservationFeed(adapter, "Drone1", recorder=rec, clock=clock)
    feed2 = ObservationFeed(adapter, "Drone2", recorder=rec, clock=clock)
    ex = FakeExecutor(block, result, clock, takes_s)
    port = ActionPort("Drone1", ex, feed1, gate=gate, recorder=rec, max_observation_age_s=max_age)
    return port, feed1, feed2, ex, rec, clock


def turn(vid="Drone1", **kw):
    return ProposedAction(vehicle_id=vid, action_type=ActionType.ROTATE_TO_HEADING,
                          heading_deg=90.0, reason="test", **kw)


def submit(port, action, based_on):
    return asyncio.run(port.submit(action, based_on=based_on))


# ------------------------------------------------------------ happy path + linkage
def test_bypassed_action_executes_and_every_step_is_linked():
    port, feed, _, ex, rec, _ = make()
    obs = feed.observe()
    a = turn()
    fb = submit(port, a, obs.observation_id)
    assert isinstance(fb, ActionFeedback)
    assert fb.status == CommandStatus.SUCCEEDED and fb.succeeded and fb.code == "completed"
    assert fb.stage == FeedbackStage.EXECUTOR and fb.sent_to_simulator
    assert fb.gate_outcome == "bypassed" and fb.gate_assessed is False
    assert [c.request_id for c in ex.calls] == [a.request_id]
    p, d, res = rec.of("proposal")[0], rec.of("gate_decision")[0], rec.of("command_result")[0]
    assert p["request_id"] == a.request_id and p["observation_id"] == obs.observation_id
    assert d["decision"]["outcome"] == "bypassed" and d["decision"]["assessed"] is False
    assert d["decision"]["observation_id"] == obs.observation_id
    assert res["result"]["request_id"] == a.request_id
    assert res["observation_id"] == obs.observation_id
    assert res["decision_id"] == d["decision"]["decision_id"]
    assert rec.of("feedback")[0]["feedback"]["request_id"] == a.request_id


def test_unmodified_observation_object_is_accepted_like_its_id():
    port, feed, _, ex, _, _ = make()
    fb = submit(port, turn(), feed.observe())
    assert fb.succeeded and len(ex.calls) == 1


# ------------------------------------------------------------ ground truth never reaches the agent
def test_feedback_carries_no_ground_truth_or_measured_detail():
    port, feed, _, _, rec, _ = make(result={
        "status": CommandStatus.TIMED_OUT, "reason_code": "settle_timeout",
        "reason": "did not settle within 15.0 s after 2 corrections (last: position error 2.31 m)",
        "fallback_applied": "hover", "corrections": 2,
        "start_snapshot": snapshot(n=1.234), "final_snapshot": snapshot(n=7.654)})
    fb = submit(port, turn(), feed.observe().observation_id)
    assert fb.status == CommandStatus.TIMED_OUT and fb.code == "timed_out"
    assert fb.fallback == "hover" and "hover" in fb.message
    text = fb.model_dump_json()
    for leak in ("2.31", "7.654", "1.234", "snapshot", "truth", "correction", "position error"):
        assert leak not in text
    full = rec.of("command_result")[0]["result"]           # ...all of it kept for evaluation
    assert full["corrections"] == 2 and full["final_truth"]["north_m"] == 7.654
    assert "2.31" in full["reason"]


@pytest.mark.parametrize("code,fallback,agent_code,agent_fallback", [
    ("completed", None, "completed", None),
    ("preflight_not_ready", None, "not_ready", None),
    ("unsupported_action", None, "unsupported_action", None),
    ("task_timeout", "hover", "timed_out", "hover"),
    ("telemetry_lost", "hover failed: RuntimeError: link down", "failed", "hover_failed"),
    ("became_grounded", "none (vehicle grounded)", "failed", "none"),
    ("internal_error", "hover", "failed", "hover"),
    ("something_new", None, "failed", None),
])
def test_executor_outcomes_map_to_a_small_fixed_vocabulary(code, fallback, agent_code,
                                                            agent_fallback):
    secret = "telemetry stale: sim progress 812 ms; ValueError: x=3.14159"
    port, feed, _, _, _, _ = make(result={"reason_code": code, "reason": secret,
                                          "fallback_applied": fallback,
                                          "status": CommandStatus.FAILED})
    fb = submit(port, turn(), feed.observe().observation_id)
    assert fb.code == agent_code and fb.fallback == agent_fallback
    assert "812" not in fb.message and "3.14159" not in fb.message and "stale" not in fb.message
    if agent_code == "unsupported_action":
        assert "rotate_to_heading" in fb.message


def test_broken_gate_fails_closed_without_exception_text():
    class Broken:
        name = "broken"

        def decide(self, action, based_on):
            raise RuntimeError("secret internals at N=12.5")
    port, feed, _, ex, rec, _ = make(gate=Broken())
    fb = submit(port, turn(), feed.observe().observation_id)
    assert fb.status == CommandStatus.REFUSED and fb.stage == FeedbackStage.GATE
    assert fb.code == "gate_error" and "12.5" not in fb.message
    assert ex.calls == [] and "12.5" in rec.of("gate_error")[0]["error"]


def test_gate_rejection_blocks_execution_and_its_reason_is_passed_on():
    class Deny:
        name = "deny"

        def decide(self, action, based_on):
            return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                                gate="deny", outcome=GateOutcome.REJECTED, assessed=True,
                                reason="too risky")
    port, feed, _, ex, rec, _ = make(gate=Deny())
    fb = submit(port, turn(), feed.observe().observation_id)
    assert fb.status == CommandStatus.REFUSED and fb.stage == FeedbackStage.GATE
    assert fb.code == "gate_rejected" and "too risky" in fb.message and ex.calls == []
    assert rec.of("gate_decision")[0]["decision"]["outcome"] == "rejected"


# ------------------------------------------------------------ observation references
def test_gate_sees_the_feeds_record_not_the_agents_edited_copy():
    seen = []

    class Spy:
        name = "spy"

        def decide(self, action, based_on):
            seen.append(based_on)
            return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                                gate="spy", outcome=GateOutcome.APPROVED, assessed=True)
    port, feed, _, _, _, _ = make(gate=Spy())
    obs = feed.observe()
    obs.payload["position_ned_m"]["x"] = 999.0          # agent scribbles on its copy...
    fb = submit(port, turn(), obs.observation_id)       # ...then refers to it by id
    assert fb.succeeded and seen[0].north_m == 1.0


@pytest.mark.parametrize("which", ["other_vehicle", "made_up", "none", "dict", "int"])
def test_unknown_observation_references_are_rejected(which):
    port, feed, feed2, ex, rec, _ = make()
    feed.observe()
    based_on = {"other_vehicle": feed2.observe().observation_id, "made_up": "f" * 32,
                "none": None, "dict": {"observation_id": "x"}, "int": 7}[which]
    fb = submit(port, turn(), based_on)
    assert fb.status == CommandStatus.REFUSED and fb.code == "unknown_observation"
    assert fb.stage == FeedbackStage.PORT and ex.calls == [] and not rec.of("gate_decision")


def test_other_vehicles_observation_object_is_unknown_here():
    port, _, feed2, ex, _, _ = make()
    fb = submit(port, turn(), feed2.observe())
    assert fb.code == "unknown_observation" and ex.calls == []


def test_altered_observation_object_is_rejected():
    port, feed, _, ex, _, _ = make()
    obs = feed.observe()
    obs.payload["altitude_local_m"] = 50.0
    fb = submit(port, turn(), obs)
    assert fb.code == "altered_observation" and ex.calls == []


def test_forgotten_observation_is_unknown():
    adapter, clock = SnapAdapter(), Clock()
    feed = ObservationFeed(adapter, "Drone1", clock=clock, remember=2)
    port = ActionPort("Drone1", FakeExecutor(), feed)
    first = feed.observe()
    feed.observe(), feed.observe()
    assert submit(port, turn(), first.observation_id).code == "unknown_observation"


def test_observation_object_itself_is_frozen():
    _, feed, _, _, _, _ = make()
    obs = feed.observe()
    with pytest.raises(Exception):
        obs.vehicle_id = "Drone2"


# ------------------------------------------------------------ freshness
def test_expired_observation_is_rejected():
    port, feed, _, ex, _, clock = make(max_age=5.0)
    obs = feed.observe()
    clock.t += 5.5
    fb = submit(port, turn(), obs.observation_id)
    assert fb.code == "observation_expired" and ex.calls == []


def test_observation_that_expires_during_gate_review_is_not_executed():
    class SlowReview:
        name = "slow"

        def __init__(self, clock):
            self.clock = clock

        def decide(self, action, based_on):
            self.clock.t += 6.0                      # e.g. a human took 6 s to approve
            return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                                gate="slow", outcome=GateOutcome.APPROVED, assessed=True)
    port, feed, _, ex, rec, clock = make(max_age=5.0)
    port._gate = SlowReview(clock)
    fb = submit(port, turn(), feed.observe().observation_id)
    assert fb.code == "observation_expired" and "reviewed" in fb.message
    assert fb.gate_outcome == "approved" and ex.calls == []


def test_observation_from_before_the_previous_command_ended_is_superseded():
    port, feed, _, ex, _, clock = make(takes_s=3.0)
    before = feed.observe()
    stale = feed.observe()                                  # also taken before the command
    assert submit(port, turn(), before.observation_id).succeeded
    assert port.last_command_ended_host_s == pytest.approx(103.0)
    assert port.last_command_ended_sequence == 2
    fb = submit(port, turn(), stale.observation_id)
    assert fb.code == "observation_superseded" and len(ex.calls) == 1
    # same clock reading as the command's end (coarse Windows clock): still fresh by sequence
    clock.t = port.last_command_ended_host_s
    assert submit(port, turn(), feed.observe().observation_id).succeeded


def test_port_rejection_does_not_move_the_command_boundary():
    port, feed, _, _, _, _ = make()
    obs = feed.observe()
    submit(port, turn("Drone2"), obs.observation_id)        # rejected at the port
    assert port.last_command_ended_host_s is None
    assert submit(port, turn(), obs.observation_id).succeeded


# ------------------------------------------------------------ action checks
def test_action_for_another_vehicle_is_rejected_before_gate_and_executor():
    port, feed, _, ex, rec, _ = make()
    fb = submit(port, turn("Drone2"), feed.observe().observation_id)
    assert fb.status == CommandStatus.REFUSED and not fb.sent_to_simulator
    assert fb.code == "wrong_vehicle" and "cannot submit an action for Drone2" in fb.message
    assert ex.calls == [] and not rec.of("gate_decision")
    assert rec.of("port_rejected")[0]["code"] == "wrong_vehicle"


def test_action_edited_after_validation_is_rejected():
    port, feed, _, ex, _, _ = make()
    a = turn()
    a.heading_deg = float("nan")                  # pydantic does not re-validate on assignment
    fb = submit(port, a, feed.observe().observation_id)
    assert fb.code == "invalid_action" and ex.calls == []


def test_action_as_plain_dict_is_validated():
    port, feed, _, ex, _, _ = make()
    obs = feed.observe()
    bad = submit(port, {"vehicle_id": "Drone1", "action_type": "rotate_to_heading",
                        "heading_deg": 90.0, "altitude_m": 5.0, "reason": "x"}, obs.observation_id)
    assert bad.code == "invalid_action" and bad.request_id is None and bad.action_type is None
    ok = submit(port, {"vehicle_id": "Drone1", "action_type": "rotate_to_heading",
                       "heading_deg": 90.0, "reason": "x"}, obs.observation_id)
    assert ok.succeeded and len(ex.calls) == 1


def test_executor_gets_a_copy_the_agent_cannot_change_mid_command():
    async def go():
        port, feed, _, ex, _, _ = make(block=True)
        a = turn()
        t = asyncio.ensure_future(port.submit(a, based_on=feed.observe().observation_id))
        await ex.started.wait()
        a.heading_deg = 270.0
        sent = ex.calls[0]
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return sent
    assert asyncio.run(go()).heading_deg == 90.0


def test_replayed_request_id_is_rejected():
    port, feed, _, ex, _, clock = make()
    a = turn()
    first = submit(port, a, feed.observe().observation_id)
    clock.t += 0.1
    again = submit(port, a, feed.observe().observation_id)
    assert first.succeeded
    assert again.code == "duplicate_request" and len(ex.calls) == 1


def test_rejected_request_id_is_not_burned():
    port, feed, feed2, _, _, _ = make()
    a = turn()
    submit(port, a, feed2.observe().observation_id)        # rejected: wrong observation
    assert submit(port, a, feed.observe().observation_id).succeeded


def test_concurrent_submission_is_rejected_while_one_is_in_flight():
    """LLM tool calls can arrive in parallel; a second command must not queue
    behind the first on an observation that the first will make stale."""
    async def go():
        port, feed, _, ex, rec, clock = make(block=True)
        obs = feed.observe()
        t = asyncio.ensure_future(port.submit(turn(), based_on=obs.observation_id))
        await ex.started.wait()
        second = await port.submit(turn(), based_on=obs.observation_id)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        ex.block = False
        clock.t += 0.1
        third = await port.submit(turn(), based_on=feed.observe().observation_id)
        return second, ex, rec, third
    second, ex, rec, third = asyncio.run(go())
    assert second.code == "busy" and len(ex.calls) == 2      # 1st (cancelled) + 3rd
    assert rec.of("command_cancelled")


def test_cancellation_during_execution_is_recorded_and_propagates():
    async def go():
        port, feed, _, ex, rec, _ = make(block=True)
        t = asyncio.ensure_future(port.submit(turn(), based_on=feed.observe().observation_id))
        await ex.started.wait()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return port, rec
    port, rec = asyncio.run(go())
    assert rec.of("command_cancelled") and not rec.of("command_result")
    assert port.last_command_ended_host_s is not None


def test_every_outcome_is_recorded_as_json_safe_feedback():
    port, feed, _, _, rec, _ = make()
    submit(port, turn("Drone2"), feed.observe().observation_id)
    submit(port, turn(), feed.observe().observation_id)
    fbs = rec.of("feedback")
    assert [f["feedback"]["code"] for f in fbs] == ["wrong_vehicle", "completed"]
    json.dumps(fbs)


def test_port_and_feed_must_belong_to_the_same_vehicle():
    adapter = SnapAdapter()
    with pytest.raises(ValueError):
        ActionPort("Drone1", FakeExecutor(), ObservationFeed(adapter, "Drone2"))


# ------------------------------------------------------------ evaluation linkage for dict actions
def test_dict_action_is_recorded_structured_with_one_request_id_throughout():
    port, feed, _, _, rec, _ = make()
    fb = submit(port, {"request_id": "llm-request-1", "vehicle_id": "Drone1",
                       "action_type": "rotate_to_heading", "heading_deg": 90.0, "reason": "x"},
                feed.observe().observation_id)
    assert fb.succeeded and fb.request_id == "llm-request-1"
    p = rec.of("proposal")[0]
    assert p["request_id"] == "llm-request-1" and p["valid"] is True
    assert isinstance(p["action"], dict) and p["action"]["heading_deg"] == 90.0
    assert rec.of("gate_decision")[0]["decision"]["request_id"] == "llm-request-1"
    assert rec.of("command_result")[0]["result"]["request_id"] == "llm-request-1"


def test_dict_action_without_request_id_gets_one_shared_by_every_record():
    port, feed, _, _, rec, _ = make()
    fb = submit(port, {"vehicle_id": "Drone1", "action_type": "rotate_to_heading",
                       "heading_deg": 90.0, "reason": "x"}, feed.observe().observation_id)
    rid = fb.request_id
    assert rid and rec.of("proposal")[0]["request_id"] == rid
    assert rec.of("command_result")[0]["result"]["request_id"] == rid


def test_invalid_dict_action_is_recorded_as_given_and_marked_invalid():
    port, feed, _, _, rec, _ = make()
    bad = {"request_id": "llm-request-2", "vehicle_id": "Drone1", "action_type": "fly_away",
           "reason": "x"}
    fb = submit(port, bad, feed.observe().observation_id)
    p = rec.of("proposal")[0]
    assert fb.code == "invalid_action" and fb.request_id == "llm-request-2"
    assert p["valid"] is False and p["action"] == bad and p["request_id"] == "llm-request-2"
    assert rec.of("port_rejected")[0]["request_id"] == "llm-request-2"


def test_summarizer_links_a_dict_action_through_to_its_result(tmp_path):
    from atal.evaluation.summarize_run import summarize
    from atal.runtime.recorder import EvaluationRecorder
    adapter, clock = SnapAdapter(), Clock()
    rec = EvaluationRecorder(run_config={}, run_dir=tmp_path)
    feed = ObservationFeed(adapter, "Drone1", recorder=rec, clock=clock)
    port = ActionPort("Drone1", FakeExecutor(), feed, recorder=rec)
    submit(port, {"request_id": "llm-request-3", "vehicle_id": "Drone1",
                  "action_type": "rotate_to_heading", "heading_deg": 90.0, "reason": "x"},
           feed.observe().observation_id)
    rec.close("pass")
    text = summarize(rec.path)
    assert "rotate_to_heading obs#1" in text and "succeeded" in text and "told: completed" in text
