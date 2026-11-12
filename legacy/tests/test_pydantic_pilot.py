"""PydanticPilot and model_config with Pydantic AI's FunctionModel (no network, no sim)."""
import asyncio
import json
import os

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.exceptions import ModelNotFoundError, ModelServerError  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from atal.agents.llm.model_config import (  # noqa: E402
    ResilientModel,
    default_rpm,
    is_transient,
    make_model,
    model_specs,
    pick_model,
)
from atal.attacks.hooks import PositionOffset  # noqa: E402
from atal.channels.action_port import ActionPort  # noqa: E402
from atal.channels.observation_feed import ObservationFeed  # noqa: E402
from atal.flight.executor import ProjectAirSimExecutor  # noqa: E402
from legacy.llm_pilot.llm_pilot import PilotLimits  # noqa: E402
from legacy.llm_pilot.pydantic_pilot import PydanticPilot  # noqa: E402
from tests.channel_fakes import ListRecorder, SnapAdapter  # noqa: E402
from tests.test_action_port import Clock, FakeExecutor  # noqa: E402
from tests.test_executor import FAST, MultiAdapter  # noqa: E402

TURN = {"action_type": "rotate_to_heading", "heading_deg": 90.0, "reason": "face east"}


def results(messages):
    """Tool results in the latest request, as dicts."""
    return [p.content for p in messages[-1].parts if isinstance(p, ToolReturnPart)]


def call(name, args=None, cid=None):
    return ToolCallPart(name, args or {}, tool_call_id=cid) if cid else ToolCallPart(name, args or {})


class Planned:
    """A well-behaved model: observe, then act on the newest observation it was given,
    one action per turn, then comment."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.seen = []

    def __call__(self, messages, info):
        res = results(messages)
        self.seen.extend(res)
        if not res:
            return ModelResponse(parts=[call("get_observation")])
        r = res[-1]
        if isinstance(r, dict) and r.get("feedback", {}).get("status") == "succeeded":
            self.plan.pop(0)
        obs = (r.get("observation") or r.get("observation_after")) if isinstance(r, dict) else None
        if not self.plan or obs is None:
            return ModelResponse(parts=[TextPart("Done: every action succeeded.")])
        return ModelResponse(parts=[call("submit_action",
                                         {"observation_id": obs["observation_id"], **self.plan[0]})])


def build(fn, executor=None, limits=PilotLimits(), hooks=(), rpm=None, on_event=None):
    adapter, rec, clock = SnapAdapter(), ListRecorder(), Clock()
    feed = ObservationFeed(adapter, "Drone1", hooks=hooks, recorder=rec, clock=clock)
    ex = executor or FakeExecutor(clock=clock, takes_s=1.0)
    port = ActionPort("Drone1", ex, feed, recorder=rec)
    model = ResilientModel(FunctionModel(fn), rpm=rpm, backoff_s=0.0,
                           on_event=on_event or (lambda k, f: rec.record(k, source="model", **f)),
                           sleep=lambda s: asyncio.sleep(0))
    pilot = PydanticPilot("Drone1", feed, port, model, task="face east", recorder=rec,
                          limits=limits, model_label="function")
    return pilot, rec, ex, feed


def run(pilot):
    return asyncio.run(pilot.run())


# ------------------------------------------------------------ the loop
def test_full_loop_records_every_step_and_ends_with_the_models_comment():
    model = Planned([TURN, {"action_type": "change_altitude", "altitude_m": 8.0, "reason": "up"}])
    pilot, rec, ex, _ = build(model)
    pilot.capture_home()
    out = run(pilot)
    assert out.ended == "completed" and out.comment == "Done: every action succeeded."
    assert out.succeeded_actions == 2 and len(ex.calls) == 2
    assert out.model_turns == 4                       # observe, act, act, comment
    sess = rec.of("llm_session")[0]
    assert sess["harness"] == "pydantic-ai" and "Launch point" in sess["task"]
    calls = rec.of("tool_call")
    rid = calls[1]["result"]["feedback"]["request_id"]
    assert rec.of("proposal")[0]["request_id"] == rid
    assert rec.of("proposal")[0]["observation_id"] == calls[0]["result"]["observation"]["observation_id"]
    turns = rec.of("llm_turn")
    assert turns[0]["tool_calls"][0]["name"] == "get_observation"
    assert turns[0]["usage"]["input_tokens"] > 0
    assert rec.of("agent_comment")[0]["text"] == out.comment


def test_model_is_offered_only_the_two_tools_and_supported_actions():
    seen = {}

    def fn(messages, info):
        seen["tools"] = {t.name: t.parameters_json_schema for t in info.function_tools}
        return ModelResponse(parts=[TextPart("nothing to do")])
    pilot, _, _, _ = build(fn)
    run(pilot)
    assert set(seen["tools"]) == {"get_observation", "submit_action"}
    schema = seen["tools"]["submit_action"]
    assert "vehicle_id" not in schema["properties"] and "request_id" not in schema["properties"]
    enum = json.dumps(schema["properties"]["action_type"])
    for a in ("rotate_to_heading", "change_altitude", "move_to_position"):
        assert a in enum
    assert set(schema["required"]) == {"observation_id", "action_type", "reason"}


def test_two_actions_in_one_turn_run_in_order_and_the_second_is_superseded():
    state = {"n": 0}

    def fn(messages, info):
        state["n"] += 1
        res = results(messages)
        if state["n"] == 1:
            return ModelResponse(parts=[call("get_observation")])
        if state["n"] == 2:
            oid = res[0]["observation"]["observation_id"]
            return ModelResponse(parts=[call("submit_action", {"observation_id": oid, **TURN}, "a"),
                                        call("submit_action", {"observation_id": oid, **TURN}, "b")])
        return ModelResponse(parts=[TextPart("ok")])
    pilot, rec, ex, _ = build(fn)
    out = run(pilot)
    assert [f["code"] for f in out.submissions] == ["completed", "observation_superseded"]
    assert len(ex.calls) == 1


def test_schema_invalid_arguments_never_reach_the_port_but_are_recorded():
    state = {"n": 0}

    def fn(messages, info):
        state["n"] += 1
        res = results(messages)
        if state["n"] == 1:
            return ModelResponse(parts=[call("get_observation")])
        if state["n"] == 2:
            oid = res[0]["observation"]["observation_id"]
            return ModelResponse(parts=[call("submit_action", {
                "observation_id": oid, "action_type": "land", "reason": "x",
                "vehicle_id": "Drone2"}, "bad")])
        return ModelResponse(parts=[TextPart(f"gave up after: {str(res)[:40]}")])
    pilot, rec, ex, _ = build(fn)
    out = run(pilot)
    assert ex.calls == [] and not rec.of("proposal")
    rej = rec.of("tool_rejected")
    assert rej and rej[0]["call_id"] == "bad" and rej[0]["name"] == "submit_action"
    assert out.ended == "completed"


def test_port_level_rejections_reach_the_model_as_feedback():
    def fn(messages, info):
        res = results(messages)
        if not res:
            return ModelResponse(parts=[call("submit_action", {"observation_id": "made-up", **TURN})])
        return ModelResponse(parts=[TextPart(res[-1]["feedback"]["code"])])
    pilot, _, ex, _ = build(fn)
    out = run(pilot)
    assert out.comment == "unknown_observation" and ex.calls == []


def test_nothing_the_model_receives_contains_ground_truth():
    model = Planned([TURN])
    pilot, _, ex, _ = build(model)
    ex.result = {"reason": "position error 0.73 m after 2 corrections", "corrections": 2}
    run(pilot)
    text = json.dumps(model.seen)
    for leak in ("truth", "snapshot", "correction", "0.73", "validation", "clean_payload"):
        assert leak not in text


def test_attacked_observation_is_what_the_model_sees():
    model = Planned([])
    pilot, _, _, _ = build(model, hooks=[PositionOffset(north_m=5.0)])
    run(pilot)
    assert model.seen[0]["observation"]["position"]["north_m"] == 6.0


# ------------------------------------------------------------ limits and failures
def test_model_turn_limit_ends_the_run():
    pilot, rec, _, _ = build(lambda m, i: ModelResponse(parts=[call("get_observation")]),
                             limits=PilotLimits(max_model_turns=3))
    out = run(pilot)
    assert out.ended == "stopped: model turn limit (3)"
    assert rec.of("agent_end")[0]["ended"] == out.ended


def test_time_budget_ends_the_run():
    async def slow(messages, info):
        await asyncio.sleep(1.0)
        return ModelResponse(parts=[TextPart("late")])
    pilot, _, _, _ = build(slow, limits=PilotLimits(time_budget_s=0.1))
    assert run(pilot).ended.startswith("stopped: time budget")


def test_transient_model_error_is_retried_below_the_harness():
    fails = {"left": 2}
    inner = Planned([TURN])

    def fn(messages, info):
        if fails["left"]:
            fails["left"] -= 1
            raise ModelServerError("function", "503 overloaded")
        return inner(messages, info)
    pilot, rec, _, _ = build(fn)
    out = run(pilot)
    assert out.ended == "completed" and out.succeeded_actions == 1
    assert [e["error"] for e in rec.of("llm_error")] == ["ModelServerError"] * 2


def test_permanent_model_error_is_not_retried_and_ends_cleanly():
    calls = {"n": 0}

    def fn(messages, info):
        calls["n"] += 1
        raise ModelNotFoundError("function", "no such model")
    pilot, rec, ex, _ = build(fn)
    out = run(pilot)
    assert calls["n"] == 1 and out.ended.startswith("stopped: model unavailable")
    assert ex.calls == []


def test_rate_limit_waits_are_recorded():
    # 60/min = 1 s apart; build() gives the limiter an instant sleep, so this is deterministic
    # (6000/min flaked: on a slow run the calls were naturally > 10 ms apart)
    pilot, rec, _, _ = build(Planned([TURN]), rpm=60)
    out = run(pilot)
    waits = rec.of("llm_rate_wait")
    assert len(waits) == out.model_turns - 1 and all(w["waited_s"] > 0.5 for w in waits)


def test_cancellation_propagates_and_is_recorded():
    async def go():
        async def hang(messages, info):
            await asyncio.sleep(3600)
        pilot, rec, _, _ = build(hang)
        t = asyncio.ensure_future(pilot.run())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return rec
    rec = asyncio.run(go())
    assert rec.of("agent_end")[0]["ended"] == "interrupted: CancelledError"


def test_real_executor_end_to_end_with_sampler_running():
    adapter, rec = MultiAdapter(["Drone1"]), ListRecorder()
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    feed = ObservationFeed(adapter, "Drone1", recorder=rec)
    port = ActionPort("Drone1", ex, feed, recorder=rec)
    model = Planned([{"action_type": "move_to_position", "north_m": 9.0, "east_m": 8.0,
                      "altitude_m": 5.0, "reason": "inspect"},
                     {"action_type": "move_to_position", "north_m": -1.0, "east_m": 8.0,
                      "altitude_m": 5.0, "reason": "return"}])
    pilot = PydanticPilot("Drone1", feed, port, ResilientModel(FunctionModel(model)),
                          task="go and return", recorder=rec)

    async def go():
        pilot.capture_home()
        stop = asyncio.Event()
        s = asyncio.ensure_future(feed.sample(50.0, stop))
        out = await pilot.run()
        stop.set()
        await s
        return out
    out = asyncio.run(go())
    assert out.ended == "completed" and out.succeeded_actions == 2 and not rec.of("port_rejected")
    assert [e["result"]["final_truth"]["north_m"] for e in rec.of("command_result")] == \
        pytest.approx([9.0, -1.0])


# ------------------------------------------------------------ model configuration
def test_model_specs_and_default_pacing(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "ollama:qwen3.5:9b, google:gemini-3.5-flash-lite")
    monkeypatch.delenv("LLM_RPM", raising=False)
    monkeypatch.delenv("GEMINI_RPM", raising=False)
    assert model_specs() == ["ollama:qwen3.5:9b", "google:gemini-3.5-flash-lite"]
    assert model_specs("google:x") == ["google:x"]
    assert default_rpm("ollama:qwen3.5:9b") is None               # local: no pacing
    assert default_rpm("google:gemini-3.5-flash-lite") == pytest.approx(12.0)
    monkeypatch.setenv("LLM_RPM", "30")
    assert default_rpm("ollama:qwen3.5:9b") == 30.0


def test_ollama_model_builds_offline_with_a_default_local_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_RPM", raising=False)
    m = make_model("ollama:qwen3.5:9b")
    assert isinstance(m, ResilientModel) and m.limiter is None
    assert "qwen3.5:9b" in m.model_name
    assert os.environ["OLLAMA_BASE_URL"] == "http://localhost:11434/v1"


def test_transient_vs_permanent_errors():
    from pydantic_ai.exceptions import ModelHTTPError, ModelOverloadedError, ModelRateLimitError
    assert is_transient(ModelOverloadedError("m", "busy"))
    assert is_transient(ModelRateLimitError("m", "slow down"))
    assert is_transient(ModelHTTPError(503, "m"))
    assert not is_transient(ModelHTTPError(400, "m"))
    assert not is_transient(ModelNotFoundError("m", "gone"))
    assert not is_transient(ValueError("bug"))


def test_pick_model_retries_overload_then_falls_back_and_skips_permanent(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "a,b,c")
    script = {"a": [ModelServerError("a", "503"), ModelServerError("a", "503")],
              "b": [ModelNotFoundError("b", "404")], "c": [0.3]}

    async def ping_fn(model):
        r = script[model].pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    sleeps = []

    async def sleep(s):
        sleeps.append(s)
    model, spec, tried = asyncio.run(pick_model(make=lambda s: s, ping_fn=ping_fn, attempts=2,
                                                log=lambda *_: None, sleep=sleep))
    assert spec == "c" and [(t["model"], t["ok"]) for t in tried] == \
        [("a", False), ("a", False), ("b", False), ("c", True)]
    assert sleeps == [3.0]


def test_pick_model_reports_when_nothing_answers(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "a")

    async def ping_fn(model):
        raise ModelNotFoundError("a", "404")
    with pytest.raises(RuntimeError, match="no model answered"):
        asyncio.run(pick_model(make=lambda s: s, ping_fn=ping_fn, log=lambda *_: None))
