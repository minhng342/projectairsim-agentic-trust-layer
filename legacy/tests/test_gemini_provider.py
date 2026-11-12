"""GeminiProvider's mapping logic against a stand-in for the google-genai SDK
(the real SDK and network are exercised by check_llm.py --harness native, live)."""
import asyncio
from types import SimpleNamespace

import pytest

from legacy.llm_pilot.llm_pilot import tool_specs
from legacy.llm_pilot.providers.base import ToolResult
from legacy.llm_pilot.providers.gemini import GeminiEmptyResponse, GeminiProvider


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):         # unset optional fields read as None, like the SDK
        return None


def fake_types(json_schema_supported=True):
    class FunctionDeclaration(_Obj):
        def __init__(self, **kw):
            if "parameters_json_schema" in kw and not json_schema_supported:
                raise TypeError("unexpected keyword parameters_json_schema")
            super().__init__(**kw)
    names = ["Tool", "AutomaticFunctionCallingConfig", "GenerateContentConfig", "Content", "Part",
             "FunctionResponse"]
    ns = {n: type(n, (_Obj,), {}) for n in names}
    return SimpleNamespace(FunctionDeclaration=FunctionDeclaration, **ns)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._gen))

    async def _gen(self, model, contents, config=None):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def response(*parts, finish="STOP", usage=None):
    content = _Obj(role="model", parts=list(parts))
    return _Obj(candidates=[_Obj(content=content, finish_reason=_Obj(value=finish))],
                usage_metadata=_Obj(**(usage or {})))


def fcall(name, args, id=None):
    return _Obj(function_call=_Obj(name=name, args=args, id=id))


def text(t, thought=False):
    return _Obj(text=t, thought=thought)


def session(responses, json_schema_supported=True):
    t = fake_types(json_schema_supported)
    client = FakeClient(responses)
    p = GeminiProvider(model="gemini-test", client=client, types_module=t)
    return p.start(system="SYS", tools=tool_specs(["rotate_to_heading"])), client


def test_config_disables_automatic_calls_and_declares_both_tools():
    s, client = session([response(text("hi"))])
    asyncio.run(s.send_user("go"))
    cfg = client.requests[0]["config"]
    assert cfg.automatic_function_calling.disable is True
    assert cfg.system_instruction == "SYS" and cfg.temperature == 0.2
    get_obs, submit = cfg.tools[0].function_declarations
    assert get_obs.name == "get_observation" and get_obs.parameters_json_schema is None
    assert submit.parameters_json_schema["required"] == ["observation_id", "action_type", "reason"]


def test_older_sdk_falls_back_to_schema_dialect():
    s, client = session([response(text("hi"))], json_schema_supported=False)
    asyncio.run(s.send_user("go"))
    submit = client.requests[0]["config"].tools[0].function_declarations[1]
    assert submit.parameters["type"] == "OBJECT"
    assert submit.parameters["properties"]["heading_deg"]["type"] == "NUMBER"


def test_tool_call_round_trip_keeps_native_history_and_ids():
    model_turn = response(fcall("get_observation", {}, id="abc"),
                          usage={"prompt_token_count": 10, "candidates_token_count": 3})
    s, client = session([model_turn, response(text("done"))])
    t1 = asyncio.run(s.send_user("go"))
    assert [(c.call_id, c.name, c.args) for c in t1.tool_calls] == [("abc", "get_observation", {})]
    assert t1.usage == {"prompt_token_count": 10, "candidates_token_count": 3}
    t2 = asyncio.run(s.send_tool_results([ToolResult("abc", "get_observation", {"ok": True})]))
    assert t2.text == "done" and not t2.tool_calls
    sent = client.requests[1]["contents"]
    assert sent[1] is model_turn.candidates[0].content          # model turn resent verbatim
    fr = sent[2].parts[0].function_response
    assert (fr.id, fr.name, fr.response) == ("abc", "get_observation", {"ok": True})


def test_call_without_id_gets_a_record_id_and_is_answered_by_name():
    s, client = session([response(fcall("get_observation", {})), response(text("ok"))])
    t1 = asyncio.run(s.send_user("go"))
    cid = t1.tool_calls[0].call_id
    assert cid.startswith("gemini-call-")
    asyncio.run(s.send_tool_results([ToolResult(cid, "get_observation", {})]))
    assert client.requests[1]["contents"][2].parts[0].function_response.id is None


def test_thoughts_are_not_returned_as_text_and_multiple_calls_are_kept_in_order():
    s, _ = session([response(text("planning...", thought=True), text("Let me look."),
                             fcall("get_observation", {}, "1"),
                             fcall("submit_action", {"action_type": "x"}, "2"))])
    t = asyncio.run(s.send_user("go"))
    assert t.text == "Let me look." and [c.call_id for c in t.tool_calls] == ["1", "2"]
    assert t.finish_reason == "STOP"


def test_failed_call_does_not_grow_history_so_a_retry_resends_the_same_turn():
    s, client = session([ConnectionError("503"), response(text("ok"))])
    with pytest.raises(ConnectionError):
        asyncio.run(s.send_user("go"))
    asyncio.run(s.send_user("go"))
    assert len(client.requests[1]["contents"]) == 1


@pytest.mark.parametrize("resp", [
    _Obj(candidates=[], prompt_feedback="blocked"),
    _Obj(candidates=[_Obj(content=None, finish_reason="SAFETY")]),
    _Obj(candidates=[_Obj(content=_Obj(parts=[]), finish_reason="MALFORMED_FUNCTION_CALL")]),
])
def test_empty_or_blocked_response_raises_for_the_pilot_to_retry(resp):
    s, client = session([resp])
    with pytest.raises(GeminiEmptyResponse):
        asyncio.run(s.send_user("go"))
    assert s._history == []


def test_missing_key_is_a_clear_error(monkeypatch):
    from legacy.llm_pilot.providers.gemini import api_key_from_env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        api_key_from_env()


def test_env_file_loader_sets_only_unset_keys(tmp_path, monkeypatch):
    from atal.env import load_env_file
    f = tmp_path / ".env"
    f.write_text('# comment\nGEMINI_API_KEY="abc"\nexport GEMINI_MODEL=m1\nALREADY=new\n\nbad line\n')
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("ALREADY", "old")
    assert load_env_file(f) == ["GEMINI_API_KEY", "GEMINI_MODEL"]
    import os
    assert os.environ["GEMINI_API_KEY"] == "abc" and os.environ["ALREADY"] == "old"
    assert load_env_file(tmp_path / "missing") == []


# ------------------------------------------------------------ preflight: pick an available model
class FakeAPIError(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{code} {msg}")
        self.code = code


class PingOnly:
    def __init__(self, model, script):
        self.model, self.script = model, script

    async def ping(self):
        r = self.script[self.model].pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def pick(models, script, monkeypatch, attempts=3):
    from legacy.llm_pilot.providers import pick_gemini
    monkeypatch.setenv("GEMINI_MODEL", models)
    sleeps = []

    async def sleep(s):
        sleeps.append(s)
    res = asyncio.run(pick_gemini(make=lambda m: PingOnly(m, script), attempts=attempts,
                                  log=lambda *_: None, sleep=sleep))
    return res, sleeps


def test_overloaded_model_is_retried_then_the_fallback_is_used(monkeypatch):
    busy = FakeAPIError(503, "UNAVAILABLE high demand")
    (provider, tried), sleeps = pick("gemini-a, gemini-b", {
        "gemini-a": [busy, busy, busy], "gemini-b": [0.4]}, monkeypatch)
    assert provider.model == "gemini-b"
    assert [(t["model"], t["ok"]) for t in tried] == [("gemini-a", False)] * 3 + [("gemini-b", True)]
    assert sleeps == [3.0, 6.0]                       # backoff between attempts on one model


def test_transient_error_then_success_on_the_same_model(monkeypatch):
    (provider, tried), _ = pick("gemini-a,gemini-b", {
        "gemini-a": [FakeAPIError(429, "rate"), 0.3], "gemini-b": []}, monkeypatch)
    assert provider.model == "gemini-a" and len(tried) == 2


def test_permanent_error_skips_to_the_next_model_without_retrying(monkeypatch):
    (provider, tried), sleeps = pick("bad-name,gemini-b", {
        "bad-name": [FakeAPIError(404, "not found")], "gemini-b": [0.2]}, monkeypatch)
    assert provider.model == "gemini-b" and len(tried) == 2 and sleeps == []


def test_no_model_available_raises_with_every_attempt_listed(monkeypatch):
    busy = FakeAPIError(503, "UNAVAILABLE")
    with pytest.raises(RuntimeError, match="no Gemini model answered.*gemini-a#2"):
        pick("gemini-a", {"gemini-a": [busy, busy]}, monkeypatch, attempts=2)


def test_ping_uses_the_real_provider_request_path():
    client = FakeClient([response(text("OK"))])
    p = GeminiProvider(model="gemini-test", client=client, types_module=fake_types())
    assert asyncio.run(p.ping()) >= 0
    assert client.requests[0]["model"] == "gemini-test"
    client2 = FakeClient([_Obj(candidates=[])])
    with pytest.raises(GeminiEmptyResponse):
        asyncio.run(GeminiProvider(model="m", client=client2, types_module=fake_types()).ping())
