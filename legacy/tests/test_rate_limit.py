"""Client-side model rate limiting: even spacing, shared quota, kept out of the
model-call timeout (no network)."""
import asyncio

import pytest

from legacy.llm_pilot.llm_pilot import LLMPilot, PilotLimits
from legacy.llm_pilot.providers import rpm_for
from legacy.llm_pilot.providers.base import ModelTurn, RateLimiter
from legacy.llm_pilot.providers.gemini import GeminiProvider
from legacy.tests.test_gemini_provider import FakeClient, fake_types, response, text
from legacy.tests.test_llm_pilot import TURN, FakeProvider, PlannedPilot, build


class FakeTime:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    async def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.t += s


def test_calls_are_evenly_spaced_never_burst():
    ft = FakeTime()
    lim = RateLimiter(12, clock=ft.clock, sleep=ft.sleep)          # one call per 5 s

    async def go():
        return [await lim.acquire() for _ in range(4)]
    waits = asyncio.run(go())
    assert waits == [0.0, 5.0, 5.0, 5.0] and ft.t == 15.0


def test_idle_time_is_not_banked_as_burst_credit():
    ft = FakeTime()
    lim = RateLimiter(12, clock=ft.clock, sleep=ft.sleep)

    async def go():
        await lim.acquire()
        ft.t += 60.0                                   # a minute of silence
        return [await lim.acquire() for _ in range(3)]
    assert asyncio.run(go()) == [0.0, 5.0, 5.0]


def test_concurrent_users_share_one_quota_in_order():
    lim = RateLimiter(600)                             # 0.1 s apart, real clock
    order = []

    async def user(name):
        await lim.acquire()
        order.append(name)

    async def go():
        t0 = asyncio.get_running_loop().time()
        await asyncio.gather(*(user(i) for i in range(4)))
        return asyncio.get_running_loop().time() - t0
    elapsed = asyncio.run(go())
    assert order == [0, 1, 2, 3] and elapsed >= 0.28


def test_bad_rate_rejected():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_default_rates_leave_headroom_and_env_overrides(monkeypatch):
    monkeypatch.delenv("GEMINI_RPM", raising=False)
    assert rpm_for("gemini-3.5-flash-lite") == pytest.approx(12.0)
    assert rpm_for("gemini-3.6-flash") == pytest.approx(4.0)
    assert rpm_for("gemini-unknown") == pytest.approx(4.0)
    monkeypatch.setenv("GEMINI_RPM", "60")
    assert rpm_for("gemini-3.6-flash") == 60.0


def test_gemini_session_reserved_slot_is_not_charged_twice():
    t = fake_types()
    client = FakeClient([response(text("a")), response(text("b"))])
    p = GeminiProvider(model="m", client=client, types_module=t, rpm=6000)
    calls = []
    orig = p.limiter.acquire

    async def counting():
        calls.append(1)
        return await orig()
    p.limiter.acquire = counting
    s = p.start(system="S", tools=[])

    async def go():
        await s.wait_turn()
        await s.send_user("x")          # uses the reserved slot
        await s.send_user("y")          # acquires its own
    asyncio.run(go())
    assert len(calls) == 2


def test_pilot_waits_for_quota_outside_the_model_timeout():
    class PacedSession:
        def __init__(self, inner):
            self.inner = inner

        async def wait_turn(self):
            await asyncio.sleep(0.2)                    # longer than the model timeout below
            return 0.2

        async def send_user(self, text):
            return await self.inner.send_user(text)

        async def send_tool_results(self, results):
            return await self.inner.send_tool_results(results)

    class Paced(FakeProvider):
        def start(self, system, tools):
            return PacedSession(super().start(system, tools))
    model = PlannedPilot([TURN])
    pilot, _, rec, _, _, _ = build(model, limits=PilotLimits(model_timeout_s=0.1))
    pilot._provider = Paced(model)
    out = asyncio.run(pilot.run())
    assert out.ended == "completed" and not rec.of("llm_error")
    assert len(rec.of("llm_rate_wait")) == out.model_turns
