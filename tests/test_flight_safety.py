"""safe_shutdown tests with a fake async drone (no simulator)."""
import asyncio

from models.telemetry import GroundState
from utils.flight_safety import safe_shutdown


class FakeFlyingDrone:
    def __init__(self, landed_raw=1, fail=()):
        self.landed_raw = landed_raw
        self.fail = set(fail)
        self.calls = []

    def get_landed_state(self):
        return self.landed_raw

    def _maybe_fail(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    async def _task(self, name):
        self._maybe_fail(name)

        async def done():
            return None
        return asyncio.ensure_future(done())

    async def hover_async(self):
        return await self._task("hover")

    async def land_async(self):
        return await self._task("land")

    def disarm(self):
        self._maybe_fail("disarm")

    def disable_api_control(self):
        self._maybe_fail("disable_api_control")


class FakeAdapter:
    def __init__(self, ground_state):
        self.ground_state = ground_state

    def get_snapshot(self, _vid):
        return type("S", (), {"ground_state": self.ground_state})()


def run(coro):
    return asyncio.run(coro)


def test_airborne_drone_hovers_lands_disarms_releases():
    d = FakeFlyingDrone(landed_raw=1)
    run(safe_shutdown(d, log=lambda *_: None))
    assert d.calls == ["hover", "land", "disarm", "disable_api_control"]


def test_grounded_per_adapter_skips_hover_and_land_even_if_raw_says_flying():
    d = FakeFlyingDrone(landed_raw=1)  # raw landed_state lagging
    run(safe_shutdown(d, log=lambda *_: None, adapter=FakeAdapter(GroundState.GROUNDED),
                      vehicle_id="Drone1"))
    assert d.calls == ["disarm", "disable_api_control"]


def test_unknown_adapter_state_falls_back_to_raw_landed_state():
    d = FakeFlyingDrone(landed_raw=0)
    run(safe_shutdown(d, log=lambda *_: None, adapter=FakeAdapter(GroundState.UNKNOWN),
                      vehicle_id="Drone1"))
    assert d.calls == ["disarm", "disable_api_control"]


def test_each_step_is_attempted_even_after_failures():
    d = FakeFlyingDrone(landed_raw=1, fail={"hover", "land", "disarm"})
    run(safe_shutdown(d, log=lambda *_: None))
    assert d.calls == ["hover", "land", "disarm", "disable_api_control"]


def test_none_drone_is_a_no_op():
    run(safe_shutdown(None))
