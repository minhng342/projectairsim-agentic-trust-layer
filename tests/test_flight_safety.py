"""safe_shutdown tests with a fake async drone (no simulator)."""
import asyncio

from models.telemetry import GroundState, ValidationStatus
from utils.flight_safety import safe_shutdown

QUIET = dict(log=lambda *_: None, confirm_timeout_s=0.2, confirm_poll_s=0.01)


class FakeFlyingDrone:
    def __init__(self, landed_raw=1, fail=(), stall_send=(), hang_task=()):
        self.landed_raw = landed_raw
        self.fail = set(fail)
        self.stall_send = set(stall_send)    # request never acknowledged
        self.hang_task = set(hang_task)      # acknowledged, task never finishes
        self.calls = []

    def get_landed_state(self):
        return self.landed_raw

    def _maybe_fail(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    async def _task(self, name):
        self._maybe_fail(name)
        if name in self.stall_send:
            await asyncio.sleep(3600)

        async def done():
            if name in self.hang_task:
                await asyncio.sleep(3600)
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
    """ground_state may be a value, or a callable(drone) evaluated on each snapshot."""

    def __init__(self, ground_state, status=ValidationStatus.VALID, raises=False, drone=None):
        self.ground_state, self.status, self.raises, self.drone = ground_state, status, raises, drone

    def get_snapshot(self, _vid):
        if self.raises:
            raise RuntimeError("not connected")
        g = self.ground_state(self.drone) if callable(self.ground_state) else self.ground_state
        return type("S", (), {"ground_state": g, "validation_status": self.status})()


def grounded_after_land(drone):
    return GroundState.GROUNDED if "land" in drone.calls else GroundState.AIRBORNE


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ normal paths
def test_airborne_without_adapter_lands_then_disarms():
    d = FakeFlyingDrone(landed_raw=1)
    r = run(safe_shutdown(d, **QUIET))
    assert d.calls == ["hover", "land", "disarm", "disable_api_control"]
    assert r.grounded_confirmed and r.disarmed and r.api_released


def test_airborne_with_adapter_disarms_only_after_ground_confirmed():
    d = FakeFlyingDrone(landed_raw=1)
    a = FakeAdapter(grounded_after_land, drone=d)
    r = run(safe_shutdown(d, adapter=a, vehicle_id="Drone1", **QUIET))
    assert d.calls == ["hover", "land", "disarm", "disable_api_control"]
    assert r.grounded_confirmed and not r.unresolved


def test_grounded_per_adapter_skips_hover_and_land_even_if_raw_says_flying():
    d = FakeFlyingDrone(landed_raw=1)  # raw landed_state lagging
    r = run(safe_shutdown(d, adapter=FakeAdapter(GroundState.GROUNDED), vehicle_id="Drone1", **QUIET))
    assert d.calls == ["disarm", "disable_api_control"] and r.grounded_confirmed


def test_none_drone_is_a_no_op():
    r = run(safe_shutdown(None))
    assert not r.disarmed


# ------------------------------------------------------------ never disarm in the air
def test_failed_landing_never_disarms():
    """Review finding: land failure previously still led to disarm (a forced fall)."""
    d = FakeFlyingDrone(landed_raw=1, fail={"land"})
    r = run(safe_shutdown(d, **QUIET))
    assert d.calls == ["hover", "land"]
    assert r.unresolved and not r.disarmed and not r.api_released
    assert any("UNRESOLVED" in s for s in r.steps)


def test_landing_that_times_out_never_disarms():
    d = FakeFlyingDrone(landed_raw=1, hang_task={"land"})
    r = run(safe_shutdown(d, land_timeout_s=0.1, **QUIET))
    assert d.calls == ["hover", "land"] and r.unresolved


def test_landing_ok_but_ground_not_confirmed_never_disarms():
    d = FakeFlyingDrone(landed_raw=1)
    r = run(safe_shutdown(d, adapter=FakeAdapter(GroundState.AIRBORNE), vehicle_id="Drone1", **QUIET))
    assert d.calls == ["hover", "land"] and r.unresolved and not r.disarmed


def test_uncertain_adapter_state_is_treated_as_airborne_and_never_confirmed():
    """Unknown, stale, invalid or erroring adapters: hover + land, but no disarm,
    and no fallback to the raw landed_state (which says LANDED here)."""
    cases = [FakeAdapter(GroundState.UNKNOWN),
             FakeAdapter(GroundState.GROUNDED, status=ValidationStatus.STALE),
             FakeAdapter(GroundState.GROUNDED, status=ValidationStatus.INVALID),
             FakeAdapter(GroundState.GROUNDED, raises=True)]
    for adapter in cases:
        d = FakeFlyingDrone(landed_raw=0)
        r = run(safe_shutdown(d, adapter=adapter, vehicle_id="Drone1", **QUIET))
        assert d.calls == ["hover", "land"], adapter.__dict__
        assert r.unresolved and not r.disarmed


def test_hover_failure_still_attempts_landing():
    d = FakeFlyingDrone(landed_raw=1, fail={"hover"})
    r = run(safe_shutdown(d, **QUIET))
    assert d.calls == ["hover", "land", "disarm", "disable_api_control"] and r.grounded_confirmed


def test_disarm_failure_still_attempts_release():
    d = FakeFlyingDrone(landed_raw=0, fail={"disarm"})
    r = run(safe_shutdown(d, **QUIET))
    assert d.calls == ["disarm", "disable_api_control"]
    assert not r.disarmed and r.api_released


# ------------------------------------------------------------ bounded end to end
def test_stalled_request_acknowledgement_is_bounded():
    """Review finding: the timeout used to start only after the request was acknowledged."""
    d = FakeFlyingDrone(landed_raw=1, stall_send={"hover", "land"})

    async def timed():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        # outer guard so a regression FAILS instead of hanging the suite
        r = await asyncio.wait_for(
            safe_shutdown(d, hover_timeout_s=0.1, land_timeout_s=0.1, **QUIET), 2.0)
        return r, loop.time() - t0
    r, elapsed = run(timed())
    assert elapsed < 1.0
    assert d.calls == ["hover", "land"] and r.unresolved
