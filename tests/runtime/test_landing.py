"""descend_before_landing retries once on lost telemetry; the land timeout grows with height."""
import asyncio
from types import SimpleNamespace

from atal.core.action import ActionType, CommandResult, CommandStatus
from atal.core.telemetry import ValidationStatus
from atal.runtime import rates
from atal.runtime.landing import descend_before_landing, land_timeout_s
from atal.runtime.rates import Timing


class _Recorder:
    def __init__(self):
        self.events = []

    def record(self, kind, **fields):
        self.events.append((kind, fields))


def _result(status, code):
    return CommandResult(request_id="r", vehicle_id="Drone2",
                         action_type=ActionType.CHANGE_ALTITUDE, status=status,
                         reason_code=code)


class _Flight:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def change_altitude(self, vid, alt, reason=""):
        self.calls.append((vid, alt))
        return self.results.pop(0)


def _ctx(flight=None, snap=None):
    adapter = SimpleNamespace(get_snapshot=lambda vid: snap)
    return SimpleNamespace(flight=flight, recorder=_Recorder(), log=lambda s: None,
                           timing=Timing(speedup=1000.0), adapter=adapter)


DRONE = SimpleNamespace(vehicle_id="Drone2", launch_altitude_m=120.0)
LOST = _result(CommandStatus.FAILED, "telemetry_lost")
OK = _result(CommandStatus.SUCCEEDED, "completed")


def test_a_descent_lost_to_invalid_telemetry_is_retried_once():
    ctx = _ctx(_Flight(LOST, OK))
    assert asyncio.run(descend_before_landing(ctx, DRONE)) == "succeeded"
    assert ctx.flight.calls == [("Drone2", 121.5), ("Drone2", 121.5)]
    assert [f["attempt"] for _, f in ctx.recorder.events] == [1, 2]


def test_the_retry_happens_only_once():
    ctx = _ctx(_Flight(LOST, LOST))
    assert asyncio.run(descend_before_landing(ctx, DRONE)) == "failed"
    assert len(ctx.flight.calls) == 2


def test_other_descent_failures_are_not_retried():
    ctx = _ctx(_Flight(_result(CommandStatus.TIMED_OUT, "settle_timeout")))
    assert asyncio.run(descend_before_landing(ctx, DRONE)) == "timed_out"
    assert len(ctx.flight.calls) == 1


def _snap(alt, status=ValidationStatus.VALID):
    return SimpleNamespace(altitude_local_m=alt, validation_status=status)


def test_land_timeout_is_the_floor_just_above_the_pad():
    assert land_timeout_s(_ctx(snap=_snap(121.5)), DRONE) == rates.LAND_TIMEOUT_MIN_S


def test_land_timeout_grows_with_height_after_a_failed_descent():
    # the 2026-11-13 live case: Drone2 left 160 m above its pad by a failed descent
    t = land_timeout_s(_ctx(snap=_snap(280.0)), DRONE)
    assert t == 160.0 / rates.LAND_SPEED_FLOOR_MPS + rates.LAND_TIMEOUT_MARGIN_S
    assert t > rates.LAND_TIMEOUT_MIN_S


def test_land_timeout_with_unknown_height_is_the_floor():
    assert land_timeout_s(_ctx(snap=None), DRONE) == rates.LAND_TIMEOUT_MIN_S
    assert land_timeout_s(_ctx(snap=_snap(280.0, ValidationStatus.STALE)),
                          DRONE) == rates.LAND_TIMEOUT_MIN_S
