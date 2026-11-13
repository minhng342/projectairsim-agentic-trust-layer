"""descend_before_landing retries once on lost telemetry; the land timeout grows with height."""
import asyncio
from types import SimpleNamespace

from atal.core.action import ActionType, CommandResult, CommandStatus
from atal.core.telemetry import ValidationStatus
from atal.runtime import rates
from atal.runtime.landing import clear_pads, descend_before_landing, land_timeout_s
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


def test_under_a_landing_reservation_the_land_timeout_is_its_validity_left_and_no_more():
    rec = SimpleNamespace(landing_validity_left_s=lambda: 40.0)
    drone = SimpleNamespace(vehicle_id="Drone2", launch_altitude_m=120.0, recovery=rec)
    ctx = _ctx(snap=_snap(280.0))
    ctx.timing = Timing(speedup=1.0)
    assert land_timeout_s(ctx, drone) == 40.0      # not the 190 s the height would give
    rec.landing_validity_left_s = lambda: 0.0      # overran into a keep-out: no land time
    assert land_timeout_s(ctx, drone) == 0.0


def test_without_a_landing_reservation_the_height_sized_timeout_applies():
    rec = SimpleNamespace(landing_validity_left_s=lambda: None)
    drone = SimpleNamespace(vehicle_id="Drone2", launch_altitude_m=120.0, recovery=rec)
    assert land_timeout_s(_ctx(snap=_snap(280.0)), drone) == 190.0


# ------------------------------------------------------------ contingency labels
class _Pads:
    def __init__(self, holder, grant):
        self.holder, self.grant, self.waits = holder, grant, 0

    def inside(self, xy):
        return abs(xy[0]) + abs(xy[1]) <= 18.0

    async def acquire(self, vid, timeout_s):
        self.waits += 1
        return self.grant


def _pad_ctx(xy):
    snap = SimpleNamespace(position_ned_m=SimpleNamespace(x=xy[0], y=xy[1]))
    return _ctx(snap=snap)


def _team_drone():
    async def leave():
        return True
    return SimpleNamespace(vehicle_id="Drone2", cleanup=None,
                           sector_executor=SimpleNamespace(leave_pad_zone=leave))


def test_a_dropout_outside_the_zone_while_a_teammate_holds_it_is_not_a_pad_zone_timeout():
    # 2026-11-13 offline dropout (corner slowdown on, seed 43): Drone2 failed 25.8 m out,
    # outside the 18 m zone, never waited; Drone3 held the zone
    ctx, pads, res = _pad_ctx((0.0, 25.8)), _Pads("Drone3", grant=False), {}
    asyncio.run(clear_pads(ctx, pads, 3, _team_drone(), res))
    assert res["contingency"] == "not_home" and pads.waits == 0
    rec = next(f for k, f in ctx.recorder.events if k == "contingency_landing")
    assert rec["held_by"] == "Drone3" and rec["inside_zone"] is False


def test_a_wait_for_the_zone_that_times_out_is_a_pad_zone_timeout():
    ctx, pads, res = _pad_ctx((0.0, 11.0)), _Pads("Drone3", grant=False), {}
    asyncio.run(clear_pads(ctx, pads, 3, _team_drone(), res))
    assert res["contingency"] == "pad_zone_timeout" and pads.waits == 1


def test_a_return_that_timed_out_on_the_way_home_keeps_its_label():
    ctx, pads, res = _pad_ctx((0.0, 25.0)), _Pads("Drone3", grant=False), {}
    res["planner"] = SimpleNamespace(return_home="pad_zone_timeout")
    asyncio.run(clear_pads(ctx, pads, 3, _team_drone(), res))
    assert res["contingency"] == "pad_zone_timeout"


def test_getting_the_zone_is_no_contingency():
    ctx, pads, res = _pad_ctx((0.0, 11.0)), _Pads(None, grant=True), {}
    asyncio.run(clear_pads(ctx, pads, 3, _team_drone(), res))
    assert "contingency" not in res
