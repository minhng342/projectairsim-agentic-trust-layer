"""Executor on_event reporting for the evaluation record (no simulator)."""
import asyncio

import pytest

from atal.core.action import CommandStatus
from atal.flight.executor import ExecutorConfig, ProjectAirSimExecutor
from tests.test_executor import FAST, FakeAdapter, FakeDrone, FakeWorld, move, rotate


def setup(on_event, **cfg):
    world = FakeWorld()
    drone = FakeDrone(world)
    logs = []
    ex = ProjectAirSimExecutor(FakeAdapter(world, drone),
                               config=ExecutorConfig(**{**FAST.__dict__, **cfg}),
                               log=logs.append, on_event=on_event)
    return world, drone, ex, logs


def collect():
    events = []
    return events, lambda kind, fields: events.append((kind, fields))


def test_correction_is_reported_with_request_id():
    events, cb = collect()
    world, _, ex, _ = setup(cb)
    world.misses = [(-1.8, 1.4, 0.0)]
    a = move(0.0, -3.0, 6.0, speed=3.0)
    r = asyncio.run(ex.execute(a))
    assert r.status == CommandStatus.SUCCEEDED
    (kind, f), = events
    assert kind == "correction" and f["request_id"] == a.request_id and f["n"] == 1
    assert f["vehicle_id"] == "Drone1" and "position error" in f["detail"] and f["sim_time_ns"] > 0


def test_fallback_is_reported_on_timeout():
    events, cb = collect()
    world, drone, ex, _ = setup(cb)
    drone.task_hangs.add("rotate_to_yaw")
    r = asyncio.run(ex.execute(rotate()))
    assert r.status == CommandStatus.TIMED_OUT
    (kind, f), = events
    assert kind == "fallback" and f["fallback"] == "hover" and f["status"] == "timed_out" and f["sent"]


def test_cancellation_after_dispatch_is_reported():
    events, cb = collect()
    world, drone, ex, _ = setup(cb, rotate_task_timeout_s=5.0)
    drone.task_hangs.add("rotate_to_yaw")

    async def go():
        t = asyncio.ensure_future(ex.execute(rotate()))
        await asyncio.sleep(0.1)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    asyncio.run(go())
    (kind, f), = events
    assert kind == "cancelled" and f["sent"] and f["fallback"] == "hover"


def test_failing_event_callback_never_breaks_a_command():
    def boom(kind, fields):
        raise RuntimeError("recorder down")
    world, _, ex, logs = setup(boom)
    world.misses = [(-1.8, 1.4, 0.0)]
    r = asyncio.run(ex.execute(move(0.0, -3.0, 6.0, speed=3.0)))
    assert r.status == CommandStatus.SUCCEEDED and r.corrections == 1
    assert any("on_event(correction) failed" in m for m in logs)


def test_successful_command_without_incidents_emits_nothing():
    events, cb = collect()
    _, _, ex, _ = setup(cb)
    assert asyncio.run(ex.execute(rotate())).status == CommandStatus.SUCCEEDED
    assert events == []
