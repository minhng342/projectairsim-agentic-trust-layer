"""run_all_or_cancel: one mission failing must stop the others before shutdown (no simulator)."""
import asyncio

import pytest

from executor.projectairsim_executor import ProjectAirSimExecutor
from models.action import ActionType, ProposedAction
from tests.test_executor import FAST, MultiAdapter
from utils.fleet import run_all_or_cancel


def run(coro):
    return asyncio.run(coro)


def test_returns_results_in_order():
    async def val(x, delay):
        await asyncio.sleep(delay)
        return x
    assert run(run_all_or_cancel([val("a", 0.03), val("b", 0.0), val("c", 0.01)])) == ["a", "b", "c"]


def test_one_failure_cancels_and_awaits_the_others_before_raising():
    events = []

    async def crash():
        await asyncio.sleep(0.01)
        raise RuntimeError("mission crashed")

    async def long_mission(name):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)          # cleanup (e.g. executor hover) takes time
            events.append(f"{name} cleaned up")
            raise

    async def main():
        with pytest.raises(RuntimeError, match="mission crashed"):
            await run_all_or_cancel([crash(), long_mission("A"), long_mission("B")])
        events.append("helper returned")
    run(main())
    # both cleanups finished BEFORE control came back to the caller (i.e. before shutdown)
    assert events == ["A cleaned up", "B cleaned up", "helper returned"]


def test_caller_cancellation_cancels_all_missions():
    events = []

    async def mission(name):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append(name)
            raise

    async def main():
        task = asyncio.ensure_future(run_all_or_cancel([mission("A"), mission("B")]))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(main())
    assert sorted(events) == ["A", "B"]


def test_real_executor_missions_stop_commanding_drones_once_one_mission_crashes():
    """The exact demo scenario: three drones, one mission crashes mid-flight."""
    ids = ["Drone1", "Drone2", "Drone3"]
    adapter = MultiAdapter(ids)
    for vid in ("Drone2", "Drone3"):
        adapter.drones[vid].task_hangs.add("move_to_position")   # long moves in progress
    ex = ProjectAirSimExecutor(adapter, config=FAST.__class__(**{**FAST.__dict__,
                               "move_task_margin_s": 30.0}), log=lambda *_: None)

    async def move(vid):
        return await ex.execute(ProposedAction(vehicle_id=vid, action_type=ActionType.MOVE_TO_POSITION,
                                               north_m=10, east_m=0, altitude_m=6, reason="t"))

    async def crashing_mission():
        await asyncio.sleep(0.1)
        raise RuntimeError("Drone1 mission crashed")

    async def main():
        with pytest.raises(RuntimeError):
            await run_all_or_cancel([crashing_mission(), move("Drone2"), move("Drone3")])
        snapshot_of_calls = {v: list(adapter.drones[v].calls) for v in ids}
        await asyncio.sleep(0.2)               # anything still running would keep calling
        return snapshot_of_calls
    calls_at_return = run(main())
    for vid in ("Drone2", "Drone3"):
        # the in-flight move was cancelled and the executor hovered, all before return
        assert calls_at_return[vid] == ["move_to_position", "hover"]
        assert adapter.drones[vid].calls == calls_at_return[vid]   # nothing afterwards
    assert adapter.drones["Drone1"].calls == []


def test_repeated_caller_cancellation_still_waits_for_full_cleanup():
    """Review finding: a second cancel used to let the helper return mid-cleanup."""
    events = []

    async def mission(name):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.15)          # slow cleanup (executor hover)
            events.append(f"{name} cleaned")
            raise

    async def main():
        task = asyncio.ensure_future(run_all_or_cancel([mission("A"), mission("B")]))
        await asyncio.sleep(0.02)
        task.cancel()                          # e.g. Ctrl+C
        await asyncio.sleep(0.05)
        task.cancel()                          # Ctrl+C again, during cleanup
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        events.append("helper returned")
    run(main())
    assert events == ["A cleaned", "B cleaned", "helper returned"]


def test_each_child_is_cancelled_exactly_once():
    """A second cancel would interrupt cleanup a child has already started."""
    cancels = {"A": 0, "B": 0}

    async def mission(name):
        while True:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancels[name] += 1
                if cancels[name] == 1:
                    try:
                        await asyncio.sleep(0.05)      # cleanup
                    except asyncio.CancelledError:
                        cancels[name] += 1             # interrupted by a second cancel
                raise

    async def crash():
        await asyncio.sleep(0.01)
        raise RuntimeError("boom")

    async def main():
        task = asyncio.ensure_future(run_all_or_cancel([crash(), mission("A"), mission("B")]))
        await asyncio.sleep(0.03)
        task.cancel()                          # caller also cancelled during cleanup
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await task
    run(main())
    assert cancels == {"A": 1, "B": 1}


def test_first_failure_is_the_error_raised():
    async def ok():
        await asyncio.sleep(0.05)
        return 1

    async def bad():
        raise ValueError("first")
    with pytest.raises(ValueError, match="first"):
        run(run_all_or_cancel([ok(), bad()]))
