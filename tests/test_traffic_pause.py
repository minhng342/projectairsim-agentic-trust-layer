"""A traffic pause in the sector executor (recovery policy section 8a): the leg stops and
the drone hovers, the action stays open, and the rest of the same path is flown after the
pause. No path_blocked, no obstacle recovery, no false "swept"; an interrupt for another
reason during a pause still ends the action."""
import asyncio
import math

from atal.core.action import ActionType, ProposedAction
from atal.core.sar import Mission
from atal.flight.executor import ProjectAirSimExecutor
from atal.flight.path_follower import FollowerConfig
from atal.flight.sector_sweep import SectorExecutor
from atal.simulation.kinematic_sim import KinematicSim
from tests.test_sar_pipeline import MISSION


def _setup(events):
    m = Mission.model_validate(MISSION)             # 40 m, 4 x 4 sectors of 10 m, at 10 m
    sim = KinematicSim({"Drone1": (0.0, 0.0)}, speedup=100)
    ex = SectorExecutor(sim, "Drone1", m, (0.0, 0.0), ProjectAirSimExecutor(sim),
                        follower_config=FollowerConfig(poll_interval_s=0.0005,
                                                       hover_timeout_s=5.0),
                        on_event=lambda kind, f: events.append((kind, f)))
    return m, sim, ex


async def _airborne(sim):
    d = sim.drone("Drone1")
    d.enable_api_control()
    d.arm()
    await (await d.takeoff_async())
    await (await d.move_to_position_async(0.0, 0.0, -10.0, 2.0))


def _sweep(sid="B2"):
    return ProposedAction(vehicle_id="Drone1", sector_id=sid,
                          action_type=ActionType.SWEEP_SECTOR, reason="t")


async def _until(cond, sim, limit_s=200.0):
    t0 = sim.now_s
    while not cond():
        if sim.now_s - t0 > limit_s:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


def test_a_pause_mid_sweep_holds_the_drone_and_the_sweep_finishes_from_where_it_stopped():
    events = []
    _, sim, ex = _setup(events)

    async def go():
        with sim:
            await _airborne(sim)
            task = asyncio.create_task(ex.execute(_sweep()))
            await _until(lambda: ex.last_progress() > 0.4, sim)
            ex.pause_for_traffic()
            await asyncio.sleep(0.02)
            before = ex.last_progress()
            p0 = sim.latest_poses()["Drone1"]
            t0 = sim.now_s
            while sim.now_s - t0 < 30.0:                  # 30 sim s: 3x the stuck limit
                await asyncio.sleep(0.005)
            p1 = sim.latest_poses()["Drone1"]
            held = (before, ex.last_progress(), task.done(),
                    math.dist(p0[1:3], p1[1:3]))
            ex.resume_traffic()
            r = await task
            return r, held
    r, (before, during, done_during, drift) = asyncio.run(go())
    assert r.reason_code == "sector_swept"
    assert not done_during and drift < 0.5               # the action stayed open, it hovered
    assert 0.4 < before <= during < 1.0                   # progress kept through the pause
    kinds = [k for k, _ in events]
    assert "traffic_pause_hold" in kinds and "traffic_pause_resume" in kinds
    assert "follower_blocked" not in kinds and "sector_failed" not in kinds


def test_a_detection_during_a_pause_ends_the_sweep_as_stopped_with_its_progress():
    events = []
    _, sim, ex = _setup(events)

    async def go():
        with sim:
            await _airborne(sim)
            task = asyncio.create_task(ex.execute(_sweep()))
            await _until(lambda: ex.last_progress() > 0.3, sim)
            ex.pause_for_traffic()
            await asyncio.sleep(0.05)
            assert ex.interrupt("victim detected") is True
            r = await asyncio.wait_for(task, 5)
            return r, ex.last_progress()
    r, progress = asyncio.run(go())
    assert r.reason_code == "sweep_stopped" and "victim detected" in r.reason
    assert 0.3 < progress < 1.0


def test_a_leg_waits_for_a_pause_that_began_before_it():
    events = []
    _, sim, ex = _setup(events)

    async def go():
        with sim:
            await _airborne(sim)
            ex.pause_for_traffic()
            task = asyncio.create_task(ex.execute(_sweep()))
            t0 = sim.now_s
            while sim.now_s - t0 < 20.0:
                await asyncio.sleep(0.005)
            moved = math.dist(sim.latest_poses()["Drone1"][1:3], (0.0, 0.0))
            ex.resume_traffic()
            return await task, moved
    r, moved = asyncio.run(go())
    assert moved < 0.5 and r.reason_code == "sector_swept"


def test_two_pauses_in_one_sweep_still_finish_it():
    events = []
    _, sim, ex = _setup(events)

    async def go():
        with sim:
            await _airborne(sim)
            task = asyncio.create_task(ex.execute(_sweep()))
            for at in (0.25, 0.6):
                await _until(lambda at=at: ex.last_progress() > at, sim)
                ex.pause_for_traffic()
                t0 = sim.now_s
                while sim.now_s - t0 < 12.0:
                    await asyncio.sleep(0.005)
                ex.resume_traffic()
            return await task
    r = asyncio.run(go())
    assert r.reason_code == "sector_swept"
    assert [k for k, _ in events].count("traffic_pause_stopped") == 2
