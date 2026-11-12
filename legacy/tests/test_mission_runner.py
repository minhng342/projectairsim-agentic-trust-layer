"""MissionRunner: climb to the layer, fly the corridor, apply maneuvers through the
ActionPort while flying (fake simulator)."""
import asyncio
import json

import pytest

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.core.action import ActionType, ProposedAction
from atal.flight.executor import ProjectAirSimExecutor
from atal.flight.path_follower import FollowerConfig, FollowMode
from legacy.corridor.mission import Corridor, DroneMission, MissionPlan
from legacy.corridor.mission_runner import MissionPhase, MissionRunner
from tests.channel_fakes import ListRecorder
from tests.flight.test_executor import FAST, FakeAdapter, FakeDrone, FakeWorld

# FakeWorld: Drone1 airborne at N=-1, E=8, altitude 5
CORRIDOR = Corridor(name="north line", waypoints=[(-1.0, 8.0), (29.0, 8.0), (29.0, 14.0)],
                    half_width_m=3.0, altitude_m=7.0, altitude_band_m=(5.0, 12.0),
                    cruise_speed_mps=1.0)
FOLLOW = FollowerConfig(poll_interval_s=0.005, hover_timeout_s=0.5)


def build(corridor=CORRIDOR):
    world = FakeWorld()
    drone = FakeDrone(world)
    adapter = FakeAdapter(world, drone)
    rec = ListRecorder()
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    runner = MissionRunner(adapter, DroneMission(vehicle_id="Drone1", corridor=corridor), ex,
                           follower_config=FOLLOW, log=lambda *_: None,
                           on_event=lambda k, f: rec.record(k, source="mission", **f))
    feed = ObservationFeed(adapter, "Drone1", recorder=rec)
    port = ActionPort("Drone1", runner, feed, recorder=rec, max_observation_age_s=5.0)
    return world, drone, runner, port, feed, rec


def act(action_type, **kw):
    return ProposedAction(vehicle_id="Drone1", action_type=action_type, reason="test", **kw)


async def propose(port, feed, action):
    obs = await feed.current(after_sequence=port.last_command_ended_sequence)
    return await port.submit(action, based_on=obs.observation_id)


async def until(cond, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.002)


# ------------------------------------------------------------ flight
def test_climbs_to_the_layer_then_flies_the_whole_corridor():
    world, drone, runner, port, feed, rec = build()
    r = asyncio.run(runner.fly())
    assert r.mode == FollowMode.DONE and runner.phase == MissionPhase.FINISHED
    assert drone.calls[0] == "move_to_position"                       # the climb (executor)
    assert world.alt == pytest.approx(7.0, abs=0.5)
    assert abs(world.n - 29.0) <= 0.5 and abs(world.e - 14.0) <= 0.5
    phases = [e["phase"] for e in rec.of("mission_phase")]
    assert phases == ["climbing", "flying", "finished"]


def test_failed_climb_fails_the_mission_without_flying():
    world, drone, runner, *_ = build()
    drone.task_raises.add("move_to_position")
    r = asyncio.run(runner.fly())
    assert r.mode == FollowMode.FAILED and "climb" in r.reason
    assert runner.phase == MissionPhase.FAILED and "move_by_velocity" not in drone.calls


# ------------------------------------------------------------ maneuvers through the port
def test_maneuvers_are_applied_through_the_port_while_flying():
    async def go():
        world, drone, runner, port, feed, rec = build()
        fly = asyncio.ensure_future(runner.fly())
        await until(lambda: runner.follower is not None and runner.follower.progress_m > 3)
        f = runner.follower
        out = {}
        out["maintain"] = await propose(port, feed, act(ActionType.MAINTAIN))
        out["hold"] = await propose(port, feed, act(ActionType.HOLD))
        await until(lambda: f.mode == FollowMode.HOLD)
        p_hold = f.progress_m
        await asyncio.sleep(0.05)
        out["progress_frozen"] = f.progress_m == p_hold
        out["resume"] = await propose(port, feed, act(ActionType.RESUME))
        out["resume_again"] = await propose(port, feed, act(ActionType.RESUME))
        out["speed"] = await propose(port, feed, act(ActionType.CHANGE_SPEED, speed_mps=2.0))
        out["too_fast"] = await propose(port, feed, act(ActionType.CHANGE_SPEED, speed_mps=9.0))
        out["layer"] = await propose(port, feed, act(ActionType.CHANGE_ALTITUDE, altitude_m=9.0))
        out["rotate"] = await propose(port, feed, act(ActionType.ROTATE_TO_HEADING, heading_deg=90))
        out["speed_now"], out["alt_target"] = f.speed_mps, f.altitude_target_m
        result = await fly
        out["after"] = await propose(port, feed, act(ActionType.MAINTAIN))
        return out, result, world, rec
    out, result, world, rec = asyncio.run(go())
    for k in ("maintain", "hold", "resume", "speed", "layer"):
        assert out[k].code == "applied" and out[k].succeeded, k
    assert out["progress_frozen"]
    assert out["resume_again"].code == "not_applicable" and not out["resume_again"].succeeded
    assert out["too_fast"].code == "out_of_range"
    assert out["rotate"].code == "unsupported_action" and "change_speed" in out["rotate"].message
    assert out["speed_now"] == 2.0 and out["alt_target"] == 9.0
    assert result.mode == FollowMode.DONE and world.alt == pytest.approx(9.0, abs=0.5)
    assert out["after"].code == "mission_finished"
    # every maneuver is in the record, linked to its proposal
    assert len(rec.of("maneuver")) == len(rec.of("command_result")) == 9


def test_hold_with_duration_resumes_by_itself():
    async def go():
        world, drone, runner, port, feed, rec = build()
        fly = asyncio.ensure_future(runner.fly())
        await until(lambda: runner.follower is not None and runner.follower.progress_m > 2)
        fb = await propose(port, feed, act(ActionType.HOLD, duration_s=0.1))
        await until(lambda: runner.follower.mode == FollowMode.HOLD)
        status_while_holding = runner.status()
        await until(lambda: runner.follower.mode == FollowMode.CRUISE)
        await fly
        return fb, status_while_holding, rec
    fb, st, rec = asyncio.run(go())
    assert fb.code == "applied" and "hold_remaining_s" in st and st["mode"] == "hold"
    autos = [e for e in rec.of("maneuver") if "automatic" in e["reason"]]
    assert len(autos) == 1


def test_a_new_hold_replaces_the_timer_of_the_previous_one():
    async def go():
        world, drone, runner, port, feed, rec = build()
        fly = asyncio.ensure_future(runner.fly())
        await until(lambda: runner.follower is not None and runner.follower.progress_m > 2)
        await propose(port, feed, act(ActionType.HOLD, duration_s=0.05))
        await propose(port, feed, act(ActionType.HOLD))            # indefinite now
        await asyncio.sleep(0.2)
        mode = runner.follower.mode
        runner.follower.stop()
        await fly
        return mode, rec
    mode, rec = asyncio.run(go())
    assert mode == FollowMode.HOLD
    assert not [e for e in rec.of("maneuver") if "automatic" in e["reason"]]


def test_maneuver_before_the_corridor_starts_is_not_flying():
    world, drone, runner, port, feed, rec = build()

    async def go():
        return await propose(port, feed, act(ActionType.MAINTAIN))
    assert asyncio.run(go()).code == "not_flying"


def test_feedback_for_maneuvers_carries_no_ground_truth():
    async def go():
        world, drone, runner, port, feed, rec = build()
        fly = asyncio.ensure_future(runner.fly())
        await until(lambda: runner.follower is not None and runner.follower.progress_m > 2)
        fb = await propose(port, feed, act(ActionType.HOLD, duration_s=0.05))
        await fly
        return fb, rec
    fb, rec = asyncio.run(go())
    text = fb.model_dump_json()
    assert "truth" not in text and "progress" not in text and "snapshot" not in text
    assert rec.of("command_result")[0]["result"]["final_truth"]       # evaluation side has it


# ------------------------------------------------------------ mission files
def test_corridor_validation():
    with pytest.raises(ValueError):
        Corridor(name="x", waypoints=[(0, 0)], half_width_m=2, altitude_m=20,
                 altitude_band_m=(5, 12), cruise_speed_mps=3)            # layer outside band
    with pytest.raises(ValueError):
        Corridor(name="x", waypoints=[(0, float("nan"))], half_width_m=2, altitude_m=6,
                 altitude_band_m=(5, 12), cruise_speed_mps=3)
    with pytest.raises(ValueError):
        Corridor(name="x", waypoints=[(0, 0)], half_width_m=2, altitude_m=6,
                 altitude_band_m=(5, 12), cruise_speed_mps=9)            # too fast


def test_mission_plan_loads_and_rejects_duplicate_vehicles(tmp_path):
    d = {"vehicle_id": "Drone1", "corridor": CORRIDOR.model_dump(mode="json")}
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"name": "t", "scene": "s.jsonc", "drones": [d]}))
    plan = MissionPlan.load(p)
    assert plan.vehicle_ids == ["Drone1"] and plan.for_vehicle("Drone1").corridor.name == "north line"
    p.write_text(json.dumps({"name": "t", "scene": "s.jsonc", "drones": [d, d]}))
    with pytest.raises(ValueError):
        MissionPlan.load(p)
