"""Phase 0 core, all together on the fake simulator: five drones fly the real mission
file's corridors, hear each other on Remote ID, and scripted agents send maneuvers
through their ports while they fly."""
import asyncio

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.channels.radio import RemoteIdBroadcaster, RemoteIdBus, RemoteIdReceiver
from atal.core.action import ActionType
from atal.flight.executor import ProjectAirSimExecutor
from atal.flight.path_follower import FollowerConfig, FollowMode
from legacy.corridor.mission import MissionPlan
from legacy.corridor.mission_runner import MissionRunner
from legacy.corridor.scripted_maneuvers import ScriptedManeuverAgent, Trigger
from tests.channel_fakes import ListRecorder
from tests.test_executor import FAST, MultiAdapter

FOLLOW = FollowerConfig(poll_interval_s=0.005, hover_timeout_s=0.5)
TRIGGERS = {
    "Drone2": [Trigger(12, ActionType.CHANGE_SPEED, "slow", {"speed_mps": 2.0})],
    "Drone3": [Trigger(15, ActionType.HOLD, "inspect", {"duration_s": 0.05})],
    "Drone4": [Trigger(20, ActionType.HOLD, "inspect"),
               Trigger(20, ActionType.RESUME, "done", after_previous_s=0.05)],
    "Drone5": [Trigger(8, ActionType.CHANGE_ALTITUDE, "mast", {"altitude_m": 19.5})],
}


def test_five_drones_fly_their_corridors_hear_each_other_and_maneuver():
    plan = MissionPlan.load("legacy/corridor/missions/inspection_5.json")
    ids = plan.vehicle_ids
    adapter, rec = MultiAdapter(ids), ListRecorder()
    for vid in ids:                                   # each fake drone starts over its pad
        n, e = plan.for_vehicle(vid).corridor.waypoints[0]
        w = adapter.worlds[vid]
        w.n, w.e, w.alt = n, e, 4.0
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    bus = RemoteIdBus(recorder=rec)
    feeds, rx, tx, runners, agents = {}, {}, {}, {}, {}
    for vid in ids:
        m = plan.for_vehicle(vid)
        slow = m.corridor.model_copy(update={"cruise_speed_mps": 1.0})   # fake moves v*0.3 per tick
        mission = m.model_copy(update={"corridor": slow})
        feeds[vid] = ObservationFeed(adapter, vid, recorder=rec)
        rx[vid] = bus.attach(RemoteIdReceiver(vid, recorder=rec))
        tx[vid] = RemoteIdBroadcaster(adapter, vid, bus, recorder=rec, self_id_text=m.self_id_text)
        runners[vid] = MissionRunner(adapter, mission, ex, follower_config=FOLLOW,
                                     log=lambda *_: None,
                                     on_event=lambda k, f: rec.record(k, source="mission", **f))
        port = ActionPort(vid, runners[vid], feeds[vid], recorder=rec, max_observation_age_s=5.0)
        agents[vid] = ScriptedManeuverAgent(vid, feeds[vid], port, rx[vid], slow,
                                            TRIGGERS.get(vid, []), recorder=rec,
                                            decide_every_s=0.01)

    async def go():
        stop = asyncio.Event()
        bg = [asyncio.ensure_future(t.run(200.0, stop)) for t in tx.values()]
        ag = [asyncio.ensure_future(a.run(stop)) for a in agents.values()]
        results = await asyncio.wait_for(
            asyncio.gather(*(r.fly(timeout_s=60) for r in runners.values())), 60)
        stop.set()
        await asyncio.gather(*bg, *ag)
        return dict(zip(runners, results))
    results = asyncio.run(go())

    assert all(r.mode == FollowMode.DONE for r in results.values()), \
        {v: (r.mode, r.reason) for v, r in results.items()}
    for vid in ids:                                    # everyone heard all four others
        assert set(rx[vid].latest()) == set(ids) - {vid}
        corridor = plan.for_vehicle(vid).corridor
        w = adapter.worlds[vid]
        end = corridor.waypoints[-1]
        assert abs(w.n - end[0]) <= 0.6 and abs(w.e - end[1]) <= 0.6
    codes = {vid: [fb.code for _, fb in agents[vid].feedback] for vid in ids}
    assert codes["Drone2"] == ["applied"] and codes["Drone3"] == ["applied"]
    assert codes["Drone4"] == ["applied", "applied"] and codes["Drone5"] == ["applied"]
    assert adapter.worlds["Drone5"].alt > 19.0                          # the layer change held
    assert not rec.of("port_rejected")
    decisions = rec.of("agent_decision")
    assert decisions and max(d.get("peers_heard", 0) for d in decisions) == 4
    # every maneuver is linked: agent step -> proposal -> gate -> result -> feedback
    steps = rec.of("agent_step")
    results_by_rid = {e["result"]["request_id"] for e in rec.of("command_result")}
    assert steps and all(s["request_id"] in results_by_rid for s in steps)
