"""ScriptedMissionAgent, PhaseBarrier, and the whole channel end to end (no simulator)."""
import asyncio

import pytest

from legacy.llm_pilot.scripted_agent import MissionStep, ScriptedMissionAgent
from channel.action_port import ActionPort
from channel.attacks import PositionOffset
from channel.observation_feed import ObservationFeed
from executor.projectairsim_executor import ProjectAirSimExecutor
from models.action import (ActionFeedback, ActionType, CommandStatus, FeedbackStage,
                           ProposedAction)
from tests.channel_fakes import ListRecorder, SnapAdapter
from tests.test_executor import FAST, MultiAdapter
from legacy.fleet import PhaseBarrier, run_all_or_cancel


# ------------------------------------------------------------ PhaseBarrier
def test_barrier_releases_only_when_all_have_arrived():
    async def go():
        b = PhaseBarrier(["A", "B"])
        order = []

        async def party(name, delay):
            await asyncio.sleep(delay)
            order.append(f"{name} arrives")
            ok = await b.wait(name)
            order.append(f"{name} passes {ok}")
        await asyncio.gather(party("A", 0), party("B", 0.05))
        return order
    order = asyncio.run(go())
    assert order[:2] == ["A arrives", "B arrives"]
    assert set(order[2:]) == {"A passes True", "B passes True"}


def test_party_departing_early_breaks_barrier_instead_of_hanging_peers():
    async def go():
        b = PhaseBarrier(["A", "B"])
        waiter = asyncio.ensure_future(b.wait("A"))
        await asyncio.sleep(0.01)
        b.depart("B")
        return await asyncio.wait_for(waiter, 1.0), b.broken_by
    assert asyncio.run(go()) == (False, "B")


def test_depart_after_trip_does_nothing():
    async def go():
        b = PhaseBarrier(["A"])
        assert await b.wait("A")
        b.depart("A")
        return b.broken_by
    assert asyncio.run(go()) is None


def test_unknown_party_rejected():
    with pytest.raises(ValueError):
        asyncio.run(PhaseBarrier(["A"]).wait("Z"))


# ------------------------------------------------------------ agent with fake port
class FakePort:
    def __init__(self, vid, statuses=None, block_on=None):
        self.vehicle_id = vid
        self.submitted = []
        self.statuses = list(statuses or [])
        self.block_on = block_on

    async def submit(self, action, based_on):
        self.submitted.append((action, based_on))
        if self.block_on and action.reason == self.block_on:
            await asyncio.sleep(3600)
        status = self.statuses.pop(0) if self.statuses else CommandStatus.SUCCEEDED
        return ActionFeedback(request_id=action.request_id, vehicle_id=action.vehicle_id,
                              action_type=action.action_type, observation_id=based_on,
                              stage=FeedbackStage.EXECUTOR, status=status,
                              code="completed" if status == CommandStatus.SUCCEEDED else "failed",
                              message="fake")


def alt_step(vid, name, alt, sync=None):
    return MissionStep(name, lambda obs, home: ProposedAction(
        vehicle_id=vid, action_type=ActionType.CHANGE_ALTITUDE, altitude_m=alt, reason=name), sync)


def agent(vid="Drone1", steps=None, port=None, adapter=None, rec=None):
    adapter = adapter or SnapAdapter()
    feed = ObservationFeed(adapter, vid)
    port = port or FakePort(vid)
    steps = steps or [alt_step(vid, "a", 7), alt_step(vid, "b", 8)]
    return ScriptedMissionAgent(vid, feed, port, steps, recorder=rec), port, adapter


def test_agent_runs_steps_in_order_each_on_a_fresh_observation():
    a, port, _ = agent()
    results = asyncio.run(a.run())
    assert [n for n, _ in results] == ["a", "b"] and a.ended == "completed"
    assert all(isinstance(oid, str) for _, oid in port.submitted)    # ids, not objects
    obs_seqs = [a._feed.resolve(oid).sequence for _, oid in port.submitted]
    assert obs_seqs == [2, 3]                       # #1 was the home observation


def test_targets_come_from_the_agents_own_home_observation():
    steps = [MissionStep("go", lambda obs, home: ProposedAction(
        vehicle_id="Drone1", action_type=ActionType.MOVE_TO_POSITION, north_m=home.north_m + 10,
        east_m=home.east_m, altitude_m=home.altitude_m, reason="go"))]
    a, port, adapter = agent(steps=steps)
    asyncio.run(a.run())
    action = port.submitted[0][0]
    assert action.north_m == adapter.state["Drone1"]["n"] + 10


def test_agent_stops_at_first_failure():
    a, port, _ = agent(port=FakePort("Drone1", [CommandStatus.TIMED_OUT]))
    results = asyncio.run(a.run())
    assert len(results) == 1 and a.ended == "stopped at a: timed_out (failed)"


def test_no_observation_aborts_without_acting():
    a, port, adapter = agent()
    adapter.raises = True
    asyncio.run(a.run())
    assert port.submitted == [] and "no initial observation" in a.ended


def test_bad_action_from_nan_observation_aborts_cleanly():
    from channel.attacks import HookContext  # noqa: F401

    def wreck(payload, ctx):
        payload["position_ned_m"] = None
        return payload
    adapter = SnapAdapter()
    feed = ObservationFeed(adapter, "Drone1", hooks=[wreck])
    steps = [MissionStep("go", lambda obs, home: ProposedAction(
        vehicle_id="Drone1", action_type=ActionType.MOVE_TO_POSITION, north_m=home.north_m,
        east_m=home.east_m, altitude_m=6.0, reason="go"))]
    port = FakePort("Drone1")
    a = ScriptedMissionAgent("Drone1", feed, port, steps)
    asyncio.run(a.run())
    assert port.submitted == [] and "could not build action" in a.ended


def test_failed_drone_releases_peers_waiting_at_the_sync_point():
    async def go():
        b = PhaseBarrier(["Drone1", "Drone2"], name="all returned")
        adapter = SnapAdapter()
        a1, _, _ = agent("Drone1", [alt_step("Drone1", "return", 7), alt_step("Drone1", "descend", 4, b)],
                         adapter=adapter)
        a2, p2, _ = agent("Drone2", [alt_step("Drone2", "return", 9), alt_step("Drone2", "descend", 4, b)],
                          port=FakePort("Drone2", [CommandStatus.FAILED]), adapter=adapter)
        await asyncio.wait_for(asyncio.gather(a1.run(), a2.run()), 2.0)
        return a1, a2
    a1, a2 = asyncio.run(go())
    assert "barrier 'all returned' broken by Drone2" in a1.ended
    assert [n for n, _ in a1.results] == ["return"]          # Drone1 never descended


def test_nobody_descends_before_everyone_has_returned():
    async def go():
        b = PhaseBarrier(["Drone1", "Drone2"])
        adapter = SnapAdapter()
        log = []

        class SlowPort(FakePort):
            async def submit(self, action, based_on):
                log.append(f"{self.vehicle_id}:{action.reason}:start")
                await asyncio.sleep(0.1 if self.vehicle_id == "Drone2" and action.reason == "return" else 0)
                log.append(f"{self.vehicle_id}:{action.reason}:end")
                return await super().submit(action, based_on)
        a1, _, _ = agent("Drone1", [alt_step("Drone1", "return", 7), alt_step("Drone1", "descend", 4, b)],
                         port=SlowPort("Drone1"), adapter=adapter)
        a2, _, _ = agent("Drone2", [alt_step("Drone2", "return", 9), alt_step("Drone2", "descend", 4, b)],
                         port=SlowPort("Drone2"), adapter=adapter)
        await asyncio.gather(a1.run(), a2.run())
        return log
    log = asyncio.run(go())
    assert log.index("Drone1:descend:start") > log.index("Drone2:return:end")


def test_cancelled_agent_departs_so_peers_are_not_stranded():
    async def go():
        b = PhaseBarrier(["Drone1", "Drone2"])
        adapter = SnapAdapter()
        a1, _, _ = agent("Drone1", [alt_step("Drone1", "return", 7), alt_step("Drone1", "descend", 4, b)],
                         adapter=adapter)
        a2, _, _ = agent("Drone2", [alt_step("Drone2", "return", 9), alt_step("Drone2", "descend", 4, b)],
                         port=FakePort("Drone2", block_on="return"), adapter=adapter)
        t2 = asyncio.ensure_future(a2.run())
        t1 = asyncio.ensure_future(a1.run())
        await asyncio.sleep(0.05)
        t2.cancel()
        await asyncio.wait_for(t1, 1.0)
        return a1, a2
    a1, a2 = asyncio.run(go())
    assert "broken by Drone2" in a1.ended and a2.ended.startswith("interrupted")


def test_agent_refuses_another_vehicles_feed_or_port():
    adapter = SnapAdapter()
    with pytest.raises(ValueError):
        ScriptedMissionAgent("Drone1", ObservationFeed(adapter, "Drone2"), FakePort("Drone1"), [])
    with pytest.raises(ValueError):
        ScriptedMissionAgent("Drone1", ObservationFeed(adapter, "Drone1"), FakePort("Drone2"), [])


# ------------------------------------------------------------ end to end
def build_fleet(hooks_for=None):
    ids = ["Drone1", "Drone2"]
    adapter = MultiAdapter(ids)
    rec = ListRecorder()
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None,
                               on_event=lambda kind, f: rec.record(kind, source="executor", **f))
    agents = {}
    for vid in ids:
        feed = ObservationFeed(adapter, vid, hooks=(hooks_for or {}).get(vid, []), recorder=rec)
        port = ActionPort(vid, ex, feed, recorder=rec)
        steps = [MissionStep("go", lambda obs, home, vid=vid: ProposedAction(
            vehicle_id=vid, action_type=ActionType.MOVE_TO_POSITION, north_m=home.north_m + 10,
            east_m=home.east_m, altitude_m=home.altitude_m, reason="go"))]
        agents[vid] = ScriptedMissionAgent(vid, feed, port, steps, recorder=rec)
    return adapter, rec, agents


def test_end_to_end_chain_is_fully_linked_in_the_record():
    adapter, rec, agents = build_fleet()
    asyncio.run(run_all_or_cancel(a.run() for a in agents.values()))
    assert all(a.ended == "completed" for a in agents.values())
    for vid in agents:
        step = [e for e in rec.of("agent_step") if e["vehicle_id"] == vid][0]
        prop = [e for e in rec.of("proposal") if e["request_id"] == step["request_id"]][0]
        dec = [e for e in rec.of("gate_decision") if e["decision"]["request_id"] == step["request_id"]][0]
        res = [e for e in rec.of("command_result") if e["result"]["request_id"] == step["request_id"]][0]
        assert prop["observation_id"] == step["observation_id"] == res["observation_id"]
        assert dec["decision"]["outcome"] == "bypassed"
        assert res["result"]["status"] == "succeeded"
        assert res["result"]["final_truth"]["north_m"] == pytest.approx(-1.0 + 10)


def test_attack_misleads_the_agent_and_executor_flies_the_wrong_target_accurately():
    """Drone1's channel reports it 5 m further north than it is. Its agent's home is
    wrong, so it commands home+10 relative to a false home; the executor (ground
    truth) flies that wrong target precisely. Drone2 is unaffected."""
    adapter, rec, agents = build_fleet({"Drone1": [PositionOffset(north_m=5.0)]})
    asyncio.run(run_all_or_cancel(a.run() for a in agents.values()))
    true_start = -1.0
    d1 = [e for e in rec.of("command_result") if e["vehicle_id"] == "Drone1"][0]["result"]
    d2 = [e for e in rec.of("command_result") if e["vehicle_id"] == "Drone2"][0]["result"]
    assert d1["status"] == "succeeded"
    assert d1["final_truth"]["north_m"] == pytest.approx(true_start + 5 + 10)   # 5 m off
    assert d2["final_truth"]["north_m"] == pytest.approx(true_start + 10)
    tampered = [e for e in rec.of("observation") if e.get("tampered_by")]
    assert tampered and all(e["vehicle_id"] == "Drone1" for e in tampered)


def test_home_captured_before_takeoff_is_reused_by_run():
    """Live Pass 5.5 bug: home was taken after takeoff (hover height), so
    'descend to launch + 1.5 m' aimed 1.3 m too high."""
    a, port, adapter = agent(steps=[MissionStep("down", lambda obs, home: ProposedAction(
        vehicle_id="Drone1", action_type=ActionType.CHANGE_ALTITUDE,
        altitude_m=home.altitude_m + 1.5, reason="down"))])
    adapter.state["Drone1"]["alt"] = 2.7            # on the ground
    home = a.capture_home()
    adapter.state["Drone1"]["alt"] = 4.0            # after takeoff
    asyncio.run(a.run())
    assert a.home is home and port.submitted[0][0].altitude_m == pytest.approx(4.2)
    acted_on = a._feed.resolve(port.submitted[0][1])
    assert acted_on.altitude_m == 4.0               # the step still acts on a fresh observation


def test_with_a_sampler_each_step_acts_on_a_sample_taken_after_the_previous_command():
    """Observation cadence is independent of the agent: samplers run at a fixed rate,
    and the agent picks the first sample received after its last command ended
    (anything older is 'superseded' at the port)."""
    ids = ["Drone1", "Drone2"]
    adapter = MultiAdapter(ids)
    rec = ListRecorder()
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    agents, feeds = {}, {}
    for vid in ids:
        feeds[vid] = ObservationFeed(adapter, vid, recorder=rec)
        port = ActionPort(vid, ex, feeds[vid], recorder=rec)
        steps = [MissionStep(name, lambda obs, home, vid=vid, dn=dn: ProposedAction(
            vehicle_id=vid, action_type=ActionType.MOVE_TO_POSITION, north_m=home.north_m + dn,
            east_m=home.east_m, altitude_m=home.altitude_m, reason=name))
            for name, dn in (("out", 10), ("back", 0))]
        agents[vid] = ScriptedMissionAgent(vid, feeds[vid], port, steps, recorder=rec)

    async def go():
        for a in agents.values():
            a.capture_home()
        stop = asyncio.Event()
        samplers = [asyncio.ensure_future(f.sample(50.0, stop)) for f in feeds.values()]
        await run_all_or_cancel(a.run() for a in agents.values())
        stop.set()
        await asyncio.gather(*samplers)
    asyncio.run(go())
    assert all(a.ended == "completed" for a in agents.values()), {v: a.ended for v, a in agents.items()}
    assert not rec.of("port_rejected")
    for vid in ids:
        seqs = [feeds[vid].resolve(e["observation_id"]).sequence
                for e in rec.of("agent_step") if e["vehicle_id"] == vid]
        total = len([e for e in rec.of("observation") if e["vehicle_id"] == vid])
        assert seqs[0] < seqs[1] and total > 3          # the stream ran between decisions
