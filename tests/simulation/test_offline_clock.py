"""The offline simulator's clock: a host stall must slow simulated time, not skip it.

One clock moves the world (physics, command expiry and the sim time every reader sees) and
only its ticks move it; reads do not. A stall moves it by at most max_lag_s, with no
catch-up burst. Protective host-time deadlines stay on the host clock, and closing the
simulator stops its timer.

Before this was fixed, a host stall of about 0.1 s during a full test run skipped 8 to 11
sim seconds at x100: velocity commands expired unseen, the drones hovered, and the path
follower reported path_blocked with no obstacle in the mission
(docs/refactor/investigations/path_blocked_without_obstacle.md). These tests reproduce the
stall on purpose, by blocking the event loop."""
import asyncio
import itertools
import json
import time

import pytest

from atal.simulation.kinematic_sim import KinematicSim
from tests.runtime.test_sar_pipeline import run as run_single

STALL_S = 0.3            # host seconds: 30 sim s at x100, three times the follower's stuck limit


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _stall_twice(monkeypatch, at_call=300):
    """Block the event loop for STALL_S host seconds inside two consecutive velocity
    commands of a path follower that is cruising mid-path (horizontal speed above 4 m/s,
    at or after the at_call-th command). The first stall lets the drone coast to a stop;
    the second comes one poll later, before it has moved 1 m again, so the follower then
    sees more than stuck_after_s of sim time without progress: the failure seen in full
    test runs, made to happen on purpose."""
    import math

    from atal.simulation import kinematic_sim
    real = kinematic_sim._Drone.move_by_velocity_async
    calls = {"n": 0, "stalls": 0}

    async def move(self, *a, **kw):
        calls["n"] += 1
        v = self._v
        cruising = math.hypot(v.vn, v.ve) > 4.0
        if calls["stalls"] == 1 or (calls["stalls"] == 0 and calls["n"] >= at_call and cruising):
            calls["stalls"] += 1
            time.sleep(STALL_S)  # noqa: ASYNC251 (blocking the loop is the point)
        return await real(self, *a, **kw)
    monkeypatch.setattr(kinematic_sim._Drone, "move_by_velocity_async", move)
    return calls


def _largest_recorded_jump_s(log_path):
    """The largest step between consecutive sim timestamps in a run log."""
    ts = [e["sim_time_ns"] for e in map(json.loads, log_path.read_text().splitlines())
          if e.get("sim_time_ns") is not None]
    return max(b - a for a, b in itertools.pairwise(ts)) / 1e9


def test_a_host_stall_does_not_make_a_drone_report_a_blocked_path(tmp_path, monkeypatch):
    calls = _stall_twice(monkeypatch)
    code, batch, summary = run_single(tmp_path)
    m = summary["runs"][0]
    log = (batch / "run_00.jsonl").read_text()
    assert calls["stalls"] == 2                # both stalls happened during the search
    assert '"path_blocked"' not in log and '"obstacle"' not in log
    assert code == 0 and m["victim_found"], m["fail_reasons"]
    # the recorded timestamps follow the same capped clock: no 30 s jump at the stalls
    assert _largest_recorded_jump_s(batch / "run_00.jsonl") < 5.0


def test_a_host_stall_does_not_make_a_team_drone_report_a_blocked_path(tmp_path, monkeypatch):
    from tests.runtime.test_sar_swarm import run as run_team
    calls = _stall_twice(monkeypatch, at_call=200)
    code, batch, summary = run_team(tmp_path)
    log = (batch / "run_00.jsonl").read_text()
    assert calls["stalls"] == 2
    assert '"path_blocked"' not in log and '"obstacle"' not in log
    assert code == 0, summary["runs"][0]["fail_reasons"]
    assert _largest_recorded_jump_s(batch / "run_00.jsonl") < 5.0


def _flying(sim):
    d = sim.drone("D")
    d.enable_api_control()
    d.arm()
    v = sim._v["D"]
    v.grounded, v.alt, v.mode = False, 10.0, "hover"
    return d


def test_the_world_moves_with_no_reads_at_all():
    async def go():
        with KinematicSim({"D": (0.0, 0.0)}, speedup=100) as sim:
            v = sim._v["D"]
            v.grounded, v.alt, v.mode, v.cmd = False, 10.0, "velocity", (5.0, 0.0, 0.0)
            v.cmd_until = 1e9
            await asyncio.sleep(0.05)            # nobody reads or commands
            return sim.now_s, v.n
    t, north = asyncio.run(go())
    assert t > 0.5 and north > 0.0             # time and the drone both moved


def test_reads_and_commands_do_not_move_the_world():
    clock = FakeClock()

    async def go():
        with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock) as sim:
            d = _flying(sim)
            sim.tick()
            before = sim.latest_poses()["D"]
            clock.t += 0.004                     # host time passes, no tick yet
            for _ in range(100):
                sim.get_snapshot("D"), sim.latest_poses(), sim.collision_log("D")
                await d.move_by_velocity_async(5.0, 0.0, 0.0, duration=0.3)
            return before, sim.latest_poses()["D"]
    before, after = asyncio.run(go())
    assert after == before                     # same time, same place


def test_how_often_the_world_is_read_does_not_change_time_or_trajectory():
    def final(reads_per_step):
        clock = FakeClock()

        async def go():
            with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock) as sim:
                d = _flying(sim)
                for i in range(60):
                    if i % 10 == 0:
                        await d.move_by_velocity_async(5.0, 1.0, 0.5, duration=0.3)
                    clock.t += 0.0013            # not a whole number of physics steps
                    sim.tick()
                    for _ in range(reads_per_step):
                        sim.get_snapshot("D")
                return sim.latest_poses()["D"]
        return asyncio.run(go())
    assert final(0) == final(1) == final(25)


def test_a_stall_moves_physics_at_most_max_lag_with_no_catch_up_burst():
    clock = FakeClock()

    async def go():
        with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock, max_lag_s=0.5,
                          max_accel_mps2=1e6) as sim:
            d = _flying(sim)
            sim.tick()
            await d.move_by_velocity_async(5.0, 0.0, 0.0, duration=100.0)
            t0, n0 = sim.now_s, sim._v["D"].n
            clock.t += 0.3                       # a host stall: 30 sim s at x100
            sim.tick()                           # the clock task resumes
            t1, n1 = sim.now_s, sim._v["D"].n
            clock.t += 0.001
            sim.tick()
            t2, n2 = sim.now_s, sim._v["D"].n
            return t0, n0, t1, n1, t2, n2, sim.longest_tick_gap_s
    t0, n0, t1, n1, t2, n2, gap = asyncio.run(go())
    assert t1 - t0 == pytest.approx(0.5, abs=0.02)            # sim time: 0.5 s, not 30 s
    assert n1 - n0 == pytest.approx(5.0 * (t1 - t0), abs=0.05)  # physics moved by the same 0.5 s
    assert t2 - t1 == pytest.approx(0.1, abs=0.02)            # then normal speed, no burst
    assert n2 - n1 == pytest.approx(5.0 * (t2 - t1), abs=0.05)
    assert gap == pytest.approx(0.3)                            # the stall is recorded


def test_a_velocity_command_expires_in_sim_time_across_a_stall():
    clock = FakeClock()

    async def go():
        with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock, max_lag_s=0.5) as sim:
            d = _flying(sim)
            sim.tick()
            await d.move_by_velocity_async(5.0, 0.0, 0.0, duration=0.3)
            until = sim._v["D"].cmd_until
            clock.t += 0.3
            sim.tick()
            return until - sim.now_s, sim._v["D"].mode, sim.now_s
    left, mode, now = asyncio.run(go())
    assert left < 0 and mode == "hover"        # expired at 0.3 sim s, inside the 0.5 s tick
    assert now < 1.0                           # and nothing else of the 30 s was skipped


def test_closing_the_simulator_stops_its_timer_and_its_clock():
    clock = FakeClock()

    async def go():
        sim = KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock)
        with sim:
            task = sim._clock_task
            await asyncio.sleep(0.01)
        await asyncio.sleep(0)
        t = sim.now_s
        clock.t += 1.0
        sim.tick()
        sim.latest_poses()
        return task, t, sim.now_s
    task, before, after = asyncio.run(go())
    assert task.done() and task.cancelled()
    assert after == before                     # no more sim time after close


def test_a_protective_host_deadline_does_not_wait_on_sim_time():
    """The stall watchdog counts host seconds: a sim clock that stops (here: closed) is
    still noticed on time."""
    from atal.runtime.watchers import watch_sim_clock

    async def go():
        sim = KinematicSim({"D": (0.0, 0.0)}, speedup=100)
        with sim:
            await asyncio.sleep(0.01)
        stalls = []
        started = time.monotonic()
        await asyncio.wait_for(watch_sim_clock(sim, asyncio.Event(), 0.05, 0.005,
                                               stalls.append), timeout=2.0)
        return stalls, time.monotonic() - started
    stalls, took = asyncio.run(go())
    assert stalls and took < 1.0


def test_without_an_event_loop_the_clock_follows_host_time():
    clock = FakeClock()
    with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock) as sim:
        t0 = sim.latest_poses()["D"][0]
        clock.t += 0.3
        assert (sim.latest_poses()["D"][0] - t0) / 1e9 == pytest.approx(30.0)
