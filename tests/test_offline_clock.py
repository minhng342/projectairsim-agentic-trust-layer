"""The offline simulator's clock: a host stall must slow simulated time, not skip it.

Before this was fixed, a host stall of about 0.1 s during a full test run skipped 8 to 11
sim seconds at x100: velocity commands expired unseen, the drones hovered, and the path
follower reported path_blocked with no obstacle in the mission
(docs/refactor/investigations/path_blocked_without_obstacle.md). These tests reproduce the
stall on purpose, by blocking the event loop."""
import asyncio
import time

import pytest

from atal.simulation.kinematic_sim import KinematicSim
from tests.test_sar_pipeline import run as run_single

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
            time.sleep(STALL_S)
        return await real(self, *a, **kw)
    monkeypatch.setattr(kinematic_sim._Drone, "move_by_velocity_async", move)
    return calls


def test_a_host_stall_does_not_make_a_drone_report_a_blocked_path(tmp_path, monkeypatch):
    calls = _stall_twice(monkeypatch)
    code, batch, summary = run_single(tmp_path)
    m = summary["runs"][0]
    log = (batch / "run_00.jsonl").read_text()
    assert calls["stalls"] == 2                # both stalls happened during the search
    assert '"path_blocked"' not in log and '"obstacle"' not in log
    assert code == 0 and m["victim_found"], m["fail_reasons"]


def test_a_host_stall_does_not_make_a_team_drone_report_a_blocked_path(tmp_path, monkeypatch):
    from tests.test_sar_swarm import run as run_team
    calls = _stall_twice(monkeypatch, at_call=200)
    code, batch, summary = run_team(tmp_path)
    log = (batch / "run_00.jsonl").read_text()
    assert calls["stalls"] == 2
    assert '"path_blocked"' not in log and '"obstacle"' not in log
    assert code == 0, summary["runs"][0]["fail_reasons"]


def test_the_clock_runs_at_most_max_lag_past_its_last_tick():
    clock = FakeClock()

    async def go():
        with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock, max_lag_s=0.5) as sim:
            await asyncio.sleep(0.01)            # the clock task is running
            t0 = sim.latest_poses()["D"][0]
            clock.t += 0.3                       # a host stall: 30 sim s at x100
            stalled = sim.latest_poses()["D"][0]
            sim.tick()                           # the clock task resumes
            clock.t += 0.001
            after = sim.latest_poses()["D"][0]
            return t0, stalled, after, sim.longest_tick_gap_s
    t0, stalled, after, gap = asyncio.run(go())
    assert (stalled - t0) / 1e9 == pytest.approx(0.5)          # not 30 s
    assert (after - stalled) / 1e9 == pytest.approx(0.1, abs=1e-6)   # then normal speed again
    assert gap == pytest.approx(0.3)                            # the stall is recorded


def test_how_often_the_world_is_read_does_not_change_the_time():
    def final_time(reads_per_step):
        clock = FakeClock()

        async def go():
            with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock) as sim:
                await asyncio.sleep(0.01)
                for _ in range(50):
                    clock.t += 0.002
                    sim.tick()
                    for _ in range(reads_per_step):
                        sim.latest_poses()
                return sim.latest_poses()["D"][0]
        return asyncio.run(go())
    assert final_time(0) == final_time(1) == final_time(25)


def test_without_an_event_loop_the_clock_follows_host_time():
    clock = FakeClock()
    with KinematicSim({"D": (0.0, 0.0)}, speedup=100, clock=clock) as sim:
        t0 = sim.latest_poses()["D"][0]
        clock.t += 0.3
        assert (sim.latest_poses()["D"][0] - t0) / 1e9 == pytest.approx(30.0)
