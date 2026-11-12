"""The recovery component with the real transition manager and a small simulated world:
successful recoveries, a taller obstacle, a descent blocked by a roof, waits that end
UNRESOLVED, retargets, simultaneous recoveries, and the episode outcomes. Sim time runs
at 100 sim s per host s."""
import asyncio
import itertools
import math

import pytest

from atal.airspace.bounds import Levels
from atal.airspace.transition_manager import Phase, Sample, State, TransitionManager
from atal.flight.recovery import Recovery, Telemetry

LEVELS = Levels(base_m=150, spacing_m=10, team_size=3, max_transit_level=2, ceiling_m=240)
SIM_PER_TICK = 0.05          # sim seconds per host millisecond tick
V_H, V_Z = 5.0, 1.0


class Box:
    """An obstacle: inside the x/y range, from bottom_m up to top_m (an overhang when
    bottom_m > 0)."""
    def __init__(self, x0, x1, y0, y1, top_m, bottom_m=0.0):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.top_m, self.bottom_m = top_m, bottom_m

    def blocks(self, x, y, alt):
        return (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
                and self.bottom_m <= alt < self.top_m)


class World:
    def __init__(self, drones: dict, boxes=(), roster=None):
        self.t = 0.0
        self.d = {v: {"x": p[0], "y": p[1], "alt": p[2], "vx": 0.0, "vy": 0.0, "vz": 0.0,
                      "grounded": False} for v, p in drones.items()}
        self.boxes = list(boxes)
        self.mgr = TransitionManager(roster or list(drones),
                                     {f"D{i}": LEVELS.of(i) for i in range(3)},
                                     ceiling_m=240)
        self.min_sep = math.inf
        self._next_survey = 0.0

    async def run(self):
        while True:
            await asyncio.sleep(0.001)
            self.t = round(self.t + SIM_PER_TICK, 6)
            for p in self.d.values():
                p["x"] += p["vx"] * SIM_PER_TICK
                p["y"] += p["vy"] * SIM_PER_TICK
                p["alt"] += p["vz"] * SIM_PER_TICK
            for a, b in itertools.combinations(self.d.values(), 2):
                self.min_sep = min(self.min_sep, math.dist((a["x"], a["y"], a["alt"]),
                                                           (b["x"], b["y"], b["alt"])))
            if self.t >= self._next_survey - 1e-9:
                self._next_survey = self.t + 0.2
                self.mgr.update(self.t, {
                    v: Sample(self.t, p["x"], p["y"], p["alt"], math.hypot(p["vx"], p["vy"]),
                              p["vz"], p["grounded"]) for v, p in self.d.items()})

    def now(self):
        return self.t


class Flight:
    def __init__(self, world: World, vid: str):
        self.w, self.vid = world, vid

    @property
    def p(self):
        return self.w.d[self.vid]

    async def hold(self):
        self.p.update(vx=0.0, vy=0.0, vz=0.0)

    def telemetry(self):
        p = self.p
        return Telemetry(self.w.t, p["x"], p["y"], p["alt"], math.hypot(p["vx"], p["vy"]),
                         p["vz"])

    def _blocked(self, x, y, alt):
        return any(b.blocks(x, y, alt) for b in self.w.boxes)

    async def change_altitude(self, alt_m):
        p = self.p
        sign = 1.0 if alt_m > p["alt"] else -1.0
        p["vz"] = sign * V_Z
        while (alt_m - p["alt"]) * sign > 0.05:
            if self._blocked(p["x"], p["y"], p["alt"] + sign * 1.0):
                p["vz"] = 0.0
                return "blocked"
            await asyncio.sleep(0.001)
        p["alt"], p["vz"] = alt_m, 0.0
        return "completed"

    async def fly_to(self, xy, alt_m):
        p = self.p
        while True:
            dx, dy = xy[0] - p["x"], xy[1] - p["y"]
            d = math.hypot(dx, dy)
            if d < 0.2:
                p.update(x=xy[0], y=xy[1], vx=0.0, vy=0.0)
                return "reached"
            if self._blocked(p["x"] + dx / d * 1.0, p["y"] + dy / d * 1.0, p["alt"]):
                p.update(vx=0.0, vy=0.0)
                return "blocked"
            s = min(V_H, d / SIM_PER_TICK)
            p.update(vx=s * dx / d, vy=s * dy / d)
            await asyncio.sleep(0.001)


def recovery(world, vid, i, records, **kw):
    transit = [t for t in (LEVELS.transit(i, k) for k in (1, 2)) if t is not None]
    kw.setdefault("poll_s", 0.001)
    return Recovery(vid, LEVELS.search(i), transit, world.mgr, Flight(world, vid),
                    world.now, lambda kind, **f: records.append((kind, f)), **kw)


async def with_world(world, body):
    task = asyncio.create_task(world.run())
    try:
        await asyncio.sleep(0.01)
        return await asyncio.wait_for(body(), timeout=20)
    finally:
        task.cancel()


def run(world, body):
    return asyncio.run(with_world(world, body))


# ---------------------------------------------------------------- one drone
def test_a_recovery_climbs_to_its_own_level_flies_on_and_comes_down_at_the_sector():
    w = World({"D0": (50, 50, 150)}, roster=["D0"])
    recs = []
    rec = recovery(w, "D0", 0, recs)

    async def body():
        w.mgr.set_phase("D0", Phase.SEARCHING)
        assert await rec.recover("impact") == "transit"
        assert rec.level == 180 and w.d["D0"]["alt"] == pytest.approx(180)
        assert await rec.descend_at((100, 60)) == "ready"
        return rec.level
    assert run(w, body) == 150
    assert (w.d["D0"]["x"], w.d["D0"]["y"]) == (100, 60)
    assert rec.summary() == {"episodes": 1, "completed": 1, "unresolved": 0, "lost": 0,
                             "open": 0}
    states = [r.state for r in w.mgr.reservations.values()]
    assert states == [State.COMPLETED, State.COMPLETED]
    assert w.mgr.flags() == {}
    assert ("recovery_outcome", {"vehicle_id": "D0", "outcome": "completed",
                                 "reason": "back at the search level", "recovery": 1}) in recs


def test_a_second_recovery_goes_one_level_higher_and_a_third_is_lost():
    w = World({"D0": (50, 50, 150)}, roster=["D0"])
    rec = recovery(w, "D0", 0, [])

    async def body():
        assert await rec.recover("impact") == "transit"
        assert await rec.recover("impact") == "transit"       # hit again up there
        assert rec.level == 210
        return await rec.recover("impact")
    assert run(w, body) == "lost"
    assert rec.outcome == "lost" and w.mgr.tracks["D0"].phase == Phase.LOST
    assert rec.summary()["lost"] == 1


def test_no_transit_level_below_the_ceiling_means_lost():
    w = World({"D0": (50, 50, 150)}, roster=["D0"])
    rec = Recovery("D0", 150, [], w.mgr, Flight(w, "D0"), w.now, poll_s=0.001)

    async def body():
        return await rec.recover("impact")
    assert run(w, body) == "lost"
    assert w.mgr.reservations == {}                            # nothing was asked for


def test_a_taller_obstacle_is_hit_again_in_transit():
    tower = Box(90, 110, 40, 80, top_m=190)                    # taller than T_0(1) = 180
    w = World({"D0": (50, 60, 150)}, boxes=[tower], roster=["D0"])
    rec = recovery(w, "D0", 0, [])

    async def body():
        assert await rec.recover("impact") == "transit"
        assert await rec.descend_at((130, 60)) == "obstacle"   # the 10 m was not enough
        assert await rec.recover("impact") == "transit"        # up to T_0(2) = 210
        return await rec.descend_at((130, 60))
    assert run(w, body) == "ready"
    assert rec.summary()["completed"] == 1 and rec.recoveries == 2


# ---------------------------------------------------------------- waits and refusals
def test_a_climb_that_is_never_granted_ends_unresolved_and_moves_nothing():
    w = World({"D0": (50, 50, 150), "D1": (53, 50, 160)})
    w.mgr.roster.discard("D2")
    rec = recovery(w, "D0", 0, [], max_hold_s=5.0)

    async def body():
        w.mgr.set_phase("D1", Phase.UNRESOLVED)               # holds inside D0's span, 3 m away
        return await rec.recover("impact")
    assert run(w, body) == "unresolved"
    assert w.d["D0"]["alt"] == pytest.approx(150)             # it never moved
    r = next(iter(w.mgr.reservations.values()))
    assert r.state == State.EXPIRED
    assert rec.summary()["unresolved"] == 1


def test_a_descent_not_granted_retargets_then_goes_home_at_the_transit_level():
    w = World({"D0": (50, 50, 150), "D1": (300, 300, 160)})
    w.mgr.roster.discard("D2")
    rec = recovery(w, "D0", 0, [], retarget_s=3.0)

    async def body():
        w.mgr.set_phase("D1", Phase.HOLDING)
        assert await rec.recover("impact") == "transit"
        w.d["D1"].update(x=102, y=60)                          # holds by the sector entry
        assert await rec.descend_at((100, 60)) == "retarget"
        return await rec.descend_at((100, 60))
    assert run(w, body) == "return_high"
    assert rec.level == 180 and w.d["D0"]["alt"] == pytest.approx(180)


def test_a_descent_onto_a_roof_goes_back_up_and_reports_the_sector_blocked():
    roof = Box(90, 110, 50, 70, top_m=165)                     # under the sector entry
    w = World({"D0": (50, 60, 150)}, boxes=[roof], roster=["D0"])
    recs = []
    rec = recovery(w, "D0", 0, recs)

    async def body():
        assert await rec.recover("impact") == "transit"
        return await rec.descend_at((100, 60))
    assert run(w, body) == "blocked"
    assert rec.level == 180 and w.d["D0"]["alt"] == pytest.approx(180)
    kinds = [r.state for r in w.mgr.reservations.values()]
    # climb, the descent that stopped (a keep-out replaced in one step), the climb back
    assert kinds == [State.COMPLETED, State.SUPERSEDED, State.COMPLETED]
    assert any(k == "transition_stuck" for k, _ in recs)


def test_a_climb_stopped_partway_with_no_way_on_stays_a_keep_out_and_ends_unresolved():
    lid = Box(40, 60, 40, 60, top_m=300, bottom_m=165)         # an overhang at 165 m
    w = World({"D0": (50, 50, 150)}, boxes=[lid], roster=["D0"])
    rec = recovery(w, "D0", 0, [], max_stuck_retries=2)

    async def body():
        return await rec.recover("impact")
    assert run(w, body) == "unresolved"
    states = [r.state for r in w.mgr.reservations.values()]
    assert states[-1] == State.KEEP_OUT                        # protection never lapsed
    assert State.SUPERSEDED in states                          # each retry took it over
    assert w.d["D0"]["alt"] == pytest.approx(164, abs=1.0)


# ---------------------------------------------------------------- a team
def test_simultaneous_recoveries_both_succeed_without_losing_separation():
    w = World({"D0": (60, 60, 150), "D1": (100, 100, 160), "D2": (140, 20, 170)})
    recs = []
    r0, r1 = recovery(w, "D0", 0, recs), recovery(w, "D1", 1, recs)

    async def body():
        w.mgr.set_phase("D2", Phase.HOLDING)
        a, b = await asyncio.gather(r0.recover("impact"), r1.recover("path_blocked"))
        c, d = await asyncio.gather(r0.descend_at((20, 20)), r1.descend_at((150, 150)))
        return a, b, c, d
    assert run(w, body) == ("transit", "transit", "ready", "ready")
    assert r0.summary()["completed"] == 1 and r1.summary()["completed"] == 1
    assert w.min_sep >= 5.0
    assert w.mgr.flags() == {}
