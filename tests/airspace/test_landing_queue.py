"""The landing queue (policy section 11a, atal/airspace/landing_queue.py): the budget and
what it is derived from, one cleanup deadline fixed once, a landing that waits longer than
60 s behind a legitimate reservation and then lands, a teammate holding in the column
(expiry, never a descent), and later conflicting requests never overtaking a waiting
landing. Uses the small world of tests/flight/test_recovery.py (pads 3 m apart, levels from
150 m every 10 m, ceiling 240 m, as on the dev mission)."""
import asyncio

import pytest

from atal.airspace.bounds import Bounds
from atal.airspace.landing_queue import MARGIN_S, CleanupDeadline, QueueBudget, landing_top_level_m
from atal.airspace.transition_manager import Kind, Phase, State
from tests.flight.test_recovery import LEVELS, World, recovery, run

BUDGET = QueueBudget.derive(LEVELS, Bounds(), pad_zone_s=600.0)


def test_the_budget_is_derived_from_the_top_transit_level_and_the_team_size():
    # top level for a landing: the highest transit level, T_2(2) = 150 + (2*3 + 2) * 10
    assert landing_top_level_m(LEVELS) == 230 == BUDGET.landing_top_level_m
    v = Bounds().validity_s(230)                            # 230 / v_z + settle + margin
    assert BUDGET.landing_v_max_s == pytest.approx(v) == pytest.approx(237.0)
    assert BUDGET.landing_queue_s == pytest.approx(3 * (v + MARGIN_S))
    assert BUDGET.startup_top_level_m == 170                # the highest search level
    assert BUDGET.startup_queue_s == pytest.approx(3 * (Bounds().validity_s(170) + MARGIN_S))
    assert BUDGET.landing_queue_s > 60.0                    # far beyond max_hold_s


def test_the_cleanup_deadline_is_fixed_once_and_the_pad_zone_wait_draws_on_it():
    t = [100.0]
    c = CleanupDeadline(BUDGET, lambda: t[0])
    first = c.start()
    assert first == pytest.approx(100.0 + BUDGET.landing_queue_s)
    t[0] = 400.0
    assert c.start() == first                               # polling never extends it
    # the pad-zone wait keeps one move's time for the landing request
    assert c.pad_zone_wait_s() == pytest.approx(min(600.0, first - 400.0
                                                    - BUDGET.landing_per_move_s))
    t[0] = first - 10.0
    assert c.pad_zone_wait_s() == 0.0


def _land(w, rec, deadline):
    """Clear a landing, fly it down and report touchdown (the runner's part)."""
    async def go():
        ok = await rec.clear_landing(deadline_s=deadline)
        if ok:
            assert await rec.flight.change_altitude(0.0) == "completed"
            w.d[rec.vehicle_id]["grounded"] = True
            await asyncio.sleep(0.01)
            w.mgr.touchdown(rec.vehicle_id)
        return ok
    return go()


async def _until(cond):
    while not cond():
        await asyncio.sleep(0.001)


def _mine(w, vid):
    return [r for r in w.mgr.reservations.values() if r.vehicle_id == vid]


def test_a_landing_behind_a_legitimate_reservation_longer_than_60_s_then_lands():
    """D0 lands from 150 m first (about 157 s of sim time on the shared column); D1, 3 m
    away, waits behind it far longer than 60 s and lands after D0's touchdown."""
    w = World({"D0": (0, -3, 150), "D1": (0, 0, 160)})
    w.mgr.roster.discard("D2")
    recs = []
    first, second = recovery(w, "D0", 0, []), recovery(w, "D1", 1, recs, max_hold_s=60.0)

    async def body():
        a = asyncio.create_task(_land(w, first, w.t + BUDGET.landing_queue_s))
        await _until(lambda: any(r.state == State.ACTIVE for r in _mine(w, "D0")))
        deadline = w.t + BUDGET.landing_queue_s
        ok = await _land(w, second, deadline)
        await a
        return ok, deadline
    ok, deadline = run(w, body)
    assert ok is True
    (r,) = [r for r in _mine(w, "D1") if r.kind == Kind.LANDING]
    assert r.state == State.COMPLETED and r.wait_until == pytest.approx(deadline)
    wait = second.landing_wait["queue_wait_s"]
    assert wait > 60.0                                      # longer than max_hold_s
    d0 = next(x for x in _mine(w, "D0") if x.kind == Kind.LANDING)
    assert r.granted_at >= d0.ended_at                      # only after D0's touchdown
    assert w.min_sep >= 3.0                                 # the pads, never closer


def test_a_teammate_holding_in_the_column_expires_the_landing_and_nothing_descends():
    """A stuck holder: D0 holds at 155 m, 3 m from D1's column, and never moves. D1's
    landing waits to its deadline, expires, and D1 never descends; an expired request
    can never be started afterwards."""
    w = World({"D0": (0, -3, 155), "D1": (0, 0, 160)})
    w.mgr.roster.discard("D2")
    recs = []
    rec = recovery(w, "D1", 1, recs)

    async def body():
        w.mgr.set_phase("D0", Phase.UNRESOLVED)              # holds, never moves
        deadline = w.t + 40.0
        ok = await rec.clear_landing(deadline_s=deadline)
        return ok, deadline
    ok, deadline = run(w, body)
    assert ok is False
    rs = [r for r in _mine(w, "D1") if r.kind == Kind.LANDING]
    assert rs and all(r.state == State.EXPIRED for r in rs)
    assert all(r.wait_until == pytest.approx(deadline) for r in rs)   # never extended
    assert "holding:D0" in rs[-1].reason
    assert w.d["D1"]["alt"] == pytest.approx(160)           # no descent
    assert w.mgr.start(rs[-1].reservation_id, rs[-1].request_id, "D1") == "expired"
    assert any(k == "landing_not_cleared" for k, _ in recs)


def test_later_conflicting_requests_never_overtake_a_waiting_landing():
    """Starvation: D1's landing waits behind D0's landing. Meanwhile D2, holding in the
    same column at 170 m, asks for a descent that conflicts with it, twice. D1's
    landing is granted first both times; D2 moves only after it."""
    w = World({"D0": (0, -3, 150), "D1": (0, 0, 160), "D2": (3, 0, 170)})
    first = recovery(w, "D0", 0, [])
    second = recovery(w, "D1", 1, [])

    async def body():
        a = asyncio.create_task(_land(w, first, w.t + BUDGET.landing_queue_s))
        await _until(lambda: any(r.state == State.ACTIVE for r in _mine(w, "D0")))
        b = asyncio.create_task(_land(w, second, w.t + BUDGET.landing_queue_s))
        await _until(lambda: any(r.kind == Kind.LANDING for r in _mine(w, "D1")))
        w.mgr.set_phase("D2", Phase.HOLDING)
        asks = []
        for i, wait in enumerate((30.0, 400.0)):   # gives up once, then waits long
            asks.append(w.mgr.request("D2", f"D2-descent-{i}", Kind.DESCENT, (3, 0), 170,
                                      150, wait_until=w.t + wait))
            await _until(lambda r=asks[-1]: r.state != State.REQUESTED)
        await a
        await b
        return asks
    asks = run(w, body)
    landing = next(r for r in _mine(w, "D1") if r.kind == Kind.LANDING)
    d0 = next(r for r in _mine(w, "D0") if r.kind == Kind.LANDING)
    assert landing.state == State.COMPLETED
    assert asks[0].state == State.EXPIRED                   # behind both landings
    # the long request was still waiting when D0 landed, and did not go first
    assert asks[1].requested_at < d0.ended_at
    assert asks[1].granted_at is not None and asks[1].granted_at >= landing.ended_at
