"""The transition manager on its own (recovery policy sections 2 to 8): bounds, levels, the
reservation lifecycle, the grant rules, freshness, the STUCK follow-up, the stop-line,
landing and the roster. Surveillance is fed by hand at 5 Hz of sim time."""
import math

import pytest

from atal.airspace.bounds import Bounds, Levels
from atal.airspace.transition_manager import (
    Kind,
    Phase,
    Sample,
    State,
    TransitionManager,
)

DT = 0.2
LEVELS = Levels(base_m=150, spacing_m=10, team_size=3, max_transit_level=2, ceiling_m=240)
TEAM = ["D0", "D1", "D2"]


def levels_of():
    return {f"D{i}": LEVELS.of(i) for i in range(3)}


class World:
    """Holds each drone's position and feeds the manager at 5 Hz of sim time."""

    def __init__(self, positions, roster=TEAM, **kw):
        self.mgr = TransitionManager(roster, levels_of(), ceiling_m=240, **kw)
        self.pos = {v: dict(p) for v, p in positions.items()}
        self.t = 0.0
        self.silent: set[str] = set()

    def step(self, seconds=DT):
        orders = []
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + DT, 6)
            samples = {v: Sample(self.t, p["x"], p["y"], p["alt"], p.get("vh", 0.0),
                                 p.get("vz", 0.0), p.get("grounded", False))
                       for v, p in self.pos.items() if v not in self.silent}
            orders += self.mgr.update(self.t, samples)
        return orders

    def move_to(self, vid, alt, seconds):
        """Change altitude linearly over `seconds` of sim time, then stop."""
        p = self.pos[vid]
        a0, n = p["alt"], max(1, round(seconds / DT))
        for k in range(1, n + 1):
            p["alt"], p["vz"] = a0 + (alt - a0) * k / n, (alt - a0) / seconds
            self.step()
        p["vz"] = 0.0


def air(x, y, alt, **kw):
    return {"x": x, "y": y, "alt": alt, **kw}


def phases(mgr, **ph):
    for v, p in ph.items():
        mgr.set_phase(v, p)


# ---------------------------------------------------------------- bounds and levels
def test_r_stop_is_derived_from_the_configured_bounds():
    b = Bounds()
    assert b.d_h == pytest.approx(8.0)
    assert b.r_stop == pytest.approx(5 + 2 * 1.5 + 6 * (1 + 0.5) + 6 ** 2 / (2 * 4))
    assert b.r_stop == pytest.approx(21.5)
    assert b.validity_s(30) == pytest.approx(37.0)
    assert b.exclusion_radius(30) == pytest.approx(243.5)
    # a changed bound changes every radius that depends on it
    slower = Bounds(v_h=3.0)
    assert slower.r_stop == pytest.approx(5 + 3 + 3 * 1.5 + 9 / 8)


def test_levels_are_distinct_and_checked_against_spacing_and_ceiling():
    assert [LEVELS.search(i) for i in range(3)] == [150, 160, 170]
    assert LEVELS.transit(0, 1) == 180 and LEVELS.transit(2, 2) == 230
    assert LEVELS.transit(0, 3) is None                       # past max_transit_level
    assert LEVELS.problems() == []
    every = [a for i in range(3) for a in LEVELS.of(i)]
    assert len(set(every)) == len(every)
    for n in range(1, 6):                                     # any team size checks itself
        top = 150 + 10 * (2 * n + n - 1)                     # T_(n-1)(2)
        assert Levels(150, 10, n, 2, ceiling_m=top + 5).problems() == []
        assert Levels(150, 10, n, 2, ceiling_m=top + 4).problems()
    assert "spacing" in Levels(150, 6, 3, 2, ceiling_m=400).problems()[0]
    assert "ceiling" in Levels(150, 10, 3, 2, ceiling_m=None).problems()[0]
    # a level above the ceiling does not exist
    assert Levels(150, 10, 3, 2, ceiling_m=220).transit(2, 2) is None


# ---------------------------------------------------------------- holding and requests
def test_a_request_needs_one_second_of_confirmed_holding_at_an_own_level():
    w = World({"D0": air(50, 50, 150)}, roster=["D0"])
    w.step(0.4)
    m = w.mgr
    m.set_phase("D0", Phase.REQUEST)
    r = m.request("D0", "q1", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r.state == State.DENIED and r.reason == "not_holding"
    w.step(1.0)
    r = m.request("D0", "q2", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r.state == State.GRANTED
    # not at one of its own levels: denied
    w2 = World({"D0": air(50, 50, 155)}, roster=["D0"])
    w2.step(2.0)
    assert w2.mgr.request("D0", "q", Kind.CLIMB, (50, 50), 155, 180,
                          wait_until=99).reason == "not_holding"


def test_duplicates_and_one_outstanding_request_per_vehicle():
    w = World({"D0": air(50, 50, 180), "D1": air(52, 50, 160)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    a = m.request("D1", "q1", Kind.CLIMB, (52, 50), 160, 190, wait_until=w.t + 60)
    assert a.state == State.REQUESTED and a.reason.startswith("holding:D0")
    assert m.request("D1", "q1", Kind.CLIMB, (52, 50), 160, 190, wait_until=w.t + 60) is a
    assert m.request("D1", "q2", Kind.CLIMB, (52, 50), 160, 190,
                     wait_until=w.t + 60).reason == "already_pending"
    assert m.request("D0", "q1", Kind.CLIMB, (50, 50), 180, 210,
                     wait_until=w.t + 60).reason == "request_id_in_use"


def test_a_request_expires_at_its_deadline_and_a_cancel_works_only_before_start():
    w = World({"D0": air(50, 50, 180), "D1": air(52, 50, 160)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    a = m.request("D1", "q1", Kind.CLIMB, (52, 50), 160, 190, wait_until=w.t + 3)
    w.step(3.2)
    assert a.state == State.EXPIRED
    b = m.request("D1", "q2", Kind.CLIMB, (52, 50), 160, 190, wait_until=w.t + 60)
    assert m.cancel("q2", "D1") == "cancelled" and b.state == State.CANCELLED
    w.pos["D0"]["x"] = 500                                   # D0 out of the way
    w.step(1.4)
    c = m.request("D1", "q3", Kind.CLIMB, (52, 50), 160, 190, wait_until=w.t + 60)
    assert c.state == State.GRANTED
    assert m.start(c.reservation_id, "q3", "D1") == "started"
    assert m.cancel("q3", "D1") == "already_started" and c.state == State.ACTIVE


# ---------------------------------------------------------------- the atomic start
def test_the_start_is_refused_after_start_by_after_a_cancel_and_when_not_holding():
    w = World({"D0": air(50, 50, 150)}, roster=["D0"])
    m = w.mgr
    m.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = m.request("D0", "q1", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert m.start(r.reservation_id, "q1", "D1") == "mismatch"
    assert m.start(r.reservation_id, "other", "D0") == "mismatch"
    w.step(2.2)                                              # past start_by
    assert m.start(r.reservation_id, "q1", "D0") == "lapsed" and r.state == State.LAPSED
    r2 = m.request("D0", "q2", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    m.cancel("q2", "D0")
    assert m.start(r2.reservation_id, "q2", "D0") == "cancelled"
    r3 = m.request("D0", "q3", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    w.pos["D0"]["vh"] = 2.0                                  # drifting: not holding
    w.step(0.2)
    assert m.start(r3.reservation_id, "q3", "D0") == "not_holding"
    assert r3.state == State.GRANTED                          # still not started


def test_a_late_grant_lapses_and_motion_without_a_started_grant_is_flagged():
    w = World({"D0": air(50, 50, 150)}, roster=["D0"])
    m = w.mgr
    m.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = m.request("D0", "q1", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    w.move_to("D0", 160, 2.4)            # climbs without calling start()
    assert r.state == State.KEEP_OUT
    assert "unauthorised_motion" in m.flags()["D0"]
    # a lapse where the drone stayed put is released cleanly
    w2 = World({"D0": air(50, 50, 150)}, roster=["D0"])
    w2.mgr.set_phase("D0", Phase.REQUEST)
    w2.step(1.4)
    r2 = w2.mgr.request("D0", "q1", Kind.CLIMB, (50, 50), 150, 180, wait_until=w2.t + 60)
    w2.step(2.4)
    assert r2.state == State.LAPSED and w2.mgr.flags() == {}


def test_the_unauthorised_motion_alarm_uses_the_per_drone_bound_e_z():
    """Layer spacing is checked as separation + 2 e_z: two drones each up to e_z off toward
    each other. A holding drone more than e_z (not 2 e_z) off its level breaks that
    assumption, so the alarm fires there."""
    within = World({"D0": air(50, 50, 150)}, roster=["D0"])
    within.step(1.0)
    within.move_to("D0", 150 + 0.9 * within.mgr.bounds.e_z, 2.0)
    within.step(1.0)
    assert within.mgr.flags() == {}
    beyond = World({"D0": air(50, 50, 150)}, roster=["D0"])
    beyond.step(1.0)
    beyond.move_to("D0", 150 + 1.5 * beyond.mgr.bounds.e_z, 2.0)   # under 2 e_z, over e_z
    assert "unauthorised_motion" in beyond.mgr.flags()["D0"]


# ---------------------------------------------------------------- completion
def test_reaching_the_target_once_is_not_enough_settling_restarts_on_drift():
    w = World({"D0": air(50, 50, 150)}, roster=["D0"])
    m = w.mgr
    m.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = m.request("D0", "q1", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert m.start(r.reservation_id, "q1", "D0") == "started"
    m.set_phase("D0", Phase.MOVING)
    w.move_to("D0", 180, 30)
    assert r.state == State.SETTLING
    w.step(1.6)
    w.pos["D0"]["alt"] = 181.6                               # drifts out of e_z
    w.step(0.4)
    w.pos["D0"]["alt"] = 180.0
    w.step(2.6)
    assert r.state == State.SETTLING                          # timer restarted
    w.step(1.0)
    assert r.state == State.COMPLETED
    assert m.flags() == {}


def test_a_move_that_overruns_becomes_a_keep_out_with_no_gap():
    w = World({"D0": air(50, 50, 150)}, roster=["D0"])
    m = w.mgr
    m.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = m.request("D0", "q1", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    m.start(r.reservation_id, "q1", "D0")
    w.move_to("D0", 165, 20)
    states = set()
    for _ in range(100):                                     # 20 s more, stalled at 165
        w.step()
        states.add(r.state)
    assert states == {State.ACTIVE, State.KEEP_OUT}           # never anything in between
    assert r.state == State.KEEP_OUT


# ---------------------------------------------------------------- grant rules
def test_simultaneous_conflicting_requests_get_exactly_one_grant():
    for k1, k2, a1, a2 in [(Kind.CLIMB, Kind.CLIMB, (150, 180), (160, 190)),
                           (Kind.CLIMB, Kind.DESCENT, (150, 180), (190, 160)),
                           (Kind.DESCENT, Kind.DESCENT, (180, 150), (190, 160))]:
        w = World({"D0": air(50, 50, a1[0]), "D1": air(53, 50, a2[0])},
                  roster=["D0", "D1"])
        phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
        w.step(1.4)
        r1 = w.mgr.request("D0", "a", k1, (50, 50), *a1, wait_until=w.t + 60)
        r2 = w.mgr.request("D1", "b", k2, (53, 50), *a2, wait_until=w.t + 60)
        granted = [r.state for r in (r1, r2)].count(State.GRANTED)
        assert granted == 1, (k1, k2, r1.reason, r2.reason)


def test_a_lower_drone_already_climbing_blocks_a_higher_request():
    w = World({"D0": air(50, 50, 150), "D1": air(60, 50, 160)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    r0 = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r0.state == State.GRANTED               # D1 holds 10 m away (>= d_h)
    m.start(r0.reservation_id, "a", "D0")
    w.pos["D1"]["x"] = 55                          # now closer than d_h to D0's column
    w.step(1.4)
    r1 = m.request("D1", "b", Kind.CLIMB, (55, 50), 160, 190, wait_until=w.t + 60)
    assert r1.state == State.REQUESTED and r1.reason == "vertical:D0"


def test_a_searching_drone_inside_the_span_blocks_within_the_exclusion_radius():
    w = World({"D0": air(0, 0, 150), "D1": air(150, 150, 160, vh=5.0)},
              roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w.step(1.4)
    r = w.mgr.request("D0", "a", Kind.CLIMB, (0, 0), 150, 180, wait_until=w.t + 60)
    assert r.state == State.REQUESTED and r.reason == "pausing:D1"   # 212 m < 243.5 m
    assert w.mgr.paused("D1")                                        # so it is paused
    # a mover at a level outside the span does not block
    w2 = World({"D0": air(0, 0, 150), "D1": air(20, 0, 200, vh=5.0)},
               roster=["D0", "D1"])
    phases(w2.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w2.step(1.4)
    assert w2.mgr.request("D0", "a", Kind.CLIMB, (0, 0), 150, 180,
                          wait_until=w2.t + 60).state == State.GRANTED


def test_a_later_request_that_does_not_conflict_goes_ahead():
    w = World({"D0": air(50, 50, 180), "D1": air(52, 50, 160), "D2": air(150, 150, 230)})
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST, D2=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    a = m.request("D1", "a", Kind.CLIMB, (52, 50), 160, 190, wait_until=w.t + 60)
    assert a.state == State.REQUESTED
    c = m.request("D2", "c", Kind.DESCENT, (150, 150), 230, 200, wait_until=w.t + 60)
    assert c.state == State.GRANTED


def test_a_descent_waiting_on_a_drone_that_waits_on_it_is_refused_as_deadlock():
    w = World({"D0": air(50, 50, 180), "D1": air(53, 50, 190)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    # D0 climbs 180 -> 210 (span holds D1 at 190); D1 descends 190 -> 160 (span holds D0)
    r0 = m.request("D0", "a", Kind.CLIMB, (50, 50), 180, 210, wait_until=w.t + 60)
    r1 = m.request("D1", "b", Kind.DESCENT, (53, 50), 190, 160, wait_until=w.t + 60)
    assert r0.state == State.REQUESTED and r0.reason == "holding:D1"
    assert r1.state == State.DENIED and r1.reason == "deadlock"     # the newer descent
    # D1 retargets: it flies off, and D0's climb goes ahead
    m.set_phase("D1", Phase.TRANSIT)
    w.pos["D1"].update(x=400, vh=5.0)
    w.step(0.2)
    assert r0.state == State.GRANTED


def test_a_later_request_is_not_stuck_behind_one_that_waits_for_its_own_drone():
    w = World({"D0": air(50, 50, 150), "D1": air(53, 50, 160)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    r0 = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r0.state == State.REQUESTED and r0.reason == "holding:D1"
    r1 = m.request("D1", "b", Kind.CLIMB, (53, 50), 160, 190, wait_until=w.t + 60)
    assert r1.state == State.GRANTED                  # D0 waits for D1 to move anyway
    m.start(r1.reservation_id, "b", "D1")
    m.set_phase("D1", Phase.MOVING)
    w.move_to("D1", 190, 30)
    w.step(3.4)
    assert r1.state == State.COMPLETED
    assert r0.state == State.GRANTED                  # then D0 goes


# ---------------------------------------------------------------- freshness
def test_a_stale_or_never_seen_member_blocks_every_grant():
    w = World({"D0": air(50, 50, 150), "D1": air(150, 150, 300)}, roster=["D0", "D1", "D2"])
    w.mgr.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = w.mgr.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r.state == State.REQUESTED and r.reason == "stale:D2"      # never seen
    w.pos["D2"] = air(150, 0, 400)
    w.step(0.2)
    assert r.state == State.GRANTED
    w.mgr.cancel("a", "D0")
    w.silent.add("D1")                                                 # stops reporting
    w.step(1.2)
    r2 = w.mgr.request("D0", "b", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r2.state == State.REQUESTED and r2.reason == "stale:D1"


def test_surveillance_lost_during_a_move_keeps_the_reservation_and_flags_the_run():
    w = World({"D0": air(50, 50, 150), "D1": air(150, 150, 300)}, roster=["D0", "D1"])
    w.mgr.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = w.mgr.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    w.mgr.start(r.reservation_id, "a", "D0")
    w.silent.add("D1")
    w.move_to("D0", 165, 15)
    assert r.state == State.ACTIVE
    assert "separation_unverified" in w.mgr.flags()["D1"]


# ---------------------------------------------------------------- STUCK
def test_a_stuck_drone_retries_from_its_measured_altitude_keeping_its_keep_out():
    w = World({"D0": air(50, 50, 150), "D1": air(54, 50, 150 + 30)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.TRANSIT)
    w.pos["D1"]["x"] = 300
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    m.start(r.reservation_id, "a", "D0")
    w.move_to("D0", 163.4, 13.4)                               # stopped between levels
    assert m.report_stuck("D0", "blocked") is r and r.state == State.KEEP_OUT
    # an ordinary request is refused; so is a follow-up while it still drifts
    assert m.request("D0", "b", Kind.CLIMB, (50, 50), 163.4, 180,
                     wait_until=w.t + 60).reason == "already_pending"
    assert m.request("D0", "c", Kind.CLIMB, (50, 50), 163.4, 180, wait_until=w.t + 60,
                     from_stationary=True).reason == "not_holding"
    w.step(1.2)                                                # confirmed stationary
    f = m.request("D0", "d", Kind.CLIMB, (50, 50), 163.4, 180, wait_until=w.t + 60,
                  from_stationary=True)
    assert f.state == State.GRANTED and r.state == State.KEEP_OUT     # still protected
    assert m.start(f.reservation_id, "d", "D0") == "started"
    assert r.state == State.SUPERSEDED and f.state == State.ACTIVE    # in one step
    assert f.span == (163.4 - 5, 185)


def test_a_stuck_follow_up_that_is_denied_leaves_the_keep_out_in_place():
    w = World({"D0": air(50, 50, 150), "D1": air(300, 50, 170)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.HOLDING)
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    m.start(r.reservation_id, "a", "D0")
    w.move_to("D0", 163, 13)
    m.report_stuck("D0")
    w.pos["D1"].update(x=54)                                   # D1 holds next to the column
    w.step(1.2)
    f = m.request("D0", "b", Kind.CLIMB, (50, 50), 163, 180, wait_until=w.t + 2,
                  from_stationary=True)
    assert f.state == State.REQUESTED and f.reason == "holding:D1"
    w.step(2.2)
    assert f.state == State.EXPIRED and r.state == State.KEEP_OUT
    # only a stuck drone may ask from where it stopped
    w2 = World({"D0": air(50, 50, 163)}, roster=["D0"])
    w2.step(1.4)
    assert w2.mgr.request("D0", "a", Kind.CLIMB, (50, 50), 163, 180, wait_until=99,
                          from_stationary=True).reason == "not_stuck"


# ---------------------------------------------------------------- stop-line and legs
def test_a_drone_crossing_toward_a_move_is_ordered_to_stop_at_r_stop():
    w = World({"D0": air(50, 50, 150), "D1": air(150, 50, 160, vh=5.0)},
              roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.HOLDING)
    w.step(1.4)
    m = w.mgr
    w.pos["D1"]["x"] = 300
    w.step(0.2)
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    m.start(r.reservation_id, "a", "D0")
    m.set_phase("D1", Phase.SEARCHING)
    orders = []
    for x in range(100, 20, -2):                              # D1 flies in at 160 m
        w.pos["D1"]["x"] = x
        orders += w.step()
        if orders:
            break
    assert orders and orders[0].vehicle_id == "D1"
    assert 21.5 - 2.0 <= orders[0].distance_m < 21.5          # stopped at the stop-line
    assert m.paused("D1") and m.pauses["D1"] == {r.reservation_id}   # a pause, owned by r
    assert not m.leg_allowed("D1", (60, 50), (60, 80), 160)    # passes 10 m from it
    assert m.leg_allowed("D1", (100, 50), (100, 80), 160)
    assert m.leg_allowed("D1", (60, 50), (60, 80), 200)        # outside the span


# ---------------------------------------------------------------- startup, landing, roster
def test_startup_starts_from_confirmed_ground_and_grounded_drones_stay_on_the_roster():
    w = World({"D0": air(0, -3, 0, grounded=True), "D1": air(0, 0, 0, grounded=True),
               "D2": air(0, 3, 0, grounded=True)})
    m = w.mgr
    w.step(0.4)
    s2 = m.request("D2", "s2", Kind.STARTUP, (0, 3), 0, 170, wait_until=w.t + 60)
    assert s2.state == State.GRANTED                    # neighbours 3 m away, own pads
    # D1's startup overlaps D2's active column only if within d_h: 3 m is, so it waits
    m.start(s2.reservation_id, "s2", "D2")
    s1 = m.request("D1", "s1", Kind.STARTUP, (0, 0), 0, 160, wait_until=w.t + 60)
    assert s1.state == State.REQUESTED and s1.reason == "vertical:D2"
    assert m.roster == {"D0", "D1", "D2"}
    # unknown ground state: refused
    w.silent.add("D0")
    w.step(1.2)
    assert m.request("D0", "s0", Kind.STARTUP, (0, -3), 0, 150,
                     wait_until=w.t + 60).reason == "not_grounded"


def test_a_grounded_drone_blocks_its_own_pad_column_only():
    w = World({"D0": air(0, -3, 0, grounded=True), "D1": air(0, 0, 160)},
              roster=["D0", "D1"])
    w.mgr.set_phase("D1", Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    ok = m.request("D1", "l1", Kind.LANDING, (0, 0), 160, 0, wait_until=w.t + 60)
    assert ok.state == State.GRANTED                    # the neighbour's pad, 3 m away
    m.cancel("l1", "D1")
    w.pos["D1"].update(y=-3)
    w.step(1.4)
    blocked = m.request("D1", "l2", Kind.LANDING, (0, -3), 160, 0, wait_until=w.t + 60)
    assert blocked.state == State.REQUESTED and blocked.reason == "ground:D0"


def test_landing_lasts_until_touchdown_then_the_roster_needs_a_disarm():
    w = World({"D0": air(0, 0, 150)}, roster=["D0"])
    m = w.mgr
    m.set_phase("D0", Phase.REQUEST)
    w.step(1.4)
    r = m.request("D0", "l", Kind.LANDING, (0, 0), 150, 0, wait_until=w.t + 60)
    m.start(r.reservation_id, "l", "D0")
    w.move_to("D0", 0.0, 30)
    w.step(3.0)
    assert r.state == State.ACTIVE                       # at the ground, not yet confirmed
    w.pos["D0"]["grounded"] = True
    w.step(0.2)
    assert m.leave_roster("D0", disarmed=True) is False  # touchdown not confirmed yet
    m.touchdown("D0")
    assert r.state == State.COMPLETED
    assert m.leave_roster("D0", disarmed=False) is False
    assert m.leave_roster("D0", disarmed=True) is True and m.roster == set()


def test_an_unresolved_drone_stays_on_the_roster_and_blocks():
    w = World({"D0": air(50, 50, 150), "D1": air(53, 50, 160)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.UNRESOLVED)
    w.step(1.4)
    assert w.mgr.leave_roster("D1", disarmed=True) is False
    r = w.mgr.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r.reason == "holding:D1"


def test_the_summary_counts_requests_waits_and_outcomes():
    w = World({"D0": air(50, 50, 150), "D1": air(53, 50, 160)}, roster=["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST)
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    m.request("D0", "a2", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    w.pos["D1"]["x"] = 300
    w.step(2.0)
    s = m.summary()
    assert s["requests"] == 2 and s["denied"] == 1 and s["granted"] == 1
    assert s["wait_total_s"] == pytest.approx(r.granted_at - r.requested_at)
    assert s["wait_total_s"] > 0


# ---------------------------------------------------------------- traffic pauses
class Calls:
    def __init__(self):
        self.log = []

    def pause(self, vid):
        self.log.append(("pause", vid))

    def resume(self, vid):
        self.log.append(("resume", vid))


def paused_world(positions, roster, bounds=None):
    calls = Calls()
    w = World(positions, roster=roster, on_pause=calls.pause, on_resume=calls.resume,
              **({"bounds": bounds} if bounds else {}))
    return w, calls


def stop(w, vid):
    w.pos[vid].update(vh=0.0)


def test_a_searching_teammate_inside_the_span_is_paused_then_the_climb_is_granted():
    w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(120, 50, 160, vh=5.0)},
                            ["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert r.state == State.REQUESTED and calls.log == [("pause", "D1")]
    w.step(0.4)
    assert r.reason == "pausing:D1"                  # still moving: not yet
    stop(w, "D1")                                    # the executor has hovered
    w.step(0.6)
    assert r.state == State.REQUESTED                # stationary for less than 1 s
    w.step(0.8)
    assert r.state == State.GRANTED                  # stationary and 70 m >= d_h
    m.start(r.reservation_id, "a", "D0")
    w.move_to("D0", 180, 30)
    w.step(3.4)
    assert r.state == State.COMPLETED
    assert calls.log == [("pause", "D1"), ("resume", "D1")]
    s = m.summary()
    assert s["pauses"] == {"D1": 1} and s["paused_s"]["D1"] > 30


def test_a_teammate_that_stops_too_close_is_not_moved_and_the_request_runs_out():
    """D1 is 6 m from D0's column (inside d_h 8 m) when D0 asks. The manager does not
    pause it there (that would hold it where it blocks D0) and never moves it; D1 stops
    there on its own, and D0's request runs out (bounded, UNRESOLVED for the drone)."""
    w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(56, 50, 160, vh=5.0)},
                            ["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 5)
    stop(w, "D1")
    w.step(2.0)
    assert r.state == State.REQUESTED and r.reason == "mover:D1"
    w.step(3.2)
    assert r.state == State.EXPIRED and "mover:D1" in r.reason
    assert calls.log == []                                           # never paused
    assert w.pos["D1"]["x"] == 56                                    # never moved
    assert ("D1", r.reservation_id) in m._skipped


def test_the_clearance_uses_the_configured_d_h():
    """d_h = 11 m here: a teammate 10 m from the column is inside it (not paused, still
    blocking); one 12 m away is paused and, once stationary, clear."""
    b = Bounds(e_h=3.0)                                  # d_h = 5 + 2 x 3 = 11 m
    for x, paused, granted in ((60, False, False), (62, True, True)):
        w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(x, 50, 160, vh=5.0)},
                                ["D0", "D1"], bounds=b)
        phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
        w.step(1.4)
        r = w.mgr.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
        stop(w, "D1")
        w.step(3.0)
        assert (("pause", "D1") in calls.log) is paused
        assert (r.state == State.GRANTED) is granted, (x, r.state, r.reason)


def test_a_drone_paused_for_two_reasons_resumes_only_when_both_end():
    w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(150, 150, 160),
                             "D2": air(100, 60, 170, vh=5.0)}, TEAM)
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST, D2=Phase.SEARCHING)
    w.step(1.4)
    m = w.mgr
    a = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    b = m.request("D1", "b", Kind.CLIMB, (150, 150), 160, 190, wait_until=w.t + 60)
    assert m.pauses["D2"] == {a.reservation_id, b.reservation_id}
    stop(w, "D2")
    w.step(1.4)
    assert a.state == b.state == State.GRANTED
    m.start(a.reservation_id, "a", "D0")
    m.start(b.reservation_id, "b", "D1")
    for k in range(1, 151):                     # D0 climbs in 15 s, D1 in 30 s
        w.pos["D0"]["alt"] = 150 + 30 * min(1.0, k / 75)
        w.pos["D1"]["alt"] = 160 + 30 * k / 150
        w.step()
        if k == 110:
            assert a.state == State.COMPLETED and b.state == State.ACTIVE
            assert m.paused("D2") and calls.log == [("pause", "D2")]   # still paused for b
    w.step(3.4)
    assert b.state == State.COMPLETED
    assert calls.log == [("pause", "D2"), ("resume", "D2")]


def test_a_cancelled_request_releases_its_pauses():
    w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(120, 50, 160, vh=5.0)},
                            ["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w.step(1.4)
    r = w.mgr.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    assert w.mgr.paused("D1")
    assert w.mgr.cancel("a", "D0") == "cancelled"
    assert not w.mgr.paused("D1") and calls.log[-1] == ("resume", "D1")
    assert r.state == State.CANCELLED


def test_a_move_that_becomes_stuck_keeps_its_pauses_and_hands_them_to_the_follow_up():
    w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(120, 50, 160, vh=5.0)},
                            ["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    stop(w, "D1")
    w.step(1.4)
    m.start(r.reservation_id, "a", "D0")
    w.move_to("D0", 163, 13)
    m.report_stuck("D0", "blocked")
    w.step(1.2)
    assert r.state == State.KEEP_OUT and m.pauses["D1"] == {r.reservation_id}
    f = m.request("D0", "b", Kind.CLIMB, (50, 50), 163, 180, wait_until=w.t + 60,
                  from_stationary=True)
    assert f.state == State.GRANTED
    m.start(f.reservation_id, "b", "D0")
    assert r.state == State.SUPERSEDED
    assert m.pauses["D1"] == {f.reservation_id}          # handed over in the same step
    assert calls.log == [("pause", "D1")]                 # never resumed in between
    w.move_to("D0", 180, 17)
    w.step(3.4)
    assert f.state == State.COMPLETED and calls.log[-1] == ("resume", "D1")


def test_a_keep_out_that_stays_keeps_its_pauses():
    w, calls = paused_world({"D0": air(50, 50, 150), "D1": air(120, 50, 160, vh=5.0)},
                            ["D0", "D1"])
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.SEARCHING)
    w.step(1.4)
    m = w.mgr
    r = m.request("D0", "a", Kind.CLIMB, (50, 50), 150, 180, wait_until=w.t + 60)
    stop(w, "D1")
    w.step(1.4)
    m.start(r.reservation_id, "a", "D0")
    w.move_to("D0", 163, 13)
    w.step(60)                                            # overran: a keep-out, for good
    assert r.state == State.KEEP_OUT and m.paused("D1")
    assert calls.log == [("pause", "D1")]



def test_a_drone_whose_move_just_completed_still_blocks_a_crossing_move():
    """Found in a team run (4.39 m in cruise): D0's climb completed in the same update in
    which D1's descent, 4 m away, was granted through D0's new altitude, while D0's phase
    still said MOVING. A drone in the span within d_h blocks in any phase."""
    w = World({"D0": air(0, 0, 180), "D1": air(4, 0, 190), "D2": air(300, 300, 170)})
    phases(w.mgr, D0=Phase.MOVING, D1=Phase.HOLDING, D2=Phase.HOLDING)
    w.step(2.0)
    down = w.mgr.request("D1", "d", Kind.DESCENT, (4, 0), 190, 160, wait_until=w.t + 60)
    w.step(1.0)
    assert down.state == State.REQUESTED
    assert "occupied:D0" in w.mgr.blockers(down)


def test_a_completed_move_leaves_its_drone_holding():
    w = World({"D0": air(0, 0, 150), "D1": air(300, 0, 160), "D2": air(300, 300, 170)})
    phases(w.mgr, D0=Phase.HOLDING, D1=Phase.HOLDING, D2=Phase.HOLDING)
    w.step(2.0)
    up = w.mgr.request("D0", "c", Kind.CLIMB, (0, 0), 150, 180, wait_until=w.t + 60)
    w.step(DT)
    assert w.mgr.start(up.reservation_id, "c", "D0") == "started"
    w.mgr.set_phase("D0", Phase.MOVING)
    w.move_to("D0", 180, 30.0)
    w.step(4.0)
    assert up.state == State.COMPLETED
    assert w.mgr.tracks["D0"].phase == Phase.HOLDING


def test_a_drone_over_the_pads_is_not_frozen_in_a_teammates_landing_column():
    """The dev deadlock (790 s): D1, just airborne over its pad 3 m from D0's landing
    column, was paused there by D0's landing request and so blocked it to its deadline.
    Now D1 is left to fly out to the pad-zone edge on its own plan; once it is beyond
    d_h it is paused, stationary, and the landing is granted."""
    w, calls = paused_world({"D0": air(0, 0, 150), "D1": air(3, 0, 140, vh=5.0),
                             "D2": air(300, 300, 170)}, TEAM)
    phases(w.mgr, D0=Phase.HOLDING, D1=Phase.RETURNING, D2=Phase.HOLDING)
    w.step(1.4)
    land = w.mgr.request("D0", "land", Kind.LANDING, (0, 0), 150, 0, wait_until=w.t + 300)
    w.step(DT)
    assert land.state == State.REQUESTED and calls.log == []
    for _ in range(15):                                 # D1 flies out at 5 m/s, 3 s
        if math.dist((w.pos["D1"]["x"], 0), (0, 0)) < w.mgr.bounds.d_h:
            # while it is inside the column, never granted, and it blocks as a mover
            assert land.state == State.REQUESTED
            assert "mover:D1" in w.mgr.blockers(land)
        w.pos["D1"]["x"] += 1.0
        w.step(DT)
    stop(w, "D1")
    w.step(3.0)
    assert ("pause", "D1") in calls.log
    assert land.state == State.GRANTED


@pytest.mark.parametrize("keep_out", [False, True])
def test_the_inside_column_exception_never_applies_to_a_move_in_progress(keep_out):
    """The exception is for pending requests only. Around an ACTIVE reservation (or a
    keep-out) a mover that comes inside the column is stopped and paused by the
    stop-line, however close it is, and stays paused while the reservation lasts."""
    w, calls = paused_world({"D0": air(0, 0, 150), "D1": air(30, 0, 140, vh=5.0),
                             "D2": air(300, 300, 170)}, TEAM)
    phases(w.mgr, D0=Phase.HOLDING, D1=Phase.HOLDING, D2=Phase.HOLDING)
    w.step(1.4)
    land = w.mgr.request("D0", "land", Kind.LANDING, (0, 0), 150, 0, wait_until=w.t + 300)
    w.step(DT)
    assert land.state == State.GRANTED
    assert w.mgr.start(land.reservation_id, "land", "D0") == "started"
    w.mgr.set_phase("D0", Phase.MOVING)
    if keep_out:
        w.mgr.report_stuck("D0", "test")
        assert land.state == State.KEEP_OUT
    w.mgr.set_phase("D1", Phase.SEARCHING)
    w.pos["D1"]["x"] = 3.0                             # inside d_h of the column
    w.step(DT)
    assert ("pause", "D1") in calls.log
    assert land.reservation_id in w.mgr.pauses["D1"]
    assert any(o.vehicle_id == "D1" for o in w.mgr.stop_orders)
    assert ("D1", land.reservation_id) not in w.mgr._skipped
    w.step(5.0)
    assert w.mgr.paused("D1")                           # still paused while it lasts


def test_two_adjacent_climbs_resolve_in_order_with_10_m_levels():
    """Policy section 4, validated configuration only (levels 10 m apart): D0 at 150 m
    and D1 at 160 m, 3 m apart, both climb. D0's span contains D1's level, D1's does not
    contain D0's, so D1 goes first and D0 after it; no cycle. This is not a claim that
    deadlock is impossible for every configuration."""
    w = World({"D0": air(0, 0, 150), "D1": air(3, 0, 160), "D2": air(300, 300, 170)})
    phases(w.mgr, D0=Phase.REQUEST, D1=Phase.REQUEST, D2=Phase.HOLDING)
    w.step(1.4)
    m = w.mgr
    a = m.request("D0", "a", Kind.CLIMB, (0, 0), 150, 180, wait_until=w.t + 120)
    b = m.request("D1", "b", Kind.CLIMB, (3, 0), 160, 190, wait_until=w.t + 120)
    w.step(DT)
    assert a.state == State.REQUESTED and "holding:D1" in m.blockers(a)
    assert b.state == State.GRANTED
    m.start(b.reservation_id, "b", "D1")
    w.mgr.set_phase("D1", Phase.MOVING)
    w.move_to("D1", 190, 30.0)
    w.step(4.0)
    assert b.state == State.COMPLETED
    assert a.state == State.GRANTED and a.granted_at >= b.ended_at
