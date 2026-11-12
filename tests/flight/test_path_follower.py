"""Velocity path follower on the fake drone (tests/flight/test_executor.py fakes)."""
import asyncio
import itertools
import math

import pytest

from atal.core.telemetry import GroundState, ValidationStatus
from atal.flight.path_follower import FollowerConfig, FollowMode, PathFollower, Polyline
from tests.flight.test_executor import FakeAdapter, FakeDrone, FakeWorld

FAST = FollowerConfig(poll_interval_s=0.0, hover_timeout_s=0.5)
# The corner slowdown and command slew (live altitude-hold fix) have their own tests below;
# the older tests check path geometry and modes without them.
PLAIN = FollowerConfig(poll_interval_s=0.0, hover_timeout_s=0.5, max_cmd_accel_mps2=None,
                       corner_speed_mps=None)
# FakeWorld starts airborne at N=-1, E=8, altitude 5, heading 315
L_PATH = [(-1.0, 8.0), (14.0, 8.0), (14.0, 20.0)]        # 15 m north, then 12 m east


class Ticks:
    """sleep() stand-in that runs callbacks at given tick numbers."""

    def __init__(self, at=None):
        self.n, self.at = 0, at or {}

    async def __call__(self, s):
        self.n += 1
        if self.n in self.at:
            self.at[self.n]()
        await asyncio.sleep(0)


def make(path=L_PATH, alt=5.0, speed=3.0, at=None, config=PLAIN):
    world = FakeWorld()
    drone = FakeDrone(world)
    events = []
    f = PathFollower(FakeAdapter(world, drone), "Drone1", path, alt, speed, config=config,
                     sleep=Ticks(at), on_event=lambda k, d: events.append((k, d)),
                     log=lambda *_: None)
    track = []
    drone.velocities = []

    async def instant(v_north, v_east, v_down, duration=0.001, **yaw):
        """The fake's own velocity command lands after 20 ms of real time; the follower's
        test sleep doesn't wait, so apply at once (moved by v * duration)."""
        track.append((world.n, world.e, world.alt))
        drone.calls.append("move_by_velocity")
        drone.velocities.append((v_north, v_east, v_down))
        world.target_ne = (world.n + v_north * duration, world.e + v_east * duration)
        world.target_alt = world.alt - v_down * duration
        done = asyncio.get_running_loop().create_future()
        done.set_result(None)
        return done
    drone.move_by_velocity_async = instant
    return f, world, drone, events, track


def run(f, **kw):
    async def guarded():                     # a regression must fail, not hang the suite
        return await asyncio.wait_for(f.run(**kw), 10.0)
    return asyncio.run(guarded())


def dist_to_path(path, n, e):
    return Polyline(path).project(n, e)[1]


# ------------------------------------------------------------ geometry
def test_polyline_length_points_and_projection():
    p = Polyline(L_PATH)
    assert p.length == pytest.approx(27.0)
    assert p.point_at(15.0) == pytest.approx((14.0, 8.0))
    assert p.point_at(21.0) == pytest.approx((14.0, 14.0))
    assert p.point_at(99.0) == pytest.approx((14.0, 20.0))
    s, d = p.project(5.0, 9.0)
    assert s == pytest.approx(6.0) and d == pytest.approx(1.0)


def test_projection_window_keeps_progress_on_the_current_leg():
    # an out-and-back path: near the start, the return leg passes the same spot
    p = Polyline([(0, 0), (10, 0), (10, 1), (0, 1)])
    s, _ = p.project(1.0, 0.6, s_min=0.0, window_m=6.0)
    assert s < 6.0                                    # not the return leg at s ~ 20


@pytest.mark.parametrize("bad", [[], [(0.0, math.nan)]])
def test_bad_paths_rejected(bad):
    with pytest.raises(ValueError):
        Polyline(bad)


# ------------------------------------------------------------ cruise
def test_flies_the_whole_path_in_order_and_hovers_at_the_end():
    f, world, drone, events, track = make()
    r = run(f)
    assert r.mode == FollowMode.DONE and r.waypoints_passed == [1, 2]
    assert math.hypot(world.n - 14.0, world.e - 20.0) <= FAST.done_tolerance_m
    assert drone.calls[-1] == "hover" and "move_to_position" not in drone.calls
    assert r.max_cross_track_m <= FAST.lookahead_m                # corner cut, bounded
    assert all(dist_to_path(L_PATH, n, e) <= FAST.lookahead_m for n, e, _ in track)
    kinds = [k for k, _ in events]
    assert kinds[0] == "follower_mode" and kinds[-1] == "follower_end"


def test_cruise_speed_is_respected_and_slows_down_near_the_end():
    f, world, drone, _, _ = make(speed=2.0)
    run(f)
    speeds = [math.hypot(vn, ve) for vn, ve, _ in drone.velocities]
    assert max(speeds) <= 2.0 + 1e-9
    assert speeds[-1] < 0.5                                       # arrived slowly


def test_progress_never_goes_backwards():
    f, world, drone, events, _ = make()
    progress = []
    orig = f._guidance

    def spy(snap):
        out = orig(snap)
        progress.append(f.progress_m)
        return out
    f._guidance = spy
    run(f)
    assert progress == sorted(progress) and progress[-1] == pytest.approx(27.0, abs=0.6)


def test_single_waypoint_path_is_a_go_to():
    f, world, *_ = make(path=[(5.0, 2.0)])
    r = run(f)
    assert r.mode == FollowMode.DONE and math.hypot(world.n - 5.0, world.e - 2.0) <= 0.5


# ------------------------------------------------------------ maneuvers
def test_hold_freezes_the_drone_mid_path_and_resume_finishes_it():
    marks = {}
    box = {}

    def do_hold():
        box["f"].hold()

    def mark():
        marks["held_at"] = (box["w"].n, box["w"].e, box["f"].progress_m, box["f"].mode)

    def check_and_resume():
        marks["later"] = (box["w"].n, box["w"].e, box["f"].progress_m)
        box["f"].resume()
    f, world, drone, events, _ = make(at={5: do_hold, 10: mark, 30: check_and_resume})
    box.update(f=f, w=world)
    r = run(f)
    n0, e0, p0, mode = marks["held_at"]
    n1, e1, p1 = marks["later"]
    assert mode == FollowMode.HOLD
    assert math.hypot(n1 - n0, e1 - e0) <= FAST.deadband_m + 0.3 and p1 == pytest.approx(p0)
    assert r.mode == FollowMode.DONE and r.waypoints_passed == [1, 2]
    modes = [d["why"] for k, d in events if k == "follower_mode"]
    assert modes == ["start", "hold", "resume"]


def test_altitude_change_while_cruising():
    box = {}
    f, world, drone, *_ = make(at={3: lambda: box["f"].set_altitude(7.0)})
    box["f"] = f
    r = run(f)
    assert r.mode == FollowMode.DONE and world.alt == pytest.approx(7.0, abs=0.5)
    assert max(abs(vd) for *_, vd in drone.velocities) <= FAST.max_vertical_mps + 1e-9


def test_speed_change_while_cruising_takes_effect():
    box = {}
    f, world, drone, *_ = make(speed=3.0, at={3: lambda: box["f"].set_speed(1.0)})
    box["f"] = f
    run(f)
    speeds = [math.hypot(vn, ve) for vn, ve, _ in drone.velocities]
    assert max(speeds[:3]) == pytest.approx(3.0) and max(speeds[4:]) <= 1.0 + 1e-9


def test_stop_abandons_the_path_and_hovers():
    box = {}
    f, world, drone, *_ = make(at={4: lambda: box["f"].stop()})
    box["f"] = f
    r = run(f)
    assert r.mode == FollowMode.STOPPED and r.progress_m < 27.0 and drone.calls[-1] == "hover"


@pytest.mark.parametrize("bad", [0.0, -1.0, 99.0, math.nan])
def test_bad_speed_rejected(bad):
    f, *_ = make()
    with pytest.raises(ValueError):
        f.set_speed(bad)


# ------------------------------------------------------------ fail-safe
def test_bad_telemetry_mid_path_fails_and_hovers():
    box = {}

    def go_stale():
        box["w"].status = ValidationStatus.STALE
    f, world, drone, *_ = make(at={4: go_stale})
    box["w"] = world
    r = run(f)
    assert r.mode == FollowMode.FAILED and "stale" in r.reason and drone.calls[-1] == "hover"


def test_grounded_fails():
    f, world, drone, *_ = make()
    world.ground = GroundState.GROUNDED
    r = run(f)
    assert r.mode == FollowMode.FAILED and "grounded" in r.reason


def test_timeout_fails_and_hovers():
    f, world, drone, *_ = make(config=FollowerConfig(poll_interval_s=0.0, hover_timeout_s=0.5,
                                                     stuck_after_s=None))
    world.stuck = True
    r = run(f, timeout_s=0.2)
    assert r.mode == FollowMode.FAILED and "within" in r.reason and drone.calls[-1] == "hover"
    assert not f.blocked


def test_no_progress_in_cruise_is_blocked_long_before_the_timeout():
    """Pressed against a wall: sim time runs, the drone does not move. The follower gives
    up after stuck_after_s of SIM time (100 ms per snapshot here) and hovers, instead of
    grinding on until the whole-path timeout."""
    f, world, drone, events, _ = make()
    world.stuck = True
    r = run(f, timeout_s=600)
    assert r.mode == FollowMode.FAILED and r.reason.startswith("blocked")
    assert f.blocked and drone.calls[-1] == "hover"
    (b,) = [d for k, d in events if k == "follower_blocked"]
    assert b["stalled_s"] >= FAST.stuck_after_s


def test_a_long_hold_is_not_blocked():
    box = {}

    def do_hold():
        box["f"].hold()

    def do_resume():
        box["f"].resume()
    # held for ~200 snapshots = 20 s of sim time, twice stuck_after_s
    f, world, drone, events, _ = make(at={5: do_hold, 205: do_resume})
    box["f"] = f
    r = run(f)
    assert r.mode == FollowMode.DONE and not f.blocked


def test_command_error_fails_and_hovers():
    f, world, drone, *_ = make()

    async def boom(*a, **k):
        raise RuntimeError("link down")
    drone.move_by_velocity_async = boom
    r = run(f)
    assert r.mode == FollowMode.FAILED and "link down" in r.reason and drone.calls[-1] == "hover"


def test_cancellation_hovers_then_propagates():
    async def go():
        world = FakeWorld()
        world.stuck = True
        drone = FakeDrone(world)
        f = PathFollower(FakeAdapter(world, drone), "Drone1", L_PATH, 5.0, 3.0,
                         config=FollowerConfig(poll_interval_s=0.01, hover_timeout_s=0.5),
                         log=lambda *_: None)
        t = asyncio.ensure_future(f.run())
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        return drone
    assert asyncio.run(go()).calls[-1] == "hover"


def test_closed_loop_path_flies_the_whole_loop():
    """Diag dry run: a rectangle that ends where it starts must not count as 'arrived'
    at the start (straight-line distance to the end is 0 there)."""
    loop = [(-1.0, 8.0), (14.0, 8.0), (14.0, 18.0), (-1.0, 18.0), (-1.0, 8.0)]
    f, world, drone, _, track = make(path=loop)
    r = run(f)
    assert r.mode == FollowMode.DONE and r.waypoints_passed == [1, 2, 3, 4]
    assert max(n for n, _, _ in track) > 13.0 and max(e for _, e, _ in track) > 17.0
    assert math.hypot(world.n + 1.0, world.e - 8.0) <= FAST.done_tolerance_m


def test_hold_brakes_first_and_stays_where_it_stopped_instead_of_flying_back():
    """Live diag 2026-10-28: at 3 m/s the drone stopped ~2 m past the hold call point and
    then flew back to it. Now it brakes (zero velocity), then pins the point where it stopped."""
    box, marks = {}, {}

    def call_hold():
        w = box["w"]
        box["f"].hold()
        marks["call"] = (w.n, w.e)
        w.vn = 2.5                                  # still moving: momentum

    def coast():                                    # the drone carries on while braking
        w = box["w"]
        w.n += 0.5
        w.target_ne = (w.n, w.e)

    def stopped():
        box["w"].vn = 0.0

    def check_and_resume():
        marks["later"] = (box["w"].n, box["w"].e)
        box["f"].resume()
    at = {5: call_hold, 6: coast, 7: coast, 8: coast, 9: coast, 10: stopped,
          30: check_and_resume}
    f, world, drone, events, _ = make(at=at)
    box.update(f=f, w=world)
    r = run(f)
    braking = drone.velocities[5:9]
    assert all(vn == 0.0 and ve == 0.0 for vn, ve, _ in braking)     # brake, don't steer back
    pin = [d for k, d in events if k == "follower_hold_point"][0]["at"]
    # pinned where it stopped: past the call point by the coast (+ the cruise step in flight)
    assert marks["call"][0] + 2.0 <= pin[0] <= marks["call"][0] + 3.0
    assert marks["later"][0] == pytest.approx(pin[0], abs=0.3)        # and stayed there
    assert r.mode == FollowMode.DONE


def test_hold_pins_the_point_after_the_brake_limit_even_if_still_moving():
    box = {}

    def call_hold():
        box["f"].hold()
        box["w"].vn = 2.5                           # never slows (e.g. wind)
    f, world, drone, events, _ = make(at={5: call_hold, 60: lambda: box["f"].stop()})
    box.update(f=f, w=world)
    run(f)
    pins = [d for k, d in events if k == "follower_hold_point"]
    assert pins and pins[0]["braking_s"] >= FAST.hold_brake_max_s


# ------------------------------------------- corners and command slew (live altitude hold)
# zeus-g 2026-11-12: at every lane corner Simple Flight climbed up to ~2 m (once 3.1 m)
# while the follower already commanded close to its full descent rate. The climbs line up
# with the horizontal command swinging at speed, so the follower now slows before a corner
# and limits how fast the horizontal command changes in CRUISE.
# cmd_duration_s 0.1 = the fake world's sim step, so distance and time agree in these tests
DEFAULTS = FollowerConfig(poll_interval_s=0.0, hover_timeout_s=0.5, cmd_duration_s=0.1)


def test_slows_before_a_corner_and_still_flies_the_path():
    f, _w, drone, _e, _ = make(speed=5.0, config=DEFAULTS)
    r = run(f)
    assert r.mode == FollowMode.DONE and r.waypoints_passed == [1, 2]
    assert r.max_cross_track_m <= DEFAULTS.lookahead_m
    # the command when the carrot first passes the corner (14, 8) is at corner speed
    swing = next(i for i, (vn, ve, _) in enumerate(drone.velocities) if ve > 0.05)
    vn, ve, _ = drone.velocities[swing - 1]
    assert math.hypot(vn, ve) <= DEFAULTS.corner_speed_mps + 0.2     # one 0.1 s step of slack


def test_the_horizontal_command_changes_at_most_max_cmd_accel_per_second_in_cruise():
    f, _w, drone, events, _ = make(speed=5.0, config=DEFAULTS)
    run(f)
    traces = [d for k, d in events if k == "follower_trace"]
    assert traces, "no follower_trace events"
    v = [(vn, ve) for vn, ve, _ in drone.velocities]
    dt = 0.1                                   # the fake world's sim step per snapshot
    lim = DEFAULTS.max_cmd_accel_mps2 * dt + 1e-9
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(v)]
    assert max(steps) <= lim


def test_braking_for_a_hold_is_not_slew_limited():
    box = {}

    def call_hold():
        box["f"].hold()
        box["w"].vn = 2.5
    f, world, drone, events, _ = make(speed=5.0, config=DEFAULTS,
                                      at={8: call_hold, 30: lambda: box["f"].stop()})
    box.update(f=f, w=world)
    run(f)
    held = next(i for i, (k, d) in enumerate(events) if k == "follower_mode" and d["mode"] == "hold")
    assert any(math.hypot(vn, ve) > 0.5 for vn, ve, _ in drone.velocities[:8])
    assert drone.velocities[8][:2] == (0.0, 0.0)          # the first command after hold: stop
    assert held is not None


def test_follower_trace_records_command_beside_measurement():
    f, _w, _d, events, _ = make(speed=3.0, config=DEFAULTS)
    run(f)
    t = [d for k, d in events if k == "follower_trace"][1]
    for key in ("alt_target_m", "alt_m", "vz_cmd_mps", "vz_mps", "vh_cmd_mps", "vh_mps",
                "cmd_gap_s", "mode", "sim_time_ns"):
        assert key in t
    times = [d["sim_time_ns"] for k, d in events if k == "follower_trace"]
    assert min(b - a for a, b in itertools.pairwise(times)) >= DEFAULTS.trace_period_s * 1e9


def test_next_corner_ignores_straight_points():
    p = Polyline([(0, 0), (5, 0), (10, 0), (10, 10), (10, 20)])
    assert p.next_corner(0.0, 30.0) == pytest.approx(10.0)
    assert p.next_corner(10.0, 30.0) is None
