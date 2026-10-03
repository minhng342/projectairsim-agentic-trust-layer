"""
Executor tests with a fake world (no simulator).

FakeWorld holds the vehicle state. Fake drone commands set a target; every
get_snapshot() advances sim time by 0.1 s and moves the state toward the target,
so settle dwell is exercised in sim time. Behaviors (exceptions, hangs, stuck
state, telemetry loss) are switched on per test.
"""
import asyncio
import math
from datetime import datetime, timezone

import pytest

from executor.projectairsim_executor import ExecutorConfig, ProjectAirSimExecutor
from models.action import ActionType, CommandStatus, ProposedAction
from models.telemetry import (GroundState, Quaternion, TelemetrySnapshot, ValidationStatus,
                              Vector3)

FAST = ExecutorConfig(poll_interval_s=0.01, invoke_timeout_s=0.2, rotate_task_timeout_s=0.5,
                      altitude_task_margin_s=0.5, settle_timeout_s=0.5, hover_timeout_s=0.2)


class FakeWorld:
    def __init__(self):
        self.t_ns = 10_000_000_000
        self.n, self.e, self.alt, self.heading = -1.0, 8.0, 5.0, 315.0
        self.vn = self.ve = self.vs = 0.0
        self.target_heading = None
        self.target_alt = None
        self.target_ne = None
        self.misses = []                   # per move call: (dn, de, dalt) the drone ends up off by
        self.stuck = False                 # state never reaches the target
        self.status = ValidationStatus.VALID
        self.ground = GroundState.AIRBORNE
        self.invalid_after_snapshots = None
        self.snapshot_raises = False
        self.snapshots = 0

    def step(self):
        self.t_ns += 100_000_000
        if self.stuck:
            return
        if self.target_heading is not None:
            self.heading = self.target_heading
        if self.target_alt is not None:
            self.alt = self.target_alt
        if self.target_ne is not None:
            self.n, self.e = self.target_ne


class FakeDrone:
    def __init__(self, world):
        self.world = world
        self.calls = []
        self.invoke_raises = set()      # command names whose invocation raises
        self.task_raises = set()        # command names whose task raises
        self.task_hangs = set()         # command names whose task never finishes
        self.active = 0
        self.max_active = 0

    async def _command(self, name, apply):
        self.calls.append(name)
        if name == "move_to_position":
            self.move_speeds = getattr(self, "move_speeds", []) + [self.last_move[3]]
        if name in self.invoke_raises:
            raise RuntimeError(f"{name} rejected")

        async def body():
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.02)
                if name in self.task_hangs:
                    await asyncio.sleep(3600)
                if name in self.task_raises:
                    raise RuntimeError(f"{name} failed in simulator")
                apply()
            finally:
                self.active -= 1
        return asyncio.ensure_future(body())

    async def rotate_to_yaw_async(self, yaw, timeout_sec=None, margin=None):
        heading = math.degrees(yaw) % 360.0
        return await self._command("rotate_to_yaw",
                                   lambda: setattr(self.world, "target_heading", heading))

    async def move_to_position_async(self, north, east, down, velocity, timeout_sec=None):
        self.last_move = (north, east, down, velocity)
        def apply():
            dn, de, da = self.world.misses.pop(0) if self.world.misses else (0.0, 0.0, 0.0)
            self.world.target_alt = -down + da
            self.world.target_ne = (north + dn, east + de)
        return await self._command("move_to_position", apply)

    async def hover_async(self):
        return await self._command("hover", lambda: None)


class FakeAdapter:
    def __init__(self, world, drone):
        self.world, self._drone = world, drone
        self.vehicle_ids = ["Drone1"]

    def drone(self, vid):
        return self._drone

    def get_snapshot(self, vid):
        w = self.world
        w.snapshots += 1
        if w.snapshot_raises:
            raise RuntimeError("request timed out")
        w.step()
        status = w.status
        if w.invalid_after_snapshots is not None and w.snapshots > w.invalid_after_snapshots:
            status = ValidationStatus.STALE
        yaw = math.radians(w.heading)
        return TelemetrySnapshot(
            vehicle_id=vid, sim_time_ns=w.t_ns, received_at_utc=datetime.now(timezone.utc),
            position_ned_m=Vector3(x=w.n, y=w.e, z=-w.alt),
            orientation_quaternion=Quaternion(w=math.cos(yaw / 2), x=0, y=0, z=math.sin(yaw / 2)),
            velocity_ned_mps=Vector3(x=w.vn, y=w.ve, z=-w.vs),
            acceleration_ned_mps2=Vector3(x=0, y=0, z=0),
            angular_velocity_rad_s=Vector3(x=0, y=0, z=0),
            latitude_deg=47.6, longitude_deg=-122.1, altitude_msl_m=120 + w.alt,
            altitude_local_m=w.alt, heading_deg=w.heading, ground_speed_mps=math.hypot(w.vn, w.ve),
            vertical_speed_mps=w.vs, ground_state=w.ground, telemetry_age_ms=1.0,
            validation_status=status,
            validation_errors=[] if status == ValidationStatus.VALID else ["simulated"])


def setup(**cfg):
    world = FakeWorld()
    drone = FakeDrone(world)
    adapter = FakeAdapter(world, drone)
    ex = ProjectAirSimExecutor(adapter, config=ExecutorConfig(**{**FAST.__dict__, **cfg}),
                               log=lambda *_: None)
    return world, drone, ex


def rotate(heading=45.0):
    return ProposedAction(vehicle_id="Drone1", action_type=ActionType.ROTATE_TO_HEADING,
                          heading_deg=heading, reason="test")


def climb(alt=7.0):
    return ProposedAction(vehicle_id="Drone1", action_type=ActionType.CHANGE_ALTITUDE,
                          altitude_m=alt, reason="test")


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ success
def test_heading_command_succeeds_after_dwell():
    world, drone, ex = setup()
    r = run(ex.execute(rotate(45.0)))
    assert r.status == CommandStatus.SUCCEEDED, r.reason
    assert r.sent_to_simulator and r.fallback_applied is None
    assert drone.calls == ["rotate_to_yaw"]
    assert r.final_snapshot.heading_deg == pytest.approx(45.0)
    # dwell is in sim time: at least 0.5 s of sim time passed after the task
    assert r.final_snapshot.sim_time_ns - r.start_snapshot.sim_time_ns >= 500_000_000


def test_heading_through_north_uses_wrapped_yaw():
    world, drone, ex = setup()
    world.heading = 350.0
    r = run(ex.rotate_to_heading("Drone1", 10.0))
    assert r.status == CommandStatus.SUCCEEDED, r.reason
    assert world.heading == pytest.approx(10.0)


def test_altitude_command_succeeds_and_holds_captured_north_east():
    world, drone, ex = setup()
    r = run(ex.change_altitude("Drone1", 7.0))
    assert r.status == CommandStatus.SUCCEEDED, r.reason
    north, east, down, velocity = drone.last_move
    assert (north, east, down) == (-1.0, 8.0, -7.0)
    assert velocity == FAST.altitude_default_speed_mps
    assert r.final_snapshot.altitude_local_m == pytest.approx(7.0)


# ------------------------------------------------------------ preflight refusals
@pytest.mark.parametrize("status", [ValidationStatus.STALE, ValidationStatus.INVALID])
def test_bad_telemetry_preflight_refuses_with_zero_drone_calls(status):
    world, drone, ex = setup()
    world.status = status
    for action in (rotate(), climb()):
        r = run(ex.execute(action))
        assert r.status == CommandStatus.REFUSED and not r.sent_to_simulator
        assert status.value in r.reason
    assert drone.calls == []


def test_telemetry_exception_preflight_refuses():
    world, drone, ex = setup()
    world.snapshot_raises = True
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.REFUSED and "unavailable" in r.reason
    assert drone.calls == []


@pytest.mark.parametrize("ground", [GroundState.GROUNDED, GroundState.UNKNOWN])
def test_not_airborne_preflight_refuses(ground):
    """Grounded (or unknown) refuses: change_altitude must never act as takeoff."""
    world, drone, ex = setup()
    world.ground = ground
    for action in (rotate(), climb(10.0)):
        r = run(ex.execute(action))
        assert r.status == CommandStatus.REFUSED and ground.value in r.reason
    assert drone.calls == []


def test_unsupported_action_refused():
    world, drone, ex = setup()
    for action in (ProposedAction(vehicle_id="Drone1", action_type=ActionType.TAKEOFF, reason="x"),
                   ProposedAction(vehicle_id="Drone1", action_type=ActionType.LAND, reason="x")):
        r = run(ex.execute(action))
        assert r.status == CommandStatus.REFUSED and "not supported" in r.reason
    assert drone.calls == []


def test_unknown_vehicle_refused():
    world, drone, ex = setup()
    r = run(ex.execute(ProposedAction(vehicle_id="Drone9", action_type=ActionType.ROTATE_TO_HEADING,
                                      heading_deg=10, reason="x")))
    assert r.status == CommandStatus.REFUSED and drone.calls == []


# ------------------------------------------------------------ failures after dispatch
def test_command_invocation_exception_fails_sent_and_hovers():
    world, drone, ex = setup()
    drone.invoke_raises.add("rotate_to_yaw")
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.FAILED and r.sent_to_simulator
    assert "invocation failed" in r.reason and r.fallback_applied == "hover"
    assert drone.calls == ["rotate_to_yaw", "hover"]


def test_simulator_task_exception_fails_and_hovers():
    world, drone, ex = setup()
    drone.task_raises.add("move_to_position")
    r = run(ex.execute(climb()))
    assert r.status == CommandStatus.FAILED and r.sent_to_simulator
    assert "simulator task failed" in r.reason and r.fallback_applied == "hover"


def test_simulator_task_timeout_times_out_and_hovers():
    world, drone, ex = setup()
    drone.task_hangs.add("rotate_to_yaw")
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.TIMED_OUT and r.sent_to_simulator
    assert "did not finish" in r.reason and r.fallback_applied == "hover"
    assert drone.calls == ["rotate_to_yaw", "hover"]


def test_settle_timeout_times_out_and_hovers():
    world, drone, ex = setup()
    world.stuck = True                     # task "completes" but state never reaches target
    r = run(ex.execute(rotate(45.0)))
    assert r.status == CommandStatus.TIMED_OUT
    assert "did not settle" in r.reason and "heading error" in r.reason
    assert r.fallback_applied == "hover"


def test_telemetry_invalid_after_dispatch_fails_and_hovers():
    world, drone, ex = setup()
    world.invalid_after_snapshots = 1      # preflight is valid, everything after is stale
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.FAILED and r.sent_to_simulator
    assert "telemetry stale" in r.reason and r.fallback_applied == "hover"


def test_telemetry_lost_while_task_running_fails_early():
    world, drone, ex = setup(rotate_task_timeout_s=5.0)
    drone.task_hangs.add("rotate_to_yaw")
    world.invalid_after_snapshots = 1
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.FAILED and "while the command was running" in r.reason
    assert r.elapsed_s < 1.0               # didn't wait for the 5 s task deadline


def test_grounded_mid_command_fails_without_hovering_on_the_ground():
    world, drone, ex = setup()
    world.stuck = True
    original_step = world.step

    def step_then_ground():
        original_step()
        if world.snapshots > 2:
            world.ground = GroundState.GROUNDED
    world.step = step_then_ground
    r = run(ex.execute(climb(0.5)))
    assert r.status == CommandStatus.FAILED and "grounded" in r.reason
    assert r.fallback_applied == "none (vehicle grounded)" and "hover" not in drone.calls


def test_hover_failure_does_not_mask_original_failure():
    world, drone, ex = setup()
    drone.task_raises.add("rotate_to_yaw")
    drone.invoke_raises.add("hover")
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.FAILED
    assert "simulator task failed" in r.reason          # original cause kept
    assert r.fallback_applied.startswith("hover failed")


# ------------------------------------------------------------ concurrency / cancellation
def test_concurrent_commands_are_serialized_per_vehicle():
    world, drone, ex = setup()

    async def both():
        return await asyncio.gather(ex.execute(rotate(45.0)), ex.execute(climb(7.0)))
    r1, r2 = run(both())
    assert r1.status == r2.status == CommandStatus.SUCCEEDED
    assert drone.max_active == 1
    assert drone.calls == ["rotate_to_yaw", "move_to_position"]


def test_cancellation_after_dispatch_hovers_then_reraises():
    world, drone, ex = setup(rotate_task_timeout_s=5.0)
    drone.task_hangs.add("rotate_to_yaw")

    async def cancel_midway():
        task = asyncio.ensure_future(ex.execute(rotate()))
        while "rotate_to_yaw" not in drone.calls:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)          # let the shielded hover finish
    run(cancel_midway())
    assert drone.calls == ["rotate_to_yaw", "hover"]


def test_cancellation_before_dispatch_sends_nothing():
    world, drone, ex = setup()

    async def cancel_while_waiting_for_lock():
        lock = ex._lock("Drone1")
        await lock.acquire()               # hold the vehicle lock
        task = asyncio.ensure_future(ex.execute(rotate()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lock.release()
    run(cancel_while_waiting_for_lock())
    assert drone.calls == []


# ------------------------------------------------------------ move_to_position
def move(n=12.0, e=-3.0, alt=6.0, speed=None):
    return ProposedAction(vehicle_id="Drone1", action_type=ActionType.MOVE_TO_POSITION,
                          north_m=n, east_m=e, altitude_m=alt, speed_mps=speed, reason="test")


def test_move_to_position_succeeds_and_sends_target():
    world, drone, ex = setup()
    r = run(ex.execute(move(12.0, -3.0, 6.0, speed=4.0)))
    assert r.status == CommandStatus.SUCCEEDED, r.reason
    assert drone.last_move == (12.0, -3.0, -6.0, 4.0)
    f = r.final_snapshot
    assert (f.position_ned_m.x, f.position_ned_m.y, f.altitude_local_m) == (12.0, -3.0, 6.0)


def test_move_to_position_default_speed():
    world, drone, ex = setup()
    run(ex.move_to_position("Drone1", 0.0, 0.0, 5.0))
    assert drone.last_move[3] == FAST.move_default_speed_mps


def test_move_not_settled_while_still_moving():
    """Pass 4.1 live finding: the task can finish while the drone still moves at 2.6 m/s."""
    world, drone, ex = setup()
    world.vn = 2.6                         # keeps reporting motion after arrival
    r = run(ex.execute(move()))
    assert r.status == CommandStatus.TIMED_OUT and "speed 2.60" in r.reason
    assert r.fallback_applied == "hover"


def test_move_task_timeout_scales_with_distance():
    world, drone, ex = setup()
    start = run_snapshot(ex)
    near = ex._move_task_timeout(move(start.position_ned_m.x + 3, start.position_ned_m.y, 5.0, 3.0), start)
    far = ex._move_task_timeout(move(start.position_ned_m.x + 30, start.position_ned_m.y, 5.0, 3.0), start)
    assert far - near == pytest.approx(9.0)


def test_move_refused_when_grounded():
    world, drone, ex = setup()
    world.ground = GroundState.GROUNDED
    r = run(ex.execute(move()))
    assert r.status == CommandStatus.REFUSED and drone.calls == []


def run_snapshot(ex):
    return ex.adapter.get_snapshot("Drone1")


# ------------------------------------------------------------ multiple vehicles
class MultiAdapter:
    """Several independent fake vehicles behind one adapter (one scene, many drones)."""

    def __init__(self, ids):
        self.vehicle_ids = list(ids)
        self.worlds = {v: FakeWorld() for v in ids}
        self.drones = {v: FakeDrone(self.worlds[v]) for v in ids}
        self._single = {v: FakeAdapter(self.worlds[v], self.drones[v]) for v in ids}

    def drone(self, vid):
        return self.drones[vid]

    def get_snapshot(self, vid):
        return self._single[vid].get_snapshot(vid)


def test_different_vehicles_run_concurrently_but_each_is_serialized():
    ids = ["Drone1", "Drone2", "Drone3"]
    adapter = MultiAdapter(ids)
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    running = {"now": 0, "max": 0}
    for d in adapter.drones.values():
        original = d._command

        async def tracked(name, apply, _orig=original):
            task = await _orig(name, apply)
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])

            def done(_):
                running["now"] -= 1
            task.add_done_callback(done)
            return task
        d._command = tracked

    async def fleet():
        jobs = []
        for i, vid in enumerate(ids):
            jobs.append(ex.execute(ProposedAction(
                vehicle_id=vid, action_type=ActionType.MOVE_TO_POSITION,
                north_m=10.0 * i, east_m=5.0, altitude_m=5.0 + 2 * i, reason="fleet")))
            jobs.append(ex.execute(ProposedAction(
                vehicle_id=vid, action_type=ActionType.ROTATE_TO_HEADING,
                heading_deg=90.0, reason="fleet")))
        return await asyncio.gather(*jobs)
    results = run(fleet())
    assert all(r.status == CommandStatus.SUCCEEDED for r in results), [r.reason for r in results]
    assert running["max"] >= 2                         # vehicles overlapped
    for d in adapter.drones.values():
        assert d.max_active == 1                       # but never two commands on one vehicle
        assert d.calls == ["move_to_position", "rotate_to_yaw"]
    for i, vid in enumerate(ids):                      # each vehicle got only its own command
        w = adapter.worlds[vid]
        assert (w.n, w.e, w.alt) == (10.0 * i, 5.0, 5.0 + 2 * i)


def test_command_for_one_vehicle_never_touches_another():
    adapter = MultiAdapter(["Drone1", "Drone2"])
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    run(ex.execute(ProposedAction(vehicle_id="Drone2", action_type=ActionType.ROTATE_TO_HEADING,
                                  heading_deg=45.0, reason="x")))
    assert adapter.drones["Drone1"].calls == [] and adapter.drones["Drone2"].calls == ["rotate_to_yaw"]


# ------------------------------------------------------------ corrections (live Pass 5.2 finding)
def test_overshoot_then_stopped_is_corrected_once():
    """Live run: the move task finished, the drone coasted ~2.3 m past and held there."""
    world, drone, ex = setup()
    world.misses = [(-1.8, 1.4, 0.0)]          # first attempt ends 2.3 m off, then stops
    r = run(ex.execute(move(0.0, -3.0, 6.0, speed=3.0)))
    assert r.status == CommandStatus.SUCCEEDED, r.reason
    assert r.corrections == 1 and "after 1 correction" in r.reason
    assert drone.calls == ["move_to_position", "move_to_position"]
    assert drone.move_speeds == [3.0, FAST.correction_speed_mps]   # correction is slow
    assert drone.last_move[:3] == (0.0, -3.0, -6.0)                # same target re-sent


def test_altitude_error_after_stop_is_corrected_at_original_hold_point():
    """Live run: Drone2 stopped 0.5 m high."""
    world, drone, ex = setup()
    world.misses = [(0.0, 0.0, 0.5)]
    r = run(ex.change_altitude("Drone1", 8.0))
    assert r.status == CommandStatus.SUCCEEDED and r.corrections == 1
    assert drone.last_move[:3] == (-1.0, 8.0, -8.0)


def test_corrections_are_bounded_then_time_out_and_hover():
    world, drone, ex = setup()
    world.misses = [(2.0, 0.0, 0.0)] * 10     # never gets there
    r = run(ex.execute(move()))
    assert r.status == CommandStatus.TIMED_OUT
    assert r.corrections == FAST.max_corrections == 2
    assert "after 2 corrections" in r.reason
    assert drone.calls == ["move_to_position"] * 3 + ["hover"]


def test_no_correction_while_still_moving():
    world, drone, ex = setup()
    world.vn = 2.6                             # off target but NOT stopped
    r = run(ex.execute(move()))
    assert r.status == CommandStatus.TIMED_OUT and r.corrections == 0
    assert drone.calls == ["move_to_position", "hover"]


def test_rotate_never_issues_corrections():
    world, drone, ex = setup()
    world.stuck = True
    r = run(ex.execute(rotate(45.0)))
    assert r.status == CommandStatus.TIMED_OUT and r.corrections == 0
    assert drone.calls == ["rotate_to_yaw", "hover"]


def test_correction_command_failure_fails_and_hovers():
    world, drone, ex = setup()
    world.misses = [(2.0, 0.0, 0.0)]
    original = drone._command
    count = {"n": 0}

    async def second_move_fails(name, apply):
        if name == "move_to_position":
            count["n"] += 1
            if count["n"] == 2:
                drone.task_raises.add("move_to_position")
        return await original(name, apply)
    drone._command = second_move_fails
    r = run(ex.execute(move()))
    assert r.status == CommandStatus.FAILED and "correction task failed" in r.reason
    assert r.corrections == 1 and r.fallback_applied == "hover"


# ------------------------------------------------------------ unexpected errors after dispatch
def test_unexpected_error_in_settle_check_fails_and_hovers():
    """Gap found reviewing Pass 5: a bug after dispatch used to escape with no hover/result."""
    world, drone, ex = setup()

    def broken_settle(snap, action, start):
        raise ZeroDivisionError("bug in a completion check")
    ex._settle_rotate = broken_settle
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.FAILED and r.sent_to_simulator
    assert r.reason == "unexpected executor error: ZeroDivisionError: bug in a completion check"
    assert r.fallback_applied == "hover"
    assert drone.calls == ["rotate_to_yaw", "hover"]


def test_unexpected_error_while_waiting_for_task_fails_and_hovers():
    world, drone, ex = setup(rotate_task_timeout_s=5.0)
    drone.task_hangs.add("rotate_to_yaw")
    calls = {"n": 0}
    original = ex._check_telemetry

    def flaky(action, state, when):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyError("pose")             # not one of the executor's expected failures
        return original(action, state, when)
    ex._check_telemetry = flaky
    r = run(ex.execute(rotate()))
    assert r.status == CommandStatus.FAILED and "KeyError" in r.reason
    assert drone.calls == ["rotate_to_yaw", "hover"]
    assert r.elapsed_s < 1.0
