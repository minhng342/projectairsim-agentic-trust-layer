"""
Mission runner: flies one drone's inspection corridor and applies maneuvers to it.

    runner = MissionRunner(adapter, drone_mission, executor)
    port = ActionPort(vid, runner, feed, gate=...)      # the runner is the port's executor
    result = await runner.fly()                         # climb to the layer, fly the corridor

fly():   1. climb in place to the corridor's altitude layer (the executor's change_altitude,
            proven live), so drones leave their pads vertically, not sideways at low altitude
         2. fly the corridor polyline with the velocity PathFollower at cruise speed

execute(action) is what the action port calls (same contract as the executor), but a
maneuver is APPLIED, not run to completion: it changes how the drone flies its corridor
and returns at once, while the corridor flight carries on.

    maintain          no change; a recorded decision to carry on
    hold              brake and stay; with duration_s, resume automatically afterwards
    resume            continue the corridor after a hold
    change_speed      new cruise speed (speed_mps)
    change_altitude   move to another layer (altitude_m) while continuing along the corridor

Result codes (CommandResult.reason_code; the port maps them to the agent's feedback):
    applied            the maneuver took effect
    not_applicable     e.g. resume while not holding; nothing changed
    out_of_range       e.g. a speed the follower can't fly; nothing changed
    not_flying         the corridor flight hasn't started (climbing) or has failed
    mission_finished   the corridor is complete; no maneuvers accepted
    unsupported_action not a mission maneuver

Policy (corridor bounds, altitude band, speed limits) is NOT enforced here: that is the
action-risk gate's job, and whether a gate is installed is an experimental condition.
Physical limits of the follower (e.g. its max speed) are.
"""
import asyncio
import time
from enum import Enum

from executor.path_follower import FollowerConfig, FollowMode, FollowResult, PathFollower
from models.action import ActionType, CommandResult, CommandStatus, ProposedAction
from legacy.corridor.mission import DroneMission


class MissionPhase(str, Enum):
    READY = "ready"
    CLIMBING = "climbing"
    FLYING = "flying"
    FINISHED = "finished"
    FAILED = "failed"


class MissionRunner:
    SUPPORTED = (ActionType.MAINTAIN, ActionType.HOLD, ActionType.RESUME,
                 ActionType.CHANGE_SPEED, ActionType.CHANGE_ALTITUDE)

    def __init__(self, adapter, mission: DroneMission, executor,
                 follower_config: FollowerConfig = FollowerConfig(), on_event=None,
                 clock=time.monotonic, log=print):
        self.adapter, self.mission, self._executor = adapter, mission, executor
        self.vehicle_id = mission.vehicle_id
        self.follower_config = follower_config
        self._on_event, self._clock, self._log = on_event, clock, log
        self.phase = MissionPhase.READY
        self.follower: PathFollower | None = None
        self.result: FollowResult | None = None
        self._hold_timer: asyncio.Task | None = None
        self._hold_until: float | None = None
        self._hold_gen = 0

    # ------------------------------------------------------------ flight
    async def fly(self, timeout_s: float | None = None) -> FollowResult:
        c = self.mission.corridor
        self._set_phase(MissionPhase.CLIMBING, altitude_m=c.altitude_m)
        try:
            climb = await self._executor.change_altitude(
                self.vehicle_id, c.altitude_m, reason=f"mission {c.name}: climb to layer")
            if climb.status != CommandStatus.SUCCEEDED:
                return self._finish(FollowResult(FollowMode.FAILED,
                                                 f"climb to layer failed: {climb.reason}",
                                                 0.0, round(c.length_m, 2), climb.elapsed_s))
            self.follower = PathFollower(self.adapter, self.vehicle_id, c.waypoints, c.altitude_m,
                                         c.cruise_speed_mps, config=self.follower_config,
                                         on_event=self._on_event, log=self._log)
            self._set_phase(MissionPhase.FLYING, corridor=c.name, length_m=round(c.length_m, 2))
            return self._finish(await self.follower.run(timeout_s=timeout_s))
        finally:
            self._cancel_hold_timer()

    def _finish(self, result: FollowResult) -> FollowResult:
        self.result = result
        self._set_phase(MissionPhase.FINISHED if result.mode == FollowMode.DONE
                        else MissionPhase.FAILED, follower=result.mode.value, reason=result.reason)
        return result

    # ------------------------------------------------------------ maneuvers (port-facing)
    async def execute(self, action: ProposedAction) -> CommandResult:
        started = self._clock()
        f = self.follower

        def result(status, code, reason, sent=False):
            r = CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                              action_type=action.action_type, status=status, reason=reason,
                              reason_code=code, sent_to_simulator=sent,
                              final_snapshot=self._truth(),
                              elapsed_s=round(self._clock() - started, 3))
            self._emit("maneuver", request_id=action.request_id,
                       action_type=action.action_type.value, code=code, reason=reason,
                       progress_m=round(f.progress_m, 2) if f else None)
            return r

        ok, refused = CommandStatus.SUCCEEDED, CommandStatus.REFUSED
        if action.vehicle_id != self.vehicle_id:
            return result(refused, "not_applicable", f"runner flies {self.vehicle_id} only")
        if action.action_type not in self.SUPPORTED:
            return result(refused, "unsupported_action",
                          f"{action.action_type.value} is not a mission maneuver")
        if self.phase == MissionPhase.FINISHED:
            return result(refused, "mission_finished", "the corridor is complete")
        if self.phase != MissionPhase.FLYING or f is None or f.mode in (
                FollowMode.DONE, FollowMode.FAILED, FollowMode.STOPPED):
            return result(refused, "not_flying", f"mission phase {self.phase.value}")

        t = action.action_type
        if t == ActionType.MAINTAIN:
            return result(ok, "applied", "maintain: no change")
        if t == ActionType.HOLD:
            self._cancel_hold_timer()
            f.hold()
            if action.duration_s:
                self._hold_gen += 1
                self._hold_until = self._clock() + action.duration_s
                self._hold_timer = asyncio.ensure_future(
                    self._auto_resume(action.duration_s, self._hold_gen))
            return result(ok, "applied", "hold" + (f" for {action.duration_s:g} s"
                                                   if action.duration_s else " until resume"), True)
        if t == ActionType.RESUME:
            if f.mode != FollowMode.HOLD and not f._hold_requested:
                return result(refused, "not_applicable", "not holding")
            self._cancel_hold_timer()
            f.resume()
            return result(ok, "applied", "resume the corridor", True)
        if t == ActionType.CHANGE_SPEED:
            try:
                f.set_speed(action.speed_mps)
            except ValueError as err:
                return result(refused, "out_of_range", str(err))
            return result(ok, "applied", f"cruise speed {action.speed_mps:g} m/s", True)
        # CHANGE_ALTITUDE: a layer change while continuing along the corridor
        try:
            f.set_altitude(action.altitude_m)
        except ValueError as err:
            return result(refused, "out_of_range", str(err))
        return result(ok, "applied", f"layer {action.altitude_m:g} m", True)

    async def _auto_resume(self, duration_s: float, gen: int) -> None:
        await asyncio.sleep(duration_s)
        f = self.follower
        if gen == self._hold_gen and f is not None and (
                f.mode == FollowMode.HOLD or f._hold_requested):
            f.resume()
            self._hold_until = None
            self._emit("maneuver", action_type="resume", code="applied",
                       reason=f"automatic, after hold of {duration_s:g} s",
                       progress_m=round(f.progress_m, 2))

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer is not None and not self._hold_timer.done():
            self._hold_timer.cancel()
        self._hold_timer, self._hold_until = None, None

    # ------------------------------------------------------------ status
    def target_altitude_m(self) -> float | None:
        """The altitude this drone is going to (the Remote ID intent field), or None on the
        ground / after the mission."""
        if self.phase == MissionPhase.CLIMBING:
            return self.mission.corridor.altitude_m
        if self.phase == MissionPhase.FLYING and self.follower is not None:
            return self.follower.altitude_target_m
        return None

    def status(self) -> dict:
        """Mission progress (the runner's own bookkeeping, from the flight controller's
        ground-truth position: evaluation / a mission tool, not a sensor reading)."""
        c, f = self.mission.corridor, self.follower
        d = {"phase": self.phase.value, "corridor": c.name, "length_m": round(c.length_m, 2)}
        if f is not None:
            d.update(mode=f.mode.value, progress_m=round(f.progress_m, 2),
                     remaining_m=round(f.remaining_m, 2), cruise_speed_mps=f.speed_mps,
                     altitude_target_m=f.altitude_target_m)
            if self._hold_until is not None:
                d["hold_remaining_s"] = round(max(0.0, self._hold_until - self._clock()), 1)
        return d

    # ------------------------------------------------------------ plumbing
    def _truth(self):
        try:
            return self.adapter.get_snapshot(self.vehicle_id)
        except Exception:
            return None

    def _set_phase(self, phase: MissionPhase, **fields) -> None:
        self.phase = phase
        self._emit("mission_phase", phase=phase.value, **fields)

    def _emit(self, kind: str, **fields) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, {"vehicle_id": self.vehicle_id, **fields})
        except Exception as err:
            self._log(f"[mission] on_event({kind}) failed: {type(err).__name__}: {err}")
