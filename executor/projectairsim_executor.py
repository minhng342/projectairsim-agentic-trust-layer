"""
Deterministic command executor for Project AirSim (Pass 5).

    ProposedAction -> [risk gate, later] -> ProjectAirSimExecutor.execute() -> adapter / drone

Every command follows the same lifecycle:

    1. PREFLIGHT   fresh snapshot must be VALID and AIRBORNE; otherwise REFUSED
                   and nothing is sent to the simulator.
    2. DISPATCH    invoke the Project AirSim command (sent_to_simulator = True).
    3. TASK WAIT   wait for the simulator's task, bounded by a host-time deadline,
                   polling telemetry (~10 Hz) so telemetry loss is caught mid-command.
    4. SETTLE      wait until the completion predicate holds continuously for its
                   dwell (SIM time), bounded by a second host-time deadline.
                   Position commands only: if the vehicle has STOPPED outside
                   tolerance, re-send the target slowly (bounded corrections).
    5. RESULT      CommandResult. On any failure after dispatch, hover first.

Outcome mapping
    preflight telemetry bad / grounded / unsupported   -> REFUSED   (sent=False)
    simulator/API exception                           -> FAILED    (sent=True), hover
    telemetry invalid, stale or unavailable mid-command -> FAILED    (sent=True), hover
    task or settle deadline exceeded                   -> TIMED_OUT (sent=True), hover
    asyncio.CancelledError after dispatch              -> shielded, bounded hover, re-raise
    any other unexpected exception after dispatch      -> FAILED (sent=True), hover

Project AirSim detail that matters here: cancelling the client-side asyncio Task
does NOT stop the command inside the simulator. The fallback therefore always
sends an explicit hover (which supersedes the running command), and every
command is sent with its own simulator-side timeout_sec as a backstop.

Commands for one vehicle never overlap: one asyncio.Lock per vehicle.
Policy limits (max altitude, speed, geofence) are NOT enforced here; that is the
risk gate's job. Takeoff is NOT implicit in any command.
"""
import asyncio
import math
import time
import traceback
from dataclasses import dataclass, field

from executor.completion import (AltitudeTolerance, Deadline, DwellTracker, HeadingTolerance,
                                 PositionTolerance, altitude_settled, heading_settled,
                                 position_settled)
from models.action import ActionType, CommandResult, CommandStatus, ProposedAction
from models.telemetry import GroundState, TelemetrySnapshot, ValidationStatus


@dataclass(frozen=True)
class ExecutorConfig:
    poll_interval_s: float = 0.1                 # ~10 Hz telemetry polling
    invoke_timeout_s: float = 5.0                # sending the request itself
    rotate_task_timeout_s: float = 20.0
    altitude_task_margin_s: float = 10.0         # task timeout = |dz| / speed + margin
    altitude_default_speed_mps: float = 1.5
    settle_timeout_s: float = 15.0
    hover_timeout_s: float = 5.0
    rotate_margin_deg: float = 2.0               # simulator-side "done" margin (< 3 deg tolerance)
    move_default_speed_mps: float = 3.0
    move_task_margin_s: float = 10.0             # task timeout = distance / speed + margin
    # Corrections (position commands only). Live Pass 5.2: move_to_position's task
    # finished while the drone still carried momentum; it overshot ~2.3 m and the
    # controller then held THAT spot. If the drone has stopped outside tolerance,
    # re-send the same target slowly, a bounded number of times.
    max_corrections: int = 2
    correction_stall_s: float = 1.0              # stopped-but-off for this long (sim time)
    correction_max_speed_mps: float = 0.35       # "stopped" means total speed <= this
    correction_speed_mps: float = 1.0
    heading: HeadingTolerance = field(default_factory=HeadingTolerance)
    altitude: AltitudeTolerance = field(default_factory=AltitudeTolerance)
    position: PositionTolerance = field(default_factory=PositionTolerance)


class _Abort(Exception):
    """Internal: stop the command with this status/reason and apply the fallback."""

    def __init__(self, status: CommandStatus, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


class ProjectAirSimExecutor:
    SUPPORTED = (ActionType.ROTATE_TO_HEADING, ActionType.CHANGE_ALTITUDE,
                 ActionType.MOVE_TO_POSITION)

    def __init__(self, adapter, config: ExecutorConfig = ExecutorConfig(),
                 clock=time.monotonic, sleep=asyncio.sleep, log=print):
        self.adapter = adapter
        self.config = config
        self._clock = clock
        self._sleep = sleep
        self._log = log
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------ public API
    async def execute(self, action: ProposedAction) -> CommandResult:
        if action.action_type == ActionType.ROTATE_TO_HEADING:
            return await self._run(action, self._start_rotate, self._settle_rotate,
                                   self.config.rotate_task_timeout_s)
        if action.action_type == ActionType.CHANGE_ALTITUDE:
            return await self._run(action, self._start_altitude, self._settle_altitude,
                                   self._altitude_task_timeout, self._correct_altitude)
        if action.action_type == ActionType.MOVE_TO_POSITION:
            return await self._run(action, self._start_move, self._settle_move,
                                   self._move_task_timeout, self._correct_move)
        return self._result(action, CommandStatus.REFUSED,
                            f"{action.action_type.value} is not supported by the executor yet",
                            started=self._clock())

    async def rotate_to_heading(self, vehicle_id: str, heading_deg: float,
                                reason: str = "direct rotate_to_heading call") -> CommandResult:
        return await self.execute(ProposedAction(
            vehicle_id=vehicle_id, action_type=ActionType.ROTATE_TO_HEADING,
            heading_deg=heading_deg, reason=reason))

    async def change_altitude(self, vehicle_id: str, altitude_m: float,
                              speed_mps: float | None = None,
                              reason: str = "direct change_altitude call") -> CommandResult:
        return await self.execute(ProposedAction(
            vehicle_id=vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
            altitude_m=altitude_m, speed_mps=speed_mps, reason=reason))

    async def move_to_position(self, vehicle_id: str, north_m: float, east_m: float,
                               altitude_m: float, speed_mps: float | None = None,
                               reason: str = "direct move_to_position call") -> CommandResult:
        return await self.execute(ProposedAction(
            vehicle_id=vehicle_id, action_type=ActionType.MOVE_TO_POSITION,
            north_m=north_m, east_m=east_m, altitude_m=altitude_m, speed_mps=speed_mps,
            reason=reason))

    # ------------------------------------------------------------ per-command pieces
    async def _start_rotate(self, drone, action, start):
        yaw = math.radians(((action.heading_deg + 180.0) % 360.0) - 180.0)  # (-pi, pi]
        return await drone.rotate_to_yaw_async(
            yaw=yaw, timeout_sec=self.config.rotate_task_timeout_s,
            margin=math.radians(self.config.rotate_margin_deg))

    def _settle_rotate(self, snap, action, start):
        return heading_settled(snap, action.heading_deg, start.position_ned_m.x,
                               start.position_ned_m.y, self.config.heading)

    def _altitude_speed(self, action) -> float:
        return action.speed_mps or self.config.altitude_default_speed_mps

    def _altitude_task_timeout(self, action, start) -> float:
        dz = abs(action.altitude_m - start.altitude_local_m)
        return dz / self._altitude_speed(action) + self.config.altitude_task_margin_s

    async def _start_altitude(self, drone, action, start):
        # Hold the north/east captured BEFORE dispatch.
        return await drone.move_to_position_async(
            north=start.position_ned_m.x, east=start.position_ned_m.y,
            down=-action.altitude_m, velocity=self._altitude_speed(action),
            timeout_sec=self._altitude_task_timeout(action, start))

    def _settle_altitude(self, snap, action, start):
        return altitude_settled(snap, action.altitude_m, start.position_ned_m.x,
                                start.position_ned_m.y, self.config.altitude)

    def _move_speed(self, action) -> float:
        return action.speed_mps or self.config.move_default_speed_mps

    def _move_task_timeout(self, action, start) -> float:
        dist = math.sqrt((action.north_m - start.position_ned_m.x) ** 2
                         + (action.east_m - start.position_ned_m.y) ** 2
                         + (action.altitude_m - start.altitude_local_m) ** 2)
        return dist / self._move_speed(action) + self.config.move_task_margin_s

    async def _start_move(self, drone, action, start):
        return await drone.move_to_position_async(
            north=action.north_m, east=action.east_m, down=-action.altitude_m,
            velocity=self._move_speed(action),
            timeout_sec=self._move_task_timeout(action, start))

    def _correction_timeout(self, snap, n, e, alt) -> float:
        dist = math.sqrt((n - snap.position_ned_m.x) ** 2 + (e - snap.position_ned_m.y) ** 2
                         + (alt - snap.altitude_local_m) ** 2)
        return dist / self.config.correction_speed_mps + self.config.move_task_margin_s

    async def _correct_move(self, drone, action, start, now):
        speed = min(self.config.correction_speed_mps, self._move_speed(action))
        return await drone.move_to_position_async(
            north=action.north_m, east=action.east_m, down=-action.altitude_m, velocity=speed,
            timeout_sec=self._correction_timeout(now, action.north_m, action.east_m, action.altitude_m))

    async def _correct_altitude(self, drone, action, start, now):
        speed = min(self.config.correction_speed_mps, self._altitude_speed(action))
        n, e = start.position_ned_m.x, start.position_ned_m.y   # still the ORIGINAL hold point
        return await drone.move_to_position_async(
            north=n, east=e, down=-action.altitude_m, velocity=speed,
            timeout_sec=self._correction_timeout(now, n, e, action.altitude_m))

    def _settle_move(self, snap, action, start):
        return position_settled(snap, action.north_m, action.east_m, action.altitude_m,
                                self.config.position)

    # ------------------------------------------------------------ lifecycle
    def _lock(self, vehicle_id: str) -> asyncio.Lock:
        if vehicle_id not in self._locks:
            self._locks[vehicle_id] = asyncio.Lock()
        return self._locks[vehicle_id]

    def _result(self, action, status, reason, started, **kw) -> CommandResult:
        return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                             action_type=action.action_type, status=status, reason=reason,
                             elapsed_s=round(self._clock() - started, 3), **kw)

    def _snapshot(self, vehicle_id: str) -> TelemetrySnapshot:
        return self.adapter.get_snapshot(vehicle_id)

    async def _run(self, action, start_cmd, settle, task_timeout, correct_cmd=None) -> CommandResult:
        vid = action.vehicle_id
        async with self._lock(vid):
            started = self._clock()

            # 1. PREFLIGHT: nothing is sent unless this passes.
            try:
                if vid not in getattr(self.adapter, "vehicle_ids", [vid]):
                    raise ValueError(f"unknown vehicle {vid!r}")
                drone = self.adapter.drone(vid)
                start = self._snapshot(vid)
            except Exception as err:
                return self._result(action, CommandStatus.REFUSED,
                                    f"preflight: telemetry unavailable ({type(err).__name__}: {err})",
                                    started)
            if start.validation_status != ValidationStatus.VALID:
                return self._result(action, CommandStatus.REFUSED,
                                    f"preflight: telemetry {start.validation_status.value}: "
                                    f"{'; '.join(start.validation_errors)}",
                                    started, start_snapshot=start, final_snapshot=start)
            if start.ground_state != GroundState.AIRBORNE:
                return self._result(action, CommandStatus.REFUSED,
                                    f"preflight: vehicle is {start.ground_state.value} "
                                    f"({start.ground_state_basis}); command requires airborne",
                                    started, start_snapshot=start, final_snapshot=start)

            timeout_s = task_timeout(action, start) if callable(task_timeout) else task_timeout
            state = {"sent": False, "task": None, "last": start, "corrections": 0}
            try:
                try:
                    await self._dispatch_and_wait(drone, action, start, start_cmd, settle,
                                                  timeout_s, state, correct_cmd)
                    n = state["corrections"]
                    reason = "settled within tolerance" + (
                        f" after {n} correction{'s' if n > 1 else ''}" if n else "")
                    return self._result(action, CommandStatus.SUCCEEDED, reason,
                                        started, sent_to_simulator=True, corrections=n,
                                        start_snapshot=start, final_snapshot=state["last"])
                except _Abort as abort:
                    self._cancel_task(state)
                    fallback = await self._fallback(drone, state["last"])
                    return self._result(action, abort.status, abort.reason, started,
                                        sent_to_simulator=state["sent"], fallback_applied=fallback,
                                        corrections=state["corrections"],
                                        start_snapshot=start, final_snapshot=state["last"])
                except Exception as err:
                    # Anything unexpected after dispatch (a bug in a settle check, a malformed
                    # snapshot, a model error building the result) must still end in the
                    # fallback and a structured result, never a drone left on its last command.
                    self._cancel_task(state)
                    self._log(f"[executor] {action.vehicle_id} {action.action_type.value}: "
                              f"unexpected {type(err).__name__}: {err}\n{traceback.format_exc()}")
                    fallback = await self._fallback(drone, state["last"])
                    return self._result(action, CommandStatus.FAILED,
                                        f"unexpected executor error: {type(err).__name__}: {err}",
                                        started, sent_to_simulator=state["sent"],
                                        fallback_applied=fallback, corrections=state["corrections"],
                                        start_snapshot=start, final_snapshot=state["last"])
            except asyncio.CancelledError:
                self._cancel_task(state)
                if state["sent"]:
                    self._log(f"[executor] {action.action_type.value} cancelled: hovering")
                    hover = asyncio.ensure_future(self._hover(drone))
                    try:
                        await asyncio.shield(hover)   # a second cancel can't stop the hover
                    except asyncio.CancelledError:
                        pass
                raise

    async def _send(self, cmd, drone, action, *args):
        cfg = self.config
        try:
            return await asyncio.wait_for(cmd(drone, action, *args), cfg.invoke_timeout_s)
        except asyncio.TimeoutError:
            raise _Abort(CommandStatus.TIMED_OUT,
                         f"simulator did not accept the command within {cfg.invoke_timeout_s:.1f} s")
        except Exception as err:
            raise _Abort(CommandStatus.FAILED, f"command invocation failed: {type(err).__name__}: {err}")

    async def _await_task(self, task, timeout_s, action, state, what="simulator task"):
        """Wait for a simulator task (host-time deadline), watching telemetry throughout."""
        cfg = self.config
        deadline = Deadline(timeout_s, clock=self._clock)
        while True:
            done, _ = await asyncio.wait({task}, timeout=cfg.poll_interval_s)
            if done:
                try:
                    task.result()
                except Exception as err:
                    raise _Abort(CommandStatus.FAILED, f"{what} failed: {type(err).__name__}: {err}")
                return
            if deadline.expired:
                raise _Abort(CommandStatus.TIMED_OUT, f"{what} did not finish within {timeout_s:.1f} s")
            self._check_telemetry(action, state, "while the command was running")

    async def _dispatch_and_wait(self, drone, action, start, start_cmd, settle, timeout_s, state,
                                 correct_cmd=None):
        cfg = self.config

        # 2. DISPATCH
        state["sent"] = True
        state["task"] = await self._send(start_cmd, drone, action, start)

        # 3. TASK WAIT
        await self._await_task(state["task"], timeout_s, action, state)

        # 4. SETTLE (host-time deadline, sim-time dwell, optional bounded corrections)
        dwell = DwellTracker(settle_dwell(action, cfg))
        stall = DwellTracker(cfg.correction_stall_s)
        deadline = Deadline(cfg.settle_timeout_s, clock=self._clock)
        detail = ""
        while True:
            snap = self._check_telemetry(action, state, "while settling")
            ok, detail = settle(snap, action, start)
            if dwell.update(snap.sim_time_ns, ok):
                return
            stopped_off_target = (not ok) and _total_speed(snap) <= cfg.correction_max_speed_mps
            if (correct_cmd is not None and state["corrections"] < cfg.max_corrections
                    and stall.update(snap.sim_time_ns, stopped_off_target)):
                state["corrections"] += 1
                self._log(f"[executor] {action.vehicle_id} {action.action_type.value}: stopped "
                          f"off target ({detail}); correction {state['corrections']}/{cfg.max_corrections}")
                state["task"] = await self._send(correct_cmd, drone, action, start, snap)
                await self._await_task(state["task"], self._correction_timeout(
                    snap, *_target(action, start)), action, state, what="correction task")
                dwell = DwellTracker(settle_dwell(action, cfg))
                stall = DwellTracker(cfg.correction_stall_s)
                deadline = Deadline(cfg.settle_timeout_s, clock=self._clock)
                continue
            if deadline.expired:
                n = state["corrections"]
                raise _Abort(CommandStatus.TIMED_OUT,
                             f"did not settle within {cfg.settle_timeout_s:.1f} s"
                             + (f" after {n} correction{'s' if n > 1 else ''}" if n else "")
                             + f" (held {dwell.held_s:.2f} s; last: {detail})")
            await self._sleep(cfg.poll_interval_s)

    def _check_telemetry(self, action, state, when: str) -> TelemetrySnapshot:
        try:
            snap = self._snapshot(action.vehicle_id)
        except Exception as err:
            raise _Abort(CommandStatus.FAILED,
                         f"telemetry unavailable {when} ({type(err).__name__}: {err})")
        state["last"] = snap
        if snap.validation_status != ValidationStatus.VALID:
            raise _Abort(CommandStatus.FAILED,
                         f"telemetry {snap.validation_status.value} {when}: "
                         f"{'; '.join(snap.validation_errors)}")
        if snap.ground_state == GroundState.GROUNDED:
            raise _Abort(CommandStatus.FAILED, f"vehicle became grounded {when}")
        return snap

    # ------------------------------------------------------------ fallback
    @staticmethod
    def _cancel_task(state) -> None:
        task = state.get("task")
        if task is not None and not task.done():
            task.cancel()

    async def _hover(self, drone) -> str:
        t = self.config.hover_timeout_s
        try:
            task = await asyncio.wait_for(drone.hover_async(), t)
            await asyncio.wait_for(task, t)
            return "hover"
        except Exception as err:
            self._log(f"[executor] hover fallback failed: {type(err).__name__}: {err}")
            return f"hover failed: {type(err).__name__}: {err}"

    async def _fallback(self, drone, last: TelemetrySnapshot | None) -> str:
        # Never try to "hover" a vehicle we are confident is on the ground.
        if (last is not None and last.validation_status == ValidationStatus.VALID
                and last.ground_state == GroundState.GROUNDED):
            return "none (vehicle grounded)"
        return await self._hover(drone)


def settle_dwell(action: ProposedAction, cfg: ExecutorConfig) -> float:
    return {ActionType.ROTATE_TO_HEADING: cfg.heading.dwell_s,
            ActionType.CHANGE_ALTITUDE: cfg.altitude.dwell_s,
            ActionType.MOVE_TO_POSITION: cfg.position.dwell_s}[action.action_type]


def _total_speed(s: TelemetrySnapshot) -> float:
    v = s.velocity_ned_mps
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _target(action: ProposedAction, start: TelemetrySnapshot) -> tuple[float, float, float]:
    """(north, east, altitude) a position command is trying to reach."""
    if action.action_type == ActionType.MOVE_TO_POSITION:
        return action.north_m, action.east_m, action.altitude_m
    return start.position_ned_m.x, start.position_ned_m.y, action.altitude_m
