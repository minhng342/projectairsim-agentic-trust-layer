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
    5. RESULT      CommandResult. On any failure after dispatch, hover first.

Outcome mapping
    preflight telemetry bad / grounded / unsupported   -> REFUSED   (sent=False)
    simulator/API exception                           -> FAILED    (sent=True), hover
    telemetry invalid, stale or unavailable mid-command -> FAILED    (sent=True), hover
    task or settle deadline exceeded                   -> TIMED_OUT (sent=True), hover
    asyncio.CancelledError after dispatch              -> shielded, bounded hover, re-raise

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
from dataclasses import dataclass, field

from executor.completion import (AltitudeTolerance, Deadline, DwellTracker, HeadingTolerance,
                                 altitude_settled, heading_settled)
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
    heading: HeadingTolerance = field(default_factory=HeadingTolerance)
    altitude: AltitudeTolerance = field(default_factory=AltitudeTolerance)


class _Abort(Exception):
    """Internal: stop the command with this status/reason and apply the fallback."""

    def __init__(self, status: CommandStatus, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


class ProjectAirSimExecutor:
    SUPPORTED = (ActionType.ROTATE_TO_HEADING, ActionType.CHANGE_ALTITUDE)

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
                                   self._altitude_task_timeout)
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

    async def _run(self, action, start_cmd, settle, task_timeout) -> CommandResult:
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
            state = {"sent": False, "task": None, "last": start}
            try:
                try:
                    await self._dispatch_and_wait(drone, action, start, start_cmd, settle,
                                                  timeout_s, state)
                    return self._result(action, CommandStatus.SUCCEEDED, "settled within tolerance",
                                        started, sent_to_simulator=True,
                                        start_snapshot=start, final_snapshot=state["last"])
                except _Abort as abort:
                    self._cancel_task(state)
                    fallback = await self._fallback(drone, state["last"])
                    return self._result(action, abort.status, abort.reason, started,
                                        sent_to_simulator=state["sent"], fallback_applied=fallback,
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

    async def _dispatch_and_wait(self, drone, action, start, start_cmd, settle, timeout_s, state):
        cfg = self.config

        # 2. DISPATCH
        state["sent"] = True
        try:
            task = await asyncio.wait_for(start_cmd(drone, action, start), cfg.invoke_timeout_s)
        except asyncio.TimeoutError:
            raise _Abort(CommandStatus.TIMED_OUT,
                         f"simulator did not accept the command within {cfg.invoke_timeout_s:.1f} s")
        except Exception as err:
            raise _Abort(CommandStatus.FAILED, f"command invocation failed: {type(err).__name__}: {err}")
        state["task"] = task

        # 3. TASK WAIT (host-time deadline; telemetry watched throughout)
        deadline = Deadline(timeout_s, clock=self._clock)
        while True:
            done, _ = await asyncio.wait({task}, timeout=cfg.poll_interval_s)
            if done:
                try:
                    task.result()
                except Exception as err:
                    raise _Abort(CommandStatus.FAILED,
                                 f"simulator task failed: {type(err).__name__}: {err}")
                break
            if deadline.expired:
                raise _Abort(CommandStatus.TIMED_OUT,
                             f"simulator task did not finish within {timeout_s:.1f} s")
            self._check_telemetry(action, state, "while the command was running")

        # 4. SETTLE (host-time deadline, sim-time dwell)
        dwell = DwellTracker(settle_dwell(action, cfg))
        deadline = Deadline(cfg.settle_timeout_s, clock=self._clock)
        detail = ""
        while True:
            snap = self._check_telemetry(action, state, "while settling")
            ok, detail = settle(snap, action, start)
            if dwell.update(snap.sim_time_ns, ok):
                return
            if deadline.expired:
                raise _Abort(CommandStatus.TIMED_OUT,
                             f"did not settle within {cfg.settle_timeout_s:.1f} s "
                             f"(held {dwell.held_s:.2f} s; last: {detail})")
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
            ActionType.CHANGE_ALTITUDE: cfg.altitude.dwell_s}[action.action_type]
