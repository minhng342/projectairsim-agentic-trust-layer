"""
Best-effort fail-safe for any script (and later the executor) that arms a drone.

Call safe_shutdown() from a `finally:` block BEFORE disconnecting:

    already grounded?  -> disarm, release API control
    otherwise          -> hover, land, then CONFIRM grounded
        confirmed      -> disarm, release API control
        not confirmed  -> do NOT disarm; leave the vehicle armed under API
                          control and report its state as UNRESOLVED. It is
                          only holding position if the hover command took
                          effect; if communication failed, its state is unknown.

Disarming an airborne multirotor drops it. A forced fall during cleanup must
never be confused with an agent-caused failure in experiment results, so a
landing that failed, timed out or can't be confirmed never leads to a disarm.

Every simulator call is bounded end to end: the timeout covers sending the
request AND waiting for the resulting task (a stalled acknowledgement can't
hang shutdown).

Ground confirmation
- With an adapter: a VALID snapshot with ground_state GROUNDED, polled for up to
  confirm_timeout_s after landing (contact + stillness needs ~1 s). Stale,
  invalid or UNKNOWN never counts, and there is no fallback to the raw
  landed_state (which in the Pass 4.2 run only changed on disarm).
- Without an adapter: a land task that completed without error is accepted,
  because the raw landed_state cannot confirm touchdown before disarm.
"""
import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class ShutdownReport:
    grounded_confirmed: bool = False
    disarmed: bool = False
    api_released: bool = False
    steps: list[str] = field(default_factory=list)

    @property
    def unresolved(self) -> bool:
        return not self.grounded_confirmed


async def _bounded_task(name, start, timeout_s, log, report) -> bool:
    """Send the request and wait for its task; the whole thing is bounded."""
    async def send_and_wait():
        task = await start()
        await task
    try:
        await asyncio.wait_for(send_and_wait(), timeout_s)
        report.steps.append(f"{name}: ok")
        log(f"[safe_shutdown] {name}: ok")
        return True
    except Exception as err:  # includes asyncio.TimeoutError
        msg = f"{name}: FAILED ({type(err).__name__}: {err})"
        report.steps.append(msg)
        log(f"[safe_shutdown] {msg}")
        return False


def is_grounded(drone, adapter=None, vehicle_id: str | None = None) -> bool | None:
    """True only when we're confident the vehicle is on the ground.

    With an adapter: trusted only if the snapshot is VALID and ground_state is
    GROUNDED/AIRBORNE. Stale, invalid or UNKNOWN returns None (treated as
    airborne by safe_shutdown). It does NOT fall back to the raw landed_state,
    because that can be wrong exactly when the adapter is unsure.
    Without an adapter: raw landed_state (may lag after touchdown).
    """
    if adapter is not None and vehicle_id is not None:
        try:
            from models.telemetry import GroundState, ValidationStatus
            snap = adapter.get_snapshot(vehicle_id)
            if snap.validation_status != ValidationStatus.VALID:
                return None
            if snap.ground_state == GroundState.UNKNOWN:
                return None
            return snap.ground_state == GroundState.GROUNDED
        except Exception:
            return None
    try:
        return int(drone.get_landed_state()) == 0  # LandedState.LANDED
    except Exception:
        return None


async def _confirm_grounded(drone, adapter, vehicle_id, timeout_s, poll_s) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if is_grounded(drone, adapter, vehicle_id) is True:
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(poll_s)


def _release(drone, log, report) -> None:
    for name, fn, attr in (("disarm", drone.disarm, "disarmed"),
                           ("disable_api_control", drone.disable_api_control, "api_released")):
        try:
            fn()
            setattr(report, attr, True)
            report.steps.append(f"{name}: ok")
            log(f"[safe_shutdown] {name}: ok")
        except Exception as err:
            msg = f"{name}: FAILED ({type(err).__name__}: {err})"
            report.steps.append(msg)
            log(f"[safe_shutdown] {msg}")


async def safe_shutdown(drone, log=print, adapter=None, vehicle_id: str | None = None,
                        hover_timeout_s: float = 5.0, land_timeout_s: float = 120.0,
                        confirm_timeout_s: float = 5.0, confirm_poll_s: float = 0.1
                        ) -> ShutdownReport:
    report = ShutdownReport()
    if drone is None:
        return report

    if is_grounded(drone, adapter, vehicle_id) is True:
        log("[safe_shutdown] already grounded: skipping hover and land")
        report.steps.append("already grounded")
        report.grounded_confirmed = True
    else:  # airborne or unknown: assume airborne
        await _bounded_task("hover", drone.hover_async, hover_timeout_s, log, report)
        landed_ok = await _bounded_task("land", drone.land_async, land_timeout_s, log, report)
        if landed_ok:
            if adapter is not None and vehicle_id is not None:
                report.grounded_confirmed = await _confirm_grounded(
                    drone, adapter, vehicle_id, confirm_timeout_s, confirm_poll_s)
            else:
                report.grounded_confirmed = True  # see module docstring
        report.steps.append(f"grounded confirmed: {report.grounded_confirmed}")

    if report.grounded_confirmed:
        _release(drone, log, report)
    else:
        hovered = "hover: ok" in report.steps
        msg = ("UNRESOLVED: vehicle not confirmed on the ground; NOT disarming. "
               + ("Hover was accepted, so it should be holding position"
                  if hovered else "Hover was NOT confirmed, so its state is unknown")
               + "; check the simulator.")
        report.steps.append(msg)
        log(f"[safe_shutdown] {msg}")
    return report
