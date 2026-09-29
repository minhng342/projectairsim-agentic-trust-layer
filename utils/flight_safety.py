"""
Best-effort fail-safe for any script (and later the executor) that arms a drone.

Call safe_shutdown() from a `finally:` block BEFORE disconnecting. Every step
is attempted independently, so one failure never skips the rest:

    1. hover            (only if not grounded)
    2. land             (only if not grounded; bounded by a timeout)
    3. disarm
    4. disable API control

Pass the adapter when you have one: its operational ground_state recognizes a
drone that has already touched down, even while Project AirSim's raw
landed_state still reports FLYING (in the Pass 4.2 run it only changed on
disarm). Hover/land is skipped ONLY for a VALID snapshot that says GROUNDED;
anything uncertain is treated as airborne. Without an adapter, the raw
landed_state is used and an already-landed drone may be hovered and landed a
second time (harmless, but slow).

The caller still disconnects afterwards.
"""
import asyncio


async def _run_task(name, start, timeout_s, log):
    try:
        task = await start()
        await asyncio.wait_for(task, timeout_s)
        log(f"[safe_shutdown] {name}: ok")
        return True
    except Exception as err:
        log(f"[safe_shutdown] {name}: FAILED ({type(err).__name__}: {err})")
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


async def safe_shutdown(drone, log=print, adapter=None, vehicle_id: str | None = None,
                        hover_timeout_s: float = 5.0, land_timeout_s: float = 120.0) -> None:
    if drone is None:
        return
    grounded = is_grounded(drone, adapter, vehicle_id)
    if grounded is True:
        log("[safe_shutdown] already grounded: skipping hover and land")
    else:  # airborne or unknown: assume airborne
        await _run_task("hover", drone.hover_async, hover_timeout_s, log)
        await _run_task("land", drone.land_async, land_timeout_s, log)
    for name, fn in (("disarm", drone.disarm), ("disable_api_control", drone.disable_api_control)):
        try:
            fn()
            log(f"[safe_shutdown] {name}: ok")
        except Exception as err:
            log(f"[safe_shutdown] {name}: FAILED ({type(err).__name__}: {err})")
