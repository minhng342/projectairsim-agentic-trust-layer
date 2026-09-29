"""
Best-effort fail-safe for any script (and later the executor) that arms a drone.

Call safe_shutdown() from a `finally:` block BEFORE disconnecting. Every step
is attempted independently, so one failure never skips the rest:

    1. hover            (only if flying)
    2. land             (only if flying; bounded by a timeout)
    3. disarm
    4. disable API control

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


def _is_landed(drone) -> bool | None:
    try:
        return int(drone.get_landed_state()) == 0  # LandedState.LANDED
    except Exception:
        return None


async def safe_shutdown(drone, log=print, hover_timeout_s: float = 5.0,
                        land_timeout_s: float = 120.0) -> None:
    if drone is None:
        return
    landed = _is_landed(drone)
    if landed is not True:  # flying or unknown: assume airborne
        await _run_task("hover", drone.hover_async, hover_timeout_s, log)
        await _run_task("land", drone.land_async, land_timeout_s, log)
    for name, fn in (("disarm", drone.disarm), ("disable_api_control", drone.disable_api_control)):
        try:
            fn()
            log(f"[safe_shutdown] {name}: ok")
        except Exception as err:
            log(f"[safe_shutdown] {name}: FAILED ({type(err).__name__}: {err})")
