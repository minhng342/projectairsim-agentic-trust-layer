"""
Running one coroutine per drone, safely.

asyncio.gather() does NOT cancel the other coroutines when one raises: they keep
running. If the caller then starts shutdown (hover/land) in a `finally:` block,
those still-running missions keep sending commands to the same drones, so two
controllers fight over each vehicle.

run_all_or_cancel() guarantees that when it returns or raises, every mission
has finished: on the first failure (or if the caller is cancelled) the rest are
cancelled and awaited. Cancelling an executor command triggers that command's
own shielded hover, so the drones are handed to shutdown holding position.
"""
import asyncio


async def run_all_or_cancel(coros) -> list:
    """Run coroutines concurrently. Returns their results in order.

    If any raises, the others are cancelled and awaited, then the first
    exception is re-raised. If the caller is cancelled, all are cancelled and
    awaited before the cancellation propagates.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    finally:
        pending = [t for t in tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            # Wait for their cleanup (e.g. executor hover) to finish. A second
            # cancellation of the caller must not abandon them half-way.
            await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
