"""
Running one coroutine per drone, safely.

asyncio.gather() does NOT cancel the other coroutines when one raises: they keep
running. If the caller then starts shutdown (hover/land) in a `finally:` block,
those still-running missions keep sending commands to the same drones, so two
controllers fight over each vehicle.

run_all_or_cancel() guarantees that when it returns or raises, every coroutine
has FINISHED, including its cleanup:
  - on the first failure, the rest are cancelled and awaited, then the error is raised;
  - if the caller is cancelled (even repeatedly, e.g. Ctrl+C twice), all are
    cancelled, their cleanup is awaited to completion, THEN the cancellation propagates.
Each child is cancelled exactly ONCE: a second cancel would interrupt the cleanup
it has already started (e.g. an executor hover).

asyncio.wait() is used instead of gather() on purpose: when the caller is
cancelled, gather() cancels its children itself, so they could be cancelled twice.
"""
import asyncio


async def _cancel_and_await(tasks) -> bool:
    """Cancel unfinished tasks once and wait until ALL have finished.

    Returns True if the caller was cancelled while waiting (the caller must then
    raise CancelledError). Repeated cancellation cannot cut the wait short.
    """
    pending = [t for t in tasks if not t.done()]
    for t in pending:
        t.cancel()
    caller_cancelled = False
    while pending:
        try:
            await asyncio.shield(asyncio.wait(pending))
        except asyncio.CancelledError:
            caller_cancelled = True        # keep waiting; propagate once cleanup is done
        pending = [t for t in pending if not t.done()]
    for t in tasks:                        # mark exceptions as retrieved (no asyncio warnings)
        if t.done() and not t.cancelled():
            t.exception()
    return caller_cancelled


async def run_all_or_cancel(coros) -> list:
    """Run coroutines concurrently. Returns their results in order.

    If any raises, the others are cancelled and awaited, then that exception is
    re-raised. If the caller is cancelled, all are cancelled and awaited before
    the cancellation propagates.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        await _cancel_and_await(tasks)
        raise
    failed = [t for t in tasks if t.done() and not t.cancelled() and t.exception() is not None]
    if failed:
        if await _cancel_and_await(tasks):
            raise asyncio.CancelledError()
        raise failed[0].exception()
    return [t.result() for t in tasks]
