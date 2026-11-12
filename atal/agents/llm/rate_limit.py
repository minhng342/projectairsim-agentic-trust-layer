"""Request pacing for a model API with a per-minute quota: how many requests per minute a
model allows (rpm_for) and a limiter that spaces calls evenly (RateLimiter)."""
import asyncio
import os
import time

# Free-tier requests per minute seen in AI Studio (2026-10-14). We pace at ~80% of these.
# GEMINI_RPM overrides for every model (e.g. after enabling billing).
FREE_TIER_RPM = {"gemini-3.6-flash": 5, "gemini-3.5-flash-lite": 15}
UNKNOWN_MODEL_RPM = 5
RPM_HEADROOM = 0.8


def rpm_for(model: str) -> float:
    env = os.environ.get("GEMINI_RPM")
    if env:
        return float(env)
    return max(1.0, FREE_TIER_RPM.get(model, UNKNOWN_MODEL_RPM) * RPM_HEADROOM)


class RateLimiter:
    """Client-side request pacing for one API quota (share ONE limiter between all
    sessions/pilots that use the same key and model).

    Even spacing, not a burst window: calls are at least 60/rpm seconds apart. A
    sliding window would let a fast model burn the whole minute's quota in a few
    seconds and then stall for up to a minute, which would age every observation
    the model is deciding from. Even spacing bounds each wait at 60/rpm."""

    def __init__(self, rpm: float, clock=None, sleep=None):
        if not rpm > 0:
            raise ValueError("rpm must be > 0")
        self.rpm = rpm
        self.interval_s = 60.0 / rpm
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._next = None
        self._lock = None

    async def acquire(self) -> float:
        """Wait for this call's slot; returns how long it waited (seconds)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:              # concurrent pilots queue in order
            now = self._clock()
            wait = 0.0 if self._next is None else max(0.0, self._next - now)
            if wait > 0:
                await self._sleep(wait)
            self._next = max(now, self._next or now) + self.interval_s
            return wait
