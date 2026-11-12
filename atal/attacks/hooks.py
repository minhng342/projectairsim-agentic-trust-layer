"""
Attack hooks for the observation channel (in-transit modification).

A hook is any callable  hook(payload: dict, ctx: HookContext) -> dict | None
    - return a (possibly modified) payload to deliver it,
    - return None to DROP the message (loss / denial of service).
It receives a private deep copy, so it cannot alter the clean record kept by the
evaluation side. Hooks run in order, BEFORE trust assessment, and they can only
change content: the receiver-side fields (observation_id, channel, sequence,
receipt time) are stamped by the feed after the hooks run.

Hooks never touch the simulator: the drone's real state is unchanged, only what
its agent is told. These few are for tests and demonstrations; research attack
scenarios belong in their own module.
"""
import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class HookContext:
    vehicle_id: str
    sequence: int
    received_host_s: float


@dataclass
class PositionOffset:
    """Shift the reported position by (north, east, up) meters from `after_sequence` on."""
    north_m: float = 0.0
    east_m: float = 0.0
    up_m: float = 0.0
    after_sequence: int = 0
    name: str = "position_offset"

    def __call__(self, payload: dict, ctx: HookContext) -> dict | None:
        if ctx.sequence < self.after_sequence:
            return payload
        p = payload.get("position_ned_m")
        if isinstance(p, dict):
            p["x"] = p.get("x", 0.0) + self.north_m
            p["y"] = p.get("y", 0.0) + self.east_m
            p["z"] = p.get("z", 0.0) - self.up_m
        if "altitude_local_m" in payload:
            payload["altitude_local_m"] += self.up_m
        return payload


@dataclass
class Replay:
    """From `after_sequence` on, keep delivering the last message seen before it (frozen feed)."""
    after_sequence: int = 0
    name: str = "replay"
    _held: dict | None = None

    def __call__(self, payload: dict, ctx: HookContext) -> dict | None:
        if ctx.sequence < self.after_sequence or self._held is None:
            self._held = copy.deepcopy(payload)
            return payload
        return copy.deepcopy(self._held)


@dataclass
class Drop:
    """Drop every message from `after_sequence` on."""
    after_sequence: int = 0
    name: str = "drop"

    def __call__(self, payload: dict, ctx: HookContext) -> dict | None:
        return None if ctx.sequence >= self.after_sequence else payload
