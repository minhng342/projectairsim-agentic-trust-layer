"""
Pad-area deconfliction for the swarm (Phase 3): one drone at a time in the airspace over
the launch pads.

The team launches from pads a few metres apart and climbs and lands in place, each on its
own layer. Layers keep drones apart in cruise, but not around the pads: a drone coming
down from its layer crosses the layers below it, and a teammate flying home on one of
them can pass straight over its pad (offline no-radio run 15, 2026-11-05: 0.55 m).

    zone = PadZone(center_xy, radius_m)
    await zone.acquire("Drone2")     # before entering the zone on the way home
    ...fly in, descend, land...
    zone.release("Drone2")           # once landed (the runner, after shutdown)

The sector executor flies a return home in two legs when it has a zone: to an approach
point on the zone's edge at the drone's layer, then (holding the zone) to its pad. A drone
that finds the zone taken hovers at the approach point until the holder has landed. The
zone is held from the moment a drone enters it until it is on the ground, so a descent
never shares the zone with a transit. It is a local stand-in for a vertiport manager: the
runner owns it, the agents never see it, and it only orders landings (takeoff is the team
climbing together, already covered by the pad separation floor).

acquire(timeout_s) gives up waiting after timeout_s host seconds (returns False, recorded
as pad_zone_timeout) so a holder that never releases cannot strand the others in the air.
A False is never permission to go in: the sector executor then holds at the zone's edge
(pad_zone_timeout) and the runner lands that drone where it is, outside the zone, as a
recorded contingency landing. The runner releases the zone only once the holder's landing
is confirmed; an unconfirmed landing keeps it held (pad_zone_held_unconfirmed), so the
others land outside rather than descend next to a drone still in the air.
"""
import asyncio
import math
import time


class PadZone:
    def __init__(self, center_xy: tuple[float, float], radius_m: float, recorder=None,
                 sim_clock=None, clock=time.monotonic):
        self.center = (float(center_xy[0]), float(center_xy[1]))
        self.radius_m = float(radius_m)
        self._lock = asyncio.Lock()
        self.holder: str | None = None
        self._recorder, self._sim_clock, self._clock = recorder, sim_clock, clock
        self.waits: list[dict] = []

    @classmethod
    def around(cls, pads, pad_zone_m: float, margin_m: float = 5.0, **kw) -> "PadZone":
        """The zone covering every pad's pad_zone_m circle, plus a margin."""
        pads = [tuple(p) for p in pads]
        c = (sum(p[0] for p in pads) / len(pads), sum(p[1] for p in pads) / len(pads))
        spread = max(math.dist(c, p) for p in pads)
        return cls(c, spread + pad_zone_m + margin_m, **kw)

    def inside(self, xy) -> bool:
        return math.dist(self.center, xy) < self.radius_m - 1e-6

    def approach_point(self, from_xy, margin_m: float = 0.0) -> tuple[float, float]:
        """The point on the zone's edge (margin_m outside it) on the way in from from_xy."""
        dx, dy = from_xy[0] - self.center[0], from_xy[1] - self.center[1]
        d, r = math.hypot(dx, dy), self.radius_m + margin_m
        if d < 1e-6:                       # over the centre: any way out will do
            return (self.center[0] + r, self.center[1])
        return (self.center[0] + dx / d * r, self.center[1] + dy / d * r)

    async def acquire(self, vid: str, timeout_s: float | None = None) -> bool:
        if self.holder == vid:
            return True
        t0, s0 = self._clock(), self._sim()
        held_by = self.holder
        try:
            if timeout_s is None:
                await self._lock.acquire()
            else:
                async with asyncio.timeout(timeout_s):
                    await self._lock.acquire()
        except TimeoutError:
            self._record("pad_zone_timeout", vehicle_id=vid, held_by=self.holder,
                         waited_host_s=round(self._clock() - t0, 2))
            return False
        self.holder = vid
        s1 = self._sim()
        waited = round(s1 - s0, 2) if (s0 is not None and s1 is not None) else None
        self.waits.append({"vehicle_id": vid, "waited_sim_s": waited, "held_by": held_by})
        self._record("pad_zone_enter", vehicle_id=vid, waited_sim_s=waited,
                     was_held_by=held_by)
        return True

    def release(self, vid: str) -> None:
        if self.holder != vid:
            return
        self.holder = None
        self._lock.release()
        self._record("pad_zone_leave", vehicle_id=vid)

    def _sim(self):
        try:
            return self._sim_clock() if self._sim_clock else None
        except Exception:
            return None

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, source="pad_zone", **fields)
