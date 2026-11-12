"""
Simulated victim sensor (Phase 1): geometric footprint detection.

The victim is detected when it lies inside the square sensor footprint below the drone
(side `footprint_m`, axis-aligned, centered on the drone; height ignored) while the drone
is at search altitude (within `altitude_tolerance_m`). Between two samples the straight
path is checked too, so a fast drone cannot skip over the victim between samples
(segment-box check adapted from search_and_rescue_stack.py).

This is the SIMULATION side: the sensor holds the scenario's true victim position and
the drone's ground-truth pose, like a camera would "hold" the scene. What it hands the
drone is a Detection with the DRONE's position, never the victim's coordinates.
Camera-based detection ("victim in camera view", deck slide 12) is a later upgrade that
replaces this class.
"""
import asyncio
import time

from atal.core.belief import Detection
from atal.core.telemetry import GroundState, ValidationStatus


def segment_box_hit(p0, p1, center, half):
    """If the straight path p0 -> p1 (x, y) passes within +/- half of `center` on both
    axes, the point in the middle of that crossing; else None."""
    t_lo, t_hi = 0.0, 1.0
    for a, b, c in ((p0[0], p1[0], center[0]), (p0[1], p1[1], center[1])):
        lo, hi = c - half, c + half
        d = b - a
        if abs(d) < 1e-12:
            if a < lo or a > hi:
                return None
        else:
            ta, tb = (lo - a) / d, (hi - a) / d
            if ta > tb:
                ta, tb = tb, ta
            t_lo, t_hi = max(t_lo, ta), min(t_hi, tb)
            if t_lo > t_hi:
                return None
    t = (t_lo + t_hi) / 2
    return p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])


class VictimSensor:
    def __init__(self, adapter, vehicle_id: str, victim_xy: tuple[float, float],
                 footprint_m: float, altitude_m: float, altitude_tolerance_m: float = 5.0,
                 grid=None, on_detection=None, recorder=None, clock=time.monotonic, log=print):
        self.adapter, self.vehicle_id = adapter, vehicle_id
        self.victim = (float(victim_xy[0]), float(victim_xy[1]))
        self.half = footprint_m / 2.0
        self.footprint_m = footprint_m
        self.altitude_m, self.altitude_tolerance_m = altitude_m, altitude_tolerance_m
        self.grid = grid
        self._on_detection, self._recorder = on_detection, recorder
        self._clock, self._log = clock, log
        self._prev: tuple[float, float] | None = None
        self.detection: Detection | None = None
        self.samples = 0

    def check(self) -> Detection | None:
        """Take one sample; returns the Detection the first time the victim is seen."""
        try:
            snap = self.adapter.get_snapshot(self.vehicle_id)
        except Exception:
            self._prev = None
            return None
        self.samples += 1
        if (snap.validation_status != ValidationStatus.VALID
                or snap.ground_state == GroundState.GROUNDED
                or abs(snap.altitude_local_m - self.altitude_m) > self.altitude_tolerance_m):
            self._prev = None            # not searching: don't bridge a gap across it
            return None
        cur = (snap.position_ned_m.x, snap.position_ned_m.y)
        prev = self._prev or cur
        self._prev = cur
        if self.detection is not None:
            return None
        hit = segment_box_hit(prev, cur, self.victim, self.half)
        if hit is None:
            return None
        d = Detection(vehicle_id=self.vehicle_id, sim_time_s=round(snap.sim_time_s, 3),
                      x_m=round(hit[0], 3), y_m=round(hit[1], 3),
                      altitude_m=round(snap.altitude_local_m, 2),
                      sector_id=self.grid.sector_at(*hit) if self.grid else None,
                      footprint_m=self.footprint_m)
        self.detection = d
        if self._recorder is not None:          # evaluation side: may hold the truth
            self._recorder.record("victim_detected", vehicle_id=self.vehicle_id,
                                  detection=d.model_dump(mode="json"),
                                  victim_truth={"x_m": self.victim[0], "y_m": self.victim[1]},
                                  sim_time_ns=snap.sim_time_ns)
        if self._on_detection is not None:
            try:
                self._on_detection(d)
            except Exception as err:
                self._log(f"[sensor] on_detection failed: {type(err).__name__}: {err}")
        return d

    async def run(self, rate_hz: float, stop: asyncio.Event) -> None:
        period = 1.0 / rate_hz
        while not stop.is_set():
            self.check()
            try:
                async with asyncio.timeout(period):
                    await stop.wait()
            except asyncio.TimeoutError:
                pass
