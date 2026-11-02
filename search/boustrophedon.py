"""
Boustrophedon ("ox-turning") sweep of one sector: parallel lanes along y (east), stepped
along x (north), flown in alternating directions, as in the Voronoi coverage baseline of
the reference paper (deck slide 11). Lanes run the same way as the serpentine sector
order (east-west), so with an odd pass count a sweep ends next to the following sector.

    path = sweep_path(mission, "F5", from_xy=(x, y))
    path.waypoints       [(x, y), ...] for the path follower (north, east), lane ends only

Geometry, for a sector sx (along x) by sy (along y) and a square footprint f with overlap o:
    passes     ceil(sx / (f * (1 - o)))              (Mission.passes_per_sector)
    lane x     x_min + (k + 0.5) * sx / passes       evenly spaced, so the actual spacing
                                                     is <= f * (1 - o) and adjacent
                                                     footprints overlap by >= o
    lane y     [y_min + inset, y_max - inset], inset = f / 2 * (1 - o) (at most sy / 2):
               the footprint still reaches past the sector's y edges by f / 2 * o, the same
               overlap the lanes have with each other, so tracking error at the turns
               does not leave a gap at the sector boundary.
The start corner is the one nearest `from_xy` (four candidates: first lane at the low or
high x edge, flown from the low or high y end), so consecutive sectors chain with short
transits.
"""
import math
from dataclasses import dataclass

from models.sar import Mission


@dataclass(frozen=True)
class SweepPath:
    sector_id: str
    waypoints: list[tuple[float, float]]     # (x / north, y / east)
    lane_xs: list[float]
    lane_y: tuple[float, float]

    @property
    def lanes(self) -> int:
        return len(self.lane_xs)

    @property
    def length_m(self) -> float:
        p = self.waypoints
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:]))

    @property
    def start(self) -> tuple[float, float]:
        return self.waypoints[0]

    @property
    def end(self) -> tuple[float, float]:
        return self.waypoints[-1]


def sweep_path(mission: Mission, sector_id: str, from_xy: tuple[float, float] | None = None
               ) -> SweepPath:
    s = mission.grid.sector(sector_id)
    sx, sy = mission.sector_size_m
    passes = mission.passes_per_sector
    step = sx / passes
    xs = [s.x_min + (k + 0.5) * step for k in range(passes)]
    inset = min(mission.sensor.footprint_m / 2 * (1 - mission.sensor.overlap), sy / 2)
    y_lo, y_hi = s.y_min + inset, s.y_max - inset

    def build(xs_order, y_first_low):
        pts, low = [], y_first_low
        for x in xs_order:
            a, b = (y_lo, y_hi) if low else (y_hi, y_lo)
            pts.append((x, a))
            if b != a:
                pts.append((x, b))
            low = not low
        return pts

    candidates = [build(o, lo) for o in (xs, xs[::-1]) for lo in (True, False)]
    if from_xy is None:
        best = candidates[0]
    else:
        fx, fy = from_xy
        best = min(candidates, key=lambda p: math.hypot(p[0][0] - fx, p[0][1] - fy))
    return SweepPath(sector_id=sector_id, waypoints=best, lane_xs=xs, lane_y=(y_lo, y_hi))
