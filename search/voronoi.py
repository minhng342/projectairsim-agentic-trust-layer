"""
Voronoi coverage over the sector grid: the only search algorithm in this project
(decided 2026-10-29), following the Voronoi coverage baseline of the reference paper.

    cells = voronoi_partition(grid, {"Drone1": (x, y), ...})        # who owns which sector
    order = coverage_order(grid, cells["Drone1"], start_xy, start_sector="A1")

Partition: a discrete Voronoi diagram on the grid. Each sector goes to the drone whose
position is nearest the sector's center (ties: the lexically smallest drone id, so the
partition is deterministic). Only sectors in `available` are assigned, so Phase 3 can
recompute the partition when sectors clear or a drone drops out. With one drone (Phase 1)
it owns the whole grid, so the same code covers both phases.

Order within a drone's own cells: serpentine by rows (a row = sectors with the same
number, i.e. the same band along x). Rows are taken from the end of the cell nearest the
reference point (the mission's start_sector if the drone owns it, else the drone's
position) to the other end, and each row is flown from whichever of its ends is nearer
the previous sector. On the full grid from A1 this is exactly A1..H1, H2..A2, A3..H3, ...;
on a Voronoi cell (convex, so each row is one contiguous run) consecutive sectors stay
adjacent. Each sector is then swept boustrophedon (search/boustrophedon.py).
"""
import math
from collections.abc import Iterable, Mapping

from models.sar import SectorGrid


def voronoi_partition(grid: SectorGrid, positions: Mapping[str, tuple[float, float]],
                      available: Iterable[str] | None = None) -> dict[str, list[str]]:
    if not positions:
        raise ValueError("voronoi_partition needs at least one drone position")
    owners = sorted(positions)
    cells = {d: [] for d in owners}
    pool = grid.ids() if available is None else [s for s in grid.ids() if s in set(available)]
    for sid in pool:
        cx, cy = grid.sector(sid).center
        best = min(owners, key=lambda d: (math.hypot(positions[d][0] - cx,
                                                     positions[d][1] - cy), d))
        cells[best].append(sid)
    return cells


def coverage_order(grid: SectorGrid, cells: Iterable[str], start_xy: tuple[float, float],
                   start_sector: str | None = None) -> list[str]:
    own = sorted(set(cells), key=lambda s: grid.sector(s).iy)
    if not own:
        return []
    ref = grid.sector(start_sector).center if start_sector in own else start_xy
    rows: dict[int, list[str]] = {}
    for sid in own:
        rows.setdefault(grid.sector(sid).ix, []).append(sid)     # west -> east within a row
    keys = sorted(rows)

    def row_dist(ix):
        return min(math.dist(grid.sector(s).center, ref) for s in rows[ix])
    if row_dist(keys[-1]) < row_dist(keys[0]):
        keys.reverse()
    order: list[str] = []
    here = ref
    for ix in keys:
        row = rows[ix]
        if math.dist(grid.sector(row[-1]).center, here) < math.dist(grid.sector(row[0]).center, here):
            row = row[::-1]
        order += row
        here = grid.sector(row[-1]).center
    return order
