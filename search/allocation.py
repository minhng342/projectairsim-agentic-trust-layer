"""
Work allocation for the swarm (Phase 3): Voronoi coverage over what is still open,
recomputed by each drone from its OWN belief every time it picks a next sector.

    cells = allocate(grid, "Drone1", positions, open_sectors, claims)
    order = coverage_order(grid, cells["Drone1"], my_xy)

    positions      this drone + every peer heard on Remote ID within peer_timeout_s (a peer
                   not heard for longer has dropped out, so its area is shared out again)
    open_sectors   not done in this drone's memory (own sweeps and accepted peer clears
                   removed), so the partition shrinks as sectors clear
    claims         sector -> claimant for live, unexpired SectorClaimed messages (and this
                   drone's own claim): a claimed sector belongs to its claimant whatever the
                   distances say, which settles ties and races at the cell borders

Partition: search/voronoi.py over the open, unclaimed sectors with the drones' CURRENT
positions as generators. Each drone runs this on its own picture, so two drones that
heard different messages can disagree; claims are what keeps them from sweeping the same
sector at the same time.

Work stealing: when this drone's cell is empty but open, unclaimed sectors remain in other
cells, it takes the one that is most "its own" among them, the largest (distance to the
nearest peer - distance to me), so it helps where the owner is least likely to go next.
A sector that is open only because a peer is sweeping it (claimed) is never stolen.

Positions come from Remote ID, i.e. they are claims too: Phase 4 can lie about position to
pull cells toward itself.
"""
import math
from collections.abc import Iterable, Mapping

from models.sar import SectorGrid
from search.voronoi import voronoi_partition


def allocate(grid: SectorGrid, me: str, positions: Mapping[str, tuple[float, float]],
             open_sectors: Iterable[str], claims: Mapping[str, str] | None = None,
             steal: bool = True) -> dict:
    """Returns {"cells": {drone: [sectors]}, "stolen": sector | None, "claimed": {...}}.
    `me` must be in positions."""
    if me not in positions:
        raise ValueError(f"{me} has no position")
    claims = {s: c for s, c in (claims or {}).items() if c in positions or c == me}
    open_set = set(open_sectors)
    pool = [s for s in grid.ids() if s in open_set and s not in claims]
    cells = voronoi_partition(grid, positions, available=pool) if pool else {
        d: [] for d in positions}
    for s, c in claims.items():
        if s in open_set:
            cells.setdefault(c, []).append(s)
    stolen = None
    if steal and not cells.get(me) and pool:
        here = positions[me]
        peers = [p for d, p in positions.items() if d != me]

        def margin(s):
            c = grid.sector(s).center
            return (min((math.dist(p, c) for p in peers), default=0.0) - math.dist(here, c), s)
        stolen = max(pool, key=margin)
        for d in cells:
            if stolen in cells[d]:
                cells[d].remove(stolen)
        cells[me] = [stolen]
    return {"cells": cells, "stolen": stolen,
            "claimed": {s: c for s, c in claims.items() if s in open_set}}
