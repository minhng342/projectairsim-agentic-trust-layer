"""The sector grid: the mission area split into sectors, and the sector ids.

Frame: the simulator's local NED frame. x_m = north, y_m = east (meters). The grid's
origin is its minimum-x, minimum-y corner.

Sector ids read like a chess board on a north-up map: the letter is the column along y
(east; A = lowest y), the number is the row along x (north; 1 = lowest x). With the deck's
800 x 800 m / 8 x 8 grid, A1 spans x 0-100, y 0-100, B1 is east of A1, A2 north of A1,
and the deck's victim at (x 430, y 560) is in F5.
"""
from atal.core.strict import Strict

MAX_SECTORS_PER_AXIS = 26          # one letter per column
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Sector id shape: column letter (along y / east) + row number (along x / north), e.g.
# "F5". Shape only; bounds are the grid's job.
SECTOR_ID_PATTERN = r"^[A-Z][1-9][0-9]?$"


class Sector(Strict):
    id: str
    ix: int                  # row index along x / north (0 = row 1)
    iy: int                  # column index along y / east (0 = column A)
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def sector_id(ix: int, iy: int) -> str:
    """(index along x, index along y) -> id: column letter from y, row number from x."""
    return f"{LETTERS[iy]}{ix + 1}"


def parse_sector_id(sid: str) -> tuple[int, int]:
    """'F5' -> (4, 5) = (index along x, index along y). Raises ValueError for a malformed id
    (bounds are the grid's job)."""
    if not isinstance(sid, str) or len(sid) < 2 or sid[0] not in LETTERS or not sid[1:].isdigit():
        raise ValueError(f"malformed sector id {sid!r}")
    ix = int(sid[1:]) - 1
    if ix < 0:
        raise ValueError(f"malformed sector id {sid!r}")
    return ix, LETTERS.index(sid[0])


class SectorGrid:
    """The mission area split into sectors; ids <-> bounds, lookups, serpentine order."""

    def __init__(self, area):
        """area: a mission's Area (origin, size_m, sectors)."""
        self.area = area
        self.nx, self.ny = area.sectors.x, area.sectors.y
        self.sx = area.size_m.width / self.nx
        self.sy = area.size_m.height / self.ny
        self.x0, self.y0 = area.origin.x_m, area.origin.y_m

    @property
    def last_id(self) -> str:
        return sector_id(self.nx - 1, self.ny - 1)

    def has(self, sid: str) -> bool:
        try:
            ix, iy = parse_sector_id(sid)
        except ValueError:
            return False
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def sector(self, sid: str) -> Sector:
        if not self.has(sid):
            raise KeyError(f"no sector {sid!r} in the {self.nx} x {self.ny} grid")
        ix, iy = parse_sector_id(sid)
        return Sector(id=sid, ix=ix, iy=iy,
                      x_min=self.x0 + ix * self.sx, x_max=self.x0 + (ix + 1) * self.sx,
                      y_min=self.y0 + iy * self.sy, y_max=self.y0 + (iy + 1) * self.sy)

    def ids(self) -> list[str]:
        return [sector_id(ix, iy) for ix in range(self.nx) for iy in range(self.ny)]

    def sectors(self) -> list[Sector]:
        return [self.sector(s) for s in self.ids()]

    def sector_at(self, x: float, y: float) -> str | None:
        """The sector containing (x, y), or None outside the grid (edges belong to the
        lower-index sector's neighbor, except on the far edges)."""
        fx, fy = (x - self.x0) / self.sx, (y - self.y0) / self.sy
        if not (0 <= fx <= self.nx and 0 <= fy <= self.ny):
            return None
        return sector_id(min(int(fx), self.nx - 1), min(int(fy), self.ny - 1))

    def serpentine(self, start: str | None = None) -> list[str]:
        """Row by row northward, each row east then west alternately: A1..H1, H2..A2,
        A3..H3, ... Rotated to begin at `start` (wrapping round), so every sector appears
        exactly once."""
        order = []
        for ix in range(self.nx):
            cols = range(self.ny) if ix % 2 == 0 else range(self.ny - 1, -1, -1)
            order += [sector_id(ix, iy) for iy in cols]
        if start is not None:
            if start not in order:
                raise KeyError(f"no sector {start!r} in the grid")
            i = order.index(start)
            order = order[i:] + order[:i]
        return order
