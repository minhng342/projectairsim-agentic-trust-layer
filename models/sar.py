"""
Search and rescue (SAR) data contracts, Phase 1 onward: the mission plan, the scenario,
the sector grid, the drone state and the typed actions.

    mission.yaml   what the drones know at launch (Mission). Frozen: peers inform beliefs,
                   never goals.
    scenario.yaml  ground truth and experiment settings (Scenario): victim, seed, runs,
                   drone starts, insider flag. Read ONLY by the runner and the simulated
                   sensor, never by a planner or agent. The victim never goes in the mission.

Frame: the simulator's local NED frame. x_m = north, y_m = east (meters), altitude_m =
height above the frame origin. The grid's origin is its minimum-x, minimum-y corner.

Sector ids read like a chess board on a north-up map: the letter is the column along y
(east; A = lowest y), the number is the row along x (north; 1 = lowest x). With the deck's
800 x 800 m / 8 x 8 grid, A1 spans x 0-100, y 0-100, B1 is east of A1, A2 north of A1,
and the deck's victim at (x 430, y 560) is in F5.

Derived in code, not written in YAML (deck slide 13):
    sector size       = area size / sector count, per axis (800 / 8 = 100 m)
    lane spacing      = footprint * (1 - overlap)            (60 * 0.8 = 48 m)
    passes per sector = ceil(sector size along x / lane spacing)   (ceil(100 / 48) = 3);
                        lanes run along y (east), stepped along x (north)

Validation (Pydantic, at load time): an area size not divisible into whole-meter sectors,
a footprint larger than a sector, and a start sector outside the grid are rejected, and
every error names the field. The swarm section (Phase 3: radio range, layers, peer-claim
rules) has defaults, so Phase 1 and 2 files stay valid and a one-drone run ignores it.
"""
import math
import random
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from models.action import SECTOR_ID_PATTERN, ActionType, ProposedAction

MAX_SECTORS_PER_AXIS = 26          # one letter per column
MAX_SPEED_MPS = 20.0
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


# ------------------------------------------------------------------ mission.yaml
class Point(_Strict):
    x_m: float
    y_m: float


class AreaSize(_Strict):
    width: float = Field(gt=0, description="extent along x (north), meters")
    height: float = Field(gt=0, description="extent along y (east), meters")


class SectorCounts(_Strict):
    x: int = Field(ge=1, le=MAX_SECTORS_PER_AXIS)
    y: int = Field(ge=1, le=MAX_SECTORS_PER_AXIS)


class Area(_Strict):
    origin: Point
    size_m: AreaSize
    sectors: SectorCounts

    @model_validator(mode="after")
    def _divisible(self):
        for axis, size, count in (("width", self.size_m.width, self.sectors.x),
                                  ("height", self.size_m.height, self.sectors.y)):
            per = size / count
            if abs(per - round(per)) > 1e-9:
                raise ValueError(f"size_m.{axis} ({size:g}) is not divisible into "
                                 f"{count} whole-meter sectors (sectors.{'x' if axis == 'width' else 'y'})")
        return self


class Flight(_Strict):
    altitude_m: float = Field(gt=0)
    speed_m_s: float = Field(gt=0, le=MAX_SPEED_MPS)


class SensorSpec(_Strict):
    footprint_m: float = Field(gt=0, description="side of the square ground footprint below the drone")
    overlap: float = Field(ge=0, lt=1, description="share of the footprint adjacent lanes overlap")


class SearchSpec(_Strict):
    pattern: Literal["boustrophedon"] = "boustrophedon"
    sector_order: Literal["serpentine"] = "serpentine"
    start_sector: str = Field(pattern=SECTOR_ID_PATTERN)
    allocation: Literal["voronoi"] = "voronoi"      # the only search algorithm (2026-10-29)


class Sigma(_Strict):
    x: float = Field(gt=0)
    y: float = Field(gt=0)


class Prior(_Strict):
    """Where the victim is believed to be. Parsed and validated, not used in Phase 1."""
    enabled: bool = False
    center: Point | None = None
    sigma_m: Sigma | None = None

    @model_validator(mode="after")
    def _complete(self):
        if self.enabled and (self.center is None or self.sigma_m is None):
            raise ValueError("an enabled prior needs center and sigma_m")
        return self


class Termination(_Strict):
    stop_on_detection: bool = True
    max_mission_time_s: float = Field(gt=0)


class Limits(_Strict):
    min_battery_pct: float = Field(ge=0, le=100)
    geofence_margin_m: float = Field(ge=0)


class Swarm(_Strict):
    """Phase 3: how the drones of a team work together. Ignored with one drone.

    radio_range_m     a message reaches a peer only if the TRUE sender is within this
                      3D distance of it (the bus decides from ground truth; None = unlimited)
    layer_spacing_m   drone i of the scenario flies at altitude_m + i * layer_spacing_m, so
                      cooperation never becomes a collision problem
    peer_timeout_s    a peer not heard for this many sim seconds has dropped out: its
                      position no longer counts in the Voronoi partition and its claims lapse
    max_claim_s       a SectorClaimed lasts until its expiry, capped at this many seconds
                      after it was received
    peer_clear_rule   when a peer's SectorCleared marks a sector done in this drone's memory:
                      single (one peer is enough: the Phase 3 cooperation baseline),
                      corroborated (corroboration_n different peers), self_only (never; peer
                      clears are recorded but this drone sweeps everything itself)
    peer_victim_rule  accept: a peer's VictimDetected ends this drone's search (Phase 3
                      baseline); ignore: recorded only
    victim_repeat_s   the finder repeats its VictimDetected this often until it lands, so a
                      peer out of range at the moment of the find still hears it later
    """
    radio_range_m: float | None = Field(default=500.0, gt=0)
    layer_spacing_m: float = Field(default=10.0, ge=0)
    peer_timeout_s: float = Field(default=10.0, gt=0)
    max_claim_s: float = Field(default=600.0, gt=0)
    peer_clear_rule: Literal["single", "corroborated", "self_only"] = "single"
    corroboration_n: int = Field(default=2, ge=2)
    peer_victim_rule: Literal["accept", "ignore"] = "accept"
    victim_repeat_s: float = Field(default=2.0, gt=0)


class Mission(_Strict):
    mission_id: str = Field(min_length=1)
    area: Area
    flight: Flight
    sensor: SensorSpec
    search: SearchSpec
    prior: Prior = Prior()
    termination: Termination
    limits: Limits
    swarm: Swarm = Swarm()

    @model_validator(mode="after")
    def _consistent(self):
        sx, sy = self.sector_size_m
        if self.sensor.footprint_m > min(sx, sy):
            raise ValueError(f"sensor.footprint_m ({self.sensor.footprint_m:g}) is larger than a "
                             f"sector ({sx:g} x {sy:g} m)")
        if not self.grid.has(self.search.start_sector):
            raise ValueError(f"search.start_sector {self.search.start_sector!r} is outside the "
                             f"{self.area.sectors.x} x {self.area.sectors.y} grid "
                             f"(A1..{self.grid.last_id})")
        return self

    # derived, never written in YAML
    @property
    def sector_size_m(self) -> tuple[float, float]:
        a = self.area
        return a.size_m.width / a.sectors.x, a.size_m.height / a.sectors.y

    @property
    def lane_spacing_m(self) -> float:
        return self.sensor.footprint_m * (1.0 - self.sensor.overlap)

    @property
    def passes_per_sector(self) -> int:
        return max(1, math.ceil(self.sector_size_m[0] / self.lane_spacing_m - 1e-9))

    @property
    def grid(self) -> "SectorGrid":
        return SectorGrid(self.area)

    def derived(self) -> dict:
        sx, sy = self.sector_size_m
        return {"sector_size_m": [sx, sy], "lane_spacing_m": round(self.lane_spacing_m, 3),
                "passes_per_sector": self.passes_per_sector,
                "sector_count": self.area.sectors.x * self.area.sectors.y}

    def for_layer(self, layer: int) -> "Mission":
        """This mission as flown by drone number `layer` of the team: the same plan at its
        own altitude (flight.altitude_m + layer * swarm.layer_spacing_m). Layer 0 is the
        mission itself."""
        if layer == 0:
            return self
        alt = self.flight.altitude_m + layer * self.swarm.layer_spacing_m
        return self.model_copy(update={"flight": self.flight.model_copy(
            update={"altitude_m": alt})})

    def geofence_excursion_m(self, x: float, y: float) -> float:
        """How far (x, y) lies outside the area (0 inside). Compare with
        limits.geofence_margin_m."""
        o, s = self.area.origin, self.area.size_m
        dx = max(o.x_m - x, 0.0, x - (o.x_m + s.width))
        dy = max(o.y_m - y, 0.0, y - (o.y_m + s.height))
        return math.hypot(dx, dy)


# ------------------------------------------------------------------ grid
class Sector(_Strict):
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

    def __init__(self, area: Area):
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


# ------------------------------------------------------------------ scenario.yaml
class DroneStart(_Strict):
    id: str = Field(min_length=1)                    # team name, e.g. "A"
    vehicle: str = "Drone1"                          # simulator robot name in the scene
    start: Point


class Insider(_Strict):
    enabled: bool = False                            # Phase 4 onward


class Scenario(_Strict):
    scenario_id: str = Field(min_length=1)
    seed: int = 0
    runs: int = Field(ge=1, default=1)
    victim: Point | None = None
    randomize_victim: bool = False                   # each run draws a victim from seed + run
    victim_edge_margin_m: float = Field(ge=0, default=0.0)
    drones: list[DroneStart] = Field(min_length=1)
    insider: Insider = Insider()

    @model_validator(mode="after")
    def _victim(self):
        if self.victim is None and not self.randomize_victim:
            raise ValueError("victim is required unless randomize_victim is true")
        ids = [d.id for d in self.drones]
        if len(set(ids)) != len(ids):
            raise ValueError("drone ids must be unique")
        if len({d.vehicle for d in self.drones}) != len(self.drones):
            raise ValueError("each drone needs its own simulator vehicle")
        return self

    def run_seed(self, run_index: int) -> int:
        return self.seed + run_index

    def victim_for_run(self, mission: Mission, run_index: int) -> Point:
        if not self.randomize_victim:
            return self.victim
        rng = random.Random(self.run_seed(run_index))
        o, s, m = mission.area.origin, mission.area.size_m, self.victim_edge_margin_m
        return Point(x_m=round(rng.uniform(o.x_m + m, o.x_m + s.width - m), 2),
                     y_m=round(rng.uniform(o.y_m + m, o.y_m + s.height - m), 2))

    def check_against(self, mission: Mission) -> None:
        """Cross-file checks: the victim and every start lie where the mission can reach."""
        if self.victim is not None and mission.geofence_excursion_m(self.victim.x_m,
                                                                     self.victim.y_m) > 0:
            raise ValueError(f"scenario victim ({self.victim.x_m:g}, {self.victim.y_m:g}) is "
                             f"outside the mission area")
        for d in self.drones:
            if mission.geofence_excursion_m(d.start.x_m, d.start.y_m) > mission.limits.geofence_margin_m:
                raise ValueError(f"drone {d.id} starts outside the area + geofence margin")


# ------------------------------------------------------------------ runtime state + actions
class DroneState(_Strict):
    """What the planner knows about its own drone (from its own observation)."""
    vehicle_id: str
    sim_time_s: float
    x_m: float
    y_m: float
    altitude_m: float
    sector_id: str | None                  # the sector below it, None outside the grid
    airborne: bool


class SweepSector(_Strict):
    """Sweep one sector with a boustrophedon pattern at mission altitude and speed."""
    kind: Literal["sweep_sector"] = "sweep_sector"
    sector_id: str = Field(pattern=SECTOR_ID_PATTERN)
    reason: str = Field(min_length=1, max_length=500)

    def to_proposed(self, vehicle_id: str) -> ProposedAction:
        return ProposedAction(vehicle_id=vehicle_id, action_type=ActionType.SWEEP_SECTOR,
                              sector_id=self.sector_id, reason=self.reason)


class ReturnHome(_Strict):
    """Fly back over the launch point at mission altitude."""
    kind: Literal["return_home"] = "return_home"
    reason: str = Field(min_length=1, max_length=500)

    def to_proposed(self, vehicle_id: str) -> ProposedAction:
        return ProposedAction(vehicle_id=vehicle_id, action_type=ActionType.RETURN_HOME,
                              reason=self.reason)


# Phase 2 adds VerifySector and ReportVictim; the agent's output type is this union.
SarAction = Annotated[Union[SweepSector, ReturnHome], Field(discriminator="kind")]


# ------------------------------------------------------------------ loading
class PlanFileError(ValueError):
    """A mission or scenario file that does not validate; the message names each field."""


def _load(path, model):
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise PlanFileError(f"{path}: cannot read: {err}") from err
    if not isinstance(data, dict):
        raise PlanFileError(f"{path}: expected a mapping at the top level")
    try:
        return model.model_validate(data)
    except ValidationError as err:
        lines = [f"{path}: {err.error_count()} problem(s)"]
        for e in err.errors():
            loc = ".".join(str(p) for p in e["loc"]) or "(file)"
            lines.append(f"  {loc}: {e['msg']}")
        raise PlanFileError("\n".join(lines)) from err


def load_mission(path) -> Mission:
    return _load(path, Mission)


def load_scenario(path, mission: Mission | None = None) -> Scenario:
    sc = _load(path, Scenario)
    if mission is not None:
        try:
            sc.check_against(mission)
        except ValueError as err:
            raise PlanFileError(f"{path}: {err}") from err
    return sc
