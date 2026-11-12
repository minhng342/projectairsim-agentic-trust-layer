"""
The mission plan (mission.yaml), Phase 1 onward, and the loader every plan file uses.

    mission.yaml   what the drones know at launch (Mission). Frozen: peers inform beliefs,
                   never goals.
    scenario.yaml  ground truth and experiment settings (Scenario): victim, seed, runs,
                   drone starts, insider flag. Read ONLY by the runner and the simulated
                   sensor, never by a planner or agent. The victim never goes in the mission.

The scenario is atal/core/scenario.py, the grid and sector ids atal/core/sector_grid.py,
the actions a chooser picks atal/core/action.py.

Frame: the simulator's local NED frame. x_m = north, y_m = east (meters), altitude_m =
height above the frame origin.

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
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from atal.core.sector_grid import MAX_SECTORS_PER_AXIS, SECTOR_ID_PATTERN, SectorGrid
from atal.core.strict import Strict

MAX_SPEED_MPS = 20.0


class Point(Strict):
    x_m: float
    y_m: float


class AreaSize(Strict):
    width: float = Field(gt=0, description="extent along x (north), meters")
    height: float = Field(gt=0, description="extent along y (east), meters")


class SectorCounts(Strict):
    x: int = Field(ge=1, le=MAX_SECTORS_PER_AXIS)
    y: int = Field(ge=1, le=MAX_SECTORS_PER_AXIS)


class Area(Strict):
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


class Flight(Strict):
    altitude_m: float = Field(gt=0)
    speed_m_s: float = Field(gt=0, le=MAX_SPEED_MPS)


class SensorSpec(Strict):
    footprint_m: float = Field(gt=0, description="side of the square ground footprint below the drone")
    overlap: float = Field(ge=0, lt=1, description="share of the footprint adjacent lanes overlap")


class SearchSpec(Strict):
    pattern: Literal["boustrophedon"] = "boustrophedon"
    sector_order: Literal["serpentine"] = "serpentine"
    start_sector: str = Field(pattern=SECTOR_ID_PATTERN)
    allocation: Literal["voronoi"] = "voronoi"      # the only search algorithm (2026-10-29)


class Sigma(Strict):
    x: float = Field(gt=0)
    y: float = Field(gt=0)


class Prior(Strict):
    """Where the victim is believed to be. Parsed and validated, not used in Phase 1."""
    enabled: bool = False
    center: Point | None = None
    sigma_m: Sigma | None = None

    @model_validator(mode="after")
    def _complete(self):
        if self.enabled and (self.center is None or self.sigma_m is None):
            raise ValueError("an enabled prior needs center and sigma_m")
        return self


class Termination(Strict):
    stop_on_detection: bool = True
    max_mission_time_s: float = Field(gt=0)


class Limits(Strict):
    """altitude_ceiling_m: the highest altitude any drone may use, plus the cruise floor
    above it (recovery policy section 2). Required for a team run with the managed
    airspace; a provisional simulation cap, checked against the loaded team's size. It
    does not establish obstacle clearance."""
    min_battery_pct: float = Field(ge=0, le=100)
    geofence_margin_m: float = Field(ge=0)
    altitude_ceiling_m: float | None = Field(default=None, gt=0)


class Swarm(Strict):
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
    peer_blocked_rule accept: a peer's SectorBlocked keeps this drone out of that sector too, so
                      one obstacle costs the team one incident (the sector stays unsearched,
                      never cleared); ignore: recorded only, each drone finds out for itself
    peer_victim_rule  accept: a peer's VictimDetected ends this drone's search (Phase 3
                      baseline); ignore: recorded only
    victim_repeat_s   the finder repeats its VictimDetected this often until it lands, so a
                      peer out of range at the moment of the find still hears it later
    identity          how a receiver knows who sent a message. authenticated (default, the
                      professor's authenticated-insider model): the radio link authenticates
                      the transmitter (a simulated stand-in for a signed link), and a peer
                      message or Remote ID report whose claimed sender_id / uas_id is not
                      that transmitter is dropped (identity_mismatch). A liar can lie about
                      content, never about who it is, so corroboration counts real senders.
                      unverified: the claimed id is taken at face value; one transmitter can
                      pose as many (a separately labelled spoofing condition)
    separation_cruise_m  acceptance: two airborne drones never closer than this (3D), except
    separation_pad_m     near the pads: while either drone is within pad_zone_m (horizontal)
    pad_zone_m           of any launch point, the floor is separation_pad_m instead (pads are
                         3 m apart and the team climbs and lands in place). Airborne = at
                         least 1 m above that drone's own launch altitude. Measured from the
                         2 Hz truth samples (a close approach between samples can be missed);
    max_truth_gap_s      a gap longer than this in the samples makes separation unmeasured,
                         which fails the run too
    allocation        dynamic (default, the Phase 3 baseline): Voronoi over the open sectors,
                      live peers and claims, recomputed at every decision, with work
                      stealing, claims and clears on the radio. static (comparison
                      condition): the grid is split once before takeoff (Voronoi over the
                      staging points, else the pads) and each drone sweeps only its own
                      part; no claims, no clears, no re-split when a drone drops out. Finds
                      are still reported, so the team stops on the first find in both
    airspace          managed (default): every vertical move of a team drone (startup,
                      obstacle recovery, landing) is a reservation from the trusted
                      transition manager (atal/airspace/, recovery policy v3.3), and a
                      recovering drone climbs to a transit level of its own. unmanaged:
                      the Phase 3 behaviour (the team climbs together, a recovery climbs
                      10 m), only to reproduce earlier results; such runs are labelled and
                      are not used for separation claims
    max_transit_level how many transit levels each drone has above the team's search
                      levels (T_i(k) = altitude_m + (k n + i) layer_spacing_m)
    staging           optional: drone i first flies (at its layer) to staging[i], then starts
                      searching. Lets the team launch from one known-clear pad area while the
                      first Voronoi split still starts from spread-out generators (drones on
                      one pad would give two of them empty cells)
    """
    radio_range_m: float | None = Field(default=500.0, gt=0)
    layer_spacing_m: float = Field(default=10.0, ge=0)
    peer_timeout_s: float = Field(default=10.0, gt=0)
    max_claim_s: float = Field(default=600.0, gt=0)
    peer_clear_rule: Literal["single", "corroborated", "self_only"] = "single"
    corroboration_n: int = Field(default=2, ge=2)
    peer_blocked_rule: Literal["accept", "ignore"] = "accept"
    peer_victim_rule: Literal["accept", "ignore"] = "accept"
    victim_repeat_s: float = Field(default=2.0, gt=0)
    allocation: Literal["dynamic", "static"] = "dynamic"
    staging: list[Point] | None = None
    identity: Literal["authenticated", "unverified"] = "authenticated"
    separation_cruise_m: float = Field(default=5.0, ge=0)
    separation_pad_m: float = Field(default=1.5, ge=0)
    pad_zone_m: float = Field(default=10.0, ge=0)
    max_truth_gap_s: float = Field(default=2.0, gt=0)
    airspace: Literal["managed", "unmanaged"] = "managed"
    max_transit_level: int = Field(default=2, ge=0)


class Mission(Strict):
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
        for i, p in enumerate(self.swarm.staging or []):
            if self.geofence_excursion_m(p.x_m, p.y_m) > 0:
                raise ValueError(f"swarm.staging[{i}] ({p.x_m:g}, {p.y_m:g}) is outside the area")
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
    def grid(self) -> SectorGrid:
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


class PlanFileError(ValueError):
    """A mission or scenario file that does not validate; the message names each field."""


def load_yaml(path, model):
    """Load and validate one YAML plan file; PlanFileError names each bad field."""
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
    return load_yaml(path, Mission)
