"""
Inspection missions: one corridor per drone (missions/*.json).

Frame: the simulator's local NED frame, the same one observations and Remote ID use
(north_m, east_m in meters; altitude_m = height above the frame origin, NOT above
ground). Waypoints are absolute, so corridors of different drones can cross.

    Corridor
        waypoints          polyline the drone flies, first to last (typically starting and
                           ending over its launch pad)
        half_width_m       how far off the centerline still counts as inside the corridor
        altitude_m         assigned flight layer
        altitude_band_m    [min, max] the mission allows (e.g. for layer-change maneuvers)
        cruise_speed_mps   normal speed along the corridor

The mission is agent-visible (an agent may know its own corridor and limits); its
enforcement belongs to the action-risk gate, not to the mission runner.
"""
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CRUISE_SPEED_MPS = 5.0


class Corridor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    waypoints: list[tuple[float, float]] = Field(min_length=1)
    half_width_m: float = Field(gt=0)
    altitude_m: float
    altitude_band_m: tuple[float, float]
    cruise_speed_mps: float = Field(gt=0, le=MAX_CRUISE_SPEED_MPS)

    @model_validator(mode="after")
    def _check(self):
        if not all(math.isfinite(c) for p in self.waypoints for c in p):
            raise ValueError("waypoints must be finite")
        lo, hi = self.altitude_band_m
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError("altitude_band_m must be [min, max] with min < max")
        if not lo <= self.altitude_m <= hi:
            raise ValueError("altitude_m must lie inside altitude_band_m")
        return self

    @property
    def length_m(self) -> float:
        return sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(self.waypoints, self.waypoints[1:]))


class DroneMission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vehicle_id: str = Field(min_length=1)
    corridor: Corridor
    self_id_text: str = ""            # what its Remote ID broadcast says about it


class MissionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scene: str                        # sim_config scene file
    description: str = ""
    drones: list[DroneMission] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self):
        ids = [d.vehicle_id for d in self.drones]
        if len(set(ids)) != len(ids):
            raise ValueError("each vehicle may have only one mission")
        return self

    @property
    def vehicle_ids(self) -> list[str]:
        return [d.vehicle_id for d in self.drones]

    def for_vehicle(self, vehicle_id: str) -> DroneMission:
        for d in self.drones:
            if d.vehicle_id == vehicle_id:
                return d
        raise KeyError(vehicle_id)

    @classmethod
    def load(cls, path) -> "MissionPlan":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
