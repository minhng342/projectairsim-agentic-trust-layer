"""The scenario (scenario.yaml): ground truth and experiment settings. Victim, seed, runs,
drone starts, insider flag. Read ONLY by the runner and the simulated sensor, never by a
chooser, an agent or the trust layer (a boundary test checks the imports). The victim
never goes in the mission.
"""
import random

from pydantic import Field, model_validator

from atal.core.mission import Mission, PlanFileError, Point, load_yaml
from atal.core.strict import Strict


class DroneStart(Strict):
    id: str = Field(min_length=1)                    # team name, e.g. "A"
    vehicle: str = "Drone1"                          # simulator robot name in the scene
    start: Point


class Insider(Strict):
    enabled: bool = False                            # Phase 4 onward


class Scenario(Strict):
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


def load_scenario(path, mission: Mission | None = None) -> Scenario:
    sc = load_yaml(path, Scenario)
    if mission is not None:
        try:
            sc.check_against(mission)
        except ValueError as err:
            raise PlanFileError(f"{path}: {err}") from err
    return sc
