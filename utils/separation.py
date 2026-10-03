"""
Pairwise separation and collision assessment for multi-drone runs.

Evaluation-side only: it consumes simulator ground truth (adapter.latest_poses()
and adapter.collision_log()), which agents must never see. The future evaluation
recorder will reuse it.

    tracker = SeparationTracker(["Drone1", "Drone2", "Drone3"])
    tracker.update(adapter.latest_poses(), phase="fan out")   # sample at ~20 Hz
    ...
    report = assess(tracker, collisions, vehicle_ids, min_separation_m=2.0)

Distances are NED meters. "3D" is straight-line distance; "horizontal" ignores
altitude (useful later for lateral-only separation rules).
"""
import itertools
import math
from dataclasses import dataclass, field

# Two pose samples are only compared if their sim timestamps are this close;
# otherwise one drone's position is from a different moment than the other's.
MAX_SAMPLE_SKEW_NS = 50_000_000


@dataclass
class PairMinimum:
    pair: tuple[str, str]
    min_3d_m: float = math.inf
    horizontal_at_min_m: float = math.nan
    vertical_at_min_m: float = math.nan
    phase_at_min: str = ""
    sim_time_ns_at_min: int | None = None
    min_horizontal_m: float = math.inf
    samples: int = 0


@dataclass
class SeparationTracker:
    vehicle_ids: list[str]
    max_skew_ns: int = MAX_SAMPLE_SKEW_NS
    pairs: dict = field(default_factory=dict)
    skipped_samples: int = 0

    def __post_init__(self):
        self.pairs = {p: PairMinimum(pair=p) for p in itertools.combinations(sorted(self.vehicle_ids), 2)}

    def update(self, poses: dict, phase: str = "") -> None:
        """poses: {vehicle_id: (sim_time_ns, north, east, down)}."""
        for (a, b), rec in self.pairs.items():
            if a not in poses or b not in poses:
                continue
            ta, na, ea, da = poses[a]
            tb, nb, eb, db = poses[b]
            if abs(ta - tb) > self.max_skew_ns:
                self.skipped_samples += 1
                continue
            horiz = math.hypot(na - nb, ea - eb)
            vert = abs(da - db)
            d3 = math.hypot(horiz, vert)
            rec.samples += 1
            rec.min_horizontal_m = min(rec.min_horizontal_m, horiz)
            if d3 < rec.min_3d_m:
                rec.min_3d_m, rec.horizontal_at_min_m, rec.vertical_at_min_m = d3, horiz, vert
                rec.phase_at_min, rec.sim_time_ns_at_min = phase, max(ta, tb)

    @property
    def overall_min(self) -> PairMinimum | None:
        sampled = [r for r in self.pairs.values() if r.samples]
        return min(sampled, key=lambda r: r.min_3d_m) if sampled else None


@dataclass
class SeparationReport:
    min_separation_m: float
    pairs: list[PairMinimum]
    vehicle_collisions: list[str]       # drone hit another drone
    impacts: list[str]                  # fast (non-resting) contact with anything
    unsampled_pairs: list[tuple[str, str]]

    @property
    def violations(self) -> list[str]:
        out = [f"{a}-{b} came within {r.min_3d_m:.2f} m (< {self.min_separation_m:.1f} m) "
               f"during '{r.phase_at_min}'"
               for r in self.pairs for (a, b) in [r.pair]
               if r.samples and r.min_3d_m < self.min_separation_m]
        out += [f"never sampled: {a}-{b}" for a, b in self.unsampled_pairs]
        return out + self.vehicle_collisions + self.impacts

    @property
    def passed(self) -> bool:
        return not self.violations


def _is_vehicle(object_name: str | None, vehicle_ids) -> bool:
    if not object_name:
        return False
    return any(v.lower() in object_name.lower() for v in vehicle_ids)


def assess(tracker: SeparationTracker, collisions: dict, vehicle_ids,
           min_separation_m: float = 2.0) -> SeparationReport:
    """collisions: {vehicle_id: [CollisionState, ...]} from adapter.collision_log()."""
    vehicle_hits, impacts = [], []
    for vid, events in collisions.items():
        for c in events:
            t = f"{c.sim_time_ns / 1e9:.2f}s" if c.sim_time_ns else "?"
            if _is_vehicle(c.object_name, [v for v in vehicle_ids if v != vid]):
                vehicle_hits.append(f"{vid} collided with {c.object_name} at {t}")
            elif c.is_resting_contact is False:
                speed = f"{c.impact_speed_mps:.1f} m/s" if c.impact_speed_mps is not None else "?"
                impacts.append(f"{vid} impact with {c.object_name} at {t} (~{speed})")
    return SeparationReport(
        min_separation_m=min_separation_m,
        pairs=sorted(tracker.pairs.values(), key=lambda r: r.min_3d_m),
        vehicle_collisions=vehicle_hits,
        impacts=impacts,
        unsampled_pairs=[p for p, r in tracker.pairs.items() if not r.samples],
    )


def format_report(report: SeparationReport) -> str:
    lines = [f"Separation (minimum allowed {report.min_separation_m:.1f} m, 3D):"]
    for r in report.pairs:
        a, b = r.pair
        if not r.samples:
            lines.append(f"  {a}-{b}: no samples")
            continue
        flag = "OK  " if r.min_3d_m >= report.min_separation_m else "FAIL"
        lines.append(f"  {flag} {a}-{b}: min {r.min_3d_m:5.2f} m "
                     f"(horizontal {r.horizontal_at_min_m:5.2f}, vertical {r.vertical_at_min_m:5.2f}) "
                     f"during '{r.phase_at_min}'  | min horizontal ever {r.min_horizontal_m:5.2f} m  "
                     f"[{r.samples} samples]")
    lines.append(f"Drone-to-drone collisions: {len(report.vehicle_collisions) or 'none'}")
    lines += [f"  {x}" for x in report.vehicle_collisions]
    lines.append(f"Impacts (non-resting contacts): {len(report.impacts) or 'none'}")
    lines += [f"  {x}" for x in report.impacts]
    return "\n".join(lines)
