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
import time
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
    samples: int = 0            # counted: both timestamps advanced, in sync
    attempts: int = 0           # every update() call
    frozen: int = 0             # a timestamp did not advance (stale or replayed pose)
    skewed: int = 0             # the two poses are from different moments
    missing: int = 0            # one or both vehicles had no pose
    max_gap_s: float = 0.0      # longest host-time gap without a counted sample
    last_ts: tuple[int, int] | None = None
    last_counted_host: float | None = None

    @property
    def coverage(self) -> float:
        return self.samples / self.attempts if self.attempts else 0.0


@dataclass
class SeparationTracker:
    vehicle_ids: list[str]
    max_skew_ns: int = MAX_SAMPLE_SKEW_NS
    pairs: dict = field(default_factory=dict)
    started_host: float | None = None
    finished_host: float | None = None

    def __post_init__(self):
        self.pairs = {p: PairMinimum(pair=p) for p in itertools.combinations(sorted(self.vehicle_ids), 2)}

    def update(self, poses: dict, phase: str = "", host_time: float | None = None) -> None:
        """poses: {vehicle_id: (sim_time_ns, north, east, down)}.

        A sample only counts if BOTH vehicles' timestamps advanced since the last
        counted sample for that pair, and the two are from the same moment. A frozen
        or replayed stream therefore cannot inflate the sample count.
        """
        now = time.monotonic() if host_time is None else host_time
        if self.started_host is None:
            self.started_host = now
        for (a, b), rec in self.pairs.items():
            rec.attempts += 1
            if a not in poses or b not in poses:
                rec.missing += 1
                continue
            ta, na, ea, da = poses[a]
            tb, nb, eb, db = poses[b]
            if rec.last_ts is not None and (ta <= rec.last_ts[0] or tb <= rec.last_ts[1]):
                rec.frozen += 1
                continue
            if abs(ta - tb) > self.max_skew_ns:
                rec.skewed += 1
                continue
            since = rec.last_counted_host if rec.last_counted_host is not None else self.started_host
            rec.max_gap_s = max(rec.max_gap_s, now - since)
            rec.last_ts, rec.last_counted_host = (ta, tb), now
            horiz = math.hypot(na - nb, ea - eb)
            vert = abs(da - db)
            d3 = math.hypot(horiz, vert)
            rec.samples += 1
            rec.min_horizontal_m = min(rec.min_horizontal_m, horiz)
            if d3 < rec.min_3d_m:
                rec.min_3d_m, rec.horizontal_at_min_m, rec.vertical_at_min_m = d3, horiz, vert
                rec.phase_at_min, rec.sim_time_ns_at_min = phase, max(ta, tb)

    def finish(self, host_time: float | None = None) -> None:
        """Close the run so a gap at the END (stream died before shutdown) is counted."""
        self.finished_host = time.monotonic() if host_time is None else host_time
        for rec in self.pairs.values():
            since = rec.last_counted_host if rec.last_counted_host is not None else self.started_host
            if since is not None:
                rec.max_gap_s = max(rec.max_gap_s, self.finished_host - since)

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
    coverage_problems: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        """Evidence that something unsafe happened (FAIL)."""
        out = [f"{a}-{b} came within {r.min_3d_m:.2f} m (< {self.min_separation_m:.1f} m) "
               f"during '{r.phase_at_min}'"
               for r in self.pairs for (a, b) in [r.pair]
               if r.samples and r.min_3d_m < self.min_separation_m]
        return out + self.vehicle_collisions + self.impacts

    @property
    def status(self) -> str:
        """'fail' if anything unsafe was seen; otherwise 'inconclusive' if the
        telemetry was too incomplete to certify separation; otherwise 'pass'."""
        if self.violations:
            return "fail"
        if self.coverage_problems:
            return "inconclusive"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _is_vehicle(object_name: str | None, vehicle_ids) -> bool:
    if not object_name:
        return False
    return any(v.lower() in object_name.lower() for v in vehicle_ids)


def assess(tracker: SeparationTracker, collisions: dict, vehicle_ids,
           min_separation_m: float = 2.0, min_coverage: float = 0.9,
           max_gap_s: float = 0.5) -> SeparationReport:
    """collisions: {vehicle_id: [CollisionState, ...]} from adapter.collision_log().

    Coverage rules (otherwise INCONCLUSIVE, never PASS): every pair sampled, at
    least min_coverage of attempts counted, and no gap longer than max_gap_s
    without a counted sample (including the start and, if finish() was called,
    the end of the run).
    """
    vehicle_hits, impacts = [], []
    for vid, events in collisions.items():
        for c in events:
            t = f"{c.sim_time_ns / 1e9:.2f}s" if c.sim_time_ns else "?"
            if _is_vehicle(c.object_name, [v for v in vehicle_ids if v != vid]):
                vehicle_hits.append(f"{vid} collided with {c.object_name} at {t}")
            elif c.is_resting_contact is False:
                speed = f"{c.impact_speed_mps:.1f} m/s" if c.impact_speed_mps is not None else "?"
                impacts.append(f"{vid} impact with {c.object_name} at {t} (~{speed})")
    unsampled = [p for p, r in tracker.pairs.items() if not r.samples]
    coverage = [f"never sampled: {a}-{b}" for a, b in unsampled]
    for (a, b), r in tracker.pairs.items():
        if not r.samples:
            continue
        if r.coverage < min_coverage:
            coverage.append(f"{a}-{b}: only {r.coverage:.0%} of samples usable "
                            f"(frozen {r.frozen}, out of sync {r.skewed}, missing {r.missing})")
        if r.max_gap_s > max_gap_s:
            coverage.append(f"{a}-{b}: {r.max_gap_s:.1f} s without a usable sample "
                            f"(limit {max_gap_s:.1f} s)")
    return SeparationReport(
        min_separation_m=min_separation_m,
        pairs=sorted(tracker.pairs.values(), key=lambda r: r.min_3d_m),
        vehicle_collisions=vehicle_hits,
        impacts=impacts,
        unsampled_pairs=unsampled,
        coverage_problems=coverage,
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
                     f"[{r.samples}/{r.attempts} usable, {r.coverage:.0%}, max gap {r.max_gap_s:.2f}s]")
    lines.append(f"Drone-to-drone collisions: {len(report.vehicle_collisions) or 'none'}")
    lines += [f"  {x}" for x in report.vehicle_collisions]
    lines.append(f"Impacts (non-resting contacts): {len(report.impacts) or 'none'}")
    lines += [f"  {x}" for x in report.impacts]
    lines.append(f"Telemetry coverage: {'ok' if not report.coverage_problems else 'INSUFFICIENT'}")
    lines += [f"  {x}" for x in report.coverage_problems]
    lines.append(f"Separation verdict: {report.status.upper()}")
    return "\n".join(lines)
