"""
Conflict detection: linear state-based projection (Kuchar & Yang's "nominal" method,
the same idea as BlueSky's SHOWTCPA that Atlas relied on; reimplemented here, not copied).

Pure functions, no I/O, so three users share exactly the same math:
    - the agent's conflict tool, on REPORTED traffic (own observation + Remote ID reports)
    - the action-risk gate (later): apply a proposed maneuver to own-ship, rerun, compare
    - the evaluator, on GROUND TRUTH, to score what really happened

Model
    Each aircraft is a point moving at constant velocity from its current state.
    Loss of separation (LOS) = horizontal distance < minima.horizontal_m AND vertical
    distance < minima.vertical_m at the same time (a cylinder around each aircraft).
    For a pair, within [0, lookahead]:
        tcpa_s        time of closest HORIZONTAL approach (>= 0)
        dcpa_m        horizontal distance at tcpa
        t_los_s       first time inside the cylinder (None: no LOS within the lookahead)
        los_end_s     when the predicted LOS ends (capped at the lookahead)
        in_los_now    already inside the cylinder

A report that is `age_s` old is first extrapolated to now along its reported velocity.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Minima:
    horizontal_m: float = 5.0
    vertical_m: float = 2.0


@dataclass(frozen=True)
class Track:
    """State of one aircraft NOW (local NED frame; altitude up, vertical speed up)."""
    id: str
    north_m: float
    east_m: float
    altitude_m: float
    v_north_mps: float = 0.0
    v_east_mps: float = 0.0
    v_up_mps: float = 0.0

    def advanced(self, dt_s: float) -> "Track":
        return Track(self.id, self.north_m + self.v_north_mps * dt_s,
                     self.east_m + self.v_east_mps * dt_s,
                     self.altitude_m + self.v_up_mps * dt_s,
                     self.v_north_mps, self.v_east_mps, self.v_up_mps)

    @property
    def finite(self) -> bool:
        return all(math.isfinite(x) for x in (self.north_m, self.east_m, self.altitude_m,
                                              self.v_north_mps, self.v_east_mps, self.v_up_mps))


@dataclass(frozen=True)
class PairPrediction:
    other_id: str
    horizontal_now_m: float
    vertical_now_m: float
    tcpa_s: float
    dcpa_m: float
    vertical_at_tcpa_m: float
    t_los_s: float | None
    los_end_s: float | None
    in_los_now: bool

    @property
    def conflict(self) -> bool:
        return self.t_los_s is not None


def _interval_below(a: float, b: float, c: float, eps: float = 1e-9):
    """{t : a t^2 + 2 b t + c < 0} for a >= 0, as (lo, hi) or None (may be infinite)."""
    if a < eps:
        if abs(b) < eps:
            return (-math.inf, math.inf) if c < 0 else None
        t0 = -c / (2 * b)
        return (-math.inf, t0) if b > 0 else (t0, math.inf)
    disc = b * b - a * c
    if disc <= 0:
        return None
    s = math.sqrt(disc)
    return ((-b - s) / a, (-b + s) / a)


def predict(own: Track, other: Track, minima: Minima = Minima(),
            lookahead_s: float = 30.0) -> PairPrediction:
    rn, re = other.north_m - own.north_m, other.east_m - own.east_m
    vn, ve = other.v_north_mps - own.v_north_mps, other.v_east_mps - own.v_east_mps
    dz, vz = other.altitude_m - own.altitude_m, other.v_up_mps - own.v_up_mps
    v2, rv = vn * vn + ve * ve, rn * vn + re * ve
    tcpa = 0.0 if v2 < 1e-9 else max(0.0, -rv / v2)
    dcpa = math.hypot(rn + vn * tcpa, re + ve * tcpa)
    h_int = _interval_below(v2, rv, rn * rn + re * re - minima.horizontal_m ** 2)
    if abs(vz) < 1e-9:
        v_int = (-math.inf, math.inf) if abs(dz) < minima.vertical_m else None
    else:
        t1, t2 = (-minima.vertical_m - dz) / vz, (minima.vertical_m - dz) / vz
        v_int = (min(t1, t2), max(t1, t2))
    t_los = los_end = None
    if h_int and v_int:
        lo = max(h_int[0], v_int[0], 0.0)
        hi = min(h_int[1], v_int[1], lookahead_s)
        if lo < hi:
            t_los, los_end = lo, hi
    horiz_now, vert_now = math.hypot(rn, re), abs(dz)
    return PairPrediction(
        other_id=other.id, horizontal_now_m=horiz_now, vertical_now_m=vert_now,
        tcpa_s=tcpa, dcpa_m=dcpa, vertical_at_tcpa_m=abs(dz + vz * tcpa),
        t_los_s=t_los, los_end_s=los_end,
        in_los_now=horiz_now < minima.horizontal_m and vert_now < minima.vertical_m)


def detect(own: Track, others, minima: Minima = Minima(),
           lookahead_s: float = 30.0) -> list[PairPrediction]:
    """Predicted conflicts with every other track, soonest first. Tracks with
    non-finite values are skipped (garbage in a report is not a conflict to resolve)."""
    out = [predict(own, o, minima, lookahead_s) for o in others
           if o.id != own.id and o.finite and own.finite]
    return sorted((p for p in out if p.conflict), key=lambda p: p.t_los_s)


# ------------------------------------------------------------ building tracks
def track_from_observation(obs) -> Track:
    """Own-ship from its own delivered observation (velocity from the payload)."""
    v = obs.payload.get("velocity_ned_mps") or {}

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return math.nan
    return Track(obs.vehicle_id, obs.north_m, obs.east_m, obs.altitude_m,
                 f(v.get("x", 0.0)), f(v.get("y", 0.0)), -f(v.get("z", 0.0)))


def track_from_peer(peer: dict) -> Track:
    """A peer from the agent traffic view (channel/remote_id.traffic_for_model):
    reported position, extrapolated by the report's age along its reported motion."""
    pos = peer.get("position") or {}
    gs = peer.get("ground_speed_mps") or 0.0
    trk = peer.get("track_deg")
    vn = ve = 0.0
    if trk is not None and gs:
        vn, ve = gs * math.cos(math.radians(trk)), gs * math.sin(math.radians(trk))

    def f(x):
        return math.nan if x is None else float(x)
    t = Track(peer.get("uas_id", "?"), f(pos.get("north_m")), f(pos.get("east_m")),
              f(pos.get("altitude_m")), vn, ve, peer.get("vertical_speed_mps") or 0.0)
    return t.advanced(peer.get("age_s") or 0.0)


def intent_track_from_peer(peer: dict, min_change_m: float = 0.5) -> Track | None:
    """The same peer at its ANNOUNCED target altitude (level, same horizontal motion), or
    None if it announced none or is already there. Planning treats a peer as occupying
    both its current altitude and its target (live 2026-10-28: two drones picked the
    same escape layer because each only saw where the other was, not where it was going)."""
    tgt = peer.get("target_altitude_m")
    now = track_from_peer(peer)
    if tgt is None or not now.finite or not math.isfinite(float(tgt))             or abs(float(tgt) - now.altitude_m) < min_change_m:
        return None
    return Track(now.id, now.north_m, now.east_m, float(tgt), now.v_north_mps, now.v_east_mps, 0.0)


def prediction_view(p: PairPrediction) -> dict:
    """Agent-facing summary of one prediction (rounded)."""
    def r(x, nd=1):
        return None if x is None or not math.isfinite(x) else round(x, nd)
    return {"with": p.other_id, "time_to_loss_of_separation_s": r(p.t_los_s),
            "time_to_closest_approach_s": r(p.tcpa_s), "closest_horizontal_m": r(p.dcpa_m),
            "vertical_at_closest_m": r(p.vertical_at_tcpa_m),
            "horizontal_now_m": r(p.horizontal_now_m), "vertical_now_m": r(p.vertical_now_m),
            "in_loss_of_separation_now": p.in_los_now}
