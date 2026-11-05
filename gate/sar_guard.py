"""
Search and rescue invariant guard (Phase 2): rules every search action must keep, whoever
proposes it. The same check runs in two places:

    1. the LLM planner's output validator: a choice that breaks a rule goes back to the
       model with the rule's code and message (Pydantic AI retry), so the model can fix it;
    2. the action port's gate slot (SarInvariantGate): nothing that breaks a rule reaches
       the executor, even if a planner skips step 1.

It is a VALIDITY guard, not a trust defense: it uses only the drone's own observation, its
own memory and the frozen mission. Trust checks on peers' claims come in Phase 5.

    code                 rule
    unknown_sector       the sector id is not in the mission grid
    not_assigned         the sector is outside this drone's Voronoi cells
    already_swept        this drone's memory already marks the sector done (its own sweep,
                         or a peer's clear the peer-claim rule accepted, Phase 3)
    search_over          the victim was detected (own sensor, or a peer report the
                         peer-claim rule accepted) and the mission stops on detection (a
                         durable stop: holds even if the detection came while deciding)
    not_enough_time      flying there, sweeping it and flying home would overrun
                         termination.max_mission_time_s (estimate, see below)
    premature_return     return_home while assigned sectors are unsearched, the victim has
                         not been detected and at least one of them still fits in the time
                         (planner choices only: the gate always lets a drone go home. That is a
                         trusted-controller privilege, not an invariant: going home is the
                         safe action after a failure, so a planner that skips the validator
                         can end its search early, never fly somewhere it shouldn't)
    blocked_by_obstacle  this drone hit something or was blocked flying to that sector
                         earlier (planner obstacle recovery); not offered again
    wrong_altitude       change_altitude to anything but the mission altitude (or, during
                         an obstacle recovery, the recovery climb the planner set)
    (move_to_position)   only to this drone's staging point at its layer (Phase 3), else
                         not_allowed
    not_allowed          any other action type

Time estimate: path length (straight transit + the sector's boustrophedon lanes + straight
flight home) / flight speed x time_factor + reserve_s. time_factor covers acceleration and
turns: live CityEnviron sweeps of the dev mission took about 2x the straight-line time.
"""
import math
from dataclasses import dataclass

from models.action import ActionType, ProposedAction
from models.belief import LocalBelief
from models.observation import GateDecision, GateOutcome, Observation
from models.sar import Mission
from search.boustrophedon import sweep_path


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str
    message: str


OK = Verdict(True, "ok", "allowed")


class SarGuard:
    def __init__(self, mission: Mission, belief: LocalBelief, home_xy: tuple[float, float],
                 time_factor: float = 2.0, reserve_s: float = 30.0, altitude_tolerance_m: float = 1.0):
        self.mission, self.belief, self.home_xy = mission, belief, home_xy
        self.time_factor, self.reserve_s = time_factor, reserve_s
        self.altitude_tolerance_m = altitude_tolerance_m
        # Set by the planner once it has planned / started:
        self.cells: set[str] | None = None        # this drone's Voronoi cells
        self.start_sim_s: float | None = None     # sim time the search started
        self.peer_victim_accepted = False         # Phase 3: a peer's VictimDetected, accepted
        self.staging_xy: tuple[float, float] | None = None   # Phase 3: the one allowed move
        self.blocked: set[str] = set()            # sectors this drone could not get through
        self.recovery_alt_m: float | None = None  # the climb allowed during a recovery

    def search_over(self) -> bool:
        """The victim was found: by this drone's sensor, or (Phase 3) by a peer whose report
        the peer-claim rule accepted (set by the swarm coordinator)."""
        return self.belief.victim_detected or self.peer_victim_accepted

    # ------------------------------------------------------------ estimates
    def sweep_time_s(self, sid: str, from_xy: tuple[float, float]) -> float:
        """Estimated sim seconds to fly from from_xy to the sector and sweep it."""
        p = sweep_path(self.mission, sid, from_xy)
        return (math.dist(from_xy, p.start) + p.length_m) / self.mission.flight.speed_m_s \
            * self.time_factor

    def sector_time_s(self, sid: str, from_xy: tuple[float, float]) -> float:
        """Estimated sim seconds to fly from from_xy to the sector, sweep it and fly home."""
        p = sweep_path(self.mission, sid, from_xy)
        dist = (math.dist(from_xy, p.start) + p.length_m + math.dist(p.end, self.home_xy))
        return dist / self.mission.flight.speed_m_s * self.time_factor + self.reserve_s

    def remaining_s(self, sim_s: float | None) -> float:
        limit = self.mission.termination.max_mission_time_s
        if sim_s is None or self.start_sim_s is None:
            return limit
        return limit - (sim_s - self.start_sim_s)

    def open_sectors(self) -> list[str]:
        """Assigned sectors not yet swept."""
        ids = self.mission.grid.ids()
        cells = self.cells if self.cells is not None else set(ids)
        return [s for s in ids if s in cells and not self.belief.is_done(s)
                and s not in self.blocked]

    def feasible(self, from_xy, sim_s) -> list[str]:
        """Open sectors that still fit in the remaining time."""
        left = self.remaining_s(sim_s)
        return [s for s in self.open_sectors() if self.sector_time_s(s, from_xy) <= left]

    # ------------------------------------------------------------ the rules
    def check(self, action: ProposedAction, x_m: float, y_m: float, sim_s: float | None,
              planner_choice: bool = True) -> Verdict:
        t = action.action_type
        if t == ActionType.SWEEP_SECTOR:
            sid = action.sector_id
            if not sid or not self.mission.grid.has(sid):
                return Verdict(False, "unknown_sector", f"{sid} is not a sector of the mission "
                               f"grid")
            if self.cells is not None and sid not in self.cells:
                return Verdict(False, "not_assigned", f"{sid} is not in your assigned sectors")
            if self.mission.termination.stop_on_detection and self.search_over():
                return Verdict(False, "search_over", "the victim has been detected; the search "
                               "is over, return home")
            if self.belief.is_done(sid):
                b = self.belief.sectors[sid]
                by = "your memory" if b.verified_by_self else f"cleared by {b.source}"
                return Verdict(False, "already_swept", f"{sid} is already swept ({by}); "
                               f"choose an unsearched sector")
            if sid in self.blocked:
                return Verdict(False, "blocked_by_obstacle", f"you were blocked by an obstacle "
                               f"flying {sid} earlier; choose another sector")
            need, left = self.sector_time_s(sid, (x_m, y_m)), self.remaining_s(sim_s)
            if need > left:
                return Verdict(False, "not_enough_time", f"{sid} needs about {need:.0f} s "
                               f"including the flight home; {max(left, 0):.0f} s of mission time "
                               f"left")
            return OK
        if t == ActionType.RETURN_HOME:
            if action.altitude_m is not None and not (
                    self.recovery_alt_m is not None
                    and abs(action.altitude_m - self.recovery_alt_m) <= self.altitude_tolerance_m):
                return Verdict(False, "wrong_altitude", f"return home flies at "
                               f"{self.mission.flight.altitude_m:g} m (a higher return only as "
                               f"an obstacle recovery)")
            if not planner_choice or self.search_over():
                return OK
            fits = self.feasible((x_m, y_m), sim_s)
            if fits:
                return Verdict(False, "premature_return", f"the victim has not been found and "
                               f"{len(fits)} unsearched sector(s) still fit in the mission time "
                               f"(for example {', '.join(fits[:3])}); keep searching")
            return OK
        if t == ActionType.CHANGE_ALTITUDE:
            alt = self.mission.flight.altitude_m
            if (self.recovery_alt_m is not None and action.altitude_m is not None
                    and abs(action.altitude_m - self.recovery_alt_m) <= self.altitude_tolerance_m):
                return OK
            if action.altitude_m is None or abs(action.altitude_m - alt) > self.altitude_tolerance_m:
                return Verdict(False, "wrong_altitude", f"the search flies at {alt:g} m only")
            return OK
        if t == ActionType.MOVE_TO_POSITION and self.staging_xy is not None:
            sx, sy = self.staging_xy
            alt = self.mission.flight.altitude_m
            if (abs(action.north_m - sx) > 1.0 or abs(action.east_m - sy) > 1.0
                    or abs(action.altitude_m - alt) > self.altitude_tolerance_m):
                return Verdict(False, "not_allowed", f"the only move allowed is to your staging "
                               f"point ({sx:g}, {sy:g}) at {alt:g} m")
            return OK
        return Verdict(False, "not_allowed", f"{t.value} is not a search action")


class SarInvariantGate:
    """The guard in the action port's gate slot. Assessed: APPROVED or REJECTED."""
    name = "sar_invariants"

    def __init__(self, guard: SarGuard, on_verdict=None):
        self.guard = guard
        self._on_verdict = on_verdict

    def decide(self, action: ProposedAction, based_on: Observation) -> GateDecision:
        sim_s = (based_on.source_sim_time_ns / 1e9
                 if based_on.source_sim_time_ns is not None else None)
        v = self.guard.check(action, based_on.north_m, based_on.east_m, sim_s,
                             planner_choice=False)
        if self._on_verdict is not None:
            self._on_verdict(action, v)
        return GateDecision(request_id=action.request_id, observation_id=based_on.observation_id,
                            gate=self.name,
                            outcome=GateOutcome.APPROVED if v.ok else GateOutcome.REJECTED,
                            assessed=True, reason=v.message if not v.ok else f"{v.code}")
