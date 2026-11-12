"""
The search loop: what one drone does from takeoff height to the flight home. The same loop
runs in every phase; two parts plug into it.

    loop = SearchLoop("Drone1", mission, feed, port, belief, chooser=FixedChooser())
    result = await loop.run()

    chooser   picks the next action from the sectors still to search:
                FixedChooser (atal/agents/fixed_chooser.py)  next one in the fixed order
                LLMChooser   (atal/agents/llm_chooser.py)    an LLM picks it
    role      says which sectors are this drone's and what the rest of the team hears:
                SoloRole (below)                              the whole grid, no radio
                TeamRole (atal/agents/team_coordinator.py)    a share of the open sectors,
                                                              claims and clears on the radio

What run() does:

    1. start: observe; the role plans this drone's sectors (recorded as `sar_plan`); climb
       in place to the mission altitude; the role's after_climb (a team drone flies to its
       staging point)
    2. one step, repeated: stop if the search is over (a detection, the time limit, a peer's
       find); ask the role what is left; ask the chooser for the next action; submit it
       through the action port; update the drone's memory (belief) from the feedback
    3. finish: fly home through the port (once more, higher, if the way home is blocked)

With the managed airspace (team runs, mission swarm.airspace: managed) a third part plugs
in: `recovery` (atal/flight/recovery.py). It owns every climb and descent after an
obstacle: the drone climbs to a transit level of its own (a reservation from the trusted
transition manager), flies on up there, and comes down to its search level only at the
next sector's entry point. The loop only asks it two things: recover(cause) after an
obstacle, and descend_at(entry) before a sweep while it is up at a transit level. The
paragraph below is the behaviour without it (one-drone runs, and unmanaged team runs kept
to reproduce earlier results).

Obstacles (wall hits, a blocked path): the runner reports an impact the simulator saw
(on_impact, which also interrupts the sweep), and the path follower gives up a path it
cannot make progress on ("path_blocked"). Either way the loop recovers instead of grinding
on: the drone is already hovering; it climbs recovery_climb_m above where it is, marks the
sector it was flying to as blocked for itself (never chosen again, the guard refuses it),
and carries on with the rest. After max_recoveries recoveries, or if the climb fails, the
drone drops out of the search ("lost": no return flight over the city; the runner lands it
where it is) and, in a team, stops taking cells so its teammates share them out. An impact
still fails the run's acceptance: recovery keeps the mission going, it does not make a
collision acceptable.

Every action is a typed Pydantic action (atal/core/action.py) converted to a ProposedAction and
submitted with the observation it was decided from, whichever chooser picked it: the port,
the executor and the log are the same for both.

The loop sees only its own observations, its belief and the frozen mission. It never sees
the scenario (victim position): a detection reaches it as a Detection from its own sensor,
via on_detection().
"""
import time
from dataclasses import dataclass, field

from atal.core.action import ActionType, DroneState, ProposedAction, ReturnHome
from atal.core.belief import Detection, LocalBelief
from atal.core.mission import Mission
from atal.search.sweep_lanes import sweep_path
from atal.search.voronoi import coverage_order, voronoi_partition

STALE = object()        # submit(): the `before` check withdrew the action
AGAIN = object()        # _come_down_for(): no sweep now, choose again

# The name each chooser and role combination is logged under (run_config.json:
# pipeline.planner). These are the class names from before the loop was split into parts,
# kept so that old and new run logs read the same.
PLANNER_NAMES = {
    ("fixed", False): "FixedSectorPlanner",
    ("llm", False): "LLMSectorPlanner",
    ("fixed", True): "SwarmFixedPlanner",
    ("llm", True): "SwarmLLMPlanner",
}


def planner_name(chooser: str, team: bool) -> str:
    """The logged name for a chooser kind ("fixed" or "llm") alone or in a team."""
    return PLANNER_NAMES[(chooser, team)]


@dataclass
class SearchResult:
    outcome: str                 # victim_found | all_swept | time_limit | returned | failed |
                                 # aborted | lost | blocked_sectors
                                 # (victim_found whenever the sensor detected the victim)
    reason: str
    order: list[str]
    swept: list[str]
    detection: Detection | None
    actions: list[dict] = field(default_factory=list)   # (action, feedback code) per submission
    ended_by: str = ""           # what ended the search, never overwritten by a detection
    return_home: str | None = None   # feedback code of the final return home
    recoveries: int = 0              # obstacle recoveries (impacts / blocked paths)
    blocked: list[str] = field(default_factory=list)   # sectors it could not get through

    @property
    def healthy(self) -> bool:
        """No failure along the way and the drone made it home."""
        return (self.ended_by not in ("failed", "aborted", "lost")
                and self.return_home == "home_reached")


class SoloRole:
    """A drone searching alone: its sectors are its Voronoi cell of the grid (one drone: all
    of it) in a fixed coverage order, and there is nobody to tell anything.

    Every role has these methods; TeamRole (atal/agents/team_coordinator.py) is the other one.
    """
    team = False

    def __init__(self, fleet_positions=None):
        self._fleet_positions = fleet_positions      # optional: peers' positions for the split
        self.loop: "SearchLoop | None" = None

    def attach(self, loop: "SearchLoop") -> None:
        self.loop = loop

    def plan(self, state) -> list[str]:
        """Voronoi partition over the fleet (one drone: the whole grid) and this drone's
        coverage order; sets the guard's cells; recorded as sar_plan."""
        loop = self.loop
        positions = {loop.vehicle_id: (state.x_m, state.y_m)}
        if self._fleet_positions:
            positions.update(self._fleet_positions())
        grid = loop.mission.grid
        cells = voronoi_partition(grid, positions)
        order = coverage_order(grid, cells[loop.vehicle_id], (state.x_m, state.y_m),
                               loop.mission.search.start_sector)
        if loop.guard is not None:
            loop.guard.start_sim_s, loop.guard.cells = loop.start_sim_s, set(order)
        loop.record("sar_plan", allocation="voronoi", fleet=sorted(positions),
                    cells=cells[loop.vehicle_id], order=order,
                    start={"x_m": state.x_m, "y_m": state.y_m}, derived=loop.mission.derived())
        return order

    def remaining(self) -> list[str]:
        """This drone's sectors still to search, in the order to fly them (blocked ones
        left out)."""
        loop = self.loop
        return [sid for sid in loop.order if not loop.belief.is_done(sid)
                and sid not in loop.blocked]

    def still_open(self, sid: str) -> bool:
        """Right before submitting: is the chosen sector still this drone's to sweep?"""
        return True

    def nothing_left(self) -> tuple[str, str]:
        loop = self.loop
        if loop.blocked and any(not loop.belief.is_done(s) for s in loop.blocked):
            return "blocked_sectors", (f"every reachable sector swept; blocked: "
                                       f"{', '.join(sorted(loop.blocked))}")
        return "all_swept", "every sector swept"

    async def after_climb(self) -> str | None:
        """At mission altitude, before the first sector. Returns a failure reason, or None."""
        return None

    def on_sweep_start(self, sid: str, request_id: str) -> None:
        """Just before a sweep is submitted."""

    def on_sector_swept(self, sid: str, request_id: str) -> None:
        """A sweep finished."""

    def peer_stop(self):
        """(outcome, reason) when peers ended the search, else None."""
        return None

    def on_finish(self) -> None:
        """The search is over and the drone is about to fly home."""

    def on_lost(self) -> None:
        """The drone dropped out."""

    def on_blocked(self, sid: str, cause: str) -> None:
        """An obstacle stopped this drone flying to sid."""

    def situation(self) -> dict | None:
        """What the LLM chooser is told about the team (None: no team)."""
        return None


class SearchLoop:
    def __init__(self, vehicle_id: str, mission: Mission, feed, port, belief: LocalBelief, *,
                 chooser, role=None, guard=None, recovery=None, sweep_progress=None,
                 recorder=None, observe_timeout_s: float = 5.0, log=print):
        self.vehicle_id, self.mission = vehicle_id, mission
        self.guard = guard               # SarGuard (atal/trust_layer/invariant_guard.py) or None
        self.return_code: str | None = None
        self.feed, self.port, self.belief = feed, port, belief
        self._sweep_progress = sweep_progress            # share of the last sweep flown
        self._recorder, self._observe_timeout_s, self.log = recorder, observe_timeout_s, log
        self.start_sim_s: float | None = None
        self.detection: Detection | None = None
        self.time_up = False
        self.order: list[str] = []
        self.actions: list[dict] = []
        self.last_state: DroneState | None = None
        self.gate_refusals_in_row = 0
        self.max_refusals_in_row = 3
        # A sweep the flight controller could not complete (it hovered) is retried from
        # wherever the drone is; only this many in a row end the search (live 2026-11-05:
        # one path_failed in G2 grounded Drone3 for the rest of the run).
        self.flight_failures_in_row = 0
        self.max_flight_failures_in_row = 2
        self.return_outcome = "returned"     # outcome when the chooser returns ReturnHome
        self.stale_choices = 0
        # Obstacle recovery (see the module docstring).
        self.phase = "starting"              # starting | search | returning | ended
        self.blocked: set[str] = set()       # sectors this drone could not get through
        self.recoveries = 0
        self.max_recoveries = 2
        self.recovery_climb_m = 10.0
        self._impact: dict | None = None     # set by on_impact, handled by the loop
        self._dropout: str | None = None     # set by on_dropout (injected failure)
        self.impacts_reported = 0
        self.recovery = recovery             # managed airspace only (see the module docstring)
        self._avoid: set[str] = set()        # sectors whose descent was not granted just now
        # The two plug-in parts. The role first: the chooser may ask it whether this drone
        # is in a team.
        self.role = role if role is not None else SoloRole()
        self.role.attach(self)
        self.chooser = chooser
        self.chooser.attach(self)

    @property
    def name(self) -> str:
        """The name this loop is logged under (see PLANNER_NAMES)."""
        return planner_name(self.chooser.kind, self.role.team)

    # ------------------------------------------------------------ events from outside
    def on_detection(self, d: Detection) -> None:
        """The drone's own sensor saw the victim."""
        if self.detection is None:
            self.detection = d

    def on_time_up(self) -> None:
        self.time_up = True

    def on_dropout(self, reason: str) -> bool:
        """Experiment: the drone fails mid-search (the dropout condition). True if it was
        still airborne before or in the search (a climb in progress finishes first); the
        caller then interrupts the move and the drone drops out as lost."""
        if self.phase not in ("starting", "search") or self._dropout is not None:
            return False
        self._dropout = reason
        return True

    def on_impact(self, info: dict) -> bool:
        """The simulator reported an impact (runner, ground truth). True if the loop will
        recover from it (it is searching); the caller then interrupts the sweep. Impacts
        while climbing, staging or flying home are only recorded."""
        self.impacts_reported += 1
        if self.phase not in ("search", "returning"):
            return False
        if self._impact is None:
            self._impact = dict(info)
        return True

    # ------------------------------------------------------------ run
    async def run(self) -> SearchResult:
        if self.guard is not None:
            self.guard.start_sim_s = None        # set again when the role plans
        ended = await self._start()
        if ended is not None:
            return ended
        self.phase = "search"
        if self.recovery is not None:
            self.recovery.searching()
        while True:
            ended = await self._step()
            if ended is not None:
                return ended

    async def _start(self) -> SearchResult | None:
        """Plan, climb to the mission altitude, and the role's after_climb. Returns a result
        if the search cannot start, else None."""
        obs, state = await self.observe()
        if state is None:
            return self._done("failed", "no observation before planning")
        self.start_sim_s = state.sim_time_s
        self.order = self.role.plan(state)

        fb = await self.submit(ProposedAction(
            vehicle_id=self.vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
            altitude_m=self.mission.flight.altitude_m,
            reason="climb in place to mission altitude"))
        if fb is None or fb.code != "completed":
            return await self._finish("failed", f"climb to mission altitude: "
                                                f"{fb.code if fb else 'no observation'}")
        if self.recovery is not None:
            if not await self.recovery.settle_startup():
                return self._unresolved("startup climb did not settle in its reservation")
            self.recovery.searching()            # from here on it moves horizontally
        failed = await self.role.after_climb()
        if failed:
            return await self._finish("failed", failed)
        return None

    async def _step(self) -> SearchResult | None:
        """Decide and fly one action. Returns a result when the search is over, else None."""
        if self._dropout is not None:
            return await self._lost(self._dropout)
        if self._impact is not None:          # a hit between sweeps (or while deciding)
            lost = await self._recover(None, "impact")
            if lost is not None:
                return lost
        stop = self._should_stop()
        if stop:
            return await self._finish(*stop)
        remaining = self.role.remaining()
        if not remaining:
            return await self._finish(*self.role.nothing_left())
        remaining = [s for s in remaining if s not in self._avoid] or remaining
        action = await self.chooser.choose(remaining)
        # The decision may have taken a while (an LLM): a detection or the time limit
        # that arrived meanwhile ends the search before anything new is flown.
        stop = self._should_stop()
        if stop:
            return await self._finish(*stop)
        if isinstance(action, ReturnHome):
            return await self._finish(self.return_outcome, action.reason)
        sid = action.sector_id
        if self.recovery is not None and self.recovery.k > 0:   # up at a transit level
            ended = await self._come_down_for(sid)
            if ended is AGAIN:
                return None
            if ended is not None:
                return ended
        self._avoid.clear()
        obs_sim = self.sim_now()

        def before(rid, sid=sid, obs_sim=obs_sim):
            if not self.role.still_open(sid):
                return False
            self.belief.mark_started(sid, obs_sim, rid)
            self.role.on_sweep_start(sid, rid)
        fb = await self.submit(action.to_proposed(self.vehicle_id), before=before)
        if fb is STALE:
            # it changed hands while deciding (in a team: a peer claimed or cleared it)
            self.stale_choices += 1
            if self.stale_choices > 100:
                return await self._finish("failed", "choices keep going stale")
            return None
        return await self._after_sweep(sid, fb)

    async def _after_sweep(self, sid: str, fb) -> SearchResult | None:
        """Update the memory from a sweep's feedback and decide whether to go on."""
        sim = self.sim_now()
        if self._dropout is not None:
            if fb is not None and fb.code != "sector_swept":
                self.belief.mark_partial(sid, sim, self._sweep_progress()
                                         if self._sweep_progress else 0.0, fb.request_id)
            elif fb is not None:
                self.belief.mark_swept(sid, sim, fb.request_id)
            return await self._lost(self._dropout)
        if fb is None:
            return await self._finish("failed", f"{sid}: no observation to decide from")
        if fb.code == "sector_swept":
            self.belief.mark_swept(sid, sim, fb.request_id)
            self.role.on_sector_swept(sid, fb.request_id)
            self.gate_refusals_in_row = 0
            self.flight_failures_in_row = 0
            return None                     # an impact at the very end: the next step recovers
        if fb.stage.value in ("port", "gate"):
            # Refused before anything flew: the sector is untouched; decide again.
            self.belief.mark_partial(sid, sim, 0.0, fb.request_id)
            self.gate_refusals_in_row += 1
            if self.gate_refusals_in_row >= self.max_refusals_in_row:
                return await self._finish("failed", f"{self.gate_refusals_in_row} refusals in "
                                                    f"a row, last {sid}: {fb.code}")
            return None
        progress = self._sweep_progress() if self._sweep_progress else 0.0
        self.belief.mark_partial(sid, sim, progress, fb.request_id)
        if self._impact is not None or fb.code == "path_blocked":
            return await self._recover(sid, "impact" if self._impact is not None
                                       else "path_blocked")
        if fb.code == "sweep_stopped":
            stop = self._should_stop()
            return await self._finish(*(stop or ("aborted", f"{sid}: sweep stopped")))
        if fb.code == "failed":        # path_failed and other flight-controller failures
            self.flight_failures_in_row += 1
            self.record("sweep_failed", sector_id=sid, message=fb.message,
                        in_row=self.flight_failures_in_row, progress=progress)
            if self.flight_failures_in_row < self.max_flight_failures_in_row:
                self.log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: {sid} "
                         f"failed ({fb.message}); deciding again")
                return None
        return await self._finish("failed", f"{sid}: {fb.code}: {fb.message}")

    # ------------------------------------------------------------ obstacle recovery
    async def _recover(self, sid: str | None, cause: str):
        """After an impact or a blocked path: climb clear, block the sector, carry on.
        Returns a SearchResult if the drone drops out, else None."""
        info, self._impact = self._impact or {}, None
        self.recoveries += 1
        if sid is not None:
            self.blocked.add(sid)
            if self.guard is not None:
                self.guard.blocked.add(sid)
            self.role.on_blocked(sid, cause)
        self.record("obstacle", cause=cause, sector_id=sid, recovery=self.recoveries,
                    object_name=info.get("object_name"),
                    impact_speed_mps=info.get("impact_speed_mps"),
                    blocked=sorted(self.blocked))
        self.log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: {cause}"
                 + (f" in {sid}" if sid else "")
                 + (f" ({info['object_name']})" if info.get("object_name") else "")
                 + f"; recovery {self.recoveries}/{self.max_recoveries}")
        if self.recovery is not None:                     # managed airspace
            return await self._ended_by(await self.recovery.recover(cause), cause)
        if self.recoveries > self.max_recoveries:
            return await self._lost(f"{self.recoveries} obstacle incidents (last: {cause}"
                                    + (f" in {sid}" if sid else "") + ")")
        _, state = await self.observe()
        if state is None:
            return await self._lost(f"{cause}: no observation to recover from")
        target = round(state.altitude_m + self.recovery_climb_m, 2)
        if self.guard is not None:
            self.guard.recovery_alt_m = target
        try:
            fb = await self.submit(ProposedAction(
                vehicle_id=self.vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
                altitude_m=target, reason=f"{cause}: climb {self.recovery_climb_m:g} m clear"))
        finally:
            if self.guard is not None:
                self.guard.recovery_alt_m = None
        if fb is None or fb.code != "completed":
            return await self._lost(f"{cause}: recovery climb "
                                    f"{fb.code if fb else 'had no observation'}")
        self.flight_failures_in_row = 0
        self.record("obstacle_recovered", cause=cause, sector_id=sid, altitude_m=target,
                    recovery=self.recoveries)
        return None

    async def _come_down_for(self, sid: str):
        """Managed airspace, up at a transit level: fly to sid's entry point up there and
        come down to the search level (atal/flight/recovery.py). None: sweep it now.
        AGAIN: choose again. A SearchResult: the search is over."""
        _, state = await self.observe()
        if state is None:
            return await self._finish("failed", f"{sid}: no observation to decide from")
        entry = sweep_path(self.mission, sid, (state.x_m, state.y_m)).waypoints[0]
        result = await self.recovery.descend_at(entry)
        if result == "ready":
            return None
        self.record("recovery_descent", sector_id=sid, result=result, level_m=self.recovery.level)
        if result == "stopped" and self._impact is None:   # a detection, the time limit
            stop = self._should_stop()
            return await self._finish(*(stop or ("aborted", f"{sid}: transit stopped")))
        if result in ("obstacle", "stopped"):   # hit something on the way to sid: as
            # without the managed airspace, sid is then blocked (for the team too)
            ended = await self._recover(sid, "impact" if self._impact else "path_blocked")
            return AGAIN if ended is None else ended
        if result == "retarget":
            self._avoid.add(sid)
            return AGAIN
        if result == "blocked":                            # its descent was blocked
            self.blocked.add(sid)
            if self.guard is not None:
                self.guard.blocked.add(sid)
            # the peer message knows two causes; an obstacle under the column is a
            # blocked path (the local record above says "descent")
            self.role.on_blocked(sid, "path_blocked")
            return AGAIN
        if result == "return_high":
            return await self._finish("returned", "descent not granted; home at the transit "
                                                  "level")
        return await self._ended_by(result, f"descent to {sid}")

    async def _ended_by(self, result: str, why: str):
        """The recovery's answer: None to carry on, else the drone is out of the search."""
        if result == "transit":
            self.flight_failures_in_row = 0
            return None
        if result == "lost":
            return await self._lost(f"{why}: {self.recovery.episodes[-1].reason}")
        return self._unresolved(f"{why}: {self.recovery.episodes[-1].reason}"
                                if self.recovery.episodes else why)

    def _unresolved(self, reason: str) -> SearchResult:
        """No safe vertical move left (managed airspace): the drone hovers where it is and
        asks for nothing more; the run fails (unresolved_airborne)."""
        self.log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: UNRESOLVED ({reason})")
        self.record("drone_unresolved", reason=reason, recoveries=self.recoveries)
        self.role.on_lost()
        self.return_code = "not_attempted"
        return self._done("unresolved", reason)

    async def _return_higher(self, fb):
        """The flight home was blocked (or hit something): climb recovery_climb_m and fly
        home at that height, once. Returns the second attempt's feedback, or None. With
        the managed airspace the climb is a recovery to the next transit level."""
        info, self._impact = self._impact or {}, None
        self.recoveries += 1
        self.record("obstacle", cause="impact" if info else "path_blocked", sector_id=None,
                    during="return_home", recovery=self.recoveries,
                    object_name=info.get("object_name"), blocked=sorted(self.blocked))
        if self.recovery is not None:
            if await self.recovery.recover("return home blocked") != "transit":
                return None
            again = await self._fly_home(f"return home at transit level "
                                         f"{self.recovery.level:g} m")
            if again is None or again.code != "home_reached":
                self.recovery.home_blocked("return home blocked at the transit level too")
            return again
        _, state = await self.observe()
        if state is None:
            return None
        target = round(state.altitude_m + self.recovery_climb_m, 2)
        if self.guard is not None:
            self.guard.recovery_alt_m = target
        try:
            climb = await self.submit(ProposedAction(
                vehicle_id=self.vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
                altitude_m=target, reason="return home blocked: climb clear"))
            if climb is None or climb.code != "completed":
                return None
            again = await self.submit(ProposedAction(
                vehicle_id=self.vehicle_id, action_type=ActionType.RETURN_HOME,
                altitude_m=target, reason=f"return home {self.recovery_climb_m:g} m higher"))
        finally:
            if self.guard is not None:
                self.guard.recovery_alt_m = None
        self.record("obstacle_recovered" if again is not None and again.code == "home_reached"
                    else "obstacle_not_recovered", during="return_home", altitude_m=target,
                    code=again.code if again else None, recovery=self.recoveries)
        return again

    async def _lost(self, reason: str) -> SearchResult:
        """Drop out of the search: no return flight across the city after repeated hits;
        the runner descends and lands the drone where it is."""
        self.log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: dropping out of the "
                 f"search ({reason})")
        self.record("drone_lost", reason=reason, recoveries=self.recoveries,
                    blocked=sorted(self.blocked))
        self.role.on_lost()
        self.return_code = "not_attempted"
        return self._done("lost", reason)

    # ------------------------------------------------------------ ending the search
    def _should_stop(self):
        if self.detection is not None and self.mission.termination.stop_on_detection:
            return "victim_found", f"victim detected in {self.detection.sector_id}"
        peer = self.role.peer_stop()
        if peer is not None:
            return peer
        if self.time_up or self._elapsed_sim() > self.mission.termination.max_mission_time_s:
            return "time_limit", (f"max_mission_time_s "
                                  f"{self.mission.termination.max_mission_time_s:g} used up")
        return None

    async def _fly_home(self, reason: str):
        """Return home through the port, at the transit level if still up there."""
        up = self.recovery is not None and self.recovery.k > 0
        if not up:
            return await self.submit(ReturnHome(reason=reason).to_proposed(self.vehicle_id))
        action = ProposedAction(vehicle_id=self.vehicle_id, action_type=ActionType.RETURN_HOME,
                                altitude_m=self.recovery.level, reason=reason)
        if self.guard is not None:
            self.guard.recovery_alt_m = self.recovery.level
        try:
            fb = await self.submit(action)
        finally:
            if self.guard is not None:
                self.guard.recovery_alt_m = None
        if fb is not None and fb.code == "home_reached":
            self.recovery.home_reached()
        return fb

    async def _finish(self, outcome: str, reason: str) -> SearchResult:
        self.role.on_finish()
        self.phase = "returning"
        self._impact = None              # one met before the search ended: already handled
        if self.recovery is not None:
            self.recovery.returning()
        fb = await self._fly_home(f"mission over: {outcome}")
        if fb is not None and (fb.code == "path_blocked" or self._impact is not None):
            fb = await self._return_higher(fb) or fb
        self.phase = "ended"
        self.return_code = fb.code if fb else None
        self.record("return_home", code=self.return_code)
        if self.recovery is not None:
            self.recovery.search_over(self.return_code)
        return self._done(outcome, reason)

    def _done(self, outcome, reason) -> SearchResult:
        self.phase = "ended"
        ended_by = outcome
        if self.detection is not None and outcome != "victim_found":
            outcome, reason = "victim_found", f"victim detected; then {reason}"
        r = SearchResult(outcome=outcome, reason=reason, order=self.order,
                         swept=self.belief.swept(), detection=self.detection,
                         actions=self.actions, ended_by=ended_by,
                         return_home=self.return_code, recoveries=self.recoveries,
                         blocked=sorted(self.blocked))
        self.record("planner_end", outcome=outcome, ended_by=ended_by, reason=reason,
                    return_home=self.return_code, healthy=r.healthy, swept=len(r.swept),
                    sectors=len(self.belief.sectors), recoveries=self.recoveries,
                    blocked=sorted(self.blocked))
        return r

    # ------------------------------------------------------------ used by the parts too
    async def observe(self):
        """(observation, DroneState) from the feed, or (None, None) if none came in time."""
        obs = await self.feed.current(after_sequence=self.port.last_command_ended_sequence,
                                      timeout_s=self._observe_timeout_s)
        if obs is None:
            return None, None
        sim_ns = obs.source_sim_time_ns
        x, y = obs.north_m, obs.east_m
        state = DroneState(vehicle_id=self.vehicle_id,
                           sim_time_s=(sim_ns or 0) / 1e9, x_m=x, y_m=y,
                           altitude_m=obs.altitude_m, sector_id=self.mission.grid.sector_at(x, y),
                           airborne=obs.ground_state != "grounded")
        self.last_state = state
        return obs, state

    async def submit(self, action: ProposedAction, before=None):
        """Submit an action through the port with a fresh observation. `before(request_id)`
        runs just before; if it returns False nothing is submitted and STALE comes back."""
        obs, _ = await self.observe()
        if obs is None:
            return None
        if before is not None and before(action.request_id) is False:
            return STALE            # nothing submitted
        fb = await self.port.submit(action, based_on=obs.observation_id)
        entry = {"action": action.action_type.value, "sector_id": action.sector_id,
                 "code": fb.code, "request_id": fb.request_id}
        self.actions.append(entry)
        self.log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: {entry['action']}"
                 + (f" {action.sector_id}" if action.sector_id else "") + f" -> {fb.code}")
        return fb

    def sim_now(self) -> float | None:
        latest = self.feed.latest()
        ns = latest.source_sim_time_ns if latest is not None else None
        return None if ns is None else round(ns / 1e9, 3)

    def _elapsed_sim(self) -> float:
        now = self.sim_now()
        if now is None or self.start_sim_s is None:
            return 0.0
        return now - self.start_sim_s

    def record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, source="planner", **fields)
