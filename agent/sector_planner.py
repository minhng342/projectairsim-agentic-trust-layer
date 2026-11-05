"""
Phase 1 planner: one drone, no LLM. Plain code decides the next sector from a fixed list.

    planner = FixedSectorPlanner("Drone1", mission, feed, port, belief)
    result = await planner.run()

    1. observe; Voronoi partition of the grid over the fleet (one drone: all 64 sectors)
       and the drone's coverage order (search/voronoi.py), recorded as `sar_plan`
    2. change_altitude to the mission altitude (climb in place)
    3. for each sector in order: SweepSector through the action port; update the local
       belief (memory) from the feedback: swept, or partial if stopped / failed
    4. stop when every sector is swept, the victim is detected (stop_on_detection), the
       mission time (sim seconds) is used up, or a sweep fails
    5. ReturnHome through the port

Obstacles (wall hits, a blocked path): the runner reports an impact the simulator saw
(on_impact, which also interrupts the sweep), and the path follower gives up a path it
cannot make progress on ("path_blocked"). Either way the planner recovers instead of
grinding on: the drone is already hovering; it climbs recovery_climb_m above where it is,
marks the sector it was flying to as blocked for itself (never chosen again, the guard
refuses it), and carries on with the rest. After max_recoveries recoveries, or if the
climb fails, the drone drops out of the search ("lost": no return flight over the city;
the runner lands it where it is) and, in the swarm, stops taking cells so its teammates
share them out. An impact still fails the run's acceptance: recovery keeps the mission
going, it does not make a collision acceptable.

Every action is a typed Pydantic action (models/sar.py) converted to a ProposedAction and
submitted with the observation it was decided from, exactly like an LLM agent's would be,
so Phase 2 only swaps step 3's "next in the list" for the agent's choice: the port, the
executor and the log stay the same.

The planner sees only its own observations, its belief and the frozen mission. It never
sees the scenario (victim position): a detection reaches it as a Detection from its own
sensor, via on_detection().
"""
import time
from dataclasses import dataclass, field

from models.action import ActionType, ProposedAction
from models.belief import Detection, LocalBelief
from models.sar import DroneState, Mission, ReturnHome, SweepSector
from search.voronoi import coverage_order, voronoi_partition


STALE = object()        # _submit: the `before` check withdrew the action


@dataclass
class PlannerResult:
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


class FixedSectorPlanner:
    name = "FixedSectorPlanner"

    def __init__(self, vehicle_id: str, mission: Mission, feed, port, belief: LocalBelief,
                 fleet_positions=None, sweep_progress=None, recorder=None,
                 observe_timeout_s: float = 5.0, log=print, guard=None):
        self.vehicle_id, self.mission = vehicle_id, mission
        self.guard = guard               # SarGuard (gate/sar_guard.py) or None
        self.return_code: str | None = None
        self.feed, self.port, self.belief = feed, port, belief
        self._fleet_positions = fleet_positions          # Phase 3: peers' positions
        self._sweep_progress = sweep_progress            # share of the last sweep flown
        self._recorder, self._observe_timeout_s, self._log = recorder, observe_timeout_s, log
        self.start_sim_s: float | None = None
        self.detection: Detection | None = None
        self.time_up = False
        self.order: list[str] = []
        self.actions: list[dict] = []
        self.gate_refusals_in_row = 0
        self.max_refusals_in_row = 3
        # A sweep the flight controller could not complete (it hovered) is retried from
        # wherever the drone is; only this many in a row end the search (live 2026-11-05:
        # one path_failed in G2 grounded Drone3 for the rest of the run).
        self.flight_failures_in_row = 0
        self.max_flight_failures_in_row = 2
        self.return_outcome = "returned"     # outcome when choose() returns ReturnHome
        self.stale_choices = 0
        # Obstacle recovery (see the module docstring).
        self.phase = "starting"              # starting | search | ended
        self.blocked: set[str] = set()       # sectors this drone could not get through
        self.recoveries = 0
        self.max_recoveries = 2
        self.recovery_climb_m = 10.0
        self._impact: dict | None = None     # set by on_impact, handled by the loop
        self._dropout: str | None = None     # set by on_dropout (injected failure)
        self.impacts_reported = 0

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
        """The simulator reported an impact (runner, ground truth). True if the planner will
        recover from it (it is searching); the caller then interrupts the sweep. Impacts
        while climbing, staging or flying home are only recorded."""
        self.impacts_reported += 1
        if self.phase not in ("search", "returning"):
            return False
        if self._impact is None:
            self._impact = dict(info)
        return True

    # ------------------------------------------------------------ run
    async def run(self) -> PlannerResult:
        obs, state = await self._observe()
        if state is None:
            return self._done("failed", "no observation before planning")
        self.start_sim_s = state.sim_time_s
        self.order = self._plan(state)

        fb = await self._submit(ProposedAction(
            vehicle_id=self.vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
            altitude_m=self.mission.flight.altitude_m,
            reason="climb in place to mission altitude"))
        if fb is None or fb.code != "completed":
            return await self._finish("failed", f"climb to mission altitude: "
                                                f"{fb.code if fb else 'no observation'}")
        failed = await self._after_climb()
        if failed:
            return await self._finish("failed", failed)

        self.phase = "search"
        while True:
            if self._dropout is not None:
                return await self._lost(self._dropout)
            if self._impact is not None:          # a hit between sweeps (or while deciding)
                lost = await self._recover(None, "impact")
                if lost is not None:
                    return lost
            stop = self._should_stop()
            if stop:
                return await self._finish(*stop)
            remaining = self._remaining()
            if not remaining:
                return await self._finish(*self._nothing_left())
            action = await self.choose(remaining)
            # The decision may have taken a while (an LLM): a detection or the time limit
            # that arrived meanwhile ends the search before anything new is flown.
            stop = self._should_stop()
            if stop:
                return await self._finish(*stop)
            if isinstance(action, ReturnHome):
                return await self._finish(self.return_outcome, action.reason)
            sid = action.sector_id
            obs_sim = self._sim_now()
            def before(rid, sid=sid, obs_sim=obs_sim):
                if not self._still_open(sid):
                    return False
                self.belief.mark_started(sid, obs_sim, rid)
                self._on_sweep_start(sid, rid)
            fb = await self._submit(action.to_proposed(self.vehicle_id), before=before)
            if fb is STALE:
                # it changed hands while deciding (Phase 3: a peer claimed or cleared it)
                self.stale_choices += 1
                if self.stale_choices > 100:
                    return await self._finish("failed", "choices keep going stale")
                continue
            sim = self._sim_now()
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
                self._on_sector_swept(sid, fb.request_id)
                self.gate_refusals_in_row = 0
                self.flight_failures_in_row = 0
                continue                    # an impact at the very end: the loop top recovers
            if fb.stage.value in ("port", "gate"):
                # Refused before anything flew: the sector is untouched; decide again.
                self.belief.mark_partial(sid, sim, 0.0, fb.request_id)
                self.gate_refusals_in_row += 1
                if self.gate_refusals_in_row >= self.max_refusals_in_row:
                    return await self._finish("failed", f"{self.gate_refusals_in_row} refusals in "
                                                        f"a row, last {sid}: {fb.code}")
                continue
            progress = self._sweep_progress() if self._sweep_progress else 0.0
            self.belief.mark_partial(sid, sim, progress, fb.request_id)
            if self._impact is not None or fb.code == "path_blocked":
                lost = await self._recover(sid, "impact" if self._impact is not None
                                           else "path_blocked")
                if lost is not None:
                    return lost
                continue
            if fb.code == "sweep_stopped":
                stop = self._should_stop()
                return await self._finish(*(stop or ("aborted", f"{sid}: sweep stopped")))
            if fb.code == "failed":        # path_failed and other flight-controller failures
                self.flight_failures_in_row += 1
                self._record("sweep_failed", sector_id=sid, message=fb.message,
                             in_row=self.flight_failures_in_row, progress=progress)
                if self.flight_failures_in_row < self.max_flight_failures_in_row:
                    self._log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: {sid} "
                              f"failed ({fb.message}); deciding again")
                    continue
            return await self._finish("failed", f"{sid}: {fb.code}: {fb.message}")

    async def choose(self, remaining: list[str]) -> SweepSector | ReturnHome:
        """The decision Phase 2 hands to the LLM: here, the next sector in the fixed order
        (with a guard: the next one the guard allows, else return home)."""
        reason = f"next sector in the fixed {self.mission.search.sector_order} order"
        if self.guard is None:
            return SweepSector(sector_id=remaining[0], reason=reason)
        _, state = await self._observe()
        if state is None:
            return SweepSector(sector_id=remaining[0], reason=reason)
        for sid in remaining:
            a = SweepSector(sector_id=sid, reason=reason)
            if self.guard.check(a.to_proposed(self.vehicle_id), state.x_m, state.y_m,
                                state.sim_time_s).ok:
                return a
        self.return_outcome = "time_limit"
        return ReturnHome(reason="no unsearched sector fits in the mission time left")

    # ------------------------------------------------------------ hooks (Phase 3 overrides)
    def _plan(self, state) -> list[str]:
        """Voronoi partition over the fleet (one drone: the whole grid) and this drone's
        coverage order; sets the guard's cells; recorded as sar_plan."""
        positions = {self.vehicle_id: (state.x_m, state.y_m)}
        if self._fleet_positions:
            positions.update(self._fleet_positions())
        grid = self.mission.grid
        cells = voronoi_partition(grid, positions)
        order = coverage_order(grid, cells[self.vehicle_id], (state.x_m, state.y_m),
                               self.mission.search.start_sector)
        if self.guard is not None:
            self.guard.start_sim_s, self.guard.cells = self.start_sim_s, set(order)
        self._record("sar_plan", allocation="voronoi", fleet=sorted(positions),
                     cells=cells[self.vehicle_id], order=order,
                     start={"x_m": state.x_m, "y_m": state.y_m}, derived=self.mission.derived())
        return order

    def _remaining(self) -> list[str]:
        """This drone's sectors still to search, in the order to fly them (blocked ones
        left out)."""
        return [sid for sid in self.order if not self.belief.is_done(sid)
                and sid not in self.blocked]

    def _still_open(self, sid: str) -> bool:
        """Right before submitting: is the chosen sector still this drone's to sweep?"""
        return True

    def _nothing_left(self) -> tuple[str, str]:
        if self.blocked and any(not self.belief.is_done(s) for s in self.blocked):
            return "blocked_sectors", (f"every reachable sector swept; blocked: "
                                       f"{', '.join(sorted(self.blocked))}")
        return "all_swept", "every sector swept"

    async def _after_climb(self) -> str | None:
        """At mission altitude, before the first sector (Phase 3: fly to the staging point).
        Returns a failure reason, or None."""
        return None

    def _on_sweep_start(self, sid: str, request_id: str) -> None:
        """Just before a sweep is submitted (Phase 3: claim the sector)."""

    def _on_sector_swept(self, sid: str, request_id: str) -> None:
        """A sweep finished (Phase 3: tell the peers it is clear)."""

    def _peer_stop(self):
        """(outcome, reason) when peers ended the search (Phase 3), else None."""
        return None

    def _on_lost(self) -> None:
        """The drone dropped out (Phase 3: stop taking cells)."""

    # ------------------------------------------------------------ obstacle recovery
    async def _recover(self, sid: str | None, cause: str):
        """After an impact or a blocked path: climb clear, block the sector, carry on.
        Returns a PlannerResult if the drone drops out, else None."""
        info, self._impact = self._impact or {}, None
        self.recoveries += 1
        if sid is not None:
            self.blocked.add(sid)
            if self.guard is not None:
                self.guard.blocked.add(sid)
        self._record("obstacle", cause=cause, sector_id=sid, recovery=self.recoveries,
                     object_name=info.get("object_name"),
                     impact_speed_mps=info.get("impact_speed_mps"),
                     blocked=sorted(self.blocked))
        self._log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: {cause}"
                  + (f" in {sid}" if sid else "")
                  + (f" ({info['object_name']})" if info.get("object_name") else "")
                  + f"; recovery {self.recoveries}/{self.max_recoveries}")
        if self.recoveries > self.max_recoveries:
            return await self._lost(f"{self.recoveries} obstacle incidents (last: {cause}"
                                    + (f" in {sid}" if sid else "") + ")")
        _, state = await self._observe()
        if state is None:
            return await self._lost(f"{cause}: no observation to recover from")
        target = round(state.altitude_m + self.recovery_climb_m, 2)
        if self.guard is not None:
            self.guard.recovery_alt_m = target
        try:
            fb = await self._submit(ProposedAction(
                vehicle_id=self.vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
                altitude_m=target, reason=f"{cause}: climb {self.recovery_climb_m:g} m clear"))
        finally:
            if self.guard is not None:
                self.guard.recovery_alt_m = None
        if fb is None or fb.code != "completed":
            return await self._lost(f"{cause}: recovery climb "
                                    f"{fb.code if fb else 'had no observation'}")
        self.flight_failures_in_row = 0
        self._record("obstacle_recovered", cause=cause, sector_id=sid, altitude_m=target,
                     recovery=self.recoveries)
        return None

    async def _return_higher(self, fb):
        """The flight home was blocked (or hit something): climb recovery_climb_m and fly
        home at that height, once. Returns the second attempt's feedback, or None."""
        info, self._impact = self._impact or {}, None
        self.recoveries += 1
        self._record("obstacle", cause="impact" if info else "path_blocked", sector_id=None,
                     during="return_home", recovery=self.recoveries,
                     object_name=info.get("object_name"), blocked=sorted(self.blocked))
        _, state = await self._observe()
        if state is None:
            return None
        target = round(state.altitude_m + self.recovery_climb_m, 2)
        if self.guard is not None:
            self.guard.recovery_alt_m = target
        try:
            climb = await self._submit(ProposedAction(
                vehicle_id=self.vehicle_id, action_type=ActionType.CHANGE_ALTITUDE,
                altitude_m=target, reason="return home blocked: climb clear"))
            if climb is None or climb.code != "completed":
                return None
            again = await self._submit(ProposedAction(
                vehicle_id=self.vehicle_id, action_type=ActionType.RETURN_HOME,
                altitude_m=target, reason=f"return home {self.recovery_climb_m:g} m higher"))
        finally:
            if self.guard is not None:
                self.guard.recovery_alt_m = None
        self._record("obstacle_recovered" if again is not None and again.code == "home_reached"
                     else "obstacle_not_recovered", during="return_home", altitude_m=target,
                     code=again.code if again else None, recovery=self.recoveries)
        return again

    async def _lost(self, reason: str) -> PlannerResult:
        """Drop out of the search: no return flight across the city after repeated hits;
        the runner descends and lands the drone where it is."""
        self._log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: dropping out of the "
                  f"search ({reason})")
        self._record("drone_lost", reason=reason, recoveries=self.recoveries,
                     blocked=sorted(self.blocked))
        self._on_lost()
        self.return_code = "not_attempted"
        return self._done("lost", reason)

    # ------------------------------------------------------------ helpers
    def _should_stop(self):
        if self.detection is not None and self.mission.termination.stop_on_detection:
            return "victim_found", f"victim detected in {self.detection.sector_id}"
        peer = self._peer_stop()
        if peer is not None:
            return peer
        if self.time_up or self._elapsed_sim() > self.mission.termination.max_mission_time_s:
            return "time_limit", (f"max_mission_time_s "
                                  f"{self.mission.termination.max_mission_time_s:g} used up")
        return None

    async def _finish(self, outcome: str, reason: str) -> PlannerResult:
        self.phase = "returning"
        self._impact = None              # one met before the search ended: already handled
        fb = await self._submit(ReturnHome(reason=f"mission over: {outcome}").to_proposed(
            self.vehicle_id))
        if fb is not None and (fb.code == "path_blocked" or self._impact is not None):
            fb = await self._return_higher(fb) or fb
        self.phase = "ended"
        self.return_code = fb.code if fb else None
        self._record("return_home", code=self.return_code)
        return self._done(outcome, reason)

    def _done(self, outcome, reason) -> PlannerResult:
        self.phase = "ended"
        ended_by = outcome
        if self.detection is not None and outcome != "victim_found":
            outcome, reason = "victim_found", f"victim detected; then {reason}"
        r = PlannerResult(outcome=outcome, reason=reason, order=self.order,
                          swept=self.belief.swept(), detection=self.detection,
                          actions=self.actions, ended_by=ended_by,
                          return_home=self.return_code, recoveries=self.recoveries,
                          blocked=sorted(self.blocked))
        self._record("planner_end", outcome=outcome, ended_by=ended_by, reason=reason,
                     return_home=self.return_code, healthy=r.healthy, swept=len(r.swept),
                     sectors=len(self.belief.sectors), recoveries=self.recoveries,
                     blocked=sorted(self.blocked))
        return r

    async def _observe(self):
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
        self._last_state = state
        return obs, state

    async def _submit(self, action: ProposedAction, before=None):
        obs, _ = await self._observe()
        if obs is None:
            return None
        if before is not None and before(action.request_id) is False:
            return STALE            # nothing submitted
        fb = await self.port.submit(action, based_on=obs.observation_id)
        entry = {"action": action.action_type.value, "sector_id": action.sector_id,
                 "code": fb.code, "request_id": fb.request_id}
        self.actions.append(entry)
        self._log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: {entry['action']}"
                  + (f" {action.sector_id}" if action.sector_id else "") + f" -> {fb.code}")
        return fb

    def _sim_now(self) -> float | None:
        latest = self.feed.latest()
        ns = latest.source_sim_time_ns if latest is not None else None
        return None if ns is None else round(ns / 1e9, 3)

    def _elapsed_sim(self) -> float:
        now = self._sim_now()
        if now is None or self.start_sim_s is None:
            return 0.0
        return now - self.start_sim_s

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, source="planner", **fields)
