"""
Obstacle recovery for one drone of a team (recovery policy v3.2, option A): after an
impact or a blocked path, climb to a transit level of its own, fly on up there, and come
back down to the search level only where the next sector starts. Every vertical move is a
reservation from the transition manager (atal/airspace/transition_manager.py).

    rec = Recovery("Drone2", search_m=160, transit_m=[190, 220], manager=mgr,
                   flight=flight, sim_now=..., record=...)
    outcome = await rec.recover("impact")          # "transit", "lost" or "unresolved"
    ready = await rec.descend_at(entry_xy)         # "ready", "retarget", "return_high",
                                                   # "obstacle", "blocked" or "unresolved"
    rec.level                                      # the altitude to fly at now

The search loop owns the search; this class owns the climbs and descents. It never picks
a sector and never changes the drone's beliefs. It talks to two things:

    manager   the trusted transition manager: phase reports, requests, the atomic start,
              STUCK reports. Its answers are about this drone's own moves only.
    flight    this drone's own flight interface (see RecoveryFlight): hold, the drone's
              own telemetry, change altitude, fly to a point at an altitude.

Waiting rules: a request is made only once the drone's own telemetry shows it holding
(confirmed holding, section 3 of the policy). A climb not granted within max_hold_s ends
UNRESOLVED (the drone hovers, asks for nothing more; the run fails). A descent not granted
within retarget_s retargets to another sector; after max_retargets the drone goes home at
its transit level. A vertical move that stops partway is STUCK: its reservation is a
keep-out, and the drone may retry from where it stopped (confirmed stationary there), at
most max_stuck_retries times. Timeouts never authorise a move.

Each recovery episode ends with an outcome of its own (completed, unresolved or lost),
recorded apart from the run's acceptance: an episode can complete in a run that fails.
"""
import asyncio
import contextlib
import itertools
from dataclasses import dataclass, field
from typing import Protocol

from atal.airspace.transition_manager import Kind, Phase, State


@dataclass
class Telemetry:
    t_s: float
    x_m: float
    y_m: float
    alt_m: float
    v_h_mps: float
    v_z_mps: float


class RecoveryFlight(Protocol):
    async def hold(self) -> None: ...
    def telemetry(self) -> Telemetry | None: ...
    async def change_altitude(self, alt_m: float) -> str: ...     # "completed" or a failure
    async def fly_to(self, xy, alt_m: float) -> str: ...           # "reached", "blocked", ...


@dataclass
class Episode:
    cause: str
    started_s: float | None
    from_m: float
    outcome: str = "open"              # completed | unresolved | lost | open
    reason: str = ""
    levels: list[float] = field(default_factory=list)
    ended_s: float | None = None


class Recovery:
    def __init__(self, vehicle_id: str, search_m: float, transit_m: list[float], manager,
                 flight: RecoveryFlight, sim_now, record=None, *, max_recoveries: int = 2,
                 max_hold_s: float = 60.0, retarget_s: float = 30.0, max_retargets: int = 2,
                 max_stuck_retries: int = 2, confirm_timeout_s: float = 15.0,
                 poll_s: float = 0.05, sleep=asyncio.sleep):
        self.vehicle_id = vehicle_id
        self.search_m = float(search_m)
        self.transit_m = [float(a) for a in transit_m]
        self.mgr, self.flight = manager, flight
        self._now, self._record_cb, self._sleep = sim_now, record, sleep
        self.max_recoveries, self.max_hold_s = max_recoveries, max_hold_s
        self.retarget_s, self.max_retargets = retarget_s, max_retargets
        self.max_stuck_retries, self.confirm_timeout_s = max_stuck_retries, confirm_timeout_s
        self.poll_s = poll_s
        self.k = 0                       # 0: at the search level; k: at transit level k
        self.recoveries = 0
        self.retargets = 0
        self.episodes: list[Episode] = []
        self.outcome: str | None = None  # "lost" or "unresolved" once the drone gives up
        self.startup = None              # its startup reservation (set by the runner)
        self.landing_wait: dict | None = None   # the landing request's queue wait
        self.landing = None              # the landing reservation it is descending under
        self._ids = itertools.count(1)

    # ------------------------------------------------------------ where it flies
    @property
    def level(self) -> float:
        return self.search_m if self.k == 0 else self.transit_m[self.k - 1]

    @property
    def episode(self) -> Episode | None:
        e = self.episodes[-1] if self.episodes else None
        return e if e is not None and e.outcome == "open" else None

    def phase(self, phase: Phase) -> None:
        self.mgr.set_phase(self.vehicle_id, phase)

    # ------------------------------------------------------------ recovering
    async def recover(self, cause: str) -> str:
        """After an impact or a blocked path: up to the next transit level. Returns
        "transit" (now holding level k at the new level), "lost" or "unresolved"."""
        await self.flight.hold()
        self.phase(Phase.HOLDING)
        self.recoveries += 1
        if self.episode is None:
            self.episodes.append(Episode(cause, self._now(), self.level))
        self._record("recovery_started", cause=cause, recovery=self.recoveries,
                     at_m=self.level)
        if self.recoveries > self.max_recoveries:
            return self._lost(f"{self.recoveries} obstacle incidents")
        if self.k >= len(self.transit_m):
            return self._lost("no transit level left below the ceiling")
        target = self.transit_m[self.k]
        result = await self._vertical(Kind.CLIMB, target, deadline_s=self.max_hold_s)
        if result == "completed":
            self.k += 1
            self.episode.levels.append(target)
            self.phase(Phase.TRANSIT)
            self._record("recovery_transit", level_m=target, k=self.k)
            return "transit"
        return self._unresolved(f"climb to {target:g} m: {result}")

    async def descend_at(self, entry_xy) -> str:
        """Before a sweep: if up at a transit level, fly there to the sector's entry point,
        then come down to the search level.

        ready        at the search level: sweep
        retarget     descent not granted in time: choose another sector
        return_high  retargeted max_retargets times: go home at this transit level
        obstacle     hit something on the way at the transit level: recover again
        stopped      interrupted on the way (a detection, the time limit): stop searching
        blocked      the descent itself was blocked; back up at the transit level:
                     mark the sector blocked and choose another
        unresolved   no safe option left (the drone hovers)
        """
        if self.outcome is not None:
            return self.outcome
        if self.k == 0:
            return "ready"
        self.phase(Phase.TRANSIT)
        flown = await self.flight.fly_to(entry_xy, self.level)
        if flown == "stopped":
            return "stopped"                 # interrupted (a detection, the time limit)
        if flown != "reached":
            return "obstacle"
        await self.flight.hold()
        self.phase(Phase.HOLDING)
        here = self.level
        result = await self._vertical(Kind.DESCENT, self.search_m, deadline_s=self.retarget_s)
        if result == "completed":
            self.k, self.retargets = 0, 0
            self.phase(Phase.SEARCHING)
            self._close("completed", "back at the search level")
            return "ready"
        if result in ("expired", "deadlock", "not_granted"):
            self.retargets += 1
            self._record("descent_retarget", retargets=self.retargets, reason=result)
            self.phase(Phase.TRANSIT)
            return "return_high" if self.retargets >= self.max_retargets else "retarget"
        if result == "back_up":
            self.phase(Phase.TRANSIT)
            return "blocked"
        return self._unresolved(f"descent from {here:g} m: {result}")

    async def settle_startup(self) -> bool:
        """After the climb to the search level: wait (holding) until the manager completes
        the startup reservation. False if it did not (it became a keep-out)."""
        r = self.startup
        if r is None:
            return True
        await self.flight.hold()
        return await self._settled(r)

    def searching(self) -> None:
        self.phase(Phase.SEARCHING)

    def returning(self) -> None:
        self.phase(Phase.RETURNING)

    def home_blocked(self, why: str) -> None:
        """The flight home at the transit level failed too: the drone drops out (LOST)
        and the runner lands it where it is, under a landing reservation."""
        if self.outcome is None:
            self._lost(why)

    def search_over(self, return_code: str | None) -> None:
        """The search and its return flight are over. An episode still open (the drone
        was up at a transit level when the search ended) gets its outcome here:
        completed if it flew home at its transit level (home_reached, or held off the
        pads by a pad_zone_timeout), otherwise lost (it lands where it is)."""
        e = self.episode
        if e is None or self.outcome is not None:
            return
        if return_code in ("home_reached", "pad_zone_timeout"):
            self._close("completed", f"returned at its transit level ({return_code})")
        else:
            self._lost(f"search over at a transit level; return home: {return_code}")

    def home_reached(self) -> None:
        """Home over the pad at the transit level: the episode is complete."""
        if self.episode is not None:
            self._close("completed", "home at its transit level")

    async def clear_landing(self, ground_m: float = 0.0, deadline_s: float | None = None,
                            ) -> bool:
        """Landing sequence steps 3 to 5 (policy section 11): the drone is above its pad
        (or at its slot after a pad_zone_timeout) holding the zone, at its current level.
        Confirm holding there, request a landing reservation bound to this position and
        level, and do the atomic start. True: descend and land now. False: no
        reservation by deadline_s (the drone stays holding; never land, never disarm).

        deadline_s is the drone's cleanup deadline in sim time, fixed once by the caller
        (atal/airspace/landing_queue.py); None means now + max_hold_s. Every request made
        here waits until that same deadline, so retries never extend it. The queue wait
        (request to grant) is recorded as landing_queue."""
        self.phase(Phase.HOLDING)
        deadline = self._now() + self.max_hold_s if deadline_s is None else deadline_s
        self.landing_wait = None
        while self._now() < deadline:
            if not await self._confirm_holding(self.level, stationary=False):
                return False
            r = await self._ask(Kind.LANDING, self.level, ground_m, until=deadline,
                                stationary=False)
            if r.state == State.DENIED and r.reason == "not_holding":
                await self._sleep(self.poll_s)
                continue
            self.landing_wait = {"requested_at": r.requested_at, "granted_at": r.granted_at,
                                 "deadline": round(deadline, 2), "state": r.state.value,
                                 "queue_wait_s": round((r.granted_at if r.granted_at
                                                        is not None else self._now())
                                                       - r.requested_at, 2)}
            self._record("landing_queue", **self.landing_wait, reason=r.reason)
            if r.state != State.GRANTED:
                self._record("landing_not_cleared", reason=r.reason or r.state.value)
                return False
            while True:            # the atomic start, within the grant's start_by
                answer = self.mgr.start(r.reservation_id, r.request_id, self.vehicle_id)
                if answer == "started":
                    self.phase(Phase.MOVING)
                    self.landing = r
                    return True
                if answer != "not_holding":                  # lapsed or cancelled
                    self._record("landing_not_cleared", reason=f"grant not started: {answer}")
                    return False
                await self._sleep(self.poll_s)
        self._record("landing_not_cleared", reason="cleanup deadline passed")
        return False

    # ------------------------------------------------------------ one vertical move
    async def _vertical(self, kind: Kind, to_m: float, deadline_s: float) -> str:
        """Hold, request, start, move, settle. Returns "completed", or why not:
        "expired" / "deadlock" / "not_granted" (nothing moved), "back_up" (a failed
        descent went back up to where it started), or "stuck" / a denial reason."""
        start_t = self._now()
        from_m = self.level
        if not await self._confirm_holding(from_m, stationary=False):
            return "not_holding"
        while True:
            left = deadline_s - (self._now() - start_t)
            if left <= 0:
                return "expired"
            r = await self._ask(kind, from_m, to_m, left, stationary=False)
            if r.state == State.DENIED and r.reason == "not_holding":
                # the manager samples a little later than the drone: its own second of
                # holding is not up yet. Nothing moved; ask again in the time left.
                await self._sleep(self.poll_s)
                continue
            if r.state != State.GRANTED:
                return r.reason if r.state == State.DENIED else r.state.value
            started = self.mgr.start(r.reservation_id, r.request_id, self.vehicle_id)
            if started == "started":
                break
            self._record("start_refused", reservation_id=r.reservation_id, answer=started)
            # nothing was commanded; confirm holding again and ask again in the time left
            if not await self._confirm_holding(from_m, stationary=False):
                return "not_holding"
        return await self._move(r, kind, from_m, to_m)

    async def _move(self, r, kind: Kind, from_m: float, to_m: float) -> str:
        self.phase(Phase.MOVING)
        done = await self._within_validity(r, self.flight.change_altitude(to_m))
        if done == "completed" and await self._settled(r):
            return "completed"
        # stopped partway (or did not settle): STUCK, a keep-out with no gap
        self.mgr.report_stuck(self.vehicle_id, f"{kind.value} to {to_m:g} m: {done}")
        self.phase(Phase.STUCK)
        await self.flight.hold()
        self._record("transition_stuck", transition=kind.value, to_m=to_m, result=done)
        retry_to = from_m if kind == Kind.DESCENT else to_m     # a failed descent goes back up
        for attempt in range(1, self.max_stuck_retries + 1):
            if not await self._confirm_holding(None, stationary=True):
                continue
            tel = self.flight.telemetry()
            f = await self._ask(kind if retry_to == to_m else Kind.CLIMB, tel.alt_m, retry_to,
                                self.max_hold_s, stationary=True)
            asked = self._now()
            while (f.state == State.DENIED and f.reason == "not_holding"
                   and self._now() - asked < self.confirm_timeout_s):
                await self._sleep(self.poll_s)            # see _vertical
                tel = self.flight.telemetry()
                f = await self._ask(kind if retry_to == to_m else Kind.CLIMB, tel.alt_m,
                                    retry_to, self.max_hold_s, stationary=True)
            if f.state != State.GRANTED:
                self._record("stuck_retry_refused", attempt=attempt, reason=f.reason
                             or f.state.value)
                continue
            if self.mgr.start(f.reservation_id, f.request_id, self.vehicle_id) != "started":
                continue
            self.phase(Phase.MOVING)
            again = await self._within_validity(f, self.flight.change_altitude(retry_to))
            if again == "completed" and await self._settled(f):
                self.phase(Phase.HOLDING)
                return "completed" if retry_to == to_m else "back_up"
            self.mgr.report_stuck(self.vehicle_id, f"retry {attempt}: {again}")
            self.phase(Phase.STUCK)
            await self.flight.hold()
        return "stuck"

    async def landing_descent(self, to_m: float, ground_m: float, deadline_s: float) -> str:
        """Landing step 6: the descent to just above the pad (to_m), under the landing
        reservation clear_landing started. Returns "descended" (holding just above the
        pad, the landing reservation still ACTIVE) or why not: "not_cleared" (no active
        landing reservation) or "stuck" (it stopped and no follow-up grant got it down).

        The move never runs past the reservation's validity time (_within_validity). If
        it stops because telemetry went invalid mid-command (telemetry_lost; the executor
        has already hovered), it is sent again ONCE under the same reservation, and only
        when that reservation is still ACTIVE inside its validity time and the drone's own
        telemetry is valid again and shows it stationary. Any other stop, a second one, or
        a reservation that overran into a keep-out is STUCK and goes through the follow-up
        process every reserved move uses: keep-out, confirmed stationary, a fresh LANDING
        grant from where it stopped, at most max_stuck_retries times, and never waiting
        past deadline_s (the drone's one cleanup deadline). No timeout authorises a move."""
        r = self.landing
        if r is None or r.state != State.ACTIVE or r.kind != Kind.LANDING:
            return "not_cleared"
        same_reservation_retry = True
        follow_ups = 0
        while True:
            self.phase(Phase.MOVING)
            done = await self._within_validity(r, self.flight.change_altitude(to_m))
            if done == "completed" and r.state == State.ACTIVE:
                return "descended"
            self._record("landing_descent_stopped", reservation_id=r.reservation_id,
                         result=done, reservation_state=r.state.value)
            if done == "telemetry_lost" and same_reservation_retry:
                same_reservation_retry = False
                if (await self._confirm_holding(None, stationary=True)
                        and r.state == State.ACTIVE and self._now() < r.valid_until):
                    self._record("landing_descent_retry", reservation_id=r.reservation_id,
                                 validity_left_s=round(r.valid_until - self._now(), 2))
                    continue
            if r.state == State.ACTIVE:
                self.mgr.report_stuck(self.vehicle_id, f"landing descent: {done}")
            self.phase(Phase.STUCK)
            await self.flight.hold()
            self._record("transition_stuck", transition=Kind.LANDING.value, to_m=to_m,
                         result=done)
            f = None
            while f is None and follow_ups < self.max_stuck_retries \
                    and self._now() < deadline_s:
                follow_ups += 1
                f = await self._landing_follow_up(ground_m, deadline_s, follow_ups)
            if f is None:
                self.landing = None
                return "stuck"
            r = self.landing = f

    async def _landing_follow_up(self, ground_m: float, deadline_s: float, attempt: int):
        """A fresh LANDING grant from where a stuck landing stopped (confirmed stationary),
        started atomically; None if there is none by deadline_s."""
        if not await self._confirm_holding(None, stationary=True):
            self._record("stuck_retry_refused", attempt=attempt, reason="not_stationary")
            return None
        while True:
            tel = self.flight.telemetry()
            if tel is None:
                self._record("stuck_retry_refused", attempt=attempt, reason="no_telemetry")
                return None
            f = await self._ask(Kind.LANDING, tel.alt_m, ground_m, stationary=True,
                                until=deadline_s)
            if f.state == State.DENIED and f.reason == "not_holding" \
                    and self._now() < deadline_s:
                await self._sleep(self.poll_s)                # see _vertical
                continue
            break
        if f.state != State.GRANTED:
            self._record("stuck_retry_refused", attempt=attempt,
                         reason=f.reason or f.state.value)
            return None
        if self.mgr.start(f.reservation_id, f.request_id, self.vehicle_id) != "started":
            self._record("stuck_retry_refused", attempt=attempt, reason="start_refused")
            return None
        return f

    def landing_validity_left_s(self) -> float | None:
        """Sim seconds left in the active landing reservation's validity; None without one
        (no manager, or not cleared): the land command must finish inside it."""
        r = self.landing
        if r is None or r.state not in (State.ACTIVE, State.KEEP_OUT) \
                or r.valid_until is None:
            return None
        return max(0.0, r.valid_until - self._now()) if r.state == State.ACTIVE else 0.0

    async def _ask(self, kind, from_m, to_m, wait_s=None, stationary=False, until=None):
        """Request and wait for the answer; the drone keeps holding meanwhile. The
        request waits wait_s from now, or until the absolute sim time until."""
        tel = self.flight.telemetry()
        now = self._now()
        rid = f"{self.vehicle_id}-{kind.value}-{next(self._ids)}"
        self.phase(Phase.REQUEST if not stationary else Phase.STUCK)
        r = self.mgr.request(self.vehicle_id, rid, kind, (tel.x_m, tel.y_m), from_m, to_m,
                             wait_until=until if until is not None else now + wait_s,
                             from_stationary=stationary)
        while r.state == State.REQUESTED:
            await self._sleep(self.poll_s)
        if r.state == State.GRANTED and self._now() > r.start_by:
            self.mgr.cancel(rid, self.vehicle_id)        # a late grant is never used
        return r

    async def _within_validity(self, r, move) -> str:
        """Run a reserved move, but never past its validity time: once the manager turns
        the reservation into a keep-out (or valid_until passes) the command is dropped
        and the drone holds, so a move that cannot finish (a roof under the column, a
        command that never returns) becomes STUCK instead of waiting on a host timeout."""
        task = asyncio.ensure_future(move)
        while not task.done():
            if r.state == State.KEEP_OUT or (r.valid_until is not None
                                             and self._now() > r.valid_until):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                return "overran_validity"
            await self._sleep(self.poll_s)
        return task.result()

    async def _settled(self, r) -> bool:
        """Wait for the manager's own surveillance to complete the reservation."""
        while r.state in (State.ACTIVE, State.SETTLING):
            await self._sleep(self.poll_s)
        return r.state == State.COMPLETED

    async def _confirm_holding(self, alt_m: float | None, stationary: bool) -> bool:
        """The drone's own telemetry: still for confirm_hold_s, at alt_m (or, stationary,
        wherever it stopped)."""
        b = self.mgr.bounds
        t0 = self._now()
        since, ref = None, None
        while self._now() - t0 <= self.confirm_timeout_s:
            tel = self.flight.telemetry()
            still = (tel is not None and tel.v_h_mps < b.hold_h_mps
                     and abs(tel.v_z_mps) < b.hold_v_mps
                     and (alt_m is None or abs(tel.alt_m - alt_m) <= b.e_z))
            if still and ref is not None and abs(tel.alt_m - ref) > b.e_z:
                still = False
            if not still:
                since, ref = None, None
            elif since is None:
                since, ref = tel.t_s, tel.alt_m
            elif tel.t_s - since >= b.confirm_hold_s:
                return True
            await self._sleep(self.poll_s)
        return False

    # ------------------------------------------------------------ outcomes
    def _lost(self, why: str) -> str:
        self.outcome = "lost"
        self.phase(Phase.LOST)
        self._close("lost", why)
        return "lost"

    def _unresolved(self, why: str) -> str:
        self.outcome = "unresolved"
        self.phase(Phase.UNRESOLVED)
        self._close("unresolved", why)
        return "unresolved"

    def _close(self, outcome: str, why: str) -> None:
        e = self.episode
        if e is not None:
            e.outcome, e.reason, e.ended_s = outcome, why, self._now()
        self._record("recovery_outcome", outcome=outcome, reason=why,
                     recovery=self.recoveries)

    def _record(self, kind: str, **fields) -> None:
        if self._record_cb is not None:
            self._record_cb(kind, vehicle_id=self.vehicle_id, **fields)

    def summary(self) -> dict:
        outs = [e.outcome for e in self.episodes]
        return {"episodes": len(outs), "completed": outs.count("completed"),
                "unresolved": outs.count("unresolved"), "lost": outs.count("lost"),
                "open": outs.count("open")}


class DroneFlight:
    """RecoveryFlight over one drone's own flight stack. Like takeoff and landing, these
    are flight-safety steps run outside the action port: the agent did not choose them and
    the gate does not judge them; the transition manager reserves every vertical move and
    the run log records each step (source "recovery")."""

    def __init__(self, adapter, vehicle_id: str, flight_executor, sector_executor,
                 hover_timeout_s: float = 5.0):
        self.adapter, self.vehicle_id = adapter, vehicle_id
        self._flight, self._sector = flight_executor, sector_executor
        self._hover_timeout_s = hover_timeout_s

    async def hold(self) -> None:
        drone = self.adapter.drone(self.vehicle_id)
        async def hover():
            await (await drone.hover_async())
        # holding is best effort; the manager's own surveillance decides "holding"
        with contextlib.suppress(Exception):
            await asyncio.wait_for(hover(), self._hover_timeout_s)

    def telemetry(self) -> Telemetry | None:
        try:
            s = self.adapter.get_snapshot(self.vehicle_id)
        except Exception:                 # noqa: BLE001
            return None
        if s is None or s.validation_status.value != "valid":
            return None                   # stale or invalid is not the drone's own state
        return Telemetry(s.sim_time_s, s.position_ned_m.x, s.position_ned_m.y,
                         s.altitude_local_m, s.ground_speed_mps, s.vertical_speed_mps)

    async def change_altitude(self, alt_m: float) -> str:
        r = await self._flight.change_altitude(self.vehicle_id, alt_m,
                                               reason="recovery: reserved vertical move")
        return "completed" if r.status.value == "succeeded" else (r.reason_code or "failed")

    async def fly_to(self, xy, alt_m: float) -> str:
        """To the next sector's entry point at the transit level, through the sector
        executor (so traffic pauses and interrupts apply). "reached", "stopped" (an
        interrupt: a detection, the time limit), or the failure code."""
        from atal.core.action import ActionType, ProposedAction
        r = await self._sector.execute(ProposedAction(
            vehicle_id=self.vehicle_id, action_type=ActionType.MOVE_TO_POSITION,
            north_m=float(xy[0]), east_m=float(xy[1]), altitude_m=float(alt_m),
            reason="recovery: transit at the transit level"))
        if r.reason_code == "position_reached":
            return "reached"
        return "stopped" if r.reason_code == "sweep_stopped" else (r.reason_code or "failed")
