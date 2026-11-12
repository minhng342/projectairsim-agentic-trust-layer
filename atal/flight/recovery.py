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

    def returning(self) -> None:
        self.phase(Phase.RETURNING)

    def home_reached(self) -> None:
        """Home over the pad at the transit level: the episode is complete."""
        if self.episode is not None:
            self._close("completed", "home at its transit level")

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
        done = await self.flight.change_altitude(to_m)
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
            again = await self.flight.change_altitude(retry_to)
            if again == "completed" and await self._settled(f):
                self.phase(Phase.HOLDING)
                return "completed" if retry_to == to_m else "back_up"
            self.mgr.report_stuck(self.vehicle_id, f"retry {attempt}: {again}")
            self.phase(Phase.STUCK)
            await self.flight.hold()
        return "stuck"

    async def _ask(self, kind, from_m, to_m, wait_s, stationary):
        """Request and wait for the answer; the drone keeps holding meanwhile."""
        tel = self.flight.telemetry()
        now = self._now()
        rid = f"{self.vehicle_id}-{kind.value}-{next(self._ids)}"
        self.phase(Phase.REQUEST if not stationary else Phase.STUCK)
        r = self.mgr.request(self.vehicle_id, rid, kind, (tel.x_m, tel.y_m), from_m, to_m,
                             wait_until=now + wait_s, from_stationary=stationary)
        while r.state == State.REQUESTED:
            await self._sleep(self.poll_s)
        if r.state == State.GRANTED and self._now() > r.start_by:
            self.mgr.cancel(rid, self.vehicle_id)        # a late grant is never used
        return r

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
