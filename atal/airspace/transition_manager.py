"""
The transition manager: reservations for every vertical move of a team (recovery policy
sections 4 to 8).

    mgr = TransitionManager(["Drone1", "Drone2"], levels={"Drone1": [150, 180], ...})
    mgr.update(now_s, {vid: Sample(t_s, x, y, alt, v_h, v_z, grounded)})   # surveillance, 5 Hz
    mgr.set_phase("Drone1", Phase.REQUEST)
    r = mgr.request("Drone1", "Drone1-7", Kind.CLIMB, (x, y), 150, 180, wait_until=now + 60)
    ...  r.state is GRANTED, or still REQUESTED (the drone keeps holding), or a denial
    mgr.start(r.reservation_id, "Drone1-7", "Drone1")   # atomic: "started", or why not
    ...  the drone moves; update() takes the reservation to SETTLING, then COMPLETED

It is trusted flight-safety infrastructure for the simulation, standing in for a UTM or
vertiport service. It sees every drone through its own surveillance (the simulator's true
positions), never through the peer radio. It answers only the requesting drone, about its
own request or move: granted, denied, stop, keep out. It never allocates search sectors,
never changes a drone's beliefs, and gives the agents no positions. The same manager runs
in every comparison condition, so its contribution to separation is the same everywhere
and is not evidence for the agents' trust layer.

Traffic pauses (policy section 8a): a queued request blocked by a drone that can move
horizontally inside its span orders that drone to pause (on_pause); a stop-line order is
the same kind of pause. Each pause has an owner, the reservation that caused it; a drone
resumes (on_resume) only when no owner is left. A paused drone stops blocking only once it
is confirmed stationary and at least d_h from the column. An owner that becomes a keep-out
keeps its pauses; one replaced by its follow-up hands them over in the same step.

Everything here runs in one event loop and nothing awaits, so a check and the state change
it allows (a grant, the start of a move) happen in one step.

Time is SIM seconds throughout (the time of the latest update()).
"""
import itertools
import math
from dataclasses import dataclass, field
from enum import Enum

from atal.airspace.bounds import Bounds


class Kind(str, Enum):
    CLIMB = "climb"
    DESCENT = "descent"
    LANDING = "landing"
    STARTUP = "startup"


class State(str, Enum):
    REQUESTED = "requested"
    GRANTED = "granted"
    ACTIVE = "active"
    SETTLING = "settling"
    COMPLETED = "completed"
    KEEP_OUT = "keep_out"
    SUPERSEDED = "superseded"      # a keep-out replaced, in one step, by its follow-up move
    LAPSED = "lapsed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    DENIED = "denied"


class Phase(str, Enum):
    """A drone's flight state, as its own executor reports it to the manager."""
    GROUNDED = "grounded"          # on its pad (before startup, or landed)
    SEARCHING = "searching"
    TRANSIT = "transit"
    RETURNING = "returning"
    HOLDING = "holding"
    REQUEST = "request"            # confirmed holding and asking for a move
    MOVING = "moving"              # climbing, descending, landing or starting up
    STUCK = "stuck"
    UNRESOLVED = "unresolved"
    LOST = "lost"


MOVERS = {Phase.SEARCHING, Phase.TRANSIT, Phase.RETURNING}     # can move horizontally
HOLDERS = {Phase.HOLDING, Phase.REQUEST, Phase.UNRESOLVED, Phase.LOST}
OPEN = {State.REQUESTED, State.GRANTED, State.ACTIVE, State.SETTLING, State.KEEP_OUT}
OCCUPYING = {State.GRANTED, State.ACTIVE, State.SETTLING, State.KEEP_OUT}
MOVING = {State.ACTIVE, State.SETTLING, State.KEEP_OUT}


@dataclass(frozen=True)
class Sample:
    """One surveillance position: sim time of the position, north/east, altitude above
    the frame origin (up), horizontal and vertical speed, and whether it is on the ground
    (a valid GROUNDED ground state)."""
    t_s: float
    x_m: float
    y_m: float
    alt_m: float
    v_h_mps: float = 0.0
    v_z_mps: float = 0.0
    grounded: bool = False

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x_m, self.y_m)


@dataclass
class Reservation:
    reservation_id: str
    request_id: str
    vehicle_id: str
    kind: Kind
    center_xy: tuple[float, float]
    from_m: float
    to_m: float
    span: tuple[float, float]
    wait_until: float
    requested_at: float
    state: State = State.REQUESTED
    reason: str = ""                       # why it was denied, or how it ended
    granted_at: float | None = None
    start_by: float | None = None
    started_at: float | None = None
    valid_until: float | None = None
    settle_since: float | None = None
    ended_at: float | None = None
    from_stationary: bool = False          # a STUCK drone's follow-up from where it stopped
    replaces: str | None = None            # the keep-out this follow-up takes over

    def overlaps(self, lo: float, hi: float) -> bool:
        return self.span[0] <= hi and lo <= self.span[1]

    def contains(self, alt: float) -> bool:
        return self.span[0] <= alt <= self.span[1]


@dataclass
class StopOrder:
    vehicle_id: str
    reservation_id: str
    distance_m: float
    t_s: float


@dataclass
class _Track:
    sample: Sample | None = None
    still_since: float | None = None       # sim time the current still period began
    still_alt: float | None = None         # altitude at the start of that period
    still_xy: tuple[float, float] | None = None
    phase: Phase = Phase.GROUNDED
    anchor_alt: float | None = None        # the altitude it is cleared to be at
    flags: set = field(default_factory=set)


class TransitionManager:
    def __init__(self, roster, levels: dict[str, list[float]], bounds: Bounds | None = None,
                 ceiling_m: float | None = None, ground_column_m: float | None = None,
                 recorder=None, on_pause=None, on_resume=None):
        self.bounds = bounds = bounds or Bounds()
        self.levels = {vid: [float(a) for a in alts] for vid, alts in levels.items()}
        self.ceiling_m = ceiling_m
        # a grounded drone blocks a column centred within this of it: its own pad only
        # (pads are 3 m apart on dev; e_h = 1.5 m)
        self.ground_column_m = bounds.e_h if ground_column_m is None else ground_column_m
        self.roster: set[str] = set(roster)
        self.tracks: dict[str, _Track] = {vid: _Track() for vid in self.roster}
        self.reservations: dict[str, Reservation] = {}
        self._by_request: dict[str, Reservation] = {}
        self._queue: list[str] = []            # reservation ids still REQUESTED, in order
        self._ids = itertools.count(1)
        self.now: float | None = None
        self.events: list[dict] = []
        self.stop_orders: list[StopOrder] = []
        self._recorder = recorder
        # traffic pauses: vehicle -> the reservations it is paused for
        self.pauses: dict[str, set[str]] = {}
        self._on_pause, self._on_resume = on_pause, on_resume
        self._paused_since: dict[str, float] = {}
        self._skipped: set = set()        # (vid, owner) pauses not made inside d_h
        self.pause_count: dict[str, int] = {}
        self.paused_s: dict[str, float] = {}

    # ------------------------------------------------------------ surveillance
    def update(self, now_s: float, samples: dict[str, Sample]) -> list[StopOrder]:
        """Take one round of surveillance, advance every reservation, try the queue, and
        return the stop orders this round produced."""
        self.now = float(now_s)
        b = self.bounds
        for vid, s in samples.items():
            if vid not in self.tracks:
                continue
            tr = self.tracks[vid]
            if s.v_h_mps > b.v_h + 1e-9 and "speed_bound_exceeded" not in tr.flags:
                tr.flags.add("speed_bound_exceeded")
                self._event("speed_bound_exceeded", vehicle_id=vid, v_h_mps=s.v_h_mps)
            still = s.v_h_mps < b.hold_h_mps and abs(s.v_z_mps) < b.hold_v_mps
            if (not still or tr.still_alt is None or abs(s.alt_m - tr.still_alt) > b.e_z
                    or math.dist(s.xy, tr.still_xy) > b.e_h):
                tr.still_since, tr.still_alt, tr.still_xy = (s.t_s, s.alt_m, s.xy) if still \
                    else (None, None, None)
            tr.sample = s
            if tr.anchor_alt is None:
                tr.anchor_alt = s.alt_m
        self._check_unverified()
        for r in list(self.reservations.values()):
            self._advance(r)
        self._check_unauthorised()
        self._grant_pass()
        return self._stop_line()

    def set_phase(self, vid: str, phase: Phase) -> None:
        tr = self.tracks.get(vid)
        if tr is not None and tr.phase != phase:
            tr.phase = phase

    def fresh(self, vid: str) -> bool:
        tr = self.tracks.get(vid)
        return (tr is not None and tr.sample is not None and self.now is not None
                and self.now - tr.sample.t_s < self.bounds.a_s)

    def holding_at(self, vid: str, alt: float, xy, stationary: bool = False) -> bool:
        """Confirmed holding (or, for a STUCK drone, confirmed stationary) at alt, within
        e_h of xy, for confirm_hold_s, in the manager's own surveillance."""
        tr = self.tracks.get(vid)
        if not self.fresh(vid) or tr.still_since is None:
            return False
        s, b = tr.sample, self.bounds
        if self.now - tr.still_since < b.confirm_hold_s - 1e-9:
            return False
        if math.dist(s.xy, xy) > b.e_h or abs(s.alt_m - alt) > b.e_z:
            return False
        if stationary:
            return abs(alt - tr.still_alt) <= b.e_z
        return any(abs(alt - lv) <= 1e-6 for lv in self.levels.get(vid, []))

    def stationary(self, vid: str) -> bool:
        """Confirmed stationary wherever it is: still for confirm_hold_s, fresh."""
        tr = self.tracks.get(vid)
        return (self.fresh(vid) and tr.still_since is not None
                and self.now - tr.still_since >= self.bounds.confirm_hold_s - 1e-9)

    def grounded_at(self, vid: str, xy) -> bool:
        tr = self.tracks.get(vid)
        return (self.fresh(vid) and tr.sample.grounded
                and math.dist(tr.sample.xy, xy) <= self.bounds.e_h)

    # ------------------------------------------------------------ requests
    def request(self, vid: str, request_id: str, kind: Kind, center_xy, from_m: float,
                to_m: float, wait_until: float, from_stationary: bool = False) -> Reservation:
        """Ask for a vertical move. The same request_id again returns the same record.
        The drone must already be confirmed holding (or grounded, for startup and a landing
        that starts on the ground is never needed); otherwise it is denied at once."""
        kind = Kind(kind)
        old = self._by_request.get(request_id)
        if old is not None:
            if old.vehicle_id != vid:
                return self._denied(vid, request_id, kind, center_xy, from_m, to_m,
                                    wait_until, "request_id_in_use", store=False)
            return old
        now = self.now if self.now is not None else 0.0
        if vid not in self.roster:
            return self._denied(vid, request_id, kind, center_xy, from_m, to_m, wait_until,
                                "not_on_roster")
        mine = [r for r in self.reservations.values()
                if r.vehicle_id == vid and r.state in OPEN]
        keep_out = next((r for r in mine if r.state == State.KEEP_OUT), None)
        if from_stationary and keep_out is None:
            return self._denied(vid, request_id, kind, center_xy, from_m, to_m, wait_until,
                                "not_stuck")
        if any(r is not keep_out for r in mine):
            return self._denied(vid, request_id, kind, center_xy, from_m, to_m, wait_until,
                                "already_pending")
        if keep_out is not None and not from_stationary:
            return self._denied(vid, request_id, kind, center_xy, from_m, to_m, wait_until,
                                "already_pending")
        if self.ceiling_m is not None and max(from_m, to_m) > self.ceiling_m:
            return self._denied(vid, request_id, kind, center_xy, from_m, to_m, wait_until,
                                "above_ceiling")
        if kind == Kind.STARTUP:
            if not self.grounded_at(vid, center_xy):
                return self._denied(vid, request_id, kind, center_xy, from_m, to_m,
                                    wait_until, "not_grounded")
        elif not self.holding_at(vid, from_m, center_xy, stationary=from_stationary):
            return self._denied(vid, request_id, kind, center_xy, from_m, to_m, wait_until,
                                "not_holding")
        f = self.bounds.separation_m
        r = Reservation(reservation_id=f"res-{next(self._ids)}", request_id=request_id,
                        vehicle_id=vid, kind=kind,
                        center_xy=(float(center_xy[0]), float(center_xy[1])),
                        from_m=float(from_m), to_m=float(to_m),
                        span=(min(from_m, to_m) - f, max(from_m, to_m) + f),
                        wait_until=float(wait_until), requested_at=now,
                        from_stationary=from_stationary,
                        replaces=keep_out.reservation_id if keep_out else None)
        self.reservations[r.reservation_id] = r
        self._by_request[request_id] = r
        self._queue.append(r.reservation_id)
        self._event("transition_requested", **self._fields(r))
        self._grant_pass()
        return r

    def cancel(self, request_id: str, vid: str) -> str:
        """Cancel a request or a GRANTED reservation before it starts."""
        r = self._by_request.get(request_id)
        if r is None or r.vehicle_id != vid:
            return "unknown"
        if r.state in (State.REQUESTED, State.GRANTED):
            self._end(r, State.CANCELLED, "cancelled by the requester")
            return "cancelled"
        if r.state in MOVING:
            return "already_started"
        return r.state.value

    def start(self, reservation_id: str, request_id: str, vid: str) -> str:
        """The atomic start: check and move to ACTIVE in one step, before any motion
        command. Anything but "started" means: send no command and keep holding."""
        r = self.reservations.get(reservation_id)
        if r is None or r.request_id != request_id or r.vehicle_id != vid:
            return "mismatch"
        if r.state != State.GRANTED:
            return r.state.value
        if self.now > r.start_by + 1e-9:
            self._lapse(r)
            return r.state.value
        if r.kind == Kind.STARTUP:
            ok = self.grounded_at(vid, r.center_xy)
        else:
            ok = self.holding_at(vid, r.from_m, r.center_xy, stationary=r.from_stationary)
        if not ok:
            return "not_holding"
        r.state, r.started_at = State.ACTIVE, self.now
        r.valid_until = self.now + self.bounds.validity_s(abs(r.to_m - r.from_m))
        if r.replaces:
            old = self.reservations[r.replaces]
            self._hand_over(old.reservation_id, r.reservation_id)
            self._end(old, State.SUPERSEDED, f"taken over by {r.reservation_id}")
        self._event("transition_started", **self._fields(r))
        return "started"

    def report_stuck(self, vid: str, why: str = "") -> Reservation | None:
        """The drone's move stopped partway (blocked, a failed command): its reservation
        becomes a keep-out at once, with the same column and span."""
        r = next((r for r in self.reservations.values()
                  if r.vehicle_id == vid and r.state in (State.ACTIVE, State.SETTLING)), None)
        if r is not None:
            self._keep_out(r, why or "stopped partway")
        return r

    def touchdown(self, vid: str) -> Reservation | None:
        """Touchdown confirmed (a valid GROUNDED snapshot after the land command): the
        landing reservation is COMPLETED. The caller then disarms, then releases the pad."""
        r = next((r for r in self.reservations.values() if r.vehicle_id == vid
                  and r.kind == Kind.LANDING and r.state in MOVING), None)
        if r is not None:
            self._end(r, State.COMPLETED, "touchdown confirmed")
        tr = self.tracks.get(vid)
        if tr is not None:
            tr.phase = Phase.GROUNDED
            if tr.sample is not None:
                tr.anchor_alt = tr.sample.alt_m
        return r

    def leave_roster(self, vid: str, disarmed: bool) -> bool:
        """Only a drone confirmed on the ground AND disarmed leaves the roster. A LOST or
        UNRESOLVED drone stays, and keeps blocking, until the run ends."""
        tr = self.tracks.get(vid)
        if tr is None or not disarmed or not self.fresh(vid) or not tr.sample.grounded:
            return False
        if any(r.vehicle_id == vid and r.state in OPEN for r in self.reservations.values()):
            return False
        self.roster.discard(vid)
        self._event("roster_left", vehicle_id=vid)
        return True

    # ------------------------------------------------------------ the grant rules
    def blockers(self, r: Reservation) -> list[str]:
        """Why r cannot be granted now (empty: it can)."""
        b, out = self.bounds, []
        stale = sorted(v for v in self.roster if not self.fresh(v))
        if stale:
            out.append(f"stale:{','.join(stale)}")                          # rule 1
        if r.kind == Kind.STARTUP:
            if not self.grounded_at(r.vehicle_id, r.center_xy):
                out.append("not_grounded")
        elif not self.holding_at(r.vehicle_id, r.from_m, r.center_xy,
                                 stationary=r.from_stationary):
            out.append("not_holding")                                       # rule 2
        for o in self.reservations.values():                               # rule 3
            if (o.vehicle_id != r.vehicle_id and o.state in OCCUPYING
                    and o.overlaps(*r.span) and math.dist(o.center_xy, r.center_xy) < b.d_h):
                out.append(f"vertical:{o.vehicle_id}")
        radius = b.exclusion_radius(abs(r.to_m - r.from_m))
        for vid in sorted(self.roster):
            tr = self.tracks[vid]
            if vid == r.vehicle_id or tr.sample is None:
                continue
            s = tr.sample
            d = math.dist(s.xy, r.center_xy)
            if self.pauses.get(vid) and not s.grounded:
                # paused: blocks until confirmed stationary, then like a holding drone
                if r.contains(s.alt_m):
                    if not self.stationary(vid):
                        out.append(f"pausing:{vid}")
                    elif d < b.d_h:
                        out.append(f"holding:{vid}")
                continue
            if s.grounded or tr.phase == Phase.GROUNDED:
                if r.contains(s.alt_m) and d <= self.ground_column_m:
                    out.append(f"ground:{vid}")                             # its pad column
            elif tr.phase in HOLDERS:
                if r.contains(s.alt_m) and d < b.d_h:
                    out.append(f"holding:{vid}")                            # rule 4
            elif tr.phase in MOVERS:
                if r.contains(s.alt_m) and d <= radius:
                    out.append(f"mover:{vid}")                              # rule 5
            elif r.contains(s.alt_m) and d < b.d_h:
                # any other phase (MOVING, STUCK, ...): wherever it is now, inside the
                # span and within d_h it blocks, with or without an open reservation
                # (a move that has just completed leaves the drone there)
                out.append(f"occupied:{vid}")
        return out

    def _grant_pass(self) -> None:
        if self.now is None:
            return
        waiting: list[tuple[Reservation, list[str]]] = []
        for rid in list(self._queue):
            r = self.reservations[rid]
            if r.state != State.REQUESTED:
                self._queue.remove(rid)
                continue
            if self.now > r.wait_until + 1e-9:
                self._end(r, State.EXPIRED, "not granted by its deadline (last waiting on "
                                            f"{r.reason or 'nothing recorded'})")
                self._queue.remove(rid)
                continue
            # a later request goes ahead only if it does not conflict with an earlier one,
            # unless that earlier one is itself waiting for this drone to move (holding in
            # its span): waiting behind it then would never end
            ahead = [w for w, w_why in waiting if w.overlaps(*r.span)
                     and math.dist(w.center_xy, r.center_xy) < self.bounds.d_h
                     and f"holding:{r.vehicle_id}" not in w_why]
            self._pause_inside(r)
            why = self.blockers(r)
            if not why and not ahead:
                r.state, r.granted_at = State.GRANTED, self.now
                r.start_by = self.now + self.bounds.start_window_s
                r.reason = ""
                self._queue.remove(rid)
                self._event("transition_granted", **self._fields(r))
                continue
            if r.kind == Kind.DESCENT and self._deadlocked(r, why):
                self._end(r, State.DENIED, "deadlock")
                self._queue.remove(rid)
                continue
            r.reason = why[0] if why else f"behind:{ahead[0].vehicle_id}"
            waiting.append((r, why))

    def _deadlocked(self, r: Reservation, why: list[str]) -> bool:
        """r waits on a holding drone that itself waits on r's drone."""
        for w in why:
            if not w.startswith("holding:"):
                continue
            other = w.split(":", 1)[1]
            theirs = next((o for o in self.reservations.values() if o.vehicle_id == other
                           and o.state == State.REQUESTED), None)
            if theirs is not None and f"holding:{r.vehicle_id}" in self.blockers(theirs) \
                    and theirs.requested_at <= r.requested_at:
                return True
        return False

    # ------------------------------------------------------------ lifecycle
    def _advance(self, r: Reservation) -> None:
        b = self.bounds
        tr = self.tracks.get(r.vehicle_id)
        s = tr.sample if tr else None
        if r.state == State.GRANTED and self.now > r.start_by + 1e-9:
            self._lapse(r)
        elif r.state == State.ACTIVE and s is not None:
            if r.kind != Kind.LANDING and abs(s.alt_m - r.to_m) <= b.e_z:
                r.state, r.settle_since = State.SETTLING, None
            elif self.now > r.valid_until + 1e-9:
                self._keep_out(r, "overran its validity time")
        if r.state == State.SETTLING and s is not None:
            if self.holding_at(r.vehicle_id, r.to_m, s.xy, stationary=True) \
                    and abs(s.alt_m - r.to_m) <= b.e_z:
                if r.settle_since is None:
                    r.settle_since = self.now
                if self.now - r.settle_since >= b.settle_s - 1e-9:
                    self._end(r, State.COMPLETED, "settled at the target")
                    tr.anchor_alt = r.to_m
                    if tr.phase == Phase.MOVING:
                        tr.phase = Phase.HOLDING   # until its drone says what it does next
            else:
                r.settle_since = None                  # drifted: the settle timer restarts
                if self.now > r.valid_until + 1e-9:
                    self._keep_out(r, "did not settle in its validity time")

    def _lapse(self, r: Reservation) -> None:
        """A grant not started by start_by. Released only once the drone is seen still
        holding where it was; if it moved anyway, it becomes a keep-out and the run
        records unauthorised_motion."""
        tr = self.tracks.get(r.vehicle_id)
        s = tr.sample if tr else None
        here = s is not None and math.dist(s.xy, r.center_xy) <= self.bounds.e_h and (
            abs(s.alt_m - r.from_m) <= self.bounds.e_z)
        if here:
            self._end(r, State.LAPSED, "not started by start_by")
        else:
            self._flag(r.vehicle_id, "unauthorised_motion", reservation_id=r.reservation_id)
            self._keep_out(r, "moved without starting its grant")

    def _keep_out(self, r: Reservation, why: str) -> None:
        r.state, r.reason = State.KEEP_OUT, why
        tr = self.tracks.get(r.vehicle_id)
        if tr is not None:
            tr.phase = Phase.STUCK
        self._event("keep_out", **self._fields(r))

    def _end(self, r: Reservation, state: State, why: str) -> None:
        r.state, r.reason, r.ended_at = state, why, self.now
        self._event(f"transition_{state.value}", **self._fields(r))
        self._release_owner(r.reservation_id)

    # ------------------------------------------------------------ traffic pauses
    def _pause(self, vid: str, owner: str, why: str) -> None:
        owners = self.pauses.setdefault(vid, set())
        if owner in owners:
            return
        first = not owners
        owners.add(owner)
        self._event("traffic_pause", vehicle_id=vid, owner=owner, reason=why,
                    owners=sorted(owners))
        if first:
            self._paused_since[vid] = self.now
            self.pause_count[vid] = self.pause_count.get(vid, 0) + 1
            if self._on_pause is not None:
                self._on_pause(vid)

    def _pause_inside(self, r: Reservation) -> None:
        """Every drone that could move horizontally inside r's span and within its
        exclusion radius (whether or not already paused for something else) is paused,
        with r as an owner, so it stays stopped for as long as r needs it.

        Except a drone already within d_h of r's column: pausing it there would hold it
        where it blocks r, and r would wait for it to its deadline (a deadlock the
        manager made). It is left to its own flight plan, never moved aside, and still
        blocks r (as a mover, or as a holder within d_h) until it is clear; it is paused
        once it is outside d_h. (Found on dev: a drone above its pad, 2.8 m from a
        teammate's landing column, paused there for 790 s.)"""
        radius = self.bounds.exclusion_radius(abs(r.to_m - r.from_m))
        for vid in sorted(self.roster):
            tr = self.tracks[vid]
            s = tr.sample
            if vid == r.vehicle_id or s is None or s.grounded:
                continue
            if not ((tr.phase in MOVERS or self.pauses.get(vid)) and r.contains(s.alt_m)):
                continue
            d = math.dist(s.xy, r.center_xy)
            if d < self.bounds.d_h:
                if (vid, r.reservation_id) not in self._skipped:
                    self._skipped.add((vid, r.reservation_id))
                    self._event("traffic_pause_skipped", vehicle_id=vid,
                                owner=r.reservation_id, distance_m=round(d, 2),
                                reason="inside d_h of the column: left to move clear")
                continue
            if d <= radius:
                self._pause(vid, r.reservation_id, "rule 5")

    def _release_owner(self, owner: str) -> None:
        for vid, owners in list(self.pauses.items()):
            if owner not in owners:
                continue
            owners.discard(owner)
            if owners:
                self._event("traffic_pause_owner_ended", vehicle_id=vid, owner=owner,
                            owners=sorted(owners))
                continue
            del self.pauses[vid]
            since = self._paused_since.pop(vid, self.now)
            self.paused_s[vid] = round(self.paused_s.get(vid, 0.0) + self.now - since, 3)
            self._event("traffic_resume", vehicle_id=vid, owner=owner,
                        paused_s=round(self.now - since, 3))
            if self._on_resume is not None:
                self._on_resume(vid)

    def _hand_over(self, old: str, new: str) -> None:
        """A keep-out replaced by its follow-up: its pauses move over in one step."""
        for owners in self.pauses.values():
            if old in owners:
                owners.add(new)

    def paused(self, vid: str) -> bool:
        return bool(self.pauses.get(vid))

    def paused_time(self) -> dict[str, float]:
        """Total paused sim time per drone, a pause still in progress counted up to now."""
        out = dict(self.paused_s)
        for vid, since in self._paused_since.items():
            out[vid] = round(out.get(vid, 0.0) + (self.now - since), 3)
        return out

    def _denied(self, vid, request_id, kind, center_xy, from_m, to_m, wait_until, why,
                store=True) -> Reservation:
        r = Reservation(reservation_id="", request_id=request_id, vehicle_id=vid, kind=kind,
                        center_xy=tuple(center_xy), from_m=from_m, to_m=to_m,
                        span=(min(from_m, to_m), max(from_m, to_m)), wait_until=wait_until,
                        requested_at=self.now or 0.0, state=State.DENIED, reason=why,
                        ended_at=self.now)
        if store:
            self._by_request[request_id] = r
        self._event("transition_denied", **self._fields(r))
        return r

    # ------------------------------------------------------------ monitoring
    def _check_unauthorised(self) -> None:
        """An airborne drone more than e_z off the altitude it is cleared for, with no
        move reserved, moved without a grant. e_z is the per-drone bound the layer spacing
        assumes (spacing >= separation + 2 e_z is the PAIR allowance, two drones each e_z
        off toward each other), so the alarm uses e_z, not 2 e_z."""
        for vid in self.roster:
            tr = self.tracks[vid]
            s = tr.sample
            if s is None or tr.anchor_alt is None:
                continue
            moving = any(r.vehicle_id == vid and r.state in MOVING
                         for r in self.reservations.values())
            if moving:
                continue
            if abs(s.alt_m - tr.anchor_alt) > self.bounds.e_z:
                self._flag(vid, "unauthorised_motion", alt_m=round(s.alt_m, 2),
                           cleared_m=tr.anchor_alt)

    def _check_unverified(self) -> None:
        """Surveillance of any member older than a_s while a move is in progress."""
        if not any(r.state in MOVING for r in self.reservations.values()):
            return
        for vid in self.roster:
            if not self.fresh(vid):
                self._flag(vid, "separation_unverified")

    def _stop_line(self) -> list[StopOrder]:
        """Every drone flying horizontally at an altitude inside the span of a move in
        progress (or a keep-out) and closer than r_stop is ordered to stop."""
        b, out = self.bounds, []
        for r in self.reservations.values():
            if r.state not in MOVING:
                continue
            for vid in self.roster:
                tr = self.tracks[vid]
                s = tr.sample
                if vid == r.vehicle_id or s is None or s.grounded or tr.phase not in MOVERS:
                    continue
                d = math.dist(s.xy, r.center_xy)
                if (r.contains(s.alt_m) and d < b.r_stop
                        and r.reservation_id not in self.pauses.get(vid, ())):
                    o = StopOrder(vid, r.reservation_id, round(d, 2), self.now)
                    out.append(o)
                    self.stop_orders.append(o)
                    self._event("stop_order", vehicle_id=vid, reservation_id=r.reservation_id,
                                distance_m=o.distance_m)
                    self._pause(vid, r.reservation_id, "stop-line")
        return out

    def leg_allowed(self, vid: str, from_xy, to_xy, alt_m: float) -> bool:
        """A horizontal leg about to start: refused if it passes within r_stop of a move in
        progress (or a keep-out) whose span contains its altitude."""
        for r in self.reservations.values():
            if r.vehicle_id == vid or r.state not in MOVING or not r.contains(alt_m):
                continue
            if _point_segment(r.center_xy, from_xy, to_xy) < self.bounds.r_stop:
                return False
        return True

    # ------------------------------------------------------------ records
    def _flag(self, vid: str, flag: str, **fields) -> None:
        tr = self.tracks.get(vid)
        if tr is not None and flag not in tr.flags:
            tr.flags.add(flag)
            self._event(flag, vehicle_id=vid, **fields)

    def flags(self) -> dict[str, list[str]]:
        return {vid: sorted(tr.flags) for vid, tr in self.tracks.items() if tr.flags}

    def _fields(self, r: Reservation) -> dict:
        return {"vehicle_id": r.vehicle_id, "reservation_id": r.reservation_id,
                "request_id": r.request_id, "transition": r.kind.value,
                "from_m": r.from_m, "to_m": r.to_m, "state": r.state.value,
                "reason": r.reason}

    def _event(self, kind: str, **fields) -> None:
        e = {"kind": kind, "t_s": self.now, **fields}
        self.events.append(e)
        if self._recorder is not None:
            self._recorder.record(kind, source="transition_manager", **fields)

    def summary(self) -> dict:
        """What section 13 of the policy records, for the run summary."""
        rs = [r for r in self.reservations.values()]
        waits = [r.granted_at - r.requested_at for r in rs if r.granted_at is not None]

        def count(st):
            return sum(1 for r in rs if r.state == st)
        denied = sum(1 for r in self._by_request.values() if r.state == State.DENIED)
        return {"requests": len(self._by_request), "granted": len(waits), "denied": denied,
                "completed": count(State.COMPLETED), "lapsed": count(State.LAPSED),
                "expired": count(State.EXPIRED), "cancelled": count(State.CANCELLED),
                "keep_outs": count(State.KEEP_OUT) + count(State.SUPERSEDED),
                "stop_orders": len(self.stop_orders),
                "pauses": dict(self.pause_count),
                "paused_s": self.paused_time(),
                "wait_total_s": round(sum(waits), 2),
                "wait_max_s": round(max(waits), 2) if waits else 0.0,
                "flags": self.flags()}


def _point_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.dist(p, (ax + t * dx, ay + t * dy))
