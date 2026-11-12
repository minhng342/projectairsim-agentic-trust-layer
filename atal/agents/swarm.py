"""
Phase 3: what makes one drone a member of the swarm. Per drone:

    SwarmReceiver     on the bus: Remote ID position reports go to the drone's
                      RemoteIdReceiver, typed peer messages (atal/core/peer_messages.py) are validated
                      (invalid ones dropped and recorded, never seen by anything else) and
                      handed to the coordinator
    PeerMessenger     this drone's transmitter for peer messages; sender hooks are the
                      insider's slot in Phase 4 (honest = none)
    SwarmCoordinator  the drone's side of cooperation:
                        incoming  every valid message -> belief (who said what, when) ->
                                  peer-claim rules (atal/trust_layer/peer_rules.py) -> belief / stop
                        allocate  Voronoi over open sectors, live peers, live claims
                                  (atal/search/allocation.py), recomputed at every decision
                        outgoing  only facts this drone verified itself: its own claim
                                  before a sweep, its own SectorCleared after one, its own
                                  sensor's VictimDetected (repeated every victim_repeat_s)
    SwarmPlannerMixin the planner hooks (atal/agents/sector_planner.py) that use the coordinator;
                      SwarmFixedPlanner / SwarmLLMPlanner are the Phase 1 / Phase 2
                      planners with it

Who knows what: the coordinator sees only delivered messages and this drone's own
observations; true senders and true positions stay on the evaluation side (bus records).
Clocks: the coordinator uses this drone's sim clock (receipt times, expiry, liveness);
a message's sim_time_s is the sender's claim.
"""
import asyncio
import copy
import math

from atal.channels.radio import RidContext, _apply_hooks, _hook_name
from atal.core.belief import Detection, LocalBelief, PeerClaim
from atal.core.peer_messages import (
    PEER_MESSAGE,
    SectorBlocked,
    SectorClaimed,
    SectorCleared,
    VictimDetected,
    is_peer_message,
)
from atal.core.sar import Mission
from atal.search.allocation import allocate
from atal.search.voronoi import coverage_order
from atal.trust_layer.peer_rules import PeerClaimRules


class _Delivered:
    """What the bus records about a delivered peer message (its id and claimed sender)."""

    def __init__(self, observation_id, claimed_uas_id):
        self.observation_id, self.claimed_uas_id = observation_id, claimed_uas_id


class SwarmReceiver:
    """authenticated=True: the bus tells this receiver who really transmitted
    (receive_linked); a message claiming another identity is dropped before anything
    reads it (identity_mismatch). False: the claimed id is believed (spoofing condition)."""

    def __init__(self, owner_id: str, rid_receiver, on_message, sim_clock, recorder=None,
                 authenticated: bool = True):
        self.owner_id = owner_id
        self.rid = rid_receiver
        self._on_message, self._sim_clock, self._recorder = on_message, sim_clock, recorder
        self.authenticated = authenticated
        self.dropped = 0
        self.identity_mismatches = 0
        self.processing_errors = 0

    def receive_linked(self, payload, link_sender: str):
        if self.authenticated:
            peer = is_peer_message(payload)
            claimed = (payload.get("sender_id") if peer else payload.get("uas_id")) \
                if isinstance(payload, dict) else None
            if claimed != link_sender:
                self.identity_mismatches += 1
                self._record("identity_mismatch", link_sender=link_sender, claimed=claimed,
                             message_kind="peer_message" if peer else "remote_id",
                             msg_id=payload.get("msg_id") if isinstance(payload, dict) else None)
                return None
        return self.receive(payload)

    def receive(self, payload):
        if not is_peer_message(payload):
            return self.rid.receive(payload)
        try:
            msg = PEER_MESSAGE.validate_python(copy.deepcopy(payload))
        except Exception as err:
            self.dropped += 1
            errors = getattr(err, "errors", None)
            self._record("peer_message_dropped", payload=payload,
                         errors=[f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                                 for e in errors()][:10] if callable(errors) else str(err)[:300])
            return None
        now = self._sim_clock()
        self._record("peer_message", msg=msg.model_dump(mode="json"), received_sim_s=now)
        try:
            self._on_message(msg, now)
        except Exception as err:
            # delivered on the radio, but this drone failed to act on it: an implementation
            # failure, counted and failing the run (never mistaken for attack resistance)
            self.processing_errors += 1
            self._record("peer_message_error", msg_id=msg.msg_id,
                         error=f"{type(err).__name__}: {err}")
        return _Delivered(msg.msg_id, msg.sender_id)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.owner_id, source="swarm", **fields)


class PeerMessenger:
    """Sends this drone's peer messages on the bus. Honest unless given sender hooks."""

    def __init__(self, vehicle_id: str, bus, hooks=(), recorder=None):
        self.vehicle_id, self._bus, self._hooks = vehicle_id, bus, list(hooks)
        self._recorder = recorder
        self._seq = 0
        self.sent = 0

    def send(self, msg) -> list[dict]:
        self._seq += 1
        clean = msg.model_dump(mode="json")
        ctx = RidContext(sender_id=self.vehicle_id, sequence=self._seq)
        sent, changed = _apply_hooks(self._hooks, [copy.deepcopy(clean)], ctx,
                                     lambda h, e: self._record(
                                         "peer_sender_hook_error", hook=_hook_name(h),
                                         error=f"{type(e).__name__}: {e}"))
        self.sent += 1
        self._record("peer_message_sent", msg=clean, tampered_by=changed or None)
        if not sent:
            return []
        return self._bus.transmit(self.vehicle_id, sent, sequence=self._seq, clean=clean,
                                  sender_hooks_changed=changed)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, source="swarm", **fields)


class SwarmCoordinator:
    def __init__(self, vehicle_id: str, mission: Mission, belief: LocalBelief, rid_receiver,
                 messenger: PeerMessenger, sim_clock, guard=None, recorder=None, log=print):
        self.vehicle_id, self.mission, self.belief = vehicle_id, mission, belief
        self.swarm = mission.swarm
        self.rules = PeerClaimRules(self.swarm)
        self.rid, self.messenger, self.sim_clock = rid_receiver, messenger, sim_clock
        self.guard = guard
        self._recorder, self._log = recorder, log
        self._seen: set[str] = set()
        self._claims: dict[str, tuple[PeerClaim, float | None]] = {}   # sender -> latest claim
        self.own_claim: tuple[str, float] | None = None
        self.peer_victim: PeerClaim | None = None
        self.on_peer_victim = None          # set by the runner: interrupt the sweep
        self._victim_msg: VictimDetected | None = None
        self._victim_repeats = 0
        self._last_cells: list[str] | None = None
        self.peer_blocked: dict[str, str] = {}      # sector -> the peer that reported it
        self._blocked_msgs: list[SectorBlocked] = []     # own reports, repeated like the find
        self.state = "searching"            # broadcast as Remote ID mission_state
        # swarm.allocation == "static": this drone's fixed part of the grid, set by the
        # runner before takeoff (the pre-mission split); no claims or clears are sent
        self.static_cells: list[str] | None = None

    # ------------------------------------------------------------ incoming
    def on_message(self, msg, received_sim_s: float | None) -> None:
        if msg.msg_id in self._seen:            # a repeat (VictimDetected) or a duplicate
            return
        self._seen.add(msg.msg_id)
        if msg.sender_id == self.vehicle_id:
            self._record("peer_rule", msg_id=msg.msg_id, msg_type=msg.msg_type,
                         accepted=False, code="claims_own_id",
                         message="a message claiming this drone's id is ignored")
            return
        base = dict(msg_id=msg.msg_id, msg_type=msg.msg_type, sender_id=msg.sender_id,
                    sector_id=getattr(msg, "sector_id", None), sent_sim_s=msg.sim_time_s,
                    received_sim_s=received_sim_s)
        if isinstance(msg, SectorClaimed):
            claim = PeerClaim(**base, expires_sim_s=msg.expires_sim_s)
            exp = self.rules.claim_expiry(claim)
            self._claims[msg.sender_id] = (claim, exp)
            self.belief.add_peer_claim(claim.model_copy(update={"accepted": True,
                                                                "rule": "claim"}))
            self._record("peer_rule", msg_id=msg.msg_id, msg_type=msg.msg_type, accepted=True,
                         code="claim", sector_id=msg.sector_id, sender_id=msg.sender_id,
                         expires_sim_s=exp)
            return
        if isinstance(msg, SectorCleared):
            claim = PeerClaim(**base)
            v = self.rules.on_cleared(self.belief, claim)
            self.belief.add_peer_claim(claim.model_copy(update={"accepted": v.accepted,
                                                                "rule": v.code}))
            changed = False
            if v.accepted:
                changed = self.belief.mark_peer_cleared(msg.sector_id, received_sim_s,
                                                        msg.sender_id, list(v.evidence))
            held = self._claims.get(msg.sender_id)
            if held and held[0].sector_id == msg.sector_id:
                self._claims.pop(msg.sender_id)         # its claim is done
            self._record("peer_rule", msg_id=msg.msg_id, msg_type=msg.msg_type,
                         accepted=v.accepted, code=v.code, message=v.message,
                         sector_id=msg.sector_id, sender_id=msg.sender_id,
                         evidence=list(v.evidence), belief_changed=changed)
            return
        if isinstance(msg, SectorBlocked):
            claim = PeerClaim(**base)
            v = self.rules.on_blocked(self.belief, claim)
            self.belief.add_peer_claim(claim.model_copy(update={"accepted": v.accepted,
                                                                "rule": v.code}))
            held = self._claims.get(msg.sender_id)
            if held and held[0].sector_id == msg.sector_id:
                self._claims.pop(msg.sender_id)         # it is not sweeping there any more
            self._record("peer_rule", msg_id=msg.msg_id, msg_type=msg.msg_type,
                         accepted=v.accepted, code=v.code, message=v.message,
                         sector_id=msg.sector_id, sender_id=msg.sender_id,
                         evidence=list(v.evidence))
            if v.accepted and msg.sector_id not in self.peer_blocked:
                self.peer_blocked[msg.sector_id] = msg.sender_id
                if self.guard is not None:
                    self.guard.blocked.add(msg.sector_id)
                self._record("peer_blocked_heard", sector_id=msg.sector_id,
                             sender_id=msg.sender_id, altitude_m=msg.altitude_m,
                             cause=msg.cause)
                self._log(f"  {self.vehicle_id}: {msg.sender_id} was stopped by an obstacle "
                          f"in {msg.sector_id}; staying out of it")
            return
        if isinstance(msg, VictimDetected):
            claim = PeerClaim(**base, x_m=msg.x_m, y_m=msg.y_m)
            v = self.rules.on_victim(self.belief, claim)
            self.belief.add_peer_claim(claim.model_copy(update={"accepted": v.accepted,
                                                                "rule": v.code}))
            self._record("peer_rule", msg_id=msg.msg_id, msg_type=msg.msg_type,
                         accepted=v.accepted, code=v.code, message=v.message,
                         sector_id=msg.sector_id, sender_id=msg.sender_id,
                         evidence=list(v.evidence))
            if v.accepted and self.peer_victim is None:
                self.peer_victim = claim
                if self.guard is not None:
                    self.guard.peer_victim_accepted = True
                self._log(f"  {self.vehicle_id}: {msg.sender_id} reports the victim in "
                          f"{msg.sector_id}" + ("; search over" if
                                                self.mission.termination.stop_on_detection else ""))
                if self.on_peer_victim is not None:
                    self.on_peer_victim(claim)

    # ------------------------------------------------------------ peers
    def live_peers(self, now: float | None) -> dict[str, tuple[float, float]]:
        """Peers heard on Remote ID within peer_timeout_s that still say they are searching:
        claimed id -> claimed (x, y)."""
        out = {}
        for uid, r in self.rid.latest().items():
            if uid == self.vehicle_id:
                continue
            if now is not None and now - r.received_host_s > self.swarm.peer_timeout_s:
                continue
            if r.payload.get("mission_state", "searching") != "searching":
                continue                    # heading home or landed: no longer takes cells
            x, y = r.num("north_m"), r.num("east_m")
            if math.isfinite(x) and math.isfinite(y):
                out[uid] = (x, y)
        return out

    def live_claims(self, now: float | None, alive) -> dict[str, str]:
        out = {}
        for sender, (c, exp) in self._claims.items():
            if sender not in alive or (exp is not None and now is not None and now > exp):
                continue
            out[c.sector_id] = sender
        if self.own_claim is not None:
            out[self.own_claim[0]] = self.vehicle_id
        return out

    # ------------------------------------------------------------ allocation
    def my_sectors(self, my_xy, now, exclude=()) -> list[str]:
        """This drone's open sectors in coverage order, from a fresh partition. `exclude`:
        sectors this drone cannot fly (it or a peer was stopped by an obstacle there); left
        out of its picture so it steals other work instead."""
        skip = set(exclude)
        if self.static_cells is not None:
            order = coverage_order(self.mission.grid,
                                   [s for s in self.static_cells
                                    if not self.belief.is_done(s) and s not in skip],
                                   tuple(my_xy))
            if order != self._last_cells:
                self._last_cells = order
                self._record("sar_allocation", sim_s=now, allocation="static", order=order,
                             open=len(order))
            return order
        peers = self.live_peers(now)
        positions = {self.vehicle_id: tuple(my_xy), **peers}
        open_s = [s for s in self.belief.unsearched() if s not in skip]
        a = allocate(self.mission.grid, self.vehicle_id, positions, open_s,
                     self.live_claims(now, peers))
        mine = [s for s in a["cells"].get(self.vehicle_id, []) if s in set(open_s)]
        order = coverage_order(self.mission.grid, mine, tuple(my_xy))
        if order != self._last_cells:
            self._last_cells = order
            self._record("sar_allocation", sim_s=now, peers=sorted(peers),
                         positions={k: [round(v[0], 1), round(v[1], 1)] for k, v in positions.items()},
                         open=len(open_s), cells={k: sorted(v) for k, v in a["cells"].items()},
                         claimed=a["claimed"], stolen=a["stolen"], order=order)
        return order

    # ------------------------------------------------------------ outgoing (own facts only)
    def claim(self, sid: str, est_s: float) -> None:
        if self.static_cells is not None:
            return                  # static split: nothing to coordinate
        now = self.sim_clock() or 0.0
        exp = now + min(est_s, self.swarm.max_claim_s)
        self.own_claim = (sid, exp)
        self.messenger.send(SectorClaimed(sender_id=self.vehicle_id, sim_time_s=now,
                                          sector_id=sid, expires_sim_s=exp))

    def cleared(self, sid: str) -> None:
        """Called only after this drone's own sweep of sid finished (verified by self)."""
        if not (self.belief.sectors[sid].verified_by_self
                and self.belief.status(sid).value == "swept"):
            raise ValueError(f"{sid} is not swept by {self.vehicle_id}: not emitting a clear")
        if self.own_claim and self.own_claim[0] == sid:
            self.own_claim = None
        if self.static_cells is not None:
            return                  # static split: no clears on the radio
        if any(d.sector_id == sid for d in self.belief.detections):
            return                  # the sweep found the victim there: it is not clear
        self.messenger.send(SectorCleared(sender_id=self.vehicle_id,
                                          sim_time_s=self.sim_clock() or 0.0, sector_id=sid))

    def blocked(self, sid: str, cause: str, altitude_m: float) -> None:
        """This drone was stopped by an obstacle flying to sid: tell the team, so that one
        obstacle costs one incident. Not a clear: the sector stays unsearched."""
        if self.own_claim and self.own_claim[0] == sid:
            self.own_claim = None
        if self.static_cells is not None:
            return                  # static split: no sector talk on the radio
        msg = SectorBlocked(sender_id=self.vehicle_id, sim_time_s=self.sim_clock() or 0.0,
                            sector_id=sid, altitude_m=altitude_m, cause=cause)
        self._blocked_msgs.append(msg)
        self.messenger.send(msg)

    def own_detection(self, d: Detection) -> None:
        if self._victim_msg is None:
            self._victim_msg = VictimDetected(sender_id=self.vehicle_id, sim_time_s=d.sim_time_s,
                                              sector_id=d.sector_id, x_m=d.x_m, y_m=d.y_m,
                                              footprint_m=d.footprint_m)
            self.messenger.send(self._victim_msg)

    def peers_still_searching(self) -> list[str]:
        """Peers whose LAST Remote ID report (however old) says they are searching. An
        out-of-range peer that never heard the find still counts: it is who the repeats are
        for. No peer heard at all also counts as "maybe", so the repeats go on."""
        latest = {u: r for u, r in self.rid.latest().items() if u != self.vehicle_id}
        if not latest:
            return ["(none heard yet)"]
        return sorted(u for u, r in latest.items()
                      if r.payload.get("mission_state", "searching") == "searching")

    async def repeat_victim(self, period_s: float, stop: asyncio.Event) -> None:
        """Repeat this drone's own VictimDetected (same msg_id) so peers that were out of
        range at the find hear it when they come back in range, until no peer is still
        searching by its last report (then nobody needs it: repeats to drones flying home or
        landed only inflated the undelivered count), or until stop. Own SectorBlocked reports
        repeat the same way (same msg_id, deduplicated by the receivers), for peers that were
        out of range when this drone hit the obstacle."""
        while not stop.is_set():
            try:
                async with asyncio.timeout(period_s):
                    await stop.wait()
            except asyncio.TimeoutError:
                pass
            if stop.is_set():
                continue
            if self._blocked_msgs and self.peers_still_searching():
                for msg in self._blocked_msgs:
                    self.messenger.send(msg)
            if self._victim_msg is None:
                continue
            if not self.peers_still_searching():
                self._record("victim_repeat_stopped", reason="no peer is still searching",
                             repeats_sent=self._victim_repeats)
                return
            self._victim_repeats += 1
            self.messenger.send(self._victim_msg)

    def _record(self, kind, **fields):
        if self._recorder is not None:
            self._recorder.record(kind, vehicle_id=self.vehicle_id, source="swarm", **fields)


class SwarmPlannerMixin:
    """Planner hooks for a swarm member. Put it before the planner class:
    class SwarmFixedPlanner(SwarmPlannerMixin, FixedSectorPlanner)."""

    def __init__(self, *args, swarm: SwarmCoordinator, staging_xy=None, **kw):
        self.swarm = swarm
        self.staging_xy = staging_xy
        super().__init__(*args, **kw)

    async def _after_climb(self):
        if self.staging_xy is None:
            return None
        from atal.core.action import ActionType, ProposedAction
        if self.guard is not None:
            self.guard.staging_xy = self.staging_xy
        x, y = self.staging_xy
        fb = await self._submit(ProposedAction(
            vehicle_id=self.vehicle_id, action_type=ActionType.MOVE_TO_POSITION,
            north_m=x, east_m=y, altitude_m=self.mission.flight.altitude_m,
            reason="fly to the staging point before searching"))
        if self.guard is not None:
            self.guard.staging_xy = None          # a one-time move
        if fb is not None and fb.code == "sweep_stopped":
            return None                           # a find meanwhile: the loop stops the search
        if fb is None or fb.code != "position_reached":
            return f"staging: {fb.code if fb else 'no observation'}"
        self._record("staged", at={"x_m": x, "y_m": y})
        return None

    def _here(self):
        obs = self.feed.latest()
        if obs is None:
            st = getattr(self, "_last_state", None)
            return ((st.x_m, st.y_m) if st else None), None
        ns = obs.source_sim_time_ns
        return (obs.north_m, obs.east_m), (None if ns is None else ns / 1e9)

    def _plan(self, state):
        order = self.swarm.my_sectors((state.x_m, state.y_m), state.sim_time_s)
        if self.guard is not None:
            self.guard.start_sim_s, self.guard.cells = self.start_sim_s, set(order)
        peers = self.swarm.live_peers(state.sim_time_s)
        self._record("sar_plan", allocation=("static" if self.swarm.static_cells is not None
                                             else "voronoi_swarm"),
                     fleet=sorted([self.vehicle_id, *peers]), cells=order, order=order,
                     start={"x_m": state.x_m, "y_m": state.y_m}, derived=self.mission.derived(),
                     altitude_m=self.mission.flight.altitude_m)
        return order

    def _remaining(self):
        xy, now = self._here()
        if xy is None:
            return []
        order = self.swarm.my_sectors(xy, now, exclude=self._all_blocked())
        if self.guard is not None:
            self.guard.cells = set(order)
        return order

    def _all_blocked(self) -> set[str]:
        """Sectors this drone stays out of: its own obstacle incidents and its peers'."""
        return self.blocked | set(self.swarm.peer_blocked)

    def _on_blocked(self, sid, cause):
        self.swarm.blocked(sid, cause, self.mission.flight.altitude_m)

    def _still_open(self, sid):
        """No await between this check and the claim, so a peer's claim, clear or obstacle
        report that arrived while this drone was deciding is never overridden."""
        if self.belief.is_done(sid) or sid in self.swarm.peer_blocked:
            return False
        _, now = self._here()
        peers = self.swarm.live_peers(now)
        holder = self.swarm.live_claims(now, peers).get(sid)
        if holder not in (None, self.vehicle_id):
            self._record("stale_choice", sector_id=sid, claimed_by=holder)
            return False
        return True

    def _nothing_left(self):
        if not self.belief.unsearched():
            return "all_swept", "every sector swept or cleared by a peer"
        stuck = sorted(set(self.belief.unsearched()) & self._all_blocked())
        if stuck:
            return "blocked_sectors", (f"nothing reachable left; blocked by obstacles: "
                                       f"{', '.join(stuck)}")
        if self.swarm.static_cells is not None:
            return "all_swept", "every sector of this drone's fixed part swept"
        return "nothing_assigned", (f"{len(self.belief.unsearched())} open sector(s), all "
                                    f"claimed by or nearer to a peer")

    def _on_sweep_start(self, sid, request_id):
        xy, _ = self._here()
        est = (self.guard.sweep_time_s(sid, xy) if (self.guard is not None and xy)
               else self.mission.swarm.max_claim_s)
        self.swarm.claim(sid, est * 1.5)

    def _on_sector_swept(self, sid, request_id):
        self.swarm.cleared(sid)

    async def _finish(self, outcome, reason):
        self.swarm.state = "returning"
        self.swarm.own_claim = None
        return await super()._finish(outcome, reason)

    def _on_lost(self):
        self.swarm.state = "lost"           # peers stop counting it: its cells are shared out
        self.swarm.own_claim = None

    def _peer_stop(self):
        v = self.swarm.peer_victim
        if v is not None and self.mission.termination.stop_on_detection:
            return "peer_found", f"{v.sender_id} reported the victim in {v.sector_id}"
        return None

    def swarm_situation(self) -> dict:
        """What the LLM planner is told about the team (Phase 3; read-only)."""
        _, now = self._here()
        peers = self.swarm.live_peers(now)
        claims = self.swarm.live_claims(now, peers)
        cleared = {}
        for c in self.belief.peer_claims:
            if c.msg_type == "sector_cleared":
                cleared.setdefault(c.sector_id, []).append(c.sender_id)
        return {
            "teammates": [{"id": p, "north_m": round(x, 1), "east_m": round(y, 1),
                           "sweeping": next((s for s, who in claims.items() if who == p), None)}
                          for p, (x, y) in sorted(peers.items())],
            "assigned_to_you": sorted(self.guard.cells) if self.guard and self.guard.cells else [],
            "blocked_for_you": sorted(self.blocked),
            "blocked_by_teammates": dict(sorted(self.swarm.peer_blocked.items())),
            "cleared_by_teammates": {s: sorted(set(v)) for s, v in sorted(cleared.items())},
        }


def _planner_classes():
    from atal.agents.llm_sector_planner import LLMSectorPlanner
    from atal.agents.sector_planner import FixedSectorPlanner

    class SwarmFixedPlanner(SwarmPlannerMixin, FixedSectorPlanner):
        name = "SwarmFixedPlanner"

    class SwarmLLMPlanner(SwarmPlannerMixin, LLMSectorPlanner):
        name = "SwarmLLMPlanner"
        team = True

    return SwarmFixedPlanner, SwarmLLMPlanner


SwarmFixedPlanner, SwarmLLMPlanner = _planner_classes()
