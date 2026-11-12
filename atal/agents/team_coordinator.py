"""
Phase 3: what makes one drone a member of the team (the swarm). Per drone:

    (transport)       SwarmReceiver and PeerMessenger, atal/channels/peer_link.py: they
                      validate, deliver and send peer messages; this file decides what
                      the messages mean
    SwarmCoordinator  the drone's side of cooperation:
                        incoming  every valid message -> belief (who said what, when) ->
                                  peer-claim rules (atal/trust_layer/peer_rules.py) -> belief / stop
                        allocate  Voronoi over open sectors, live peers, live claims
                                  (atal/search/allocation.py), recomputed at every decision
                        outgoing  only facts this drone verified itself: its own claim
                                  before a sweep, its own SectorCleared after one, its own
                                  sensor's VictimDetected (repeated every victim_repeat_s)
    TeamRole          the search loop's role (atal/agents/search_loop.py) for a team drone:
                      it asks the coordinator which sectors are this drone's, claims a
                      sector before sweeping it and clears it after; it works with either
                      chooser (fixed or LLM)

Who knows what: the coordinator sees only delivered messages and this drone's own
observations; true senders and true positions stay on the evaluation side (bus records).
Clocks: the coordinator uses this drone's sim clock (receipt times, expiry, liveness);
a message's sim_time_s is the sender's claim.
"""
import asyncio
import math

from atal.channels.peer_link import PeerMessenger
from atal.core.action import ActionType, ProposedAction
from atal.core.belief import Detection, LocalBelief, PeerClaim
from atal.core.peer_messages import (
    SectorBlocked,
    SectorClaimed,
    SectorCleared,
    VictimDetected,
)
from atal.core.sar import Mission
from atal.search.allocation import allocate
from atal.search.voronoi import coverage_order
from atal.trust_layer.peer_rules import PeerClaimRules


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


class TeamRole:
    """The search loop's role for a team drone (the other role is SoloRole in
    atal/agents/search_loop.py, with the same methods)."""
    team = True

    def __init__(self, coordinator: SwarmCoordinator, staging_xy=None):
        self.coordinator = coordinator
        self.staging_xy = staging_xy
        self.loop = None

    def attach(self, loop) -> None:
        self.loop = loop

    async def after_climb(self):
        """Fly to this drone's staging point before searching (a one-time move)."""
        loop = self.loop
        if self.staging_xy is None:
            return None
        if loop.guard is not None:
            loop.guard.staging_xy = self.staging_xy
        x, y = self.staging_xy
        fb = await loop.submit(ProposedAction(
            vehicle_id=loop.vehicle_id, action_type=ActionType.MOVE_TO_POSITION,
            north_m=x, east_m=y, altitude_m=loop.mission.flight.altitude_m,
            reason="fly to the staging point before searching"))
        if loop.guard is not None:
            loop.guard.staging_xy = None          # a one-time move
        if fb is not None and fb.code == "sweep_stopped":
            return None                           # a find meanwhile: the loop stops the search
        if fb is None or fb.code != "position_reached":
            return f"staging: {fb.code if fb else 'no observation'}"
        loop.record("staged", at={"x_m": x, "y_m": y})
        return None

    def _here(self):
        loop = self.loop
        obs = loop.feed.latest()
        if obs is None:
            st = loop.last_state
            return ((st.x_m, st.y_m) if st else None), None
        ns = obs.source_sim_time_ns
        return (obs.north_m, obs.east_m), (None if ns is None else ns / 1e9)

    def plan(self, state):
        loop, team = self.loop, self.coordinator
        order = team.my_sectors((state.x_m, state.y_m), state.sim_time_s)
        if loop.guard is not None:
            loop.guard.start_sim_s, loop.guard.cells = loop.start_sim_s, set(order)
        peers = team.live_peers(state.sim_time_s)
        loop.record("sar_plan", allocation=("static" if team.static_cells is not None
                                            else "voronoi_swarm"),
                    fleet=sorted([loop.vehicle_id, *peers]), cells=order, order=order,
                    start={"x_m": state.x_m, "y_m": state.y_m}, derived=loop.mission.derived(),
                    altitude_m=loop.mission.flight.altitude_m)
        return order

    def remaining(self):
        """A fresh share of the open sectors, at every decision."""
        loop = self.loop
        xy, now = self._here()
        if xy is None:
            return []
        order = self.coordinator.my_sectors(xy, now, exclude=self._all_blocked())
        if loop.guard is not None:
            loop.guard.cells = set(order)
        return order

    def _all_blocked(self) -> set[str]:
        """Sectors this drone stays out of: its own obstacle incidents and its peers'."""
        return self.loop.blocked | set(self.coordinator.peer_blocked)

    def on_blocked(self, sid, cause):
        self.coordinator.blocked(sid, cause, self.loop.mission.flight.altitude_m)

    def still_open(self, sid):
        """No await between this check and the claim, so a peer's claim, clear or obstacle
        report that arrived while this drone was deciding is never overridden."""
        loop, team = self.loop, self.coordinator
        if loop.belief.is_done(sid) or sid in team.peer_blocked:
            return False
        _, now = self._here()
        peers = team.live_peers(now)
        holder = team.live_claims(now, peers).get(sid)
        if holder not in (None, loop.vehicle_id):
            loop.record("stale_choice", sector_id=sid, claimed_by=holder)
            return False
        return True

    def nothing_left(self):
        belief = self.loop.belief
        if not belief.unsearched():
            return "all_swept", "every sector swept or cleared by a peer"
        stuck = sorted(set(belief.unsearched()) & self._all_blocked())
        if stuck:
            return "blocked_sectors", (f"nothing reachable left; blocked by obstacles: "
                                       f"{', '.join(stuck)}")
        if self.coordinator.static_cells is not None:
            return "all_swept", "every sector of this drone's fixed part swept"
        return "nothing_assigned", (f"{len(belief.unsearched())} open sector(s), all "
                                    f"claimed by or nearer to a peer")

    def on_sweep_start(self, sid, request_id):
        loop = self.loop
        xy, _ = self._here()
        est = (loop.guard.sweep_time_s(sid, xy) if (loop.guard is not None and xy)
               else loop.mission.swarm.max_claim_s)
        self.coordinator.claim(sid, est * 1.5)

    def on_sector_swept(self, sid, request_id):
        self.coordinator.cleared(sid)

    def on_finish(self):
        self.coordinator.state = "returning"
        self.coordinator.own_claim = None

    def on_lost(self):
        self.coordinator.state = "lost"     # peers stop counting it: its cells are shared out
        self.coordinator.own_claim = None

    def peer_stop(self):
        v = self.coordinator.peer_victim
        if v is not None and self.loop.mission.termination.stop_on_detection:
            return "peer_found", f"{v.sender_id} reported the victim in {v.sector_id}"
        return None

    def situation(self) -> dict:
        """What the LLM chooser is told about the team (read-only)."""
        loop, team = self.loop, self.coordinator
        _, now = self._here()
        peers = team.live_peers(now)
        claims = team.live_claims(now, peers)
        cleared = {}
        for c in loop.belief.peer_claims:
            if c.msg_type == "sector_cleared":
                cleared.setdefault(c.sector_id, []).append(c.sender_id)
        return {
            "teammates": [{"id": p, "north_m": round(x, 1), "east_m": round(y, 1),
                           "sweeping": next((s for s, who in claims.items() if who == p), None)}
                          for p, (x, y) in sorted(peers.items())],
            "assigned_to_you": sorted(loop.guard.cells) if loop.guard and loop.guard.cells else [],
            "blocked_for_you": sorted(loop.blocked),
            "blocked_by_teammates": dict(sorted(team.peer_blocked.items())),
            "cleared_by_teammates": {s: sorted(set(v)) for s, v in sorted(cleared.items())},
        }
