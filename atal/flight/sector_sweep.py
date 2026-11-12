"""
Sector executor (Phase 1): runs one typed SAR action at a time, to completion. It is the
action port's executor, so every action still goes proposal -> gate -> executor -> result
-> feedback and is recorded.

    executor = SectorExecutor(adapter, "Drone1", mission, home_xy, flight_executor)
    port = ActionPort("Drone1", executor, feed, gate=...)

    sweep_sector     boustrophedon lanes for the sector (atal/search/sweep_lanes.py), flown
                     as one polyline by the velocity PathFollower at mission altitude and
                     speed; the path starts at the drone's position, so the transit from
                     the previous sector is part of it
    return_home      straight back over the launch point at mission altitude (or at
                     action.altitude_m if higher: the planner's obstacle recovery). With a
                     pad_zone (swarm, atal/flight/pad_zone.py) in two legs: to the zone's edge,
                     wait there until no teammate is landing, then in to the pad. A drone
                     that is inside the zone when a teammate holds it first flies out to
                     the edge. If the wait times out it never goes in (pad_zone_timeout)
    move_to_position straight to (north, east) at mission altitude (Phase 3 staging), or
                     at one of this drone's transit levels (obstacle recovery); result
                     position_reached
    change_altitude  delegated to the flight executor (climb in place to mission altitude,
                     the move the corridor missions proved live)

interrupt(reason) stops the sweep in progress (the drone hovers where it is). The runner
wires it to the victim sensor when termination.stop_on_detection is set, and to the
mission-time watchdog.

pause_for_traffic() / resume_traffic(): a traffic pause from the transition manager
(recovery policy section 8a). The leg in progress stops and the drone hovers, but the
action does not end: when the pause is lifted the drone flies the rest of the same path
from where it stopped. So a pause never produces path_blocked, obstacle recovery or a
"sector swept" result, and the sweep's progress counts the part flown before it. An
interrupt for another reason (a detection, the time limit) during a pause still ends the
action as sweep_stopped. A leg that has not started yet waits for the pause to end.

Result codes (CommandResult.reason_code, mapped to agent feedback by the port):
    sector_swept      every lane flown
    sweep_stopped     interrupted (victim detected, time limit); progress in the result reason
    home_reached      over the launch point
    unknown_sector    not a sector of this mission's grid; nothing sent
    path_blocked      the follower made no progress for stuck_after_s (an obstacle); it
                      hovered
    path_failed       the follower failed (telemetry, timeout, grounded); it hovered
    pad_zone_timeout  return home: a teammate held the pad zone for pad_wait_s; the drone
                      did not go in and is holding at the zone's edge
    unsupported_action, not_applicable
Policy (which sector may be swept now, geofence) is not checked here: in Phase 2 that is
the invariant guard's job, in the port's gate slot.
"""
import asyncio
import itertools
import math
import time
from types import SimpleNamespace as _Leg

from atal.core.action import ActionType, CommandResult, CommandStatus, ProposedAction
from atal.core.sar import Mission
from atal.flight.path_follower import (
    FollowerConfig,
    FollowMode,
    FollowResult,
    PathFollower,
    Polyline,
)
from atal.search.sweep_lanes import sweep_path

PAD_EDGE_MARGIN_M = 2.0     # wait this far outside the pad zone (the follower stops ~0.1 m short)
TRAFFIC = "traffic_pause"


class SectorExecutor:
    SUPPORTED = (ActionType.SWEEP_SECTOR, ActionType.RETURN_HOME, ActionType.CHANGE_ALTITUDE,
                 ActionType.MOVE_TO_POSITION)

    def __init__(self, adapter, vehicle_id: str, mission: Mission, home_xy: tuple[float, float],
                 flight_executor, follower_config: FollowerConfig | None = None,
                 timeout_factor: float = 3.0, timeout_margin_s: float = 60.0, on_event=None,
                 sleep=None, clock=time.monotonic, log=print, pad_zone=None,
                 pad_wait_s: float | None = 600.0):
        self.adapter, self.vehicle_id, self.mission = adapter, vehicle_id, mission
        self.home_xy = home_xy
        self._flight = flight_executor
        speed = mission.flight.speed_m_s
        cfg = follower_config or FollowerConfig()
        if cfg.max_speed_mps < speed:            # the mission speed is the follower's cruise
            cfg = FollowerConfig(**{**cfg.__dict__, "max_speed_mps": speed})
        self.follower_config = cfg
        self.timeout_factor, self.timeout_margin_s = timeout_factor, timeout_margin_s
        self._on_event, self._sleep, self._clock, self._log = on_event, sleep, clock, log
        self.follower: PathFollower | None = None
        self._interrupt_reason: str | None = None
        self.current_sector: str | None = None
        self.pad_zone, self.pad_wait_s = pad_zone, pad_wait_s
        self.traffic_paused = False
        self._waiting_for_traffic = False
        self.transit_altitudes: list[float] = []     # set by the recovery (atal/flight/recovery.py)
        self._done_m, self._total_m = 0.0, 0.0

    def interrupt(self, reason: str) -> bool:
        """Stop the sweep / return in progress. True if something was running."""
        if self._waiting_for_traffic:              # paused for traffic: end the action
            self._interrupt_reason = reason
            self._emit("sweep_interrupt", reason=reason, sector_id=self.current_sector)
            return True
        f = self.follower
        if f is None or f.mode in (FollowMode.DONE, FollowMode.FAILED, FollowMode.STOPPED):
            return False
        self._interrupt_reason = reason
        f.stop()
        self._emit("sweep_interrupt", reason=reason, sector_id=self.current_sector)
        return True

    # ------------------------------------------------------------ traffic pauses
    def pause_for_traffic(self) -> None:
        """The transition manager paused this drone: hover, keep the action open."""
        if self.traffic_paused:
            return
        self.traffic_paused = True
        f = self.follower
        if f is not None and f.mode not in (FollowMode.DONE, FollowMode.FAILED,
                                            FollowMode.STOPPED):
            if self._interrupt_reason is None:
                self._interrupt_reason = TRAFFIC
            f.stop()
        self._emit("traffic_pause_hold", sector_id=self.current_sector,
                   progress=round(self.last_progress(), 4))

    def resume_traffic(self) -> None:
        """No owner of a pause is left: the leg goes on."""
        if not self.traffic_paused:
            return
        self.traffic_paused = False
        self._emit("traffic_pause_resume", sector_id=self.current_sector,
                   progress=round(self.last_progress(), 4))

    async def _wait_for_traffic(self) -> bool:
        """Hover until the pause ends. False if interrupted for another reason meanwhile."""
        self._waiting_for_traffic = True
        try:
            while self.traffic_paused:
                if self._interrupt_reason not in (None, TRAFFIC):
                    return False
                await (self._sleep or asyncio.sleep)(self.follower_config.poll_interval_s)
            return self._interrupt_reason in (None, TRAFFIC)
        finally:
            self._waiting_for_traffic = False

    async def _follow(self, waypoints, alt, timeout_for) -> FollowResult:
        """Fly the waypoints with the path follower, through any traffic pauses. The result
        counts progress along the whole path, the parts before each pause included."""
        self._done_m = 0.0
        self._total_m = _length(waypoints)
        left = list(waypoints)
        kw = {"sleep": self._sleep} if self._sleep is not None else {}
        while True:
            if self.traffic_paused and not await self._wait_for_traffic():
                return FollowResult(FollowMode.STOPPED, "stopped on request (paused)",
                                    self._done_m, self._total_m, 0.0)
            if self._interrupt_reason == TRAFFIC:
                self._interrupt_reason = None
            self.follower = PathFollower(self.adapter, self.vehicle_id, left, alt,
                                         self.mission.flight.speed_m_s,
                                         config=self.follower_config, on_event=self._on_event,
                                         log=self._log, **kw)
            r = await self.follower.run(timeout_s=timeout_for(left))
            if r.mode == FollowMode.STOPPED and self._interrupt_reason == TRAFFIC:
                path = self.follower.path
                snap = self._truth()
                s_at = r.progress_m
                if snap is not None:
                    here = (snap.position_ned_m.x, snap.position_ned_m.y)
                    s_proj, _ = path.project(*here, r.progress_m,
                                             2 * self.follower_config.lookahead_m)
                    s_at = max(s_at, s_proj)
                else:
                    here = path.point_at(s_at)
                self._done_m += min(s_at, path.length)
                left = [here] + _rest_of(path, s_at)
                self._emit("traffic_pause_stopped", sector_id=self.current_sector,
                           progress=round(self.last_progress(), 4))
                continue
            r.progress_m += self._done_m
            r.path_length_m = self._total_m
            return r

    async def execute(self, action: ProposedAction) -> CommandResult:
        started = self._clock()

        def result(status, code, reason, sent=False):
            return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                                 action_type=action.action_type, status=status, reason=reason,
                                 reason_code=code, sent_to_simulator=sent,
                                 final_snapshot=self._truth(),
                                 elapsed_s=round(self._clock() - started, 3))

        if action.vehicle_id != self.vehicle_id:
            return result(CommandStatus.REFUSED, "not_applicable",
                          f"executor flies {self.vehicle_id} only")
        t = action.action_type
        if t == ActionType.CHANGE_ALTITUDE:
            return await self._flight.execute(action)
        if t not in self.SUPPORTED:
            return result(CommandStatus.REFUSED, "unsupported_action",
                          f"{t.value} is not a search action")
        snap = self._truth()
        if snap is None:
            return result(CommandStatus.REFUSED, "path_failed", "no telemetry")
        here = (snap.position_ned_m.x, snap.position_ned_m.y)
        fly_alt = self.mission.flight.altitude_m

        if t == ActionType.SWEEP_SECTOR:
            sid = action.sector_id
            if not self.mission.grid.has(sid):
                return result(CommandStatus.REFUSED, "unknown_sector",
                              f"{sid} is not a sector of {self.mission.mission_id}")
            path = sweep_path(self.mission, sid, here)
            waypoints = [here] + path.waypoints
            self.current_sector = sid
            self._emit("sector_started", sector_id=sid, lanes=path.lanes,
                       sweep_length_m=round(path.length_m, 2), waypoints=path.waypoints,
                       request_id=action.request_id)
        elif t == ActionType.MOVE_TO_POSITION:
            allowed = [self.mission.flight.altitude_m, *self.transit_altitudes]
            if all(abs(action.altitude_m - a) > 1.0 for a in allowed):
                return result(CommandStatus.REFUSED, "not_applicable",
                              f"moves fly at the mission altitude "
                              f"{self.mission.flight.altitude_m:g} m only (or a transit "
                              f"level during an obstacle recovery)")
            fly_alt = float(action.altitude_m)
            sid, path = None, None
            target = (action.north_m, action.east_m)
            waypoints = [here, target]
            self.current_sector = None
            self._emit("move_started", target=target, request_id=action.request_id)
        else:
            sid, path = None, None
            if action.altitude_m is not None and action.altitude_m > fly_alt:
                fly_alt = float(action.altitude_m)
            waypoints = [here, self.home_xy]
            z = self.pad_zone
            if z is not None and z.holder != self.vehicle_id:
                # leg 1: to the zone's edge on this layer. A drone already inside (its
                # search ended near the pads) waits at the edge too, not over the pads.
                edge = (None if z.inside(here) and z.holder is None else self._pad_edge(here))
                if edge is not None:
                    r = await self._fly([here, edge], fly_alt, action)
                    if r is not None:
                        return result(*r)
                    snap = self._truth()
                    here = ((snap.position_ned_m.x, snap.position_ned_m.y) if snap is not None
                            else edge)
                    waypoints = [here, self.home_xy]
                self._emit("pad_zone_wait", request_id=action.request_id, held_by=z.holder)
                if not await z.acquire(self.vehicle_id, timeout_s=self.pad_wait_s):
                    # never fly in without the zone: hold outside it; the runner lands
                    # the drone here, outside the zone (a recorded contingency landing)
                    self.current_sector = None
                    self._emit("return_home_failed", sector_id=None,
                               request_id=action.request_id, reason="pad_zone_timeout",
                               held_by=z.holder, position=here)
                    return result(CommandStatus.FAILED, "pad_zone_timeout",
                                  f"the pad zone stayed held by {z.holder} for "
                                  f"{self.pad_wait_s:g} s; holding outside it", True)
            self.current_sector = None
            self._emit("return_home_started", home=self.home_xy, request_id=action.request_id)

        length = _length(waypoints)
        self._interrupt_reason = None
        r = await self._follow(waypoints, fly_alt, self._timeout_for)
        progress = 1.0 if length <= 0 else min(1.0, r.progress_m / length)

        if r.mode == FollowMode.DONE:
            if sid is not None:
                self._emit("sector_done", sector_id=sid, request_id=action.request_id,
                           elapsed_s=r.elapsed_s, max_cross_track_m=r.max_cross_track_m)
                return result(CommandStatus.SUCCEEDED, "sector_swept",
                              f"{sid} swept ({path.lanes} lanes, {r.elapsed_s:.1f} s)", True)
            if t == ActionType.MOVE_TO_POSITION:
                return result(CommandStatus.SUCCEEDED, "position_reached", "at the target", True)
            return result(CommandStatus.SUCCEEDED, "home_reached", "over the launch point", True)
        if r.mode == FollowMode.STOPPED and self._interrupt_reason:
            self._emit("sector_stopped" if sid else "return_home_stopped", sector_id=sid,
                       request_id=action.request_id, reason=self._interrupt_reason,
                       progress=round(progress, 4))
            return result(CommandStatus.SUCCEEDED, "sweep_stopped",
                          f"stopped: {self._interrupt_reason} (progress {progress:.2f})", True)
        blocked = self.follower.blocked
        self._emit("sector_failed" if sid else "return_home_failed", sector_id=sid,
                   request_id=action.request_id, reason=r.reason, progress=round(progress, 4),
                   blocked=blocked)
        return result(CommandStatus.FAILED, "path_blocked" if blocked else "path_failed",
                      f"{r.mode.value}: {r.reason}", True)

    def _pad_edge(self, here):
        """The waiting point outside the pad zone, inside the geofence (None if there is none)."""
        return self.pad_zone.exit_point(
            here, PAD_EDGE_MARGIN_M,
            allowed=lambda p: self.mission.geofence_excursion_m(*p) <= 0.0)

    async def leave_pad_zone(self) -> bool:
        """Fly straight out to the pad zone's edge on this drone's layer (the runner, when a
        drone must land without holding the zone). True once outside it."""
        z, snap = self.pad_zone, self._truth()
        if z is None or snap is None:
            return z is None
        here = (snap.position_ned_m.x, snap.position_ned_m.y)
        if not z.inside(here):
            return True
        edge = self._pad_edge(here)
        if edge is None:
            return False
        self._emit("pad_zone_exit", position=here, edge=edge)
        r = await self._fly([here, edge], self.mission.flight.altitude_m,
                            _Leg(request_id=None))
        return r is None

    async def _fly(self, waypoints, alt, action):
        """One leg (the return home's approach to the pad zone). None when it got there,
        else the (status, code, reason, sent) to return."""
        self._interrupt_reason = None
        r = await self._follow(waypoints, alt, self._timeout_for)
        if r.mode == FollowMode.DONE:
            return None
        if r.mode == FollowMode.STOPPED and self._interrupt_reason:
            self._emit("return_home_stopped", sector_id=None, request_id=action.request_id,
                       reason=self._interrupt_reason, leg="pad_approach")
            return (CommandStatus.SUCCEEDED, "sweep_stopped",
                    f"stopped: {self._interrupt_reason} (approaching the pads)", True)
        blocked = self.follower.blocked
        self._emit("return_home_failed", sector_id=None, request_id=action.request_id,
                   reason=r.reason, leg="pad_approach", blocked=blocked)
        return (CommandStatus.FAILED, "path_blocked" if blocked else "path_failed",
                f"{r.mode.value}: {r.reason} (approaching the pads)", True)

    def _timeout_for(self, waypoints) -> float:
        return (_length(waypoints) / self.mission.flight.speed_m_s * self.timeout_factor
                + self.timeout_margin_s)

    def last_progress(self) -> float:
        """Share of the current action's path flown, parts before a traffic pause included."""
        f = self.follower
        if f is None or self._total_m <= 0:
            return 0.0
        running = 0.0 if f.mode == FollowMode.STOPPED and self.traffic_paused else f.progress_m
        return min(1.0, (self._done_m + running) / self._total_m)

    def _truth(self):
        try:
            return self.adapter.get_snapshot(self.vehicle_id)
        except Exception:
            return None

    def _emit(self, kind: str, **fields) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, {"vehicle_id": self.vehicle_id, **fields})
        except Exception as err:
            self._log(f"[sector] on_event({kind}) failed: {type(err).__name__}: {err}")


def _length(points) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points))


def _rest_of(path: Polyline, s: float) -> list:
    """The path's points from arc length s on (the point at s first)."""
    out, acc = [path.point_at(s)], 0.0
    pts = path.points
    for a, b in itertools.pairwise(pts):
        acc += math.hypot(b[0] - a[0], b[1] - a[1])
        if acc > s + 1e-6:
            out.append(b)
    return out
