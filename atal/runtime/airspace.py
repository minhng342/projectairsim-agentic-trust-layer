"""
The team runner's side of the managed airspace (mission swarm.airspace: managed): it builds
the transition manager, feeds it surveillance, orders the startups top-down, and connects
traffic pauses to each drone's sector executor. The rules themselves live in
atal/airspace/ (the manager) and atal/flight/recovery.py (each drone's climbs, descents and
landing clearance).

    air = TeamAirspace.build(mission, vids, recorder, timing)     # None if unmanaged
    air.attach(team)                       # pauses go to each drone's sector executor
    task = air.survey(adapter, stop)       # surveillance at 5 Hz of sim time
    r = await air.startup(vid, pad_xy, launch_alt_m)   # before takeoff: granted and started

The manager is trusted flight-safety infrastructure. It runs the same way in every
comparison condition, and what it contributes to separation is not evidence for the
agents' trust layer.
"""
import asyncio

from atal.airspace.bounds import Bounds, Levels
from atal.airspace.landing_queue import CleanupDeadline, QueueBudget
from atal.airspace.transition_manager import Kind, Phase, Sample, State, TransitionManager
from atal.core.telemetry import GroundState, ValidationStatus

SURVEY_SIM_S = 0.2          # the manager's surveillance period (5 Hz of sim time)


class TeamAirspace:
    def __init__(self, manager: TransitionManager, levels: Levels, vids: list[str], timing,
                 recorder, budget: QueueBudget):
        self.manager, self.levels, self.vids = manager, levels, list(vids)
        self.budget = budget                 # startup and landing waits (landing_queue.py)
        self.timing, self.recorder = timing, recorder
        self._startups: dict | None = None
        self._lock = asyncio.Lock()
        self.startup_wait_s: dict[str, float] = {}

    @classmethod
    def build(cls, mission, vids, recorder, timing, pad_zone_s: float
              ) -> "TeamAirspace | None":
        sw = mission.swarm
        if sw.airspace != "managed":
            return None
        levels = Levels(base_m=mission.flight.altitude_m, spacing_m=sw.layer_spacing_m,
                        team_size=len(vids), max_transit_level=sw.max_transit_level,
                        ceiling_m=mission.limits.altitude_ceiling_m,
                        separation_m=sw.separation_cruise_m)
        problems = levels.problems()
        if problems:
            raise ValueError("managed airspace: " + "; ".join(problems))
        bounds = Bounds(separation_m=sw.separation_cruise_m, v_h=1.2 * mission.flight.speed_m_s)
        mgr = TransitionManager(vids, {v: levels.of(i) for i, v in enumerate(vids)},
                                bounds=bounds, ceiling_m=mission.limits.altitude_ceiling_m,
                                recorder=recorder)
        budget = QueueBudget.derive(levels, bounds, pad_zone_s)
        recorder.record("airspace", source="transition_manager", managed=True,
                        levels={v: levels.of(i) for i, v in enumerate(vids)},
                        ceiling_m=mission.limits.altitude_ceiling_m, d_h=bounds.d_h,
                        r_stop=round(bounds.r_stop, 2), v_h=bounds.v_h,
                        queue_budget=budget.record(),
                        note="trusted flight-safety infrastructure; the same in every "
                             "condition; not evidence for the agents' trust layer")
        return cls(mgr, levels, vids, timing, recorder, budget)

    def cleanup_deadline(self) -> CleanupDeadline:
        """A new drone's overall cleanup deadline (fixed when its cleanup begins)."""
        return CleanupDeadline(self.budget, lambda: self.manager.now or 0.0)

    # ------------------------------------------------------------ wiring
    def attach(self, team: dict) -> None:
        """Traffic pauses act on each drone's sector executor."""
        self.manager._on_pause = lambda v: team[v].sector_executor.pause_for_traffic()
        self.manager._on_resume = lambda v: team[v].sector_executor.resume_traffic()

    def survey(self, adapter, stop: asyncio.Event) -> asyncio.Task:
        return asyncio.create_task(self._survey(adapter, stop))

    async def _survey(self, adapter, stop):
        period = self.timing.poll(SURVEY_SIM_S)
        while not stop.is_set():
            samples = {}
            for v in self.vids:
                try:
                    s = adapter.get_snapshot(v)
                except Exception:          # noqa: BLE001 - a missing sample is "stale"
                    continue
                if s is None or s.validation_status != ValidationStatus.VALID:
                    continue
                samples[v] = Sample(s.sim_time_s, s.position_ned_m.x, s.position_ned_m.y,
                                    s.altitude_local_m, s.ground_speed_mps,
                                    s.vertical_speed_mps,
                                    s.ground_state == GroundState.GROUNDED)
            if samples:
                self.manager.update(max(x.t_s for x in samples.values()), samples)
            await asyncio.sleep(period)

    # ------------------------------------------------------------ startup
    async def startup(self, vid: str, pad_xy, launch_alt_m: float):
        """Wait for this drone's startup reservation (granted top-down: the highest level
        first) and start it. Returns the ACTIVE reservation, or None if it was not
        granted in time (the drone then stays on its pad)."""
        async with self._lock:
            if self._startups is None:
                self._startups = await self._request_all(self.budget.startup_queue_s)
        r = self._startups.get(vid)
        if r is None:
            return None
        while r.state == State.REQUESTED:
            await asyncio.sleep(self.timing.poll(0.1))
        if r.state != State.GRANTED:
            return None
        self.startup_wait_s[vid] = round((r.granted_at or 0) - r.requested_at, 2)
        while True:
            answer = self.manager.start(r.reservation_id, r.request_id, vid)
            if answer == "started":
                self.manager.set_phase(vid, Phase.MOVING)
                return r
            if answer != "not_holding":           # lapsed, cancelled, expired
                return None
            await asyncio.sleep(self.timing.poll(0.1))

    async def _request_all(self, wait_s: float) -> dict:
        """Once every drone is seen on its pad: one startup request each, highest level
        first, so a lower climb never crosses an occupied level (policy section 10)."""
        m = self.manager
        while not all(m.fresh(v) and m.tracks[v].sample.grounded for v in self.vids):
            await asyncio.sleep(self.timing.poll(0.1))
        out = {}
        for i in sorted(range(len(self.vids)), key=self.levels.search, reverse=True):
            v = self.vids[i]
            s = m.tracks[v].sample
            out[v] = m.request(v, f"{v}-startup", Kind.STARTUP, s.xy, s.alt_m,
                               self.levels.search(i), wait_until=m.now + wait_s)
        return out

    # ------------------------------------------------------------ landing
    async def landed(self, vid: str, disarmed: bool, wait_s: float = 3.0) -> bool:
        """After a confirmed touchdown and the disarm: leave the roster once the manager's
        own surveillance shows the drone on the ground."""
        m = self.manager
        t0 = m.now or 0.0
        while True:
            if m.leave_roster(vid, disarmed=disarmed):
                return True
            if not disarmed or (m.now or 0.0) - t0 > wait_s:
                return False
            await asyncio.sleep(self.timing.poll(SURVEY_SIM_S))

    def summary(self) -> dict:
        s = self.manager.summary()
        s["startup_wait_s"] = dict(self.startup_wait_s)
        s["on_roster"] = sorted(self.manager.roster)
        return s
