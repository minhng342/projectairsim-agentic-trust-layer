"""Building one drone: every part it needs, wired together.

A drone is the same chain in every run:

    observation feed -> planner -> (guard) -> action port -> gate -> sector executor -> simulator
    victim sensor -> memory (belief) + planner

A team drone adds the radio side: a Remote ID broadcaster, a receiver, a messenger and the
coordinator that applies the peer-claim rules.
"""
import math
from dataclasses import dataclass
from typing import Any

from atal.agents.llm_sector_planner import LLMSectorPlanner
from atal.agents.sector_planner import FixedSectorPlanner
from atal.agents.swarm import (
    PeerMessenger,
    SwarmCoordinator,
    SwarmFixedPlanner,
    SwarmLLMPlanner,
    SwarmReceiver,
)
from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.channels.radio import RemoteIdBroadcaster, RemoteIdReceiver
from atal.core.belief import LocalBelief
from atal.flight.sector_sweep import SectorExecutor
from atal.runtime import rates
from atal.simulation.victim_sensor import VictimSensor
from atal.trust_layer.action_gate import BypassGate
from atal.trust_layer.assessor import NoTrustAssessor
from atal.trust_layer.invariant_guard import SarGuard, SarInvariantGate


@dataclass
class Drone:
    """One drone's parts, as the run needs them."""
    vehicle_id: str
    mission: Any                   # this drone's mission (a team drone has its own altitude layer)
    feed: ObservationFeed          # the only way observations reach the planner
    belief: LocalBelief            # the drone's memory
    sector_executor: SectorExecutor
    planner: Any
    sensor: VictimSensor
    launch_altitude_m: float
    coordinator: Any = None        # team only
    broadcaster: Any = None        # team only


@dataclass
class TeamParts:
    """What the drones of one team share."""
    bus: Any                       # the radio bus
    pads: Any                      # the pad zone (one drone at a time over the pads)
    static_cells: dict | None      # the pre-mission split, in the static-allocation condition
    staging: list                  # staging points, by drone index (may be empty)
    size: int                      # number of drones


def read_home(ctx, spec, feed) -> tuple:
    """The drone's first observation, on the ground: where home is and how high. Refuses to
    fly if the drone is not where the scenario says it starts."""
    vid = spec.vehicle
    home = feed.observe()
    if home is None:
        raise RuntimeError(f"{vid}: no home observation")
    home_xy = (home.north_m, home.east_m)
    off = math.dist(home_xy, (spec.start.x_m, spec.start.y_m))
    ctx.recorder.record("agent_home", vehicle_id=vid, observation_id=home.observation_id,
                        home={"x_m": home_xy[0], "y_m": home_xy[1]},
                        offset_from_scenario_start_m=round(off, 2))
    if off > rates.SPAWN_TOLERANCE_M:
        raise RuntimeError(f"{vid} spawned at ({home_xy[0]:.1f}, {home_xy[1]:.1f}), "
                           f"{off:.1f} m from the scenario start; fix the scene or "
                           f"scenario before flying")
    return home, home_xy


def build_single_drone(ctx, spec, mission, victim, gate_on: bool) -> Drone:
    """The one drone of a one-drone run."""
    vid, recorder, llm = spec.vehicle, ctx.recorder, ctx.llm
    feed = ObservationFeed(ctx.adapter, vid, assessor=NoTrustAssessor(), recorder=recorder)
    home, home_xy = read_home(ctx, spec, feed)
    belief = LocalBelief(vid, mission, path=ctx.stem.with_suffix(".memory.json"),
                         on_change=recorder.recorder_for(source="belief"))
    sector_executor = SectorExecutor(ctx.adapter, vid, mission, home_xy, ctx.flight,
                                     follower_config=ctx.timing.follower(),
                                     on_event=recorder.recorder_for(source="mission"))
    # The planner-side guard (the LLM's output validator; the fixed planner's choice
    # under the guard) and the execution-side gate are separate switches: --gate
    # none bypasses only the gate. The LLM planner always validates its own output.
    guard = SarGuard(mission, belief, home_xy) if (llm or gate_on) else None
    gate = SarInvariantGate(guard) if gate_on else BypassGate()
    port = ActionPort(vid, sector_executor, feed, gate=gate, recorder=recorder,
                      max_observation_age_s=rates.MAX_OBSERVATION_AGE_S)
    if llm:
        if hasattr(llm["model"], "on_event"):
            llm["model"].on_event = recorder.recorder_for(source="model")
        planner = LLMSectorPlanner(vid, mission, feed, port, belief, model=llm["model"],
                                   guard=guard, model_label=llm["label"],
                                   model_settings=llm["settings"],
                                   decision_timeout_s=ctx.args.decision_timeout,
                                   recorder=recorder,
                                   sweep_progress=sector_executor.last_progress, log=ctx.log)
    else:
        planner = FixedSectorPlanner(vid, mission, feed, port, belief, recorder=recorder,
                                     sweep_progress=sector_executor.last_progress, log=ctx.log,
                                     guard=guard)

    def detected(d):
        belief.add_detection(d)
        planner.on_detection(d)
        ctx.log(f"  victim detected by {vid} in {d.sector_id} at sim {d.sim_time_s:.1f} s")
        if mission.termination.stop_on_detection:
            sector_executor.interrupt("victim detected")
    sensor = VictimSensor(ctx.adapter, vid, (victim.x_m, victim.y_m), mission.sensor.footprint_m,
                          mission.flight.altitude_m, grid=mission.grid,
                          on_detection=detected, recorder=recorder)
    return Drone(vid, mission, feed, belief, sector_executor, planner, sensor, home.altitude_m)


def build_team_drone(ctx, spec, index: int, mission, victim, gate_on: bool,
                     team: TeamParts) -> Drone:
    """One drone of a team. `mission` is this drone's own (its altitude layer)."""
    vid, recorder, llm, log = spec.vehicle, ctx.recorder, ctx.llm, ctx.log
    adapter = ctx.adapter

    def sim_clock():                       # sim time in seconds, as this drone last heard it
        p = adapter.latest_poses().get(vid)
        return None if p is None else round(p[0] / 1e9, 3)

    feed = ObservationFeed(adapter, vid, assessor=NoTrustAssessor(), recorder=recorder)
    home, home_xy = read_home(ctx, spec, feed)
    belief = LocalBelief(vid, mission, path=ctx.stem.with_suffix(f".{vid}.memory.json"),
                         on_change=recorder.recorder_for(source="belief"))
    sector_executor = SectorExecutor(adapter, vid, mission, home_xy, ctx.flight,
                                     follower_config=ctx.timing.follower(),
                                     on_event=recorder.recorder_for(source="mission"),
                                     pad_zone=team.pads if team.size > 1 else None,
                                     pad_wait_s=rates.PAD_WAIT_S / ctx.timing.k)
    guard = SarGuard(mission, belief, home_xy)
    gate = SarInvariantGate(guard) if gate_on else BypassGate()
    port = ActionPort(vid, sector_executor, feed, gate=gate, recorder=recorder,
                      max_observation_age_s=rates.MAX_OBSERVATION_AGE_S)

    # the radio side
    remote_id_receiver = RemoteIdReceiver(vid, recorder=None, clock=sim_clock)
    messenger = PeerMessenger(vid, team.bus, recorder=recorder)
    coordinator = SwarmCoordinator(vid, mission, belief, remote_id_receiver, messenger, sim_clock,
                                   guard=guard, recorder=recorder, log=log)
    if team.static_cells is not None:
        coordinator.static_cells = list(team.static_cells[vid])
    team.bus.attach(SwarmReceiver(vid, remote_id_receiver, coordinator.on_message, sim_clock,
                                  recorder=recorder,
                                  authenticated=mission.swarm.identity == "authenticated"))

    stage = team.staging
    common = dict(recorder=recorder, sweep_progress=sector_executor.last_progress,
                  log=log, swarm=coordinator,
                  staging_xy=(stage[index].x_m, stage[index].y_m) if index < len(stage) else None)
    if llm:
        planner = SwarmLLMPlanner(vid, mission, feed, port, belief, model=llm["model"],
                                  guard=guard, model_label=llm["label"],
                                  model_settings=llm["settings"],
                                  decision_timeout_s=ctx.args.decision_timeout, **common)
    else:
        planner = SwarmFixedPlanner(vid, mission, feed, port, belief, guard=guard, **common)

    # Stop only a SEARCH move: a drone already heading home (it found nothing
    # left, or an earlier report ended its search) keeps flying home even if it
    # passes over the victim on the way.
    def peer_victim(claim):
        if mission.termination.stop_on_detection and coordinator.state == "searching":
            sector_executor.interrupt(f"{claim.sender_id} reported the victim")
    coordinator.on_peer_victim = peer_victim

    def detected(det):
        log(f"  victim detected by {vid} in {det.sector_id} at sim {det.sim_time_s:.1f} s")
        belief.add_detection(det)
        planner.on_detection(det)
        coordinator.own_detection(det)
        if mission.termination.stop_on_detection and coordinator.state == "searching":
            sector_executor.interrupt("victim detected")
    sensor = VictimSensor(adapter, vid, (victim.x_m, victim.y_m),
                          mission.sensor.footprint_m, mission.flight.altitude_m, grid=mission.grid,
                          on_detection=detected, recorder=recorder)
    broadcaster = RemoteIdBroadcaster(adapter, vid, team.bus, recorder=recorder,
                                      status=lambda: coordinator.state)
    return Drone(vid, mission, feed, belief, sector_executor, planner, sensor, home.altitude_m,
                 coordinator=coordinator, broadcaster=broadcaster)
