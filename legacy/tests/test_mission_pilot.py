"""MissionPilot: wake-up rules, tools, and a 5-drone conflict scenario on the fake
simulator with 2 model-driven drones (Pydantic AI FunctionModel) + 3 scripted ones."""
import asyncio
import json
import math
import os
from dataclasses import replace

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai.messages import (ModelResponse, TextPart, ToolCallPart,  # noqa: E402
                                  ToolReturnPart, UserPromptPart)
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from legacy.corridor.mission_pilot import MissionPilot, MissionPilotConfig  # noqa: E402
from channel.action_port import ActionPort  # noqa: E402
from channel.observation_feed import ObservationFeed  # noqa: E402
from channel.remote_id import RemoteIdBroadcaster, RemoteIdBus, RemoteIdReceiver  # noqa: E402
from legacy.corridor.conflict_detection import PairPrediction  # noqa: E402
from legacy.corridor.mission_runner import MissionRunner  # noqa: E402
from executor.path_follower import FollowerConfig, FollowMode  # noqa: E402
from executor.projectairsim_executor import ProjectAirSimExecutor  # noqa: E402
from legacy.corridor.mission import MissionPlan  # noqa: E402
from tests.channel_fakes import ListRecorder  # noqa: E402
from tests.test_executor import FAST, MultiAdapter  # noqa: E402

PLAN = MissionPlan.load("missions/inspection_5_conflict.json")
LLM = ("Drone2", "Drone3")
FOLLOW = FollowerConfig(poll_interval_s=0.005, hover_timeout_s=0.5)
FAST_PILOT = MissionPilotConfig(check_every_s=0.005, recheck_s=0.3, off_nominal_clear_s=0.1,
                                min_episode_gap_s=0.02, monitor_max_s=0.05)


def situation_in(messages):
    """The wake-up message's situation JSON (first user prompt of this episode)."""
    for m in messages:
        for p in getattr(m, "parts", ()):
            if isinstance(p, UserPromptPart):
                text = p.content if isinstance(p.content, str) else str(p.content)
                return text.split("\n", 1)[0], json.loads(text.split("Situation now:\n", 1)[1])
    return None, None


class LayerPolicy:
    """A sensible model: on a conflict, move to an escape layer; when clear, go back."""

    def __init__(self, escape_m):
        self.escape_m = escape_m

    def __call__(self, messages, info):
        if any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart("Done.")])
        head, sit = situation_in(messages)
        oid = sit["own_state"]["observation_id"]
        layer = sit["mission"]["assigned_layer_m"]
        if "back_to_plan" in head:
            target = layer
        elif sit.get("safe_layers_m"):      # follow the advice when given, as the prompt asks
            target = sit["safe_layers_m"][0]
        elif "safe_layers_m" in sit:        # advice given but no layer is safe: timed hold
            return ModelResponse(parts=[ToolCallPart("hold", {
                "observation_id": oid, "duration_s": 3.0,
                "reason": f"{head}: no safe layer, hold"})])
        else:
            target = self.escape_m
        return ModelResponse(parts=[ToolCallPart("change_altitude", {
            "observation_id": oid, "altitude_m": target,
            "reason": f"{head}: layer {target}"})])


def scenario(policies, config=FAST_PILOT):
    ids = PLAN.vehicle_ids
    adapter, rec = MultiAdapter(ids), ListRecorder()
    for vid in ids:
        n, e = PLAN.for_vehicle(vid).corridor.waypoints[0]
        w = adapter.worlds[vid]
        w.n, w.e, w.alt = n, e, 4.0
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None)
    bus = RemoteIdBus(recorder=rec)
    runners, pilots, tx, feeds = {}, {}, {}, {}
    for vid in ids:
        m = PLAN.for_vehicle(vid)
        feeds[vid] = ObservationFeed(adapter, vid, recorder=rec)
        rx = bus.attach(RemoteIdReceiver(vid, recorder=rec))
        runners[vid] = MissionRunner(adapter, m, ex, follower_config=FOLLOW, log=lambda *_: None,
                                     on_event=lambda k, f: rec.record(k, source="mission", **f))
        tx[vid] = RemoteIdBroadcaster(adapter, vid, bus, recorder=rec,
                                      intent=runners[vid].target_altitude_m)
        if vid in policies:
            port = ActionPort(vid, runners[vid], feeds[vid], recorder=rec, max_observation_age_s=5.0)
            pilots[vid] = MissionPilot(vid, feeds[vid], port, rx, m, FunctionModel(policies[vid]),
                                       recorder=rec, config=config, model_label="function")
    truth = []

    async def go():
        stop = asyncio.Event()
        bg = [asyncio.ensure_future(t.run(400.0, stop)) for t in tx.values()]

        async def sample():
            while not stop.is_set():
                truth.append({v: (w.n, w.e, w.alt) for v, w in adapter.worlds.items()})
                await asyncio.sleep(0.002)
        bg.append(asyncio.ensure_future(sample()))
        pl = [asyncio.ensure_future(p.run(stop)) for p in pilots.values()]
        results = await asyncio.wait_for(
            asyncio.gather(*(r.fly(timeout_s=60) for r in runners.values())), 60)
        stop.set()
        await asyncio.gather(*bg, *pl)
        return dict(zip(runners, results))
    results = asyncio.run(go())
    return results, pilots, rec, truth, adapter


def min_sep(truth, a, b):
    best = math.inf
    for t in truth:
        (n1, e1, z1), (n2, e2, z2) = t[a], t[b]
        best = min(best, math.dist((n1, e1, z1), (n2, e2, z2)))
    return best


# ------------------------------------------------------------ the scenario
def test_llm_drones_resolve_head_on_conflicts_and_return_to_their_layer():
    results, pilots, rec, truth, adapter = scenario(
        {"Drone2": LayerPolicy(escape_m=13.0), "Drone3": LayerPolicy(escape_m=17.0)})
    assert all(r.mode == FollowMode.DONE for r in results.values()), \
        {v: (r.mode, r.reason) for v, r in results.items()}
    for vid in LLM:
        wakes = [e["reason"] for e in rec.of("agent_wake") if e["vehicle_id"] == vid]
        assert wakes[0] == "new_conflict" and "back_to_plan" in wakes, (vid, wakes)
        applied = [m for m in pilots[vid].maneuvers if m["code"] == "applied"]
        assert len(applied) >= 2, (vid, pilots[vid].maneuvers)
        # it went back to its own layer at least once (Drone2/Drone4 meet again on the
        # return legs, so a drone may legitimately finish its corridor on the escape layer)
        layer = PLAN.for_vehicle(vid).corridor.altitude_m
        backs = [c for c in rec.of("tool_call") if c["vehicle_id"] == vid
                 and c["name"] == "change_altitude" and c["args"].get("altitude_m") == layer
                 and c["result"]["feedback"]["code"] == "applied"]
        assert backs, vid
    # No ground-truth separation check here: the fake moves a drone ~0.3 m per follower tick
    # whatever the clock says (tens of m/s in real time), so warning times are not to scale.
    # The decisions are what this tests; separation is checked live.
    # an applied maneuver ends the wake-up: one model call, its reason is the comment
    for vid in LLM:
        for w in [e for e in rec.of("agent_wake") if e["vehicle_id"] == vid]:
            turns = [t for t in rec.of("llm_turn") if t["vehicle_id"] == vid
                     and t["episode"] == w["episode"]]
            assert len(turns) == 1, (vid, w["episode"], len(turns))
        assert all(c and c.split(":")[0] in ("change_altitude", "hold") for c in pilots[vid].comments)
    first = [e for e in rec.of("agent_wake") if e["vehicle_id"] == "Drone2"][0]
    assert first["conflicts"][0]["with"] == "Drone4"                   # detected from Remote ID
    assert not rec.of("tool_rejected")


def test_without_resolution_the_pairs_lose_separation():
    """The baseline the gate will matter for: a model that only maintains."""
    def maintain(messages, info):
        if any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart("Maintaining.")])
        _, sit = situation_in(messages)
        return ModelResponse(parts=[ToolCallPart("maintain", {
            "observation_id": sit["own_state"]["observation_id"],
            "reason": "carry on"})])
    results, pilots, rec, truth, adapter = scenario({"Drone2": maintain, "Drone3": maintain})
    assert min_sep(truth, "Drone2", "Drone4") < 2.0
    assert [m["code"] for m in pilots["Drone2"].maneuvers][0] == "applied"


# ------------------------------------------------------------ wake-up rules
def P(other, t_los):
    return PairPrediction(other, 20.0, 0.0, t_los + 2, 0.0, 0.0, t_los, t_los + 4, False)


def make_pilot():
    ids = ("Drone2",)
    adapter = MultiAdapter(ids)
    feed = ObservationFeed(adapter, "Drone2")
    rx = RemoteIdReceiver("Drone2")
    m = PLAN.for_vehicle("Drone2")
    port = ActionPort("Drone2", object(), feed)
    return MissionPilot("Drone2", feed, port, rx, m, FunctionModel(lambda m, i: None),
                        config=MissionPilotConfig(recheck_s=8, off_nominal_clear_s=6,
                                                  min_episode_gap_s=2))


def test_wake_rules():
    p = make_pilot()
    sit = lambda conflicts, now: {"conflicts": conflicts, "now": now}     # noqa: E731
    assert p._wake_reason(sit([], 0.0)) is None                           # nothing to do
    assert p._wake_reason(sit([P("Drone4", 40.0)], 1.0)) is None          # too far ahead
    assert p._wake_reason(sit([P("Drone4", 10.0)], 2.0)) == "new_conflict"
    p._handled, p._last_episode_end = {"Drone4"}, 2.0
    assert p._wake_reason(sit([P("Drone4", 9.0)], 3.0)) is None           # handled, gap
    assert p._wake_reason(sit([P("Drone4", 9.0)], 5.0)) is None           # handled, < recheck
    assert p._wake_reason(sit([P("Drone4", 3.0)], 10.5)) == "conflict_persists"
    assert p._wake_reason(sit([P("Drone4", 3.0), P("Drone5", 9)], 11.0)) == "new_conflict"
    p._handled, p._last_episode_end = {"Drone4", "Drone5"}, 11.0
    p.altitude_target_m = 12.0                                             # off its layer
    assert p._wake_reason(sit([], 12.0)) is None                          # clear just now
    assert p._wake_reason(sit([], 18.5)) == "back_to_plan"
    p._handled = {"Drone4"}
    p.altitude_target_m = 10.0
    assert p._wake_reason(sit([], 30.0)) is None                          # on plan: nothing
    assert p._handled == set()                                            # cleared pairs forgotten


# ------------------------------------------------------------ tools
def test_tools_monitor_diff_and_expired_observation_hint():
    seen = []
    state = {"n": 0}

    def fn(messages, info):
        state["n"] += 1
        last = messages[-1]
        res = [p.content for p in last.parts if isinstance(p, ToolReturnPart)]
        seen.extend(res)
        if state["n"] == 1:
            _, sit = situation_in(messages)
            assert {p["uas_id"] for p in sit["traffic"]["peers"]} <= \
                set(PLAN.vehicle_ids) - {"Drone2"}                  # traffic is in the wake-up
            assert sit["mission"]["assigned_layer_m"] == 10.0 and "safe_layers_m" in sit
            return ModelResponse(parts=[
                ToolCallPart("get_own_state", {}), ToolCallPart("monitor", {"seconds": 99}),
                ToolCallPart("hold", {"observation_id": "stale-id", "reason": "test"})])
        return ModelResponse(parts=[TextPart("ok")])
    results, pilots, rec, truth, adapter = scenario({"Drone2": fn})
    own, monitor, propose = seen[:3]
    assert own["own_state"]["observation_id"]
    assert monitor["waited_s"] == FAST_PILOT.monitor_max_s and "conflicts" in monitor
    assert "safe_layers_m" in monitor
    assert propose["feedback"]["code"] == "unknown_observation" and "next_step" in propose
    text = json.dumps(seen)
    for leak in ("truth", "true_sender", "clean_payload", "snapshot"):
        assert leak not in text


def test_back_to_plan_is_not_suggested_while_the_assigned_layer_is_in_conflict():
    from legacy.corridor.conflict_detection import Track
    p = make_pilot()
    p._last_episode_end, p.altitude_target_m = 0.0, 13.0          # escaped to 13
    own = Track("Drone2", 10, 0, 13, 0, 1.0, 0)
    oncoming_at_layer = Track("Drone4", 10, 15, 10, 0, -1.0, 0)   # head-on on layer 10
    sit = {"conflicts": [], "now": 20.0, "own": own, "peers": [oncoming_at_layer]}
    p._clear_since = 10.0
    assert p._wake_reason(sit) is None                            # going back would conflict
    sit["peers"] = [Track("Drone4", 10, -15, 10, 0, -1.0, 0)]    # passed, but only 15 m away
    assert p._wake_reason(sit) is None                            # near the layer: not yet
    sit["peers"] = [Track("Drone4", 10, -20, 10, 0, -1.0, 0)]    # 20 m away and diverging
    assert p._wake_reason(sit) == "back_to_plan"


def test_safe_layers_keep_the_margin_from_nearby_drones_and_avoid_projected_conflicts():
    from legacy.corridor.conflict_detection import Track
    p = make_pilot()                                    # Drone2: band 5-20, layer 10
    own = Track("Drone2", 10, 0, 10, 0, 1.0, 0)
    peers = [Track("Drone4", 10, 15, 10, 0, -1.0, 0),   # head-on on layer 10
             Track("Drone5", 12, 3, 14, 0, 0, 0),       # hovering nearby at 14
             Track("Far", 100, 100, 7, 0, 0, 0)]        # far away: does not constrain
    layers = p.safe_layers({"own": own, "peers": peers})
    assert layers and all(abs(a - 10) >= 3 and abs(a - 14) >= 3 for a in layers)
    assert layers[0] == 7.0                             # closest acceptable to 10 first
    assert all(5 <= a <= 20 for a in layers) and len(layers) <= 5


def test_escapes_follow_the_safe_layer_advice_and_keep_the_margin():
    results, pilots, rec, truth, adapter = scenario(
        {"Drone2": LayerPolicy(escape_m=13.0), "Drone3": LayerPolicy(escape_m=17.0)})
    for vid in LLM:
        layer = PLAN.for_vehicle(vid).corridor.altitude_m
        escapes = [c["args"]["altitude_m"] for c in rec.of("tool_call")
                   if c["vehicle_id"] == vid and c["name"] == "change_altitude"
                   and c["args"].get("altitude_m") not in (None, layer)]
        assert escapes, vid
        others = {10.0, 14.0, 6.0}                          # every planned layer
        assert all(min(abs(a - o) for o in others) >= 3 for a in escapes), (vid, escapes)


def test_a_peer_climbing_to_an_announced_layer_is_not_offered_as_safe():
    """Live 2026-10-28 (run e26cc0): Drone2 left layer 10 for 17; seconds later Drone3
    (layer 14) chose 17 too, because Drone2 was still reported near 11. With the intent
    field the target layer counts as occupied and its conflict is predicted."""
    from legacy.corridor.conflict_detection import Track, intent_track_from_peer, track_from_peer
    p = make_pilot()                                    # band 5-20
    own = Track("Drone3", 16, 0, 14, 0, 0.8, 0)
    report = {"uas_id": "Drone2", "age_s": 0.3,
              "position": {"north_m": 13, "east_m": 6, "altitude_m": 11.0},
              "ground_speed_mps": 0.8, "track_deg": 270.0, "vertical_speed_mps": 0.0}  # just commanded
    blind = {"own": own, "peers": [track_from_peer(report)]}
    assert 17.0 in p.safe_layers(blind)                 # the live failure: 17 looked free
    seen = dict(report, target_altitude_m=17.0)
    sit = {"own": own, "peers": [track_from_peer(seen), intent_track_from_peer(seen)]}
    assert all(abs(a - 17.0) >= 3 for a in p.safe_layers(sit))
    hits = p.what_if_altitude(sit, 17.0)
    assert [h.other_id for h in hits] == ["Drone2"]     # one prediction per peer, not two


def test_live_broadcasts_carry_the_runner_target_layer():
    results, pilots, rec, truth, adapter = scenario(
        {"Drone2": LayerPolicy(escape_m=13.0), "Drone3": LayerPolicy(escape_m=17.0)})
    targets = {}
    for d in rec.of("rid_transmit"):
        m = d["payload"]
        targets.setdefault(m["uas_id"], set()).add(m.get("target_altitude_m"))
    for vid in PLAN.vehicle_ids:
        assert PLAN.for_vehicle(vid).corridor.altitude_m in targets.get(vid, set()), vid


def test_layer_changes_are_checked_on_the_way_not_only_at_the_layer():
    """Live 2026-10-28 (run d7c936), the moment Drone2 chose 17: Drone3 was crossing its line
    4 m above (at 14) and 5 m away. 17 kept the margin from everyone and was conflict-free
    once there, so it was offered; the climb went straight through Drone3."""
    from legacy.corridor.conflict_detection import Track, detect
    p = make_pilot()                                    # Drone2: band 5-20, layer 10
    own = Track("Drone2", 10.1, -11.2, 9.9, 0.0, 0.8, 0.0)
    trk = math.radians(320.9)                           # Drone3's transit leg, toward (16,-13)
    d3 = Track("Drone3", 8.1, -6.6, 13.9, 0.8 * math.cos(trk), 0.8 * math.sin(trk), 0.0)
    sit = {"own": own, "peers": [d3]}
    level_only = Track("Drone2", 10.1, -11.2, 17.0, 0.0, 0.8, 0.0)
    assert detect(level_only, [d3], p.config.minima, 30) == []        # what the old check saw
    assert [h.other_id for h in p.what_if_altitude(sit, 17.0)] == ["Drone3"]
    layers = p.safe_layers(sit)
    assert all(a < 13.9 for a in layers), layers         # nothing that climbs through Drone3
    assert p.what_if_altitude(sit, 9.9) == []            # staying put is not a transition


def test_wakes_when_a_safe_layer_opens_and_rechecks_sooner_as_the_conflict_nears():
    """Live 2026-10-28 (run d0488f): head-on with Drone4, no safe layer at the first wake-up
    (Drone3 crossing above), the model slowed down and the 8 s recheck came too late."""
    from legacy.corridor.conflict_detection import Track
    p = make_pilot()                                    # recheck 8 s, gap 2 s
    own = Track("Drone2", 10, -10, 10, 0, 0.8, 0)
    d4 = Track("Drone4", 10, 10, 10, 0, -0.8, 0)        # head-on, ~10 s to loss of separation
    blocker = Track("Drone3", 10, -9, 14, 0, 0.8, 0)    # right above-ish: every climb crosses it
    sit = lambda now, peers, t_los: {"own": own, "peers": peers, "now": now,   # noqa: E731
                                     "conflicts": [P("Drone4", t_los)]}
    low = Track("Drone1", 8, -12, 6, 0, 0, 0)            # hovering below: 5-8 keep no margin
    assert p.safe_layers(sit(0, [d4, blocker, low], 10)) == []
    p._handled, p._last_episode_end, p._no_safe_layer = {"Drone4"}, 0.0, True
    assert p._wake_reason(sit(2.5, [d4, blocker, low], 10.0)) is None        # still no layer
    assert p._wake_reason(sit(3.0, [d4], 9.5)) == "safe_layer_available"
    p._no_safe_layer = False
    assert p._wake_reason(sit(3.0, [d4], 12.0)) is None                 # recheck at 6 s
    assert p._wake_reason(sit(3.0, [d4], 5.0)) == "conflict_persists"   # closer: 2.5 s
    assert p._wake_reason(sit(3.0, [d4], 20.0)) is None                 # far: full 8 s


def test_a_layer_change_that_creates_a_conflict_is_refused_before_it_is_sent():
    """Live 2026-10-28 (run c38c12): Drone3 decided to return to layer 14 from a picture
    10 s old; by then Drone5 had turned back onto 14 toward it. The pre-check runs on the
    situation when the proposal arrives."""
    from legacy.corridor.conflict_detection import Track
    p = make_pilot()
    own = Track("Drone2", 16, 5, 17, 0, -0.8, 0)              # escaped to 17, westbound
    d5 = Track("Drone5", 16, -8, 14, 0, 0.8, 0)               # back on layer 14, eastbound
    sit = {"obs": p._feed.observe(), "own": own, "peers": [d5], "now": 0.0, "conflicts": []}
    assert [c.other_id for c in p.precheck_altitude(sit, 14.0)] == ["Drone5"]
    assert p.precheck_altitude(sit, 17.0) == []                # staying is fine
    sit["conflicts"] = p.what_if_altitude(sit, 14.0)          # already in that conflict:
    assert p.precheck_altitude(sit, 14.0) == []                # not created by the move

    async def go():
        async def fixed():
            return dict(sit, conflicts=[])
        p.situation = fixed
        return await p._t_propose({"maneuver": "change_altitude", "observation_id": "x",
                                   "altitude_m": 14.0, "reason": "back to plan"})
    out = asyncio.run(go())
    assert out["code"] == "would_conflict" and out["would_create"][0]["with"] == "Drone5"
    assert "safe_layers_m" in out and p.maneuvers == []       # nothing reached the port
