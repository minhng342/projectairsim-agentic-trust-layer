"""Phase 3: three honest drones. Peer messages, the radio-range bus, belief with sources,
peer-claim rules, Voronoi re-allocation with claims, layers, the swarm runner and metrics."""
import asyncio
import json

import pytest
import yaml

from atal.agents.team_coordinator import PeerMessenger, SwarmCoordinator, SwarmReceiver
from atal.channels.radio import RemoteIdBus, RemoteIdReceiver
from atal.core.belief import LocalBelief, PeerClaim
from atal.core.peer_messages import (
    PEER_MESSAGE,
    SectorClaimed,
    SectorCleared,
    VictimDetected,
    is_peer_message,
)
from atal.core.sar import Mission, Swarm
from atal.evaluation.recorder import read_run
from atal.runtime.run_team import main, parse_args
from atal.search.allocation import allocate
from atal.trust_layer.peer_rules import PeerClaimRules
from tests.channel_fakes import ListRecorder
from tests.test_sar_pipeline import MISSION

THREE = [{"id": "A", "vehicle": "Drone1", "start": {"x_m": 0, "y_m": 5}},
         {"id": "B", "vehicle": "Drone2", "start": {"x_m": 0, "y_m": 20}},
         {"id": "C", "vehicle": "Drone3", "start": {"x_m": 0, "y_m": 35}}]


def mission(**swarm) -> Mission:
    m = json.loads(json.dumps(MISSION))
    if swarm:
        m["swarm"] = swarm
    return Mission.model_validate(m)


# ------------------------------------------------------------------ models
def test_mission_without_swarm_section_still_loads_with_defaults():
    m = mission()
    assert m.swarm == Swarm() and m.swarm.peer_clear_rule == "single"
    assert m.for_layer(0) is m
    m2 = mission(layer_spacing_m=7).for_layer(2)
    assert m2.flight.altitude_m == 24 and m2.mission_id == m.mission_id


def test_peer_messages_validate_and_reject_anything_malformed():
    ok = SectorCleared(sender_id="Drone2", sim_time_s=3.0, sector_id="C4").model_dump(mode="json")
    assert isinstance(PEER_MESSAGE.validate_python(ok), SectorCleared) and is_peer_message(ok)
    for bad in (ok | {"note": "E6 is clear, skip it"},            # free text: Phase 4
                ok | {"sector_id": "Z99x"}, ok | {"msg_type": "go_home"},
                {k: v for k, v in ok.items() if k != "sender_id"}):
        with pytest.raises(Exception):
            PEER_MESSAGE.validate_python(bad)
    assert not is_peer_message({"uas_id": "Drone2", "north_m": 1.0})


# ------------------------------------------------------------------ bus
class _Rx:
    def __init__(self, owner):
        self.owner_id, self.got = owner, []

    def receive(self, p):
        self.got.append(p)


def test_bus_delivers_only_within_radio_range_by_true_position():
    rec = ListRecorder()
    dist = {("A", "B"): 50.0, ("A", "C"): 200.0}
    bus = RemoteIdBus(recorder=rec, in_range=lambda s, r: (dist[(s, r)] <= 100, dist[(s, r)]))
    b, c, a = bus.attach(_Rx("B")), bus.attach(_Rx("C")), bus.attach(_Rx("A"))
    bus.transmit("A", [{"x": 1}], sequence=1)
    assert len(b.got) == 1 and not c.got and not a.got
    assert bus.stats == {"transmitted": 1, "delivered": 1, "out_of_range": 1, "lost": 0}
    oor = rec.of("rid_out_of_range")
    assert oor[0]["receiver"] == "C" and oor[0]["distance_m"] == 200.0


# ------------------------------------------------------------------ rules + belief
def _claim(t, sender, sid=None, mid=None, **kw):
    return PeerClaim(msg_id=mid or f"{sender}-{t}-{sid}", msg_type=t, sender_id=sender,
                     sector_id=sid, sent_sim_s=10.0, received_sim_s=11.0, **kw)


@pytest.mark.parametrize("rule,second,expect", [("single", False, True),
                                                ("corroborated", False, False),
                                                ("corroborated", True, True),
                                                ("self_only", True, False)])
def test_peer_clear_rules(rule, second, expect):
    m = mission(peer_clear_rule=rule)
    b = LocalBelief("Drone1", m)
    rules = PeerClaimRules(m.swarm)
    c1 = _claim("sector_cleared", "Drone2", "B2")
    b.add_peer_claim(c1)
    v = rules.on_cleared(b, c1)
    if second:
        c2 = _claim("sector_cleared", "Drone3", "B2")
        v = rules.on_cleared(b, c2)
        b.add_peer_claim(c2)
    assert v.accepted is expect and c1.msg_id in v.evidence
    assert b.cleared_by("B2") == (["Drone2", "Drone3"] if second else ["Drone2"])


def test_a_peer_clear_never_overrides_own_sweep_and_counts_as_done():
    m = mission()
    b = LocalBelief("Drone1", m)
    b.mark_swept("A1", 5.0, "r1")
    assert not b.mark_peer_cleared("A1", 6.0, "Drone2", ["m1"])
    assert b.sectors["A1"].verified_by_self
    assert b.mark_peer_cleared("B1", 6.0, "Drone2", ["m1"])
    s = b.sectors["B1"]
    assert s.status.value == "peer_cleared" and s.source == "Drone2" and not s.verified_by_self
    assert b.is_done("B1") and "B1" not in b.unsearched() and b.swept() == ["A1"]


def test_claim_expiry_is_measured_from_receipt_and_capped():
    rules = PeerClaimRules(Swarm(max_claim_s=100))
    c = _claim("sector_claimed", "Drone2", "B2", expires_sim_s=10.0 + 30)
    assert rules.claim_expiry(c) == 11.0 + 30
    squat = _claim("sector_claimed", "Drone2", "B2", expires_sim_s=1e9)
    assert rules.claim_expiry(squat) == 11.0 + 100


# ------------------------------------------------------------------ allocation
def test_allocation_claims_win_dropouts_reshare_and_stealing_fills_an_empty_cell():
    g = mission().grid
    pos = {"Drone1": (5.0, 5.0), "Drone2": (5.0, 35.0)}
    a = allocate(g, "Drone1", pos, g.ids())
    assert "A1" in a["cells"]["Drone1"] and "D1" in a["cells"]["Drone2"]
    a = allocate(g, "Drone1", pos, g.ids(), claims={"A1": "Drone2"})
    assert "A1" in a["cells"]["Drone2"] and "A1" not in a["cells"]["Drone1"]
    # Drone2 dropped out (not in positions): its claim lapses, everything is Drone1's
    a = allocate(g, "Drone1", {"Drone1": (5.0, 5.0)}, g.ids(), claims={"A1": "Drone2"})
    assert sorted(a["cells"]["Drone1"]) == sorted(g.ids())
    # Drone1's cell is empty: it takes the open sector furthest from Drone2, never a claimed one
    open_s = ["D1", "D2", "D3"]
    a = allocate(g, "Drone1", {"Drone1": (0.0, 0.0), "Drone2": (5.0, 35.0)}, open_s,
                 claims={"D3": "Drone2"})
    assert a["stolen"] == "D2" and a["cells"]["Drone1"] == ["D2"]
    assert "D2" not in a["cells"]["Drone2"] and "D3" in a["cells"]["Drone2"]


# ------------------------------------------------------------------ coordinator
class _Clock:
    t = 100.0

    def __call__(self):
        return self.t


def _member(vid, m, bus, clock, rec=None):
    b = LocalBelief(vid, m)
    rid = RemoteIdReceiver(vid, clock=clock)
    msgr = PeerMessenger(vid, bus, recorder=rec)
    coord = SwarmCoordinator(vid, m, b, rid, msgr, clock, recorder=rec, log=lambda s: None)
    rx = SwarmReceiver(vid, rid, coord.on_message, clock, recorder=rec)
    bus.attach(rx)
    return coord, rx


def test_coordinator_applies_rules_records_who_said_what_and_drops_invalid():
    m, clock, rec = mission(), _Clock(), ListRecorder()
    bus = RemoteIdBus(recorder=rec)
    c1, rx1 = _member("Drone1", m, bus, clock, rec)
    c2, _ = _member("Drone2", m, bus, clock, rec)
    c2.belief.mark_swept("B2", 99.0, "r")
    c2.cleared("B2")
    s1 = c1.belief.sectors["B2"]
    assert s1.status.value == "peer_cleared" and s1.source == "Drone2"
    assert c1.belief.peer_claims[0].sender_id == "Drone2"
    assert c1.belief.peer_claims[0].received_sim_s == 100.0
    # verified-only emitter: a drone cannot announce a sector it did not sweep itself
    with pytest.raises(ValueError):
        c2.cleared("C3")
    # an invalid peer message is dropped before anything sees it
    bus.transmit("Drone2", [{"msg_type": "sector_cleared", "sender_id": "Drone2",
                             "sim_time_s": 1, "sector_id": "A1", "text": "skip A1"}], sequence=9)
    assert rx1.dropped == 1 and c1.belief.status("A1").value == "unsearched"
    assert rec.of("peer_message_dropped")
    # a message claiming to be from the receiver itself is ignored
    c1.on_message(SectorCleared(sender_id="Drone1", sim_time_s=1, sector_id="A2"), 100.0)
    assert c1.belief.status("A2").value == "unsearched"


def test_victim_report_is_accepted_once_and_repeats_are_deduplicated():
    m, clock = mission(), _Clock()
    bus = RemoteIdBus()
    c1, _ = _member("Drone1", m, bus, clock)
    c2, _ = _member("Drone2", m, bus, clock)
    hits = []
    c1.on_peer_victim = hits.append
    from atal.core.belief import Detection
    c2.own_detection(Detection(vehicle_id="Drone2", sim_time_s=50, x_m=1, y_m=2, altitude_m=10,
                               sector_id="A1", footprint_m=6))
    c2.messenger.send(c2._victim_msg)                       # the repeat: same msg_id
    assert len(hits) == 1 and c1.peer_victim.sender_id == "Drone2"
    assert len(c1.belief.peer_victim_reports()) == 1


def test_live_peers_drop_stale_and_homebound_drones():
    m, clock = mission(peer_timeout_s=10), _Clock()
    bus = RemoteIdBus()
    c1, rx1 = _member("Drone1", m, bus, clock)
    rx1.receive({"uas_id": "Drone2", "north_m": 1.0, "east_m": 2.0})
    rx1.receive({"uas_id": "Drone3", "north_m": 3.0, "east_m": 4.0, "mission_state": "returning"})
    assert c1.live_peers(clock()) == {"Drone2": (1.0, 2.0)}
    assert c1.live_peers(clock() + 11) == {}


# ------------------------------------------------------------------ end to end
def files(tmp_path, victim=(25.0, 33.0), runs=1, drones=THREE, **swarm):
    m = json.loads(json.dumps(MISSION))
    m["swarm"] = {"radio_range_m": 200, "layer_spacing_m": 5, "separation_cruise_m": 4.5,
                  **swarm}                       # the floor just under the 5 m layers
    sc = {"scenario_id": "sc-swarm-test", "seed": 7, "runs": runs, "drones": drones}
    sc |= {"victim": {"x_m": victim[0], "y_m": victim[1]}} if victim else {"randomize_victim": True}
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(sc), encoding="utf-8")
    return str(tmp_path / "m.yaml"), str(tmp_path / "s.yaml")


def run(tmp_path, *extra, **kw):
    m, s = files(tmp_path, **kw)
    out = tmp_path / "out"
    code = asyncio.run(main(parse_args(["--offline", "--speedup", "100", "--mission", m,
                                        "--scenario", s, "--out", str(out), *extra])))
    batch = next(out.iterdir())
    return code, batch, json.loads((batch / "batch_summary.json").read_text())


def test_three_drones_find_the_victim_on_separate_layers_and_all_land(tmp_path):
    code, batch, summary = run(tmp_path)
    assert code == 0 and summary["all_passed"], summary["runs"][0]["fail_reasons"]
    m = summary["runs"][0]
    assert m["victim_found"] and m["victim_sector"] == "D3" and m["duplicate_sweeps"] == 0
    assert {d["altitude_m"] for d in m["drones"].values()} == {10, 15, 20}
    assert m["min_separation_m"] >= 4.9                    # layers 5 m apart at the least
    assert all(d["landed"] and d["return_home"] == "home_reached" for d in m["drones"].values())
    others = [v for v in m["drones"] if v != m["finder"]]
    assert set(m["time_to_hear_s"]) == set(others)
    assert all(t is not None for t in m["time_to_hear_s"].values())
    assert m["messages"]["peer_dropped_invalid"] == 0 and m["messages"]["delivery_ratio"] == 1.0
    log = read_run(batch / "run_00.jsonl")
    # every drone swept with claims first, and the others heard its clears
    sent = [e["msg"]["msg_type"] for e in log.of_kind("peer_message_sent")]
    assert "sector_claimed" in sent and "sector_cleared" in sent and "victim_detected" in sent
    assert log.of_kind("sar_allocation") and log.of_kind("peer_rule")
    for d in ("Drone1", "Drone2", "Drone3"):
        mem = json.loads((batch / f"run_00.{d}.memory.json").read_text())
        assert all(c["sender_id"] != d for c in mem["peer_claims"])
    assert (batch / "run_00.svg").read_text().startswith("<svg")


def test_full_sweep_shares_the_grid_without_duplicates(tmp_path):
    code, batch, summary = run(tmp_path, "--full-sweep")
    m = summary["runs"][0]
    assert code == 0, m["fail_reasons"]
    assert m["sectors_swept"] == 16 and m["area_coverage"] >= 0.99
    # the finder does not call the victim's sector clear, so a peer may sweep it again
    assert set(m["duplicated_sectors"]) <= {m["detected_sector"]}
    assert all(d["sectors_swept"] >= 3 for d in m["drones"].values())   # work was shared


def test_without_cooperation_drones_sweep_the_same_sectors(tmp_path):
    code, batch, summary = run(tmp_path, "--full-sweep", "--peer-clear-rule", "self_only")
    m = summary["runs"][0]
    assert m["sectors_swept"] == 16
    # peer clears are ignored, so once a drone's own cell is done it steals already-swept
    # sectors: wasted work that the single-peer rule avoids
    assert m["duplicate_sweeps"] > 0


def test_out_of_range_peers_still_finish_and_the_finder_is_heard_later_or_never(tmp_path):
    code, batch, summary = run(tmp_path, "--radio-range", "8")
    m = summary["runs"][0]
    assert m["victim_found"] and m["messages"]["out_of_range"] > 0
    assert all(d["landed"] for d in m["drones"].values())


def test_llm_swarm_with_the_scripted_model(tmp_path):
    code, batch, summary = run(tmp_path, "--planner", "llm", "--model", "scripted")
    m = summary["runs"][0]
    assert code == 0, m["fail_reasons"]
    assert sum(d["llm_decisions"] for d in m["drones"].values()) > 0
    log = read_run(batch / "run_00.jsonl")
    prompt = log.of_kind("llm_decision")[0]["prompt"]
    assert '"team"' in prompt and "assigned_to_you" in prompt


def test_one_drone_through_the_swarm_runner(tmp_path):
    code, batch, summary = run(tmp_path, drones=THREE[:1])
    m = summary["runs"][0]
    assert code == 0 and m["victim_found"] and m["messages"]["delivery_ratio"] is None


def test_team_launches_from_one_pad_area_and_flies_to_staging_points(tmp_path):
    pads = [{"id": k, "vehicle": f"Drone{i + 1}", "start": {"x_m": 0, "y_m": -3 + 3 * i}}
            for i, k in enumerate("ABC")]
    stage = [{"x_m": 0, "y_m": 5}, {"x_m": 0, "y_m": 20}, {"x_m": 0, "y_m": 35}]
    code, batch, summary = run(tmp_path, drones=pads, staging=stage)
    m = summary["runs"][0]
    assert code == 0, m["fail_reasons"]
    log = read_run(batch / "run_00.jsonl")
    staged = {e["vehicle_id"]: e["at"]["y_m"] for e in log.of_kind("staged")}
    assert staged == {"Drone1": 5, "Drone2": 20, "Drone3": 35}
    # the first split after staging gives everyone sectors, not two empty cells
    assert all(d["sectors_swept"] >= 1 for d in m["drones"].values())
    moves = [e for e in log.of_kind("gate_decision")
             if e["decision"]["outcome"] == "approved"]
    assert moves


def test_guard_allows_only_the_staging_move():
    from atal.core.action import ActionType, ProposedAction
    from atal.trust_layer.invariant_guard import SarGuard
    m = mission()
    g = SarGuard(m, LocalBelief("Drone1", m), (0, 0))

    def mv(n, e, alt=10):
        return ProposedAction(vehicle_id="Drone1", action_type=ActionType.MOVE_TO_POSITION,
                              north_m=n, east_m=e, altitude_m=alt, reason="t")
    assert not g.check(mv(0, 20), 0, 0, None).ok
    g.staging_xy = (0, 20)
    assert g.check(mv(0, 20), 0, 0, None).ok
    assert not g.check(mv(5, 20), 0, 0, None).ok and not g.check(mv(0, 20, 30), 0, 0, None).ok


def test_sim_clock_watchdog_marks_stall_and_stops():
    import asyncio

    from atal.runtime.watchers import watch_sim_clock as _watch_sim_clock

    class A:
        stalled = None
        ns = 1_000

        def latest_poses(self):
            return {"Drone1": (self.ns, 0.0, 0.0, 0.0)}

    now = [0.0]
    a, calls = A(), []

    async def go():
        stop = asyncio.Event()
        task = asyncio.create_task(_watch_sim_clock(a, stop, 30.0, 0.001, calls.append,
                                                    clock=lambda: now[0]))
        for _ in range(5):                       # clock moving: no stall
            a.ns += 10
            now[0] += 20.0
            await asyncio.sleep(0.005)
        assert not calls
        now[0] += 31.0                           # frozen for 31 s of host time
        await asyncio.wait_for(task, 1.0)

    asyncio.run(go())
    assert calls and a.stalled and "stopped" in a.stalled


def test_one_failed_sweep_does_not_end_a_drones_search(tmp_path, monkeypatch):
    """Live 2026-11-05: one path_failed in G2 sent Drone3 home for the rest of the run."""
    from atal.core.action import ActionType, CommandResult, CommandStatus
    from atal.flight.sector_sweep import SectorExecutor
    real, failed = SectorExecutor.execute, []

    async def execute(self, action):
        if (action.action_type == ActionType.SWEEP_SECTOR and action.vehicle_id == "Drone3"
                and not failed):
            failed.append(action.sector_id)
            return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                                 action_type=action.action_type, status=CommandStatus.FAILED,
                                 reason_code="path_failed", sent_to_simulator=True,
                                 reason="injected")
        return await real(self, action)
    monkeypatch.setattr(SectorExecutor, "execute", execute)
    code, batch, summary = run(tmp_path, "--full-sweep")
    m = summary["runs"][0]
    assert failed and code == 0, m["fail_reasons"]
    d3 = m["drones"]["Drone3"]
    assert d3["ended_by"] not in ("failed", "aborted") and d3["sectors_swept"] >= 1
    assert m["sectors_swept"] == 16
    ev = read_run(batch / "run_00.jsonl").of_kind("sweep_failed")
    assert [e["sector_id"] for e in ev] == failed and ev[0]["in_row"] == 1


def test_two_failed_sweeps_in_a_row_still_end_the_search(tmp_path, monkeypatch):
    from atal.core.action import ActionType, CommandResult, CommandStatus
    from atal.flight.sector_sweep import SectorExecutor
    real = SectorExecutor.execute

    async def execute(self, action):
        if action.action_type == ActionType.SWEEP_SECTOR and action.vehicle_id == "Drone3":
            return CommandResult(request_id=action.request_id, vehicle_id=action.vehicle_id,
                                 action_type=action.action_type, status=CommandStatus.FAILED,
                                 reason_code="path_failed", sent_to_simulator=True,
                                 reason="injected")
        return await real(self, action)
    monkeypatch.setattr(SectorExecutor, "execute", execute)
    code, batch, summary = run(tmp_path, "--full-sweep")
    m = summary["runs"][0]
    assert m["drones"]["Drone3"]["ended_by"] == "failed" and code == 1
    assert len(read_run(batch / "run_00.jsonl").of_kind("sweep_failed")) == 2
    assert m["sectors_swept"] == 16                       # the other two covered its cell


# ------------------------------------------------------------ audit 2026-11-05 (Phase 3)
def _spoof_two_names(identity):
    """The audit's reproduction: one transmitter (Drone3) sends SectorCleared(B2) as FakeA,
    then as FakeB, to a receiver using the corroborated rule."""
    m, clock = mission(peer_clear_rule="corroborated", identity=identity), _Clock()
    bus = RemoteIdBus()
    b = LocalBelief("Drone1", m)
    rid = RemoteIdReceiver("Drone1", clock=clock)
    coord = SwarmCoordinator("Drone1", m, b, rid, PeerMessenger("Drone1", bus), clock,
                             log=lambda s: None)
    rx = SwarmReceiver("Drone1", rid, coord.on_message, clock,
                       authenticated=identity == "authenticated")
    bus.attach(rx)
    liar = PeerMessenger("Drone3", bus)
    for name in ("FakeA", "FakeB"):
        liar.send(SectorCleared(sender_id=name, sim_time_s=1, sector_id="B2"))
    return b, rx


def test_one_transmitter_cannot_pose_as_two_peers_when_identity_is_authenticated():
    b, rx = _spoof_two_names("authenticated")
    assert b.status("B2").value == "unsearched"
    assert rx.identity_mismatches == 2 and not b.peer_claims


def test_unverified_identity_is_the_labelled_spoofing_condition():
    b, rx = _spoof_two_names("unverified")
    assert b.status("B2").value == "peer_cleared" and rx.identity_mismatches == 0


def test_remote_id_reports_are_bound_to_the_transmitter_too():
    m, clock = mission(), _Clock()
    bus = RemoteIdBus()
    _, rx1 = _member("Drone1", m, bus, clock)
    bus.attach(type("Tx", (), {"owner_id": "Drone3", "receive": lambda self, p: None})())
    bus.transmit("Drone3", [{"uas_id": "Drone2", "north_m": 1.0, "east_m": 2.0}], sequence=1)
    assert rx1.identity_mismatches == 1 and "Drone2" not in rx1.rid.latest()
    bus.transmit("Drone3", [{"uas_id": "Drone3", "north_m": 1.0, "east_m": 2.0}], sequence=2)
    assert "Drone3" in rx1.rid.latest()


def test_a_peer_processing_error_fails_the_run(tmp_path, monkeypatch):
    real = SwarmCoordinator.on_message
    calls = []

    def on_message(self, msg, t):
        if not calls:
            calls.append(msg.msg_id)
            raise RuntimeError("injected")
        return real(self, msg, t)
    monkeypatch.setattr(SwarmCoordinator, "on_message", on_message)
    code, batch, summary = run(tmp_path)
    m = summary["runs"][0]
    assert calls and code == 1 and m["messages"]["processing_errors"] == 1
    assert any("processing failed" in r for r in m["fail_reasons"])


def _sep_log(tmp_path, pairs):
    """A run log with truth samples: pairs = [(t_s, {vid: (north, east, alt)})]."""
    from atal.evaluation.recorder import EvaluationRecorder
    m = mission()
    rec = EvaluationRecorder(path=tmp_path / "r.jsonl", run_config={})
    for t, poses in pairs:
        rec.record("truth", sim_time_ns=int(t * 1e9), poses={
            v: {"sim_time_ns": int(t * 1e9), "north_m": p[0], "east_m": p[1],
                "altitude_m": p[2]} for v, p in poses.items()})
    rec.close("searched")
    drones = [{"vehicle": "Drone1", "start": {"x_m": 0, "y_m": 0}},
              {"vehicle": "Drone2", "start": {"x_m": 0, "y_m": 3}}]
    return read_run(tmp_path / "r.jsonl"), m, drones


def test_separation_floors_split_pad_zone_and_cruise(tmp_path):
    from atal.evaluation.swarm_metrics import separation
    ground = {"Drone1": (0, 0, 0), "Drone2": (0, 3, 0)}
    climb = {"Drone1": (0, 0, 20), "Drone2": (0, 3, 21)}          # 3.2 m apart on the pads
    cruise_ok = {"Drone1": (50, 50, 30), "Drone2": (50, 60, 35)}
    cruise_close = {"Drone1": (50, 50, 30), "Drone2": (52, 50, 31)}  # 2.2 m in cruise
    log, m, drones = _sep_log(tmp_path, [(0, ground), (0.5, climb), (1, cruise_ok)])
    s = separation(log, m, drones)
    assert s["measured"] and s["pad_violations"] == 0 and s["cruise_violations"] == 0
    assert s["min_pad_m"] == 3.16 and s["min_cruise_m"] == 11.18
    log, m, drones = _sep_log(tmp_path, [(0, ground), (0.5, cruise_close)])
    s = separation(log, m, drones)
    assert s["cruise_violations"] == 1 and s["violations"][0]["zone"] == "cruise"
    log, m, drones = _sep_log(tmp_path, [(0, ground), (9, cruise_ok)])     # 9 s gap
    assert not separation(log, m, drones)["measured"]


def test_acceptance_fails_on_separation_errors_and_identity_mismatches():
    from atal.runtime.acceptance import team_run_failures as run_failures
    base = {"victim_found": True, "drones": {}, "geofence_breaches": 0, "impacts": 0,
            "run_complete": True, "run_log_outcome": "victim_found",
            "messages": {}, "separation": {"measured": True, "pad_violations": 0,
                                           "cruise_violations": 0,
                                           "floors": {"pad_m": 1.5, "cruise_m": 5.0}}}
    assert run_failures(base, False) == []
    bad = dict(base, separation=dict(base["separation"], cruise_violations=1, min_cruise_m=0.1))
    assert any("cruise" in r for r in run_failures(bad, False))
    bad = dict(base, separation=dict(base["separation"], measured=False, max_gap_s=9))
    assert any("not measured" in r for r in run_failures(bad, False))
    spoofed = dict(base, messages={"identity_mismatch": 2})
    assert any("identity" in r for r in run_failures(spoofed, False))
    assert run_failures(dict(spoofed, insider_enabled=True), False) == []


# ------------------------------------------------------------ follow-up review (04be8d7)
def _truth_log(tmp_path, events):
    """events = [(t_s or None, {vid: (north, east, alt, pose_t_s or None)})]."""
    from atal.evaluation.recorder import EvaluationRecorder
    rec = EvaluationRecorder(path=tmp_path / "t.jsonl", run_config={})
    for t, poses in events:
        rec.record("truth", sim_time_ns=None if t is None else int(t * 1e9), poses={
            v: {"sim_time_ns": None if p[3] is None else int(p[3] * 1e9), "north_m": p[0],
                "east_m": p[1], "altitude_m": p[2]} for v, p in poses.items()})
    rec.close("searched")
    drones = [{"vehicle": "Drone1", "start": {"x_m": 0, "y_m": 0}},
              {"vehicle": "Drone2", "start": {"x_m": 0, "y_m": 3}}]
    return read_run(tmp_path / "t.jsonl"), mission(), drones


def test_samples_of_one_drone_never_make_separation_measured(tmp_path):
    from atal.evaluation.swarm_metrics import separation
    log, m, d = _truth_log(tmp_path, [(t, {"Drone1": (0, 0, 0, t)}) for t in (0, 0.5, 1)])
    s = separation(log, m, d)
    assert not s["measured"] and s["unmeasured_pairs"] == ["Drone1-Drone2"]


def test_skewed_airborne_pairs_are_gaps_not_measurements(tmp_path):
    from atal.evaluation.swarm_metrics import separation
    ev = [(0, {"Drone1": (0, 0, 0, 0), "Drone2": (0, 3, 0, 0)})]
    ev += [(t, {"Drone1": (50, 50, 30, t), "Drone2": (50, 60, 35, t - 1)}) for t in (3, 6)]
    log, m, d = _truth_log(tmp_path, ev)
    s = separation(log, m, d)
    assert s["skewed_pairs"] == 2 and not s["measured"]
    assert s["pairs"]["Drone1-Drone2"]["airborne"] == 0


def test_a_long_missing_pair_interval_in_a_well_sampled_run_is_unmeasured(tmp_path):
    from atal.evaluation.swarm_metrics import separation
    both = lambda t: {"Drone1": (50, 50, 30, t), "Drone2": (50, 60, 35, t)}
    ev = [(0, {"Drone1": (0, 0, 0, 0), "Drone2": (0, 3, 0, 0)})]
    ev += [(t / 2, both(t / 2)) for t in range(1, 10)]                     # to 4.5 s
    ev += [(t / 2, {"Drone1": (50, 50, 30, t / 2)}) for t in range(10, 20)]  # Drone2 gone 5 s
    ev += [(t / 2, both(t / 2)) for t in range(20, 24)]
    log, m, d = _truth_log(tmp_path, ev)
    s = separation(log, m, d)
    assert not s["measured"] and s["max_gap_s"] == 5.5
    log, m, d = _truth_log(tmp_path, [(0, {"Drone1": (0, 0, 0, 0), "Drone2": (0, 3, 0, 0)})]
                           + [(t / 2, both(t / 2)) for t in range(1, 24)])
    s = separation(log, m, d)
    assert s["measured"] and s["pairs"]["Drone1-Drone2"]["airborne"] == 23


def test_missing_timestamps_and_stale_poses_are_not_measurements(tmp_path):
    from atal.evaluation.swarm_metrics import separation
    ev = [(0, {"Drone1": (0, 0, 0, 0), "Drone2": (0, 3, 0, 0)})]
    ev += [(None, {"Drone1": (50, 50, 30, 1), "Drone2": (50, 60, 35, 1)})]   # event: no time
    ev += [(t, {"Drone1": (50, 50, 30, t), "Drone2": (50, 60, 35, None)}) for t in (1, 2, 3)]
    ev += [(t, {"Drone1": (50, 50, 30, t), "Drone2": (50, 60, 35, 0.2)}) for t in (4, 5)]
    log, m, d = _truth_log(tmp_path, ev)
    s = separation(log, m, d)
    pr = s["pairs"]["Drone1-Drone2"]
    assert s["samples_without_time"] == 1 and pr["airborne"] == 0 and pr["stale"] >= 1
    assert not s["measured"]
