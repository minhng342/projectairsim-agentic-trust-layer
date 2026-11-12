"""ObservationFeed, attack hooks and the placeholder trust assessor (no simulator)."""
import json
import math

import pytest

from atal.attacks.hooks import Drop, PositionOffset, Replay
from atal.channels.observation_feed import VERDICT_FIELDS, ObservationFeed
from atal.core.observation import TrustAssessment, TrustStatus
from atal.core.telemetry import ValidationStatus
from tests.channel_fakes import ListRecorder, SnapAdapter


def make(hooks=(), assessor=None, vid="Drone1"):
    adapter, rec = SnapAdapter(), ListRecorder()
    return adapter, rec, ObservationFeed(adapter, vid, hooks=hooks, assessor=assessor, recorder=rec)


def test_observation_carries_receiver_fields_and_raw_payload_but_no_verdict():
    _, rec, feed = make()
    obs = feed.observe()
    assert obs.vehicle_id == "Drone1" and obs.sequence == 1
    assert obs.channel == "projectairsim_ground_truth:Drone1"
    assert obs.received_host_s > 0 and obs.received_at_utc is not None
    assert obs.source_sim_time_ns == obs.payload["sim_time_ns"] > 0
    assert obs.north_m == 1.0 and obs.east_m == 2.0 and obs.altitude_m == 6.0
    assert not VERDICT_FIELDS & set(obs.payload)           # adapter verdict never delivered...
    assert not any(k.startswith("claimed_") for k in obs.payload)   # ...not even renamed
    assert obs.assessment.status == TrustStatus.NOT_ASSESSED and not obs.assessment.assessed
    e = rec.of("observation")[0]
    assert e["observation"]["observation_id"] == obs.observation_id
    assert e["source_validation"]["status"] == "valid"     # kept on the evaluation side only
    assert "tampered_by" not in e


def test_sequences_and_ids_are_unique_per_observation():
    _, _, feed = make()
    a, b = feed.observe(), feed.observe()
    assert (a.sequence, b.sequence) == (1, 2) and a.observation_id != b.observation_id


def test_stale_source_is_delivered_without_a_verdict_and_recorded_as_stale():
    adapter, rec, feed = make()
    adapter.status = ValidationStatus.STALE
    obs = feed.observe()
    assert obs is not None and not VERDICT_FIELDS & set(obs.payload)
    assert rec.of("observation")[0]["source_validation"]["status"] == "stale"


def test_position_attack_changes_what_the_agent_sees_and_only_the_recorder_knows():
    _, rec, feed = make(hooks=[PositionOffset(north_m=5.0, up_m=2.0)])
    obs = feed.observe()
    assert obs.north_m == 6.0 and obs.altitude_m == 8.0
    dumped = json.dumps(obs.model_dump(mode="json"))
    assert "tamper" not in dumped and "position_offset" not in dumped and "clean" not in dumped
    e = rec.of("observation")[0]
    assert e["tampered_by"] == ["position_offset"]
    assert e["clean_payload"]["position_ned_m"]["x"] == 1.0
    assert e["clean_payload"]["altitude_local_m"] == 6.0


def test_attack_starts_at_the_configured_sequence():
    _, _, feed = make(hooks=[PositionOffset(north_m=5.0, after_sequence=2)])
    assert feed.observe().north_m == 1.0
    assert feed.observe().north_m == 6.0


def test_hook_cannot_forge_a_verdict_and_old_valid_is_not_carried_forward():
    def forge(payload, ctx):
        payload["position_ned_m"]["x"] = 99.0
        payload["validation_status"] = "valid"
        payload["validation_errors"] = []
        return payload
    _, rec, feed = make(hooks=[forge])
    obs = feed.observe()
    assert not VERDICT_FIELDS & set(obs.payload)
    assert obs.payload["claimed_validation_status"] == "valid"   # kept as a claim, visible to the assessor
    assert obs.assessment.status == TrustStatus.NOT_ASSESSED
    assert rec.of("observation")[0]["source_validation"]["status"] == "valid"   # clean source was valid...
    assert obs.north_m == 99.0                                                # ...but this content was never checked


def test_hook_gets_a_private_copy_and_cannot_alter_the_clean_record():
    seen = {}

    def grab(payload, ctx):
        seen["p"] = payload
        return payload
    _, rec, feed = make(hooks=[grab, PositionOffset(north_m=1.0)])
    feed.observe()
    seen["p"]["position_ned_m"]["x"] = -500.0       # mutate after the fact
    assert rec.of("observation")[0]["clean_payload"]["position_ned_m"]["x"] == 1.0


def test_hook_cannot_change_receiver_fields():
    def rewrite(payload, ctx):
        payload["observation_id"] = "forged"
        payload["sequence"] = 999
        return payload
    _, _, feed = make(hooks=[rewrite])
    obs = feed.observe()
    assert obs.observation_id != "forged" and obs.sequence == 1 and obs.vehicle_id == "Drone1"


@pytest.mark.parametrize("hook", [Drop(after_sequence=1), lambda p, c: 1 / 0])
def test_dropped_or_crashing_hook_delivers_nothing(hook):
    _, rec, feed = make(hooks=[hook])
    assert feed.observe() is None
    assert len(rec.of("observation_dropped")) == 1 and not rec.of("observation")


def test_replay_freezes_content_while_receiver_sequence_keeps_advancing():
    _, rec, feed = make(hooks=[Replay(after_sequence=3)])
    obs = [feed.observe() for _ in range(5)]
    ts = [o.source_sim_time_ns for o in obs]
    assert ts[0] < ts[1] and ts[2] == ts[3] == ts[4]        # sender's clock frozen from #3
    assert [o.sequence for o in obs] == [1, 2, 3, 4, 5]      # receiver still counts
    assert [e.get("tampered_by") for e in rec.of("observation")][3:] == [["replay"], ["replay"]]


def test_adapter_error_gives_none_and_is_recorded():
    adapter, rec, feed = make()
    adapter.raises = True
    assert feed.observe() is None
    assert "timed out" in rec.of("observation_error")[0]["error"]


def test_resolve_returns_the_feeds_own_record_for_its_own_ids_only():
    adapter, _, feed = make()
    other = ObservationFeed(adapter, "Drone2")
    obs = feed.observe()
    rec = feed.resolve(obs.observation_id)
    assert rec is not None and rec is not obs and rec == obs     # same content, separate copy
    assert feed.resolve(other.observe().observation_id) is None  # another vehicle's id
    assert feed.resolve("f" * 32) is None
    assert feed.resolve(None) is None and feed.resolve(obs) is None   # ids only


def test_editing_the_agents_copy_cannot_change_the_record():
    _, rec, feed = make()
    obs = feed.observe()
    obs.payload["position_ned_m"]["x"] = 999.0
    obs.payload["altitude_local_m"] = -1.0
    assert feed.resolve(obs.observation_id).north_m == 1.0
    assert feed.latest().north_m == 1.0 and feed.history()[-1].altitude_m == 6.0
    assert rec.of("observation")[0]["observation"]["payload"]["position_ned_m"]["x"] == 1.0


def test_observation_fields_cannot_be_reassigned():
    _, _, feed = make()
    obs = feed.observe()
    for field, value in [("vehicle_id", "Drone2"), ("payload", {}), ("assessment", None),
                         ("received_host_s", 0.0)]:
        with pytest.raises(Exception):
            setattr(obs, field, value)
    with pytest.raises(Exception):
        obs.assessment.status = TrustStatus.TRUSTED


def test_resolve_forgets_old_observations_beyond_its_memory():
    adapter = SnapAdapter()
    feed = ObservationFeed(adapter, "Drone1", remember=2)
    first = feed.observe()
    feed.observe(), feed.observe()
    assert feed.resolve(first.observation_id) is None


# ------------------------------------------------------------ cadence and history
def test_history_and_latest_do_not_sample_and_keep_gaps_from_drops():
    adapter = SnapAdapter()
    feed = ObservationFeed(adapter, "Drone1", hooks=[lambda p, c: None if c.sequence == 2 else p],
                           history_len=3)
    assert feed.latest() is None and feed.history() == []
    for _ in range(5):
        feed.observe()
    calls = adapter.calls
    assert [o.sequence for o in feed.history()] == [3, 4, 5]     # bounded, oldest first
    assert [o.sequence for o in feed.history(2)] == [4, 5] and feed.history(0) == []
    assert feed.latest().sequence == 5
    assert adapter.calls == calls                                 # reading never samples


def test_sampler_collects_at_its_own_rate_and_assessor_sees_the_stream():
    import asyncio
    pairs = []

    class Spy:
        name = "spy"

        def assess(self, obs, previous):
            pairs.append((obs.sequence, previous.sequence if previous else None))
            return TrustAssessment(observation_id=obs.observation_id, assessor="spy",
                                   assessed=True, status=TrustStatus.TRUSTED)

    async def go():
        feed = ObservationFeed(SnapAdapter(), "Drone1", assessor=Spy(), recorder=ListRecorder())
        stop = asyncio.Event()
        task = asyncio.ensure_future(feed.sample(50.0, stop))
        await asyncio.sleep(0.25)
        running = feed.sample_hz
        stop.set()
        await task
        return feed, running
    feed, running = asyncio.run(go())
    n = len(pairs)
    assert running == 50.0 and feed.sample_hz is None
    assert 6 <= n <= 16                                     # ~12 at 50 Hz over 0.25 s
    assert pairs == [(i, i - 1 if i > 1 else None) for i in range(1, n + 1)]
    states = [e["status"] for e in feed._recorder.of("sampling")]
    assert states == ["started", "stopped"]


def test_current_waits_for_a_sample_newer_than_the_given_time():
    import asyncio

    async def go():
        feed = ObservationFeed(SnapAdapter(), "Drone1")
        stop = asyncio.Event()
        task = asyncio.ensure_future(feed.sample(20.0, stop))
        await asyncio.sleep(0.12)
        boundary = feed.last_sequence
        newer = await feed.current(after_sequence=boundary, timeout_s=1.0)
        any_ = await feed.current()
        stop.set()
        await task
        return boundary, newer, any_
    boundary, newer, any_ = asyncio.run(go())
    assert newer is not None and newer.sequence > boundary
    assert any_ is not None


def test_current_times_out_when_every_message_is_dropped():
    import asyncio

    async def go():
        feed = ObservationFeed(SnapAdapter(), "Drone1", hooks=[Drop()])
        stop = asyncio.Event()
        task = asyncio.ensure_future(feed.sample(50.0, stop))
        got = await feed.current(timeout_s=0.2)
        stop.set()
        await task
        return got
    assert asyncio.run(go()) is None


def test_current_without_a_sampler_observes_now():
    import asyncio
    adapter = SnapAdapter()
    feed = ObservationFeed(adapter, "Drone1")
    obs = asyncio.run(feed.current(after_sequence=10**9))
    assert obs is not None and adapter.calls == 1


def test_sampler_rejects_a_bad_rate():
    import asyncio
    with pytest.raises(ValueError):
        asyncio.run(ObservationFeed(SnapAdapter(), "Drone1").sample(0))


def test_unknown_vehicle_rejected():
    with pytest.raises(ValueError):
        ObservationFeed(SnapAdapter(), "Drone9")


def test_assessor_sees_previous_observation_and_a_crash_is_not_assessed():
    calls = []

    class Spy:
        name = "spy"

        def assess(self, obs, previous):
            calls.append((obs.sequence, previous.sequence if previous else None))
            if obs.sequence == 2:
                raise RuntimeError("boom")
            return TrustAssessment(observation_id=obs.observation_id, assessor="spy",
                                   assessed=True, status=TrustStatus.TRUSTED)
    _, _, feed = make(assessor=Spy())
    a, b = feed.observe(), feed.observe()
    assert calls == [(1, None), (2, 1)]
    assert a.assessment.status == TrustStatus.TRUSTED
    assert b.assessment.status == TrustStatus.NOT_ASSESSED and "boom" in b.assessment.reasons[0]


def test_missing_or_malformed_fields_read_as_nan_not_defaults():
    def wreck(payload, ctx):
        del payload["position_ned_m"]
        payload["altitude_local_m"] = "high"
        return payload
    _, _, feed = make(hooks=[wreck])
    obs = feed.observe()
    assert math.isnan(obs.north_m) and math.isnan(obs.altitude_m)


def test_resolved_copies_cannot_change_the_record_or_later_reads():
    _, _, feed = make()
    obs = feed.observe()
    got = feed.resolve(obs.observation_id)
    got.payload["position_ned_m"]["x"] = 999.0
    assert feed.resolve(obs.observation_id).north_m == 1.0
    assert feed.latest().north_m == 1.0 and feed.history()[-1].north_m == 1.0


def test_assessor_cannot_change_the_record_or_its_history():
    class Vandal:
        name = "vandal"

        def assess(self, obs, previous):
            obs.payload["altitude_local_m"] = -50.0
            if previous is not None:
                previous.payload["altitude_local_m"] = -60.0
            return TrustAssessment(observation_id=obs.observation_id, assessor="vandal",
                                   assessed=True, status=TrustStatus.TRUSTED)
    _, rec, feed = make(assessor=Vandal())
    a, b = feed.observe(), feed.observe()
    assert [o.altitude_m for o in feed.history()] == [6.0, 6.0]
    assert feed.resolve(a.observation_id).altitude_m == 6.0 == b.altitude_m
    assert all(e["observation"]["payload"]["altitude_local_m"] == 6.0
               for e in rec.of("observation"))
