"""EvaluationRecorder: written during the run, readable when incomplete (no simulator)."""
import json
import math
from dataclasses import dataclass
from enum import Enum

import pytest

from atal.core.action import ActionType, CommandResult, CommandStatus
from atal.evaluation.run_log import read_run
from atal.runtime.recorder import EvaluationRecorder


def rec(tmp_path, **kw):
    return EvaluationRecorder(run_config={"min_separation_m": 2.0}, path=tmp_path / "run.jsonl",
                              include_git=False, **kw)


def test_run_start_config_and_event_fields(tmp_path):
    clock = iter([0.0, 1.5, 2.0, 3.0, 4.0])
    r = rec(tmp_path, sim_clock=lambda: 42, clock=lambda: next(clock))
    r.record("proposal", vehicle_id="Drone1", request_id="abc")
    r.close("pass")
    log = read_run(r.path)
    assert log.complete and log.outcome == "pass" and not log.truncated_tail
    start, prop = log.events[0], log.events[1]
    assert start["kind"] == "run_start" and start["config"] == {"min_separation_m": 2.0}
    assert "python" in start["versions"]
    assert [e["event_id"] for e in log.events] == [0, 1, 2]
    assert prop["sim_time_ns"] == 42 and prop["host_s"] == 2.0 and prop["run_id"] == r.run_id
    assert log.events[-1]["event_counts"] == {"run_start": 1, "proposal": 1}


def test_events_are_on_disk_before_close_and_an_unclosed_run_reads_as_incomplete(tmp_path):
    r = rec(tmp_path)
    r.record("observation", vehicle_id="Drone1")
    log = read_run(r.path)                       # simulates a crash: never closed
    assert [e["kind"] for e in log.events] == ["run_start", "observation"]
    assert not log.complete and log.outcome is None


def test_cut_off_last_line_is_reported_not_fatal(tmp_path):
    r = rec(tmp_path)
    r.record("truth", poses={})
    r._file.write('{"run_id": "x", "kind": "trut')      # power cut mid-write
    r._file.flush()
    log = read_run(r.path)
    assert log.truncated_tail and len(log.events) == 2 and log.bad_lines == 0


def test_values_are_json_safe(tmp_path):
    class Color(Enum):
        RED = "red"

    @dataclass
    class D:
        x: float
    r = rec(tmp_path)
    res = CommandResult(request_id="r", vehicle_id="Drone1", action_type=ActionType.LAND,
                        status=CommandStatus.REFUSED)
    r.record("misc", nan=math.nan, inf=math.inf, enum=Color.RED, dc=D(1.0), model=res,
             tup=(1, 2), nested={"a": [math.nan]}, obj=object())
    r.close("x")
    e = read_run(r.path).events[1]
    assert e["nan"] is None and e["inf"] is None and e["enum"] == "red"
    assert e["dc"] == {"x": 1.0} and e["model"]["status"] == "refused" and e["tup"] == [1, 2]
    assert e["nested"] == {"a": [None]} and e["obj"].startswith("<object")
    for line in r.path.read_text().splitlines():
        json.loads(line)                         # strict JSON: no NaN tokens


def test_events_after_close_are_ignored_and_close_is_idempotent(tmp_path):
    r = rec(tmp_path)
    r.close("pass")
    assert r.record("late") is None
    r.close("fail")
    log = read_run(r.path)
    assert log.outcome == "pass" and len(log.events) == 2


def test_context_manager_marks_exceptions_as_aborted(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with rec(tmp_path) as r:
            raise KeyboardInterrupt()
    log = read_run(r.path)
    assert log.complete and log.outcome == "aborted" and "KeyboardInterrupt" in log.events[-1]["error"]


def test_failing_sim_clock_never_breaks_recording(tmp_path):
    def boom():
        raise RuntimeError("disconnected")
    r = rec(tmp_path, sim_clock=boom)
    assert r.record("truth") is not None
    assert read_run(r.path).events[-1]["sim_time_ns"] is None


def test_explicit_sim_time_wins_over_sim_clock(tmp_path):
    r = rec(tmp_path, sim_clock=lambda: 1)
    r.record("correction", sim_time_ns=777)
    assert read_run(r.path).events[-1]["sim_time_ns"] == 777


def test_recorder_for_binds_fields(tmp_path):
    r = rec(tmp_path)
    on_event = r.recorder_for(source="executor")
    on_event("correction", n=1)
    e = read_run(r.path).events[-1]
    assert e["source"] == "executor" and e["n"] == 1


def test_default_path_is_unique_per_run(tmp_path):
    a = EvaluationRecorder(run_dir=tmp_path, include_git=False)
    b = EvaluationRecorder(run_dir=tmp_path, include_git=False)
    assert a.path != b.path and a.path.parent == tmp_path
    a.close("x"), b.close("x")


def test_real_executor_events_reach_the_file(tmp_path):
    """Live Pass 5.5 bug: on_event(kind, fields) did not match recorder_for's signature,
    so corrections were missing from the run log. Uses the real executor + recorder."""
    import asyncio

    from atal.flight.executor import ExecutorConfig, ProjectAirSimExecutor
    from tests.test_executor import FAST, FakeAdapter, FakeDrone, FakeWorld, move
    world = FakeWorld()
    drone = FakeDrone(world)
    r = rec(tmp_path)
    ex = ProjectAirSimExecutor(FakeAdapter(world, drone), config=FAST, log=lambda *_: None,
                               on_event=r.recorder_for(source="executor"))
    world.misses = [(-1.8, 1.4, 0.0)]
    a = move(0.0, -3.0, 6.0, speed=3.0)
    assert asyncio.run(ex.execute(a)).corrections == 1
    r.close("x")
    corr = read_run(r.path).of_kind("correction")
    assert len(corr) == 1 and corr[0]["request_id"] == a.request_id
    assert corr[0]["source"] == "executor" and corr[0]["n"] == 1


def test_summary_survives_events_whose_result_field_is_plain_text(tmp_path):
    """Live 2026-10-27: the executor's final_approach event has result="settled" (a string);
    the summarizer crashed treating every 'result' as a nested record."""
    from atal.evaluation.summarize_run import summarize
    from atal.runtime.recorder import EvaluationRecorder
    rec = EvaluationRecorder(run_config={}, run_dir=tmp_path)
    rec.record("proposal", vehicle_id="Drone1", request_id="r1", observation_id="o1",
               action={"action_type": "move_to_position"})
    rec.record("final_approach", vehicle_id="Drone1", request_id="r1", result="settled", commands=12)
    rec.record("command_result", vehicle_id="Drone1", observation_id="o1",
               result={"request_id": "r1", "status": "succeeded", "elapsed_s": 9.4})
    rec.close("pass")
    text = summarize(rec.path)
    assert "move_to_position" in text and "succeeded 9.4s" in text
