"""
Evaluation recorder: the experiment's record of truth. One JSONL file per run.

    rec = EvaluationRecorder(run_config={...}, sim_clock=lambda: ...)
    rec.record("proposal", vehicle_id="Drone1", request_id=..., ...)
    ...
    rec.close("pass")

Each line is one event:
    {"run_id", "event_id", "kind", "host_s", "utc", "sim_time_ns", ...fields}
    host_s       monotonic host seconds since the recorder started
    sim_time_ns  simulator clock (ground truth) at recording time, from sim_clock;
                 None if unavailable. Events may also carry their own sim times.

Written DURING the run: every line is flushed as it is written, so a crash,
Ctrl+C or power-off keeps everything up to that point. The first line is
run_start (run id, config, thresholds, git commit + dirty files, versions);
a complete run ends with run_end. read_run() reports a run without run_end, or
with a cut-off last line, as incomplete instead of failing to load it.

The recorder is evaluation side: it may hold ground truth, clean payloads and
attack details that agents, the trust assessor and the gate must never see.
Recording problems never stop a flight: a failed write is counted, not raised.
"""
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean(v):
    """JSON-safe: pydantic models, dataclasses, enums, datetimes, tuples; NaN/inf -> None."""
    if hasattr(v, "model_dump"):
        v = v.model_dump(mode="json")
    elif is_dataclass(v) and not isinstance(v, type):
        v = asdict(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x) for x in v]
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Enum):
        return _clean(v.value)
    if isinstance(v, datetime):
        return v.isoformat()
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return repr(v)


def git_info(root: Path = REPO_ROOT) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD") or None
        dirty = [line[3:] for line in run("status", "--porcelain").splitlines() if line.strip()]
        return {"commit": commit, "dirty": bool(dirty), "dirty_files": dirty[:50]}
    except Exception as err:
        return {"commit": None, "error": f"{type(err).__name__}: {err}"}


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        from importlib.metadata import version
        for pkg in ("projectairsim", "pydantic"):
            try:
                out[pkg] = version(pkg)
            except Exception:
                out[pkg] = None
    except Exception:
        pass
    return out


class EvaluationRecorder:
    def __init__(self, run_config: dict | None = None, path: str | os.PathLike | None = None,
                 run_dir: str | os.PathLike = REPO_ROOT / "runs", sim_clock=None,
                 clock=time.monotonic, include_git: bool = True):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{uuid4().hex[:6]}"
        self.path = Path(path) if path else Path(run_dir) / f"run_{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sim_clock = sim_clock
        self._clock = clock
        self._t0 = clock()
        self._lock = threading.Lock()
        self._next_id = 0
        self._file = open(self.path, "w", encoding="utf-8", newline="\n")
        self.closed = False
        self.write_failures = 0
        self.counts: dict[str, int] = {}
        self.record("run_start", config=run_config or {},
                    git=git_info() if include_git else None, versions=versions(),
                    argv=sys.argv, pid=os.getpid())

    def set_sim_clock(self, sim_clock) -> None:
        self._sim_clock = sim_clock

    def record(self, kind: str, *, sim_time_ns: int | None = None, **fields) -> int | None:
        """Write one event now. Returns its event_id (None if closed or the write failed)."""
        if sim_time_ns is None and self._sim_clock is not None:
            try:
                sim_time_ns = self._sim_clock()
            except Exception:
                sim_time_ns = None
        with self._lock:
            if self.closed:
                return None
            event_id = self._next_id
            self._next_id += 1
            event = {"run_id": self.run_id, "event_id": event_id, "kind": kind,
                     "host_s": round(self._clock() - self._t0, 4),
                     "utc": datetime.now(timezone.utc).isoformat(),
                     "sim_time_ns": sim_time_ns}
            event.update(fields)
            try:
                self._file.write(json.dumps(_clean(event), allow_nan=False) + "\n")
                self._file.flush()
            except Exception:
                self.write_failures += 1
                return None
            self.counts[kind] = self.counts.get(kind, 0) + 1
            return event_id

    def recorder_for(self, **bound):
        """A record() that always adds these fields (e.g. source='executor').

        Accepts both call styles: rec(kind, **fields) and rec(kind, fields_dict),
        the latter being the executor's on_event(kind, fields) contract.
        """
        def rec(kind, fields: dict | None = None, **kw):
            return self.record(kind, **{**bound, **(fields or {}), **kw})
        return rec

    def close(self, outcome: str, **fields) -> None:
        if self.closed:
            return
        self.record("run_end", outcome=outcome, event_counts=dict(self.counts),
                    write_failures=self.write_failures, **fields)
        with self._lock:
            self.closed = True
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except Exception:
                pass
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.closed:
            if exc_type is None:
                self.close("finished")
            else:
                self.close("aborted", error=f"{exc_type.__name__}: {exc}")
        return False


@dataclass
class RunLog:
    path: Path
    events: list[dict]
    complete: bool            # ends with run_end
    truncated_tail: bool      # last line was cut off mid-write
    bad_lines: int

    @property
    def outcome(self) -> str | None:
        return self.events[-1].get("outcome") if self.complete else None

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("kind") == kind]


def read_run(path) -> RunLog:
    """Load a run log, tolerating an incomplete run (crash, Ctrl+C, cut-off last line)."""
    path = Path(path)
    events, bad, truncated = [], 0, False
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                truncated = True
            else:
                bad += 1
    complete = bool(events) and events[-1].get("kind") == "run_end"
    return RunLog(path=path, events=events, complete=complete, truncated_tail=truncated,
                  bad_lines=bad)
