"""The contract of a run log: one JSON object per line (JSONL), one file per run.

Every event has these fields, then its own:
    run_id       the run's id (UTC start stamp + 6 hex digits)
    event_id     0, 1, 2, ... in the order written
    kind         what happened ("run_start", "proposal", "peer_message", ...)
    host_s       monotonic host seconds since the recorder started
    utc          wall-clock time of writing, ISO 8601
    sim_time_ns  simulator clock (ground truth) at recording time; None if unavailable.
                 Events may also carry their own sim times.

The first event is RUN_START (config, git commit and dirty files, versions); a complete
run ends with RUN_END. Values are made JSON safe by json_safe() before writing.

Written by atal/runtime/recorder.py (during a run), read by atal/evaluation/run_log.py
(afterwards). Neither imports the other; both follow this file.
"""
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum

RUN_START = "run_start"
RUN_END = "run_end"
HEADER_FIELDS = ("run_id", "event_id", "kind", "host_s", "utc", "sim_time_ns")


def json_safe(v):
    """JSON-safe: pydantic models, dataclasses, enums, datetimes, tuples; NaN/inf -> None."""
    if hasattr(v, "model_dump"):
        v = v.model_dump(mode="json")
    elif is_dataclass(v) and not isinstance(v, type):
        v = asdict(v)
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [json_safe(x) for x in v]
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Enum):
        return json_safe(v.value)
    if isinstance(v, datetime):
        return v.isoformat()
    if v is None or isinstance(v, (str, int, bool)):
        return v
    return repr(v)
