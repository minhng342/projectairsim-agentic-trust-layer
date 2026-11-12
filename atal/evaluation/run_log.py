"""Reading a run log after the run (the format is atal/core/events.py).

read_run() reports a run without run_end, or with a cut-off last line, as incomplete
instead of failing to load it.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from atal.core.events import RUN_END


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
    complete = bool(events) and events[-1].get("kind") == RUN_END
    return RunLog(path=path, events=events, complete=complete, truncated_tail=truncated,
                  bad_lines=bad)
