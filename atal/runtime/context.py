"""What every part of one run needs to share."""
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from atal.runtime.rates import Timing


@dataclass
class RunContext:
    args: Any                      # the command-line options
    adapter: Any                   # the simulator (offline or Project AirSim)
    recorder: Any                  # the run log
    flight: Any                    # the flight executor shared by every drone
    timing: Timing
    log: Callable[[str], None]     # console output
    llm: dict | None               # the prepared model, or None for the fixed planner
    stem: Path                     # output path of this run without a suffix (.../run_00)
    stop: asyncio.Event = field(default_factory=asyncio.Event)   # set when the run is over
