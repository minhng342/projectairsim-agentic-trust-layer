# Project AirSim Agentic Trust Layer

Port of [uas-agentic-trust-layer](https://github.com/krolva/uas-agentic-trust-layer)
from BlueSky to [Project AirSim](https://github.com/iamaisim/ProjectAirSim).
An agent proposes drone actions; deterministic layers validate telemetry and
gate actions before anything reaches the simulator.

```
Agent proposal -> Action risk gate -> Executor -> Project AirSim adapter -> Simulator
                                                        |
                                    TelemetrySnapshot + validation
```

## Status

| Pass | Scope | State |
|---|---|---|
| 1-3 | Connect, fly, explore telemetry | done |
| 4 | Telemetry adapter, snapshot model, validation | done |
| 4.1 | Review fixes: robust callbacks, reconnect reset, landed state, request lock, vendored config | done |
| 5 | Deterministic command primitives (no agent) | next |

## Setup (Windows)

1. Download the Project AirSim **v1.0.1** Blocks environment and unzip it.
2. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Start the simulator and wait for the Blocks window:
   ```powershell
   <path>\Blocks-Windows-1.0.1\Blocks.exe -windowed -ResX=1280 -ResY=720
   ```

Scene and robot configs are vendored in `sim_config/` (copied from the
Project AirSim v1.0.1 samples), so no sibling checkout is required.

## Running

All commands from the repo root.

| Command | Needs sim | What it checks |
|---|---|---|
| `python -m pytest tests` | no | Adapter math, validation, malformed messages, reconnects, threading |
| `python smoke_projectairsim_connection.py` | yes | Connect / disconnect |
| `python test_drone_creation.py` | yes | Scene load, drone creation, one telemetry read |
| `python test_navigation.py` | yes | Takeoff, rotate nose, move along tracks, return, land |
| `python test_adapter_live.py [--fly]` | yes | Snapshots at 2 Hz, optionally during a flight |
| `python telemetry_explorer.py` | yes | Dumps every non-camera topic to `telemetry_dump.json` |

Every script that arms the drone calls `utils.flight_safety.safe_shutdown()`
in a `finally` block (hover, land, disarm, release API control).

## Conventions

- **Frame:** NED (x north, y east, z down). Positive `vertical_speed_mps` = climbing.
- **Units:** SI inside the system; convert to ft / kt only for display.
- **Heading vs. track:** `heading_deg` is where the nose points (yaw);
  `track_deg` is where the drone moves. A multirotor can differ by any angle.
- **Validation:** snapshots are always built, even from bad data.
  `validation_status` is `valid`, `stale` or `invalid`; only `valid` should
  ever justify a command. Warnings (collision impacts, landed-state
  inconsistencies) don't change the status.

## Layout

```
adapters/projectairsim_adapter.py   connect, cache push topics, pull kinematics/geo/landed state
models/telemetry.py                 TelemetrySnapshot contract
validation/telemetry_validator.py   deterministic data-quality checks
utils/flight_safety.py              fail-safe shutdown for armed-flight scripts
sim_config/                         scene + robot configs
tests/                              offline tests with a fake projectairsim module
```
