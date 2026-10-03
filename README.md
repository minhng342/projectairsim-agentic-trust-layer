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
| 4.2 | Client pin 1.0.2, operational ground state, sim-progress staleness, action/result contract, completion predicates | done |
| 4.3 | Ground-state hardening: raw LANDED must be still, support-surface normals only, latch release on movement, reset on clock reset; strict action fields | done |
| 5.1 | Executor: `rotate_to_heading`, `change_altitude` (preflight, bounded task + settle waits, hover fallback, per-vehicle lock, cancellation) | done |
| 5.2 | `move_to_position` with bounded corrections (1.0 m tolerance: Simple Flight stops within ~0.5-0.75 m by design); three-drone scene and concurrent multi-drone flight | done |
| 5.3+ | `move_along_track`, `takeoff`, `land`; observation channel | next |

## Setup (Windows)

1. Download the Project AirSim **v1.0.1** Blocks environment and unzip it. Its release notes pair it with the Python client `projectairsim==1.0.2`.
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
| `python -m pytest tests` | no | Adapter, validation, ground detector, action contract, completion predicates, fail-safe |
| `python smoke_projectairsim_connection.py` | yes | Connect / disconnect |
| `python test_drone_creation.py` | yes | Scene load, drone creation, one telemetry read |
| `python test_navigation.py` | yes | Takeoff, rotate nose, move along tracks, return, land |
| `python test_adapter_live.py [--fly]` | yes | Snapshots at 2 Hz, optionally during a flight |
| `python test_executor_live.py` | yes | Executor: takeoff (setup), rotate +90 deg, climb 2 m, safe shutdown |
| `python test_multi_drone_live.py` | yes | Three drones (scene_three_drones.jsonc): climb to separate layers, fan out, turn, return, land, all concurrently |
| `python telemetry_explorer.py` | yes | Dumps every non-camera topic to `telemetry_dump.json` |

Every script that arms the drone calls `utils.flight_safety.safe_shutdown()`
in a `finally` block: hover, land, and disarm only once the drone is confirmed
on the ground. If landing fails, times out or can't be confirmed, it does NOT
disarm (that would drop the drone); it reports UNRESOLVED and leaves the drone
armed and hovering. Every step is time-bounded end to end.

## Conventions

- **Frame:** NED (x north, y east, z down). Positive `vertical_speed_mps` = climbing.
- **Units:** SI inside the system; convert to ft / kt only for display.
- **Heading vs. track:** `heading_deg` is where the nose points (yaw);
  `track_deg` is where the drone moves. A multirotor can differ by any angle.
- **Ground state:** `landed_state` is Project AirSim's raw value and can stay
  `flying` for ~12 s after touchdown. `ground_state` is the operational
  decision. It requires the drone to be still, plus either raw `landed` or a
  slow contact with an upward-facing surface (collision normal z <= -0.7, so
  walls don't count) followed by >= 1 s of stillness near that contact.
  Decisions use `ground_state`, and only from a `valid` snapshot.
- **Freshness:** `telemetry_age_ms` is time since the last pose message;
  `sim_progress_age_ms` is time since the pose timestamp last advanced. Both
  must be under the limit for `valid`, so a frozen or replayed stream goes stale.
- **Validation:** snapshots are always built, even from bad data.
  `validation_status` is `valid`, `stale` or `invalid`; only `valid` should
  ever justify a command. Warnings (collision impacts, landed-state
  inconsistencies) don't change the status.

## Layout

```
adapters/projectairsim_adapter.py   connect, cache push topics, pull kinematics/geo/landed state
models/telemetry.py                 TelemetrySnapshot contract
models/action.py                    ProposedAction (per-type parameters) and CommandResult
executor/completion.py              pure settle predicates, dwell tracker, deadline
executor/projectairsim_executor.py  command lifecycle: preflight -> dispatch -> task wait -> settle -> result
validation/telemetry_validator.py   deterministic data-quality checks
utils/flight_safety.py              fail-safe shutdown for armed-flight scripts
sim_config/                         scene + robot configs
tests/                              offline tests with a fake projectairsim module
```
