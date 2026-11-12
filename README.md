# Project AirSim Agentic Trust Layer

Port of [uas-agentic-trust-layer](https://github.com/krolva/uas-agentic-trust-layer)
from BlueSky to [Project AirSim](https://github.com/iamaisim/ProjectAirSim).
An agent proposes drone actions; deterministic layers validate telemetry and
gate actions before anything reaches the simulator.

**New here?** Read [docs/walkthrough.md](docs/walkthrough.md): how one search and rescue
run works, with the real file and function names. The repository is being reorganised to
be easier to read; the starting point and the checks are in
[docs/refactor/baseline.md](docs/refactor/baseline.md).

```
                 Simulator  <-- executor (flight control, ground truth) <--+
                     |                                                    |
          adapter (ground truth snapshot)                          action-risk gate
                     |                                              (BypassGate now)
      ObservationFeed per drone                                           |
  strip verdict -> attack hooks -> trust assessor --> Agent --> ActionPort per drone
  (sampled at a fixed rate,       (none yet)    (by observation_id)   |
   history kept per drone)                 <-- ActionFeedback (no ground truth)

  EvaluationRecorder (runs/*.jsonl): every observation, proposal, decision,
  result, correction, cancellation, shutdown, ground-truth pose and collision
```

## Who may see what

| Component | Ground truth | Its own drone's delivered observations | Raw payload | Clean (pre-attack) payload / attack details | Other drones' observations |
|---|---|---|---|---|---|
| Agent | No (feedback is ActionFeedback: status + fixed codes, no positions or measured errors) | Yes (its own deep copies) | Yes | No | No (explicit peer reports later) |
| Trust / provenance assessor | No | Yes (+ previous one) | Yes | No | No (peer reports later) |
| Action-risk gate | No | Yes, the feed's protected record (not the agent's copy), with its trust assessment | Yes | No | No (peer reports later) |
| Executor | **Yes** (preflight, settle, corrections) | No | No | No | No |
| Recorder / separation | Yes (full CommandResult, snapshots, reasons) | Yes | Yes | Yes | Yes |

The executor deliberately uses simulator ground truth: it plays the role of a
reliable flight controller. Consequence for experiments: when an agent's
observations are corrupted, its commands are still flown accurately, so an
attack shows up as the agent commanding the WRONG target, executed precisely
(see `test_attack_misleads_the_agent_and_executor_flies_the_wrong_target_accurately`).
Experiments that attack the flight controller's own sensing would need a
different executor.

How an agent refers to what it saw: proposals carry an `observation_id`. The
port resolves it through the agent's own feed to the feed's protected record and
rejects unknown ids (including other drones'), altered observation objects,
expired observations (`max_observation_age_s`, checked again after the gate),
observations received before the previous command ended, concurrent submissions
and replayed request ids. Observations are sampled at a fixed rate
(`ObservationFeed.sample`), independent of how often the agent decides; the
evaluator's 20 Hz ground-truth samples are not observations.

Placeholders are explicit, never silent: with no assessor every observation is
`not_assessed`, and with no gate every action is `bypassed` (`assessed=false`).
A run with `bypassed` decisions is an unprotected run, not one where actions
passed a risk check. Conflict decisions will later need permitted traffic
information (peer reports) on a modeled channel; the isolated missions here
don't.

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
| 5.3 | Pairwise separation + collision assessment; executor catch-all after dispatch; missions cancelled and awaited before shutdown | done |
| 5.4 | Cancellation survives repeated Ctrl+C and cancels each drone's task once (takeoff included); separation counts only fresh, in-sync samples and reports INCONCLUSIVE on poor coverage | done |
| 5.5 | Observation feed (verdict stripped, attack hooks before assessment, receiver-stamped ids), vehicle-bound action port (ownership, observation linkage, no replays, explicit gate bypass), JSONL evaluation recorder written during the run; three-drone demo runs through agents, with a sync point before descending; agents capture home on the ground before takeoff; altitude tolerance 0.5 m (Simple Flight accuracy) | done |
| 5.6a | Interface hardening for external (LLM) agents: agent-facing `ActionFeedback` (no ground truth, fixed codes; full `CommandResult` evaluation side only, executor `reason_code`), proposals by `observation_id` resolved to frozen, feed-protected records, freshness and supersession checks, one command in flight per drone, fixed-rate observation sampling with history | done (offline) |
| 5.6b | LLM pilot agent (`legacy/llm_pilot/llm_pilot.py`): tools `get_observation` / `submit_action`, final comment, bounded turns / submissions / time, model retries, provider interface (`legacy/llm_pilot/providers/`) with Gemini; dict proposals recorded with one request_id; protected copies from `resolve()` and to the assessor; supersession by feed sequence (Windows clock resolution) | offline done; live next |
| 5.6c | Executor velocity final approach (default): Simple Flight's position hold orbited the target (~1 m, 0.6-0.7 m/s, never settling) at diagonal headings 315/45; holding the heading explicitly did not help; steering in with velocity commands settled at 315/45/0 in ~9.4 s with < 0.05 m/s residual motion (`legacy/diagnostics/diag_heading_settle.py`) | done (diag); demo rerun next |
| 5.6d | Pydantic AI harness (default): any model via one setting (`LLM_MODEL`: Ollama, OpenAI-compatible, Gemini), pacing and transient-error retries below the framework, optional Logfire tracing; same prompt, tools, port and recording as the reference `LLMPilot` (`PilotCore`); schema-rejected tool calls recorded | offline done; live next |
| 0.a | Velocity path follower (`atal/flight/path_follower.py`): continuous polyline flight by pure pursuit, live maneuvers (hold = brake then stay, resume, altitude, speed, stop), fail-safe hover; live diag at 3 m/s: 2.90 m/s on straights, 0.22 m median cross-track (1.15 m out of corners), no orbit after arrival. Python 3.11 `asyncio.wait_for` could swallow a cancellation: replaced by `asyncio.timeout` in executor, feed, follower, native pilot | done (hold fix: live recheck) |
| 0.b | Remote ID broadcast channel (`atal/channels/radio.py`): honest 1 Hz broadcasters, bus, per-drone receivers keyed by CLAIMED id, sender hooks (malicious drone: lie, impersonate, Sybil, drop) and link hooks (in transit), assessor slot per stream, agent traffic view with unverified free text | done (offline) |
| 0.c | Mission layer: maneuver actions (`maintain`, `hold` [optional duration], `resume`, `change_speed`, `change_altitude` as a layer change); `MissionRunner` (climb to layer, fly the corridor with the path follower, apply maneuvers via the port: applied / not_applicable / out_of_range / not_flying / mission_finished); mission files (`missions/inspection_5.json`: 5 crossing corridors on separate layers) and `scene_inspection_5.jsonc`; `ScriptedManeuverAgent` (own observations + corridor + Remote ID traffic, maneuvers through the port); 5-drone offline integration test | offline done; live next |
| 0.d | Conflict detection (`legacy/corridor/conflict_detection.py`: linear projection, TCPA/DCPA/time to loss of separation, 5 m / 2 m minima; Kuchar & Yang, Atlas ideas, no Atlas code), `MissionPilot` LLM mission agent (deterministic watch loop wakes the model on new / persisting conflicts or to return to plan; tools own state, traffic, conflicts, mission, real-time `monitor` diff, `propose_maneuver`; what-if check of the assigned layer), conflict mission `inspection_5_conflict.json` v2 (`scene_inspection_5_wide.jsonc`, pads 6 m apart; 2 pairs fly a shared line out and back, LLM vs scripted, ~13 s warning). After live run 1 (Drone2/Drone4 collided: 5-10 s free-tier decisions vs 8 s observation limit, a too-early return to layer, a 4.5 s-warning crossing): one model call per decision, safe-layer advice, stricter back-to-plan, contact episodes in the separation report. After live run 2 (Drone2 and Drone3 both escaped to 17 m within seconds: each saw only where the other was): Remote ID intent field `target_altitude_m` (modeled on ADS-B selected altitude; a claim like any field), and planning treats a peer as occupying its current altitude and its announced target. After live run 3 (Drone2 climbed 10 to 17, a clear layer, through Drone3 crossing above at 14): layer changes are also checked along the climb or descent (at the nominal and half vertical rate); the adapter keeps the first 2000 collision records so a long contact no longer hides when it started. After live run 4 (no safe layer at the first wake-up, the model slowed down, the 8 s recheck came after the head-on): new wake reason `safe_layer_available` when one opens, recheck interval shrinks with time to loss of separation, own-ship projected to level off at its target altitude, prompt says speed/hold do not resolve a head-on and to decide from the wake-up message. After live run 5 (model latency 5-10 s, three information calls before acting, a return to layer decided from a 10 s old picture): one tool per maneuver with required parameters, get_traffic/get_conflicts/get_mission removed (the wake-up message carries them), layer changes pre-checked on the current situation and refused with `would_conflict` if they create a new or sooner conflict | offline done; live rerun next |
| 0.e | Top-down track plot (`atal/evaluation/plot_tracks.py`): SVG of planned corridors and flown ground-truth tracks for any run log, no plotting library. Mission-track CD&R metrics dropped: the corridor scenario is parked after the search-and-rescue pivot (roadmap rev 5) | done |
| 0 live | Clean recorded 5-drone LLM run (0.d live rerun), hold-fix recheck (0.a) | parked with the corridor scenario; rerun on zeus-g only if the corridor demo is still wanted |
| **SAR 1** | **Phase 1, one drone, no LLM** (docs/phase1.md): `missions/sar/*.yaml` validated by Pydantic (`atal/core/sar.py`: Mission, Scenario, SectorGrid, DroneState, SweepSector, ReturnHome), 8 x 8 sector grid A1..H8, Voronoi coverage (`atal/search/voronoi.py`), boustrophedon lanes from footprint and overlap (`atal/search/sweep_lanes.py`), geometric victim sensor (`atal/simulation/victim_sensor.py`), memory of swept sectors and detections with sources (`atal/core/belief.py`), `SectorExecutor` on the path follower, `FixedSectorPlanner` submitting typed actions through the port, `run_sar_mission.py` (N seeded runs, run log + memory + metrics + plot per run, batch verdict), `atal/simulation/kinematic_sim.py` for offline runs. Offline: dev mission 10 of 10 runs found the victim; the deck mission's full sweep needs ~80 min, more than its 3600 s limit | offline done; live next (spawn point / area in CityEnviron) |
| **SAR 2** | **Phase 2, one LLM drone** (docs/phase2.md): `LLMSectorPlanner` (Pydantic AI, any model) chooses each next sector from its memory; output `SweepSector \| ReturnHome`; invariant guard (`atal/trust_layer/invariant_guard.py`) as output validator (refusals go back to the model) and in the port's gate slot; Phase 1 choice as safe fallback; `--planner llm`, scripted model for offline runs; Logfire spans per run and decision | offline done; live next |
| **SAR 3** | **Phase 3, three honest drones** (docs/phase3.md): typed peer messages `SectorClaimed` / `SectorCleared` / `VictimDetected` (`atal/core/peer_messages.py`) on the Remote ID bus with a radio-range filter from true positions; invalid messages dropped before anything sees them; belief records who said what (`PeerClaim`); peer-claim rules (`atal/trust_layer/peer_rules.py`: one honest peer's clear accepted, a peer's find ends the search); Voronoi re-allocation over open sectors, live peers and claims at every decision, with work stealing (`atal/search/allocation.py`); each drone on its own layer; `run_sar_swarm.py` (fixed or Pydantic AI planners) and `atal/evaluation/swarm_metrics.py`. Offline baseline over 20 seeded runs in docs/phase3.md | offline done; live next (spawn points in CityEnviron) |
| next | Phase 4 (insider), 5 (defenses) | planned |

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

LLM pilots run on Pydantic AI, so any model is one setting. `pip install -r
requirements.txt`, then put these in the repo's `.env` (gitignored; never commit a key):
```
# preference order; the first model that answers the preflight ping is used
LLM_MODEL=ollama:qwen3.5:9b,google:gemini-3.5-flash-lite
OLLAMA_BASE_URL=http://localhost:11434/v1     # or the GPU box, e.g. http://192.168.1.50:11434/v1
GEMINI_API_KEY=...                            # only for google:* models
# LLM_RPM=60                                  # request pacing; default: Gemini free tier, none for local
# LOGFIRE_TOKEN=...                           # turns on Logfire tracing of every model and tool call
```
`python -m legacy.llm_pilot.check_llm --list-models` lists what the server behind `LLM_MODEL` offers.
Requests are paced client-side (one limiter per model, shared by every pilot) and
transient errors (overload, rate limit, 5xx, connection) are retried below the agent
framework (`atal/agents/llm/model_config.py`). Each action costs one model call: the action's
result carries the next observation.

Local models (Ollama): Ollama's default context window is small, and a pilot's
conversation reaches 10-20k tokens. Raise it on the server (e.g. `OLLAMA_CONTEXT_LENGTH=32768`
before `ollama serve`, or `num_ctx` in a Modelfile); otherwise the model silently loses
the start of its mission.

Scene and robot configs are vendored in `sim_config/` (copied from the
Project AirSim v1.0.1 samples), so no sibling checkout is required.

## Running

All commands from the repo root. On zeus-g prefix them with `pas` (runs them in the lab image).

### What's where

| Phase | Entry point | Docs |
|---|---|---|
| 1. One drone, no LLM (SAR) | `run_sar_mission.py` | [docs/phase1.md](docs/phase1.md) |
| 2. One drone, LLM planner (SAR) | `run_sar_mission.py --planner llm` | [docs/phase2.md](docs/phase2.md) |
| 3. Three honest drones (SAR swarm) | `run_sar_swarm.py [--planner llm]` | [docs/phase3.md](docs/phase3.md) |
| 0. Corridor scenario (parked; Remote ID, conflict detection reused later) | `legacy/corridor/corridors_live.py` | Status table above |
| Model check | `legacy/llm_pilot/check_llm.py` | below |
| Simulator bring-up checks, diagnostics, the earlier LLM pilot | `legacy/` | below |

### Demo: Phases 1 and 2 live

Project AirSim (CityEnviron) and Ollama running. Same seed, so both search for the same victim.

```bash
# Phase 1: fixed serpentine order
python run_sar_mission.py --mission missions/sar/mission_dev.yaml --scenario missions/sar/scenario_dev.yaml --runs 1
# Phase 2: the LLM picks each next sector (invariant guard on by default)
python run_sar_mission.py --planner llm --model ollama:qwen3.5:9b --thinking off \
    --mission missions/sar/mission_dev.yaml --scenario missions/sar/scenario_dev.yaml --runs 1
```

Each run writes `runs/sar/<batch>/run_00.jsonl`, `.memory.json`, `.summary.json` and a top-down `.svg`.

### Current commands

| Command | Needs sim | What it checks |
|---|---|---|
| `python -m pytest` | no | The whole offline test suite (`tests/` and `legacy/tests/`); real model calls are blocked |
| `python run_sar_mission.py --offline [--mission missions/sar/mission_dev.yaml --scenario missions/sar/scenario_dev.yaml] [--runs N] [--full-sweep]` | no | Phase 1 search and rescue on the kinematic sim: one drone sweeps the 8 x 8 grid (Voronoi order, boustrophedon lanes) until it detects the victim; PASS = found, home, landed, no failures (`--full-sweep`: also every sector swept and 95% measured coverage) |
| `python run_sar_mission.py --offline --planner llm --model scripted[:stubborn\|broken]` | no | Phase 2 pipeline without a model server: guard refusals sent back to the model (`stubborn`), fallback to the Phase 1 order (`broken`); `--model test` uses Pydantic AI's TestModel |
| `python run_sar_mission.py [--planner llm --model SPEC --thinking off] [--mission ... --scenario ...]` | yes | The same missions in Project AirSim (scene `sim_config/scene_sar_one_drone.jsonc`); refuses to fly if the drone spawns > 5 m from the scenario start |
| `python run_sar_swarm.py --offline [--runs N] [--planner llm --model scripted] [--radio-range M] [--peer-clear-rule single\|corroborated\|self_only] [--allocation static] [--dropout Drone2@200] [--obstacle NAME:N0,N1,E0,E1,TOP] [--reference BATCH] [--full-sweep]` | no | Phase 3: three drones on the dev mission share the search over the bus (claims, clears, finds; Voronoi re-split at every decision); PASS = found by some drone, every drone home and landed, no failures; `batch_summary.json` has the baseline (time to find, coverage, duplicate sweeps, delivery) and, with `--reference`, the paired speedup with a 95% interval; every batch writes `manifest.json` (commit, config hashes, model, seeds, counts). Compare batches: `python -m atal.evaluation.compare REF BATCH...`; pack evidence: `python -m atal.evaluation.archive BATCH... --reference REF` |
| `python run_sar_swarm.py [--planner llm --model SPEC --thinking off] --runs 1` | yes | The same with three drones in Project AirSim (scene `sim_config/scene_sar_three_drones.jsonc`, layers 150 / 160 / 170 m) |
| `python -m legacy.llm_pilot.check_llm [--harness pydantic\|native] [--model SPEC]` | no | The model answers and calls tools correctly (against a fake drone); `--model test` needs no network |
| `python -m atal.evaluation.plot_tracks runs/<...>.jsonl` | no | Top-down SVG of a run: SAR grid with swept sectors, track, victim and detection; or corridors and tracks for Phase 0 runs |
| `python -m legacy.corridor.corridors_live [--mission missions/inspection_5.json] [--no-maneuvers]` | yes | Phase 0 base scenario: 5 drones fly crossing corridors (scene_inspection_5.jsonc), Remote ID between all of them, scripted mission agents; PASS = all corridors done, all landed, separation >= 2 m with coverage |
| `python -m legacy.corridor.corridors_live --mission missions/inspection_5_conflict.json --llm Drone2,Drone3` | yes | 2 LLM mission agents + 3 scripted drones that do not yield; two head-on encounters on shared layers |
| `python -m atal.evaluation.summarize_run runs/run_<id>.jsonl` | no | Per-drone observation counts and the chain observation -> proposal -> gate decision -> result -> feedback; works on incomplete runs |

### Archived scripts

Kept because they still work and help when something breaks, but no longer on the main path. Run them as modules from the repo root.

| Command | Needs sim | What it checks |
|---|---|---|
| `python -m legacy.diagnostics.smoke_projectairsim_connection` | yes | Connect / disconnect |
| `python -m legacy.diagnostics.test_drone_creation` | yes | Scene load, drone creation, one telemetry read |
| `python -m legacy.diagnostics.test_navigation` | yes | Takeoff, rotate nose, move along tracks, return, land |
| `python -m legacy.diagnostics.test_adapter_live [--fly]` | yes | Snapshots at 2 Hz, optionally during a flight |
| `python -m legacy.diagnostics.test_executor_live` | yes | Executor: takeoff (setup), rotate +90 deg, climb 2 m, safe shutdown |
| `python -m legacy.diagnostics.test_multi_drone_live` | yes | Three drones (scene_three_drones.jsonc), scripted agents through their own feeds and ports; PASS also requires >= 2 m separation between every pair |
| `python -m legacy.diagnostics.telemetry_explorer` | yes | Dumps every non-camera topic to `telemetry_dump.json` |
| `python -m legacy.diagnostics.diag_path_follow [--speed 3] [--no-maneuvers]` | yes | 50 m rectangle with the velocity path follower; cruise speed, cross-track, stopping distance, drift |
| `python -m legacy.diagnostics.diag_heading_settle [--headings 315,45,0]` | yes | Does Simple Flight settle after move_to_position at every heading |
| `python -m legacy.llm_pilot.llm_pilot_live [--harness pydantic\|native] [--model SPEC]` | yes | The pre-SAR LLM pilot: one drone flown by an LLM through its feed and port |

Every script that arms the drone calls `atal.flight.shutdown.safe_shutdown()`
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

The current code is one package, `atal/`. Start reading at `atal/runtime/`; the reading guide
is [docs/walkthrough.md](docs/walkthrough.md). Where each old file went:
[docs/refactor/moved.md](docs/refactor/moved.md).

```
run_sar_mission.py            one-drone run (wrapper around atal/runtime/run_single.py)
run_sar_swarm.py              three-drone run (wrapper around atal/runtime/run_team.py)

atal/
  runtime/                    building and running a mission
    run_single.py             one drone: a batch of runs, and the six steps of one run
    run_team.py               three drones on one radio bus, same six steps
    build_drone.py            every part of one drone, wired together
    watchers.py               background tasks: truth record, time limit, stalled simulator, dropout
    landing.py                takeoff, pad turn-taking, descent, shutdown
    acceptance.py             when a run passes, and why it does not
    rates.py                  rates and time limits; scaling for the fast offline simulator
    simulator.py              offline simulator or Project AirSim
    llm_setup.py              choose and ping the model before anything flies
    context.py                what the parts of one run share
  agents/                     the deciding part of a drone
    sector_planner.py         FixedSectorPlanner: fixed sector order, no LLM
    llm_sector_planner.py     LLMSectorPlanner: an LLM picks each next sector
    swarm.py                  team member: claims, clears, finds, re-split of open sectors
    llm/model_config.py       model from LLM_MODEL (any backend), pacing, retries, preflight ping
    llm/rate_limit.py         request pacing for a per-minute quota
    llm/scripted_model.py     scripted stand-in for the LLM (offline runs and tests)
  trust_layer/                what may be believed and what may be done
    assessor.py               trust assessor interface + NoTrustAssessor placeholder
    action_gate.py            action gate interface + BypassGate placeholder
    invariant_guard.py        SarGuard: the checks every search action must pass
    peer_rules.py             what a peer's "cleared" or "found" may change
  channels/                   the only ways in and out of a drone agent
    observation_feed.py       observations in
    action_port.py            actions out: validate, gate, execute, feedback
    radio.py                  Remote ID bus, broadcasters, receivers, peer messages
  flight/                     turning an accepted action into flight
    executor.py               command lifecycle: preflight, dispatch, task wait, settle, result
    completion.py             settle predicates, dwell tracker, deadline
    path_follower.py          velocity path follower for polylines
    sector_sweep.py           SectorExecutor: sweep_sector / return_home
    pad_zone.py               one drone at a time over the landing pads
    shutdown.py               fail-safe shutdown: hover, land, disarm only when on the ground
  search/                     search geometry, no drone state
    voronoi.py                Voronoi partition of the sector grid + coverage order
    allocation.py             split of open sectors between team members
    sweep_lanes.py            lanes for one sector from footprint and overlap
  simulation/                 the world the drones fly in
    airsim.py                 Project AirSim adapter: connect, topics, telemetry snapshots
    kinematic_sim.py          offline point-mass stand-in (faster than real time)
    telemetry_checks.py       deterministic data-quality checks
    victim_sensor.py          footprint detection (the only reader of the victim's true position)
  attacks/hooks.py            in-transit tampering hooks (offset, replay, drop)
  core/                       shared data types
    telemetry.py              TelemetrySnapshot
    observation.py            Observation, TrustAssessment, GateDecision
    action.py                 ProposedAction and CommandResult
    sar.py                    Mission / Scenario (YAML), SectorGrid, SweepSector, ReturnHome
    belief.py                 LocalBelief: the drone's memory, with the source of each fact
    remote_id.py              RemoteIdReport
    peer_messages.py          SectorClaimed / SectorCleared / VictimDetected
  evaluation/                 run evidence: written during a run, read after it
    recorder.py               JSONL run log; read_run()
    sar_metrics.py            time to find, coverage, sectors swept, geofence
    swarm_metrics.py          team metrics: duplicates, message delivery, per-drone work
    compare.py                paired comparison of batches
    evidence.py, archive.py   manifest per batch; evidence archive
    plot_tracks.py            top-down SVG of a run
    summarize_run.py          readable run summary
    separation.py             pairwise separation and collision assessment
  observability.py            optional Logfire tracing (LOGFIRE_TOKEN)
  env.py                      load the repo's .env without printing it

legacy/                       code from before the search and rescue work; atal/ never imports it
  llm_pilot/                  the first LLM pilot, its Gemini provider, check_llm
  corridor/                   the corridor scenario (Phase 0)
  diagnostics/, demos/        simulator bring-up scripts, ten-drone demo
  tests/                      tests of the code above

search_and_rescue_stack.py    Sam's SAR prototype
missions/                     mission and scenario files (missions/sar/*.yaml)
sim_config/                   scene and robot configs
tests/                        offline tests with a fake projectairsim module
tools/reference_cases.py      offline reference runs used to check the restructuring
docs/                         walkthrough, phase notes, restructuring baseline
```
