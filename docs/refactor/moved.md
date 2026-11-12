# Where every file went

Step 3 of the restructuring moved every file without changing what it does. The current
code is in `atal/`; code from before the search and rescue work is in `legacy/`.

## Commands

The old spellings in the left column no longer work, except where the right column says
unchanged.

| Before | Now |
|---|---|
| `python run_sar_mission.py ...` | unchanged (or `python -m atal.runtime.run_single ...`) |
| `python run_sar_swarm.py ...` | unchanged (or `python -m atal.runtime.run_team ...`) |
| `python -m pytest` | unchanged; collects `tests/` and `legacy/tests/` |
| `python -m evaluation.<tool> ...` | `python -m atal.evaluation.<tool> ...` |
| `python check_llm.py ...` | `python -m legacy.llm_pilot.check_llm ...` |
| `python test_corridors_live.py ...` | `python -m legacy.corridor.corridors_live ...` |
| `python test_multi_drone_10.py` | `python -m legacy.demos.multi_drone_10` |
| `python -m archive.diagnostics.<name>` | `python -m legacy.diagnostics.<name>` |
| `python -m archive.llm_pilot.test_llm_pilot_live` | `python -m legacy.llm_pilot.llm_pilot_live` |
| `python search_and_rescue_stack.py` | unchanged |

Run logs still record `run_sar_mission.py` and `run_sar_swarm.py` as the script name, so
old and new logs read the same.

## Imports

A module name follows its path: `from models.sar import Mission` is now
`from atal.core.sar import Mission`.

## Files

| Before | Now |
|---|---|
| `adapters/kinematic_sim.py` | `atal/simulation/kinematic_sim.py` |
| `adapters/projectairsim_adapter.py` | `atal/simulation/airsim.py` |
| `agent/llm_pilot.py` | `legacy/llm_pilot/llm_pilot.py` |
| `agent/mission_pilot.py` | `legacy/corridor/mission_pilot.py` |
| `agent/model_config.py` | `atal/agents/llm/model_config.py` |
| `agent/pilot_setup.py` | `legacy/llm_pilot/pilot_setup.py` |
| `agent/providers/__init__.py` | `legacy/llm_pilot/providers/__init__.py` |
| `agent/providers/base.py` | `legacy/llm_pilot/providers/base.py` |
| `agent/providers/gemini.py` | `legacy/llm_pilot/providers/gemini.py` |
| `agent/pydantic_pilot.py` | `legacy/llm_pilot/pydantic_pilot.py` |
| `agent/rate_limit.py` | `atal/agents/llm/rate_limit.py` |
| `agent/sar_llm_planner.py` | `atal/agents/llm_chooser.py` (step 5; it was `atal/agents/llm_sector_planner.py` after step 3) |
| `agent/sar_scripted_model.py` | `atal/agents/llm/scripted_model.py` |
| `agent/scripted_agent.py` | `legacy/llm_pilot/scripted_agent.py` |
| `agent/scripted_maneuvers.py` | `legacy/corridor/scripted_maneuvers.py` |
| `agent/sector_planner.py` | `atal/agents/search_loop.py` and `atal/agents/fixed_chooser.py` (step 5) |
| `agent/swarm.py` | `atal/agents/team_coordinator.py` (step 5; it was `atal/agents/swarm.py` after step 3) |
| `archive/diagnostics/diag_heading_settle.py` | `legacy/diagnostics/diag_heading_settle.py` |
| `archive/diagnostics/diag_path_follow.py` | `legacy/diagnostics/diag_path_follow.py` |
| `archive/diagnostics/smoke_projectairsim_connection.py` | `legacy/diagnostics/smoke_projectairsim_connection.py` |
| `archive/diagnostics/telemetry_explorer.py` | `legacy/diagnostics/telemetry_explorer.py` |
| `archive/diagnostics/test_adapter_live.py` | `legacy/diagnostics/test_adapter_live.py` |
| `archive/diagnostics/test_drone_creation.py` | `legacy/diagnostics/test_drone_creation.py` |
| `archive/diagnostics/test_executor_live.py` | `legacy/diagnostics/test_executor_live.py` |
| `archive/diagnostics/test_multi_drone_live.py` | `legacy/diagnostics/test_multi_drone_live.py` |
| `archive/diagnostics/test_navigation.py` | `legacy/diagnostics/test_navigation.py` |
| `archive/llm_pilot/test_llm_pilot_live.py` | `legacy/llm_pilot/llm_pilot_live.py` |
| `channel/action_port.py` | `atal/channels/action_port.py` |
| `channel/attacks.py` | `atal/attacks/hooks.py` |
| `channel/observation_feed.py` | `atal/channels/observation_feed.py` |
| `channel/remote_id.py` | `atal/channels/radio.py` |
| `check_llm.py` | `legacy/llm_pilot/check_llm.py` |
| `conflict/detection.py` | `legacy/corridor/conflict_detection.py` |
| `evaluation/archive.py` | `atal/evaluation/archive.py` |
| `evaluation/compare.py` | `atal/evaluation/compare.py` |
| `evaluation/evidence.py` | `atal/evaluation/evidence.py` |
| `evaluation/plot_tracks.py` | `atal/evaluation/plot_tracks.py` |
| `evaluation/recorder.py` | `atal/evaluation/recorder.py` |
| `evaluation/sar_metrics.py` | `atal/evaluation/sar_metrics.py` |
| `evaluation/summarize_run.py` | `atal/evaluation/summarize_run.py` |
| `evaluation/swarm_metrics.py` | `atal/evaluation/swarm_metrics.py` |
| `executor/completion.py` | `atal/flight/completion.py` |
| `executor/mission_runner.py` | `legacy/corridor/mission_runner.py` |
| `executor/pad_zone.py` | `atal/flight/pad_zone.py` |
| `executor/path_follower.py` | `atal/flight/path_follower.py` |
| `executor/projectairsim_executor.py` | `atal/flight/executor.py` |
| `executor/sector_executor.py` | `atal/flight/sector_sweep.py` |
| `gate/action_gate.py` | `atal/trust_layer/action_gate.py` |
| `gate/peer_rules.py` | `atal/trust_layer/peer_rules.py` |
| `gate/sar_guard.py` | `atal/trust_layer/invariant_guard.py` |
| `models/action.py` | `atal/core/action.py` |
| `models/belief.py` | `atal/core/belief.py` |
| `models/mission.py` | `legacy/corridor/mission.py` |
| `models/observation.py` | `atal/core/observation.py` |
| `models/remote_id.py` | `atal/core/remote_id.py` |
| `models/sar.py` | `atal/core/sar.py` |
| `models/swarm.py` | `atal/core/peer_messages.py` |
| `models/telemetry.py` | `atal/core/telemetry.py` |
| `observability.py` | `atal/observability.py` |
| `search/allocation.py` | `atal/search/allocation.py` |
| `search/boustrophedon.py` | `atal/search/sweep_lanes.py` |
| `search/detection.py` | `atal/simulation/victim_sensor.py` |
| `search/voronoi.py` | `atal/search/voronoi.py` |
| `test_corridors_live.py` | `legacy/corridor/corridors_live.py` |
| `test_multi_drone_10.py` | `legacy/demos/multi_drone_10.py` |
| `tests/test_conflict_detection.py` | `legacy/tests/test_conflict_detection.py` |
| `tests/test_fleet.py` | `legacy/tests/test_fleet.py` |
| `tests/test_gemini_provider.py` | `legacy/tests/test_gemini_provider.py` |
| `tests/test_llm_pilot.py` | `legacy/tests/test_llm_pilot.py` |
| `tests/test_mission_pilot.py` | `legacy/tests/test_mission_pilot.py` |
| `tests/test_mission_runner.py` | `legacy/tests/test_mission_runner.py` |
| `tests/test_phase0_integration.py` | `legacy/tests/test_phase0_integration.py` |
| `tests/test_pydantic_pilot.py` | `legacy/tests/test_pydantic_pilot.py` |
| `tests/test_rate_limit.py` | `legacy/tests/test_rate_limit.py` |
| `tests/test_scripted_agent.py` | `legacy/tests/test_scripted_agent.py` |
| `trust/assessor.py` | `atal/trust_layer/assessor.py` |
| `utils/env.py` | `atal/env.py` |
| `utils/fleet.py` | `legacy/fleet.py` |
| `utils/flight_safety.py` | `atal/flight/shutdown.py` |
| `utils/separation.py` | `atal/evaluation/separation.py` |
| `validation/telemetry_validator.py` | `atal/simulation/telemetry_checks.py` |
| `run_sar_mission.py` | `atal/runtime/run_single.py` (a short wrapper stays at the root) |
| `run_sar_swarm.py` | `atal/runtime/run_team.py` (a short wrapper stays at the root) |
| `agent/providers/base.py` (`RateLimiter`), `agent/providers/__init__.py` (`rpm_for`) | `atal/agents/llm/rate_limit.py` |

## Step 4: inside `atal/runtime/`

Step 4 split the two runner files into small files. The flow of a run did not change.

| Before (in `run_single.py` / `run_team.py`) | Now |
|---|---|
| `OBSERVATION_HZ`, `PAD_WAIT_S` and the other constants, `Timing` | `rates.py` |
| choosing `KinematicSim` or `ProjectAirSimAdapter` | `simulator.py: open_simulator` |
| the per-drone setup inside `_run_one()` | `build_drone.py: build_single_drone`, `build_team_drone`, `read_home` |
| `_sample_truth` (two copies), `_watch_time` (two copies), `_watch_sim_clock`, the dropout task | `watchers.py: sample_truth`, `watch_mission_time`, `watch_sim_clock`, `inject_dropout`, `stop_background` |
| `_takeoff`, and the nested `fly()` and `clear_pads()` | `landing.py: takeoff`, `fly_team_drone`, `clear_pads`, `descend_before_landing`, `shut_down` |
| `run_failures` (two functions with one name), `run_passes` | `acceptance.py: single_run_failures`, `team_run_failures`, `single_run_passes` |
| `prepare_llm` | `llm_setup.py` |
| the `run_config` dictionary, the printed tables | functions `run_config`, `print_run_line`, `print_batch` in each run file |

One visible difference: in a one-drone run log, when a geofence breach is recorded it now
comes just before the `truth` record of the same instant instead of just after it (the one-drone
and team runs now share one truth sampler, and that is the team order).

## Step 5: inside `atal/agents/`

Step 5 replaced the planner class hierarchy (a fixed planner, an LLM planner that inherited
from it, and a team mixin that overrode eleven of its hooks) with one search loop and two
parts that plug into it. The behaviour did not change.

| Before | Now |
|---|---|
| `sector_planner.py: FixedSectorPlanner.run()` and its recovery, finish and submit code | `search_loop.py: SearchLoop` (run is now `_start`, then `_step` repeated, with `_after_sweep` for the feedback) |
| `sector_planner.py: FixedSectorPlanner.choose()` | `fixed_chooser.py: FixedChooser.choose()` |
| `sector_planner.py`: the hooks `_plan`, `_remaining`, `_still_open`, `_nothing_left`, ... | `search_loop.py: SoloRole` (`plan`, `remaining`, `still_open`, `nothing_left`, ...) |
| `sector_planner.py: PlannerResult` | `search_loop.py: SearchResult` |
| `llm_sector_planner.py: LLMSectorPlanner` | `llm_chooser.py: LLMChooser` (the prompt functions are unchanged) |
| `swarm.py: SwarmPlannerMixin`, `SwarmFixedPlanner`, `SwarmLLMPlanner` | `team_coordinator.py: TeamRole`, used with either chooser |
| `swarm.py: SwarmReceiver`, `PeerMessenger`, `SwarmCoordinator` | `team_coordinator.py` (same names) |
| `planner._observe()`, `_submit()`, `_record()`, `_sim_now()`, `_log` | `loop.observe()`, `submit()`, `record()`, `sim_now()`, `log` (the parts call them, so they are public) |
| `swarm_situation()` on the team planner | `TeamRole.situation()`, read by `LLMChooser.team_situation()` |

How a drone is put together now (`atal/runtime/build_drone.py`):

    SearchLoop(vehicle_id, mission, feed, port, belief,
               chooser=FixedChooser() or LLMChooser(model=...),
               role=TeamRole(coordinator, staging_xy) for a team drone (default SoloRole()),
               guard=..., recorder=..., sweep_progress=..., log=...)

The run logs still name the planner `FixedSectorPlanner`, `LLMSectorPlanner`,
`SwarmFixedPlanner` or `SwarmLLMPlanner` (`run_config.json`: `pipeline.planner`), from the
table `PLANNER_NAMES` in `search_loop.py`, so old and new logs read the same.

## Step 6: the remaining boundaries

The old module paths below are removed, not forwarded. Run logs are unchanged.

| Before | Now |
|---|---|
| `atal/agents/team_coordinator.py: SwarmReceiver`, `PeerMessenger`, `_Delivered` | `atal/channels/peer_link.py: SwarmReceiver`, `PeerMessenger`, `Delivered` |
| `atal/channels/radio.py: _apply_hooks`, `_hook_name` | `apply_hooks`, `hook_name` (public: the peer link uses them too) |
| `atal/evaluation/recorder.py: EvaluationRecorder`, `git_info`, `versions` | `atal/runtime/recorder.py` (written during a run) |
| `atal/evaluation/recorder.py: RunLog`, `read_run` | `atal/evaluation/run_log.py` (read after it) |
| `atal/evaluation/recorder.py: _clean`, the event fields | `atal/core/events.py: json_safe`, `HEADER_FIELDS`, `RUN_START`, `RUN_END` |
| `atal/evaluation/recorder.py: REPO_ROOT` | `atal/core/paths.py: REPO_ROOT` |
| `atal/core/sar.py`: `Point` ... `Swarm`, `Mission`, `PlanFileError`, `load_mission` | `atal/core/mission.py` (plus `load_yaml`, the loader both files use) |
| `atal/core/sar.py`: `DroneStart`, `Insider`, `Scenario`, `load_scenario` | `atal/core/scenario.py` (the ground truth; agent and trust code never import it) |
| `atal/core/sar.py`: `Sector`, `SectorGrid`, `sector_id`, `parse_sector_id`, `LETTERS` | `atal/core/sector_grid.py` (with `SECTOR_ID_PATTERN`, from `action.py`) |
| `atal/core/sar.py`: `DroneState`, `SweepSector`, `ReturnHome`, `SarAction` | `atal/core/action.py` |
| `atal/core/sar.py: _Strict` | `atal/core/strict.py: Strict` |
| `atal/trust_layer/assessor.py`: `TrustAssessor`, `NoTrustAssessor` | `atal/core/contracts.py` |
| `atal/trust_layer/action_gate.py`: `ActionGate`, `BypassGate` | `atal/core/contracts.py` |
| `atal/attacks/hooks.py: HookContext` | `atal/core/contracts.py` |
| `team_coordinator.py`: the messenger's type `PeerMessenger` | `atal/core/contracts.py: PeerSender` |
| `atal/evaluation/sar_metrics.py: _is_impact`, `_tracks` | `is_impact`, `truth_tracks` (public: other files use them) |

`tests/architecture/test_boundaries.py` checks the folder rules: what each folder may import, that
truth (`core/scenario.py`, `simulation/`) never reaches agent, trust, search, channel,
flight or airspace code, that only `flight/` and `runtime/` use `airspace/`, and that no
file imports a name starting with `_` from another file.

## Step 6b: packaging

| Before | Now |
|---|---|
| `missions/sar/*.yaml` | `atal/configs/missions/sar/*.yaml`, named `bundled:missions/sar/<file>` |
| `sim_config/*.jsonc` | `atal/configs/sim_config/*.jsonc`, named `bundled:sim_config/<file>` |
| `missions/inspection_5*.json` (Phase 0) | `legacy/corridor/missions/` |
| `--mission missions/sar/mission_dev.yaml` (relative to the checkout) | `--mission bundled:missions/sar/mission_dev.yaml`, or a path to your own file relative to where you run |
| `--scene scene_sar_three_drones.jsonc` (a name in `sim_config/`) | `--scene bundled:sim_config/scene_sar_three_drones.jsonc`, or a path to your own scene with its robot files next to it |
| default output `<checkout>/runs/` | `./runs/` in the folder you run in |
| `python run_sar_swarm.py ...`, `python -m atal.evaluation.compare ...` | still work from the checkout; installed: `atal swarm ...`, `atal compare ...` (`atal/runtime/cli.py`) |

The old checkout-relative spellings are not looked up anywhere else: a path that does
not exist is an error that lists the bundled files.

## Step 6c: reading

| Before | Now |
|---|---|
| `tests/test_*.py` (flat) | `tests/<folder>/test_*.py`, one folder per `atal/` folder, plus `tests/architecture/`; whole-run tests in `tests/runtime/` |
| `docs/phase1.md` to `phase3.md` | `docs/history/` |
| README status table (the build log) | `docs/history/build_log.md` |
| README "Who may see what", "Conventions", "Layout" | `docs/architecture.md` (with the reading order, diagram, rules and ownership) |
| README "Setup", archived scripts | `docs/setup.md` |
| README "Running" and the command tables | `docs/commands.md` |
