# Restructuring baseline

The repository is being reorganised so the team can read, navigate and explain it. This
file records the starting point, so every later step can be checked against it. The reading
guide for the code is [../walkthrough.md](../walkthrough.md).

File paths in this document are the ones at the baseline commit. Since step 3 the files have
new places: see [moved.md](moved.md).

## Starting point

| Item | Value |
|---|---|
| Repository | `minhng342/projectairsim-agentic-trust-layer` |
| Branch and commit | `main` at `5a0655638fdf3f3993613f3bdfb0fe8356c0a180` (2026-11-05) |
| Scope | Phases 0 to 3. Phases 4 and 5 are paused and are not part of this work. |
| Remote branches | `main` only, plus the restructuring working branch |
| Python | 3.12 (3.13 does not install the `projectairsim` client) |
| Key packages | `projectairsim` 1.0.2, `pydantic` 2.14.3, `pydantic-ai-slim` 2.67.0, `pytest` 9.2.0 |
| Offline tests | 557 passed, 0 failed (`python -m pytest -n 2`, about 90 s on 4 CPUs) |

Run the tests with at most two workers. With four workers on a four-CPU machine some
timing-sensitive tests fail at random.

## Rules for the restructuring

1. A structural change moves or renames code. It does not change behaviour.
2. A defect is fixed in its own change, with its own test, never inside a move.
3. Every step keeps the offline tests passing and the reference cases below the same.
4. Old commands keep working until their replacement has been checked.
5. Old run logs and evidence archives are never rewritten.

## Supported commands

The commands at the baseline commit. **Rule 4 was not kept in full in step 3.** The two run
commands, `python -m pytest` and `python search_and_rescue_stack.py` work unchanged. Every
other command in this table was replaced, and the old spelling now fails (for example
`python -m evaluation.compare` gives `ModuleNotFoundError`; the replacement is
`python -m atal.evaluation.compare`). The replacements are listed in [moved.md](moved.md).
Lab notes and scripts that use the old spellings need the new ones.

| Command | Needs the simulator | Purpose |
|---|---|---|
| `python -m pytest` | no | Offline test suite |
| `python run_sar_mission.py [--offline] [--planner llm --model SPEC]` | live unless `--offline` | One drone. Defaults: `missions/sar/mission.yaml`, `scenario.yaml`, scene `scene_sar_one_drone.jsonc`, gate `auto` |
| `python run_sar_swarm.py [--offline] [--planner llm --model SPEC]` | live unless `--offline` | Three drones. Defaults: `mission_dev.yaml`, `scenario_swarm_dev.yaml`, scene `scene_sar_three_drones.jsonc`, gate `invariants` |
| `python -m evaluation.plot_tracks RUN.jsonl` | no | Top-down plot of a run |
| `python -m evaluation.summarize_run RUN.jsonl` | no | Readable summary of a run log |
| `python -m evaluation.compare REF BATCH...` | no | Paired comparison of batches |
| `python -m evaluation.archive BATCH...` | no | Pack batches into an evidence archive |
| `python check_llm.py` | no | Check a model answers and calls tools |
| `python test_corridors_live.py` | yes | Corridor scenario (parked) |
| `python search_and_rescue_stack.py` | yes | Sam's SAR prototype |
| `python -m archive.diagnostics.*` | yes | Simulator bring-up and diagnostics |

On zeus-g the live commands are prefixed with `pas`.

## Evidence contracts

Evaluation code and past results depend on these. They are preserved, or changed only with
an explicit version note.

- **Run log:** one JSONL file per run. Each line has `kind`, `event_id`, `host_s`,
  `sim_time_ns`, and usually `vehicle_id` and `source`.
- **Event kinds read by the metrics**, including `proposal`, `gate_decision`,
  `command_result`, `feedback`, `belief_update`, `peer_message`, `peer_rule`,
  `sar_allocation`, `truth`, `victim_detected`, `planner_end`, `shutdown`.
- **Reason codes** returned to the planner (`sector_swept`, `sweep_stopped`, `path_blocked`,
  `pad_zone_timeout`, `gate_rejected` and the rest of `_EXECUTOR_FEEDBACK` in
  `channel/action_port.py`) and guard codes (top of `gate/sar_guard.py`).
- **Per-batch files:** `batch_summary.json`, `manifest.json`, `run_NN.summary.json`,
  `run_NN.<drone>.memory.json`, `run_NN.svg`.
- **Names written into logs:** `run_config.script`, `pipeline.planner`,
  `pipeline.action_gate`, `pipeline.peer_rules`. Renaming a class or script changes these,
  so comparisons across the restructuring use metrics, not these names.

## Reference cases

Eight offline cases, three seeded runs each, recorded in
[reference_baseline.json](reference_baseline.json). Rerun and compare with:

```
python tools/reference_cases.py --check docs/refactor/reference_baseline.json
```

Checked in every run of every case, whatever the reference says:

- every drone landed and disarmed, and its descent succeeded;
- the run log is complete and did not end `aborted`;
- no run error, no failed background task, no stalled simulator;
- in team runs, no peer-message processing error and no pad zone left held;
- the runner's exit status agrees with its own verdicts (0 when every run passed, 1 when a
  run failed acceptance). Anything else is treated as a crash.

Compared with the reference, per run:

- the run index, seed and victim sector, which must be identical;
- whether the victim was found;
- whether the run passed acceptance. In the obstacle case a run may fail, but only for the
  reason `N impacts`. Any other failure reason there is reported;
- the time to find within a tolerance (15 % or 30 s, whichever is larger), only where the
  sweep order is fixed: one drone, and the static team split.

The script exits with status 1 when anything is reported, and it refuses to record a
reference that is not sound by the first list. `tests/test_reference_cases.py` checks these
rules.

Why not exact numbers: the offline simulator is paced by the host clock, so the same seed
gives slightly different times on each execution. One drone repeats within about 2 %. A
cooperating team does not: the drones re-split the open sectors at every decision, so a
small timing difference changes who sweeps what. Between two executions of this same commit
the team time to find moved by up to 150 % (run 2 of the obstacle case: 313 s and 781 s).
Team times are therefore recorded but not compared. The same seed fixes the victim, not the
team's execution.

| Case | What it exercises | Found | Passed | Time to find (s), runs 0 to 2 | Time compared |
|---|---|---|---|---|---|
| `one_fixed` | One drone, fixed sector order | 3/3 | 3/3 | 726, 202, 624 | yes |
| `one_llm` | One drone, scripted stand-in for the LLM, invariant guard on | 3/3 | 3/3 | 194, 1029, 799 | yes |
| `team_fixed` | Three drones cooperating over the bus | 3/3 | 3/3 | 361, 151, 328 | no |
| `team_llm` | Three drones, scripted stand-in for the LLM | 3/3 | 3/3 | 337, 482, 199 | no |
| `team_dropout` | One drone fails 200 s into the search | 3/3 | 3/3 | 521, 155, 437 | no |
| `team_static` | Fixed pre-mission split, no claims | 3/3 | 3/3 | 364, 150, 260 | yes |
| `team_obstacle` | An obstacle in the grid: impacts, recovery, shared blocked sectors | 3/3 | 1/3 | 425, 156, 311 | no |
| `team_self_only` | Peer clears are recorded but never accepted | 3/3 | 3/3 | 758, 156, 589 | no |

In the obstacle case, whether a drone touches the obstacle varies between executions (in
five executions between zero and two of the three runs had an impact), and any impact fails
a run by design. That is the only failure allowed there. It is in the set because it
exercises obstacle recovery and shared blocked sectors, and the victim must still be found.

What this means for the restructuring: the reference cases catch a broken run path, a lost
find or a new failure. They cannot detect a small change in team search efficiency. The
557 tests are the main protection for behaviour; a team efficiency comparison needs a
20-run batch (`python run_sar_swarm.py --offline`, then `python -m atal.evaluation.compare`).
A 20-run batch estimates the variation between executions. It does not remove it.

The reference cases cover the offline path only. The live path (Project AirSim on zeus-g)
is checked once before any old entry point is removed.

## What the tests and cases must keep true

- The outcome of a run: found or not, coverage, why it ended, pass or fail, every drone home
  and landed.
- A drone acts only for itself and sees only its own observations.
- The evidence chain in the log: proposal, gate decision, command, result, linked by ids.
- Peer messages: authenticated identity, the peer-claim rules, the effect of a missed
  message, and the difference between self-verified and peer-reported beliefs.
- Refused actions, stale observations, and the fallback when the LLM fails.
- The hard cases: simulator stall, obstacle recovery, dropout, occupied pad, pad timeout,
  unconfirmed landing.
- Fixed and scripted-LLM planner behaviour, and the existing message-tampering hooks.

## Known issues at the baseline

Recorded here so they are neither silently fixed nor silently kept. Each one is handled in
its own change.

| Issue | Where | Status |
|---|---|---|
| In one execution of the obstacle reference case, a run ended with a drone making a contingency landing outside the pads (`pad_zone_timeout`) after `return home sweep_stopped`. It did not recur in the next execution. | `run_sar_swarm.py`, `executor/pad_zone.py` | Observed offline on 2026-11-12. Not investigated. May be the intended fail-safe. |
| Team runs are not repeatable for a given seed: time to find varies widely between executions because allocation depends on task timing. | `agent/swarm.py`, offline simulator pacing | A question of experimental method, separate from the restructuring: a 20-run batch estimates this variation, it does not remove it, and two batches with the same seeds are not the same executions. |
| The deck mission (60 m altitude) has never been flown live; the altitude is thought to be too low for the city scene. | `missions/sar/mission.yaml` | Open since Phase 3. |
| A live 20-run team baseline and a live one-drone run have not been done. | zeus-g | Open since Phase 3. |
| Three root scripts are named `test_*.py` but are live demos. `pytest.ini` works around it with `testpaths = tests`. | repo root | Resolved by the restructuring (they move to `legacy/`). |
| The test suite is timing-sensitive above two workers. | `tests/` | Open. |
| `path_blocked` with no obstacle in offline team runs under load (found 2026-11-12). A host stall of about 0.1 s skips 8 to 11 sim seconds at ×100. The drones' velocity commands expire, they hover, and the follower reports a blocked path, so the drone recovers and blocks a sector (15/16 swept). In a team, that recovery can also break separation (4.39 m once). About 2 in 8 full-suite runs, on the code before and after step 5. Never alone in 185 runs. | `atal/simulation/kinematic_sim.py` (clock = host time × speedup), `atal/flight/path_follower.py` (`stuck_after_s`) | Fixed in the offline simulator's clock (its own change, 2026-11-12; the runs that failed before the fix stay failed). Investigation and fix in `docs/refactor/investigations/path_blocked_without_obstacle.md`. A run where it happens is a failed run. |
| Obstacle recovery breaks the team's separation (found 2026-11-12, when the reference checker began to report every failure reason; cause traced in recorded runs). The drones are kept apart only by flying at levels `layer_spacing_m` (10 m) apart. (1) A drone that hits an obstacle climbs `recovery_climb_m` = 10 m, which is exactly its neighbour's level; two drones then flew home and waited for the pads on one level, 4.5 m apart for about 90 s (0.93 m in another run). (2) The same 10 m does not clear an obstacle taller than that: from the 150 m level the second try home hit the 165 m tower again and the drone landed where it was (the contingency landing in the first row). (3) A recovered drone descends back to its own level while flying home, crossing other levels next to drones waiting at the pad zone edge, whose waiting points are only 2 to 3 m apart horizontally (3.3 m in one run). About 4 of 42 runs of the obstacle case fail for one of these; all were the run whose victim is next to the obstacle. Not seen in runs without an obstacle, except one unexplained 1.19 m near the pads in the self-only case (once in 54 runs). | `atal/agents/search_loop.py` (`_recover`, `_return_higher`), `atal/flight/sector_sweep.py` (return home), `atal/flight/pad_zone.py` (waiting points), `layer_spacing_m` in the mission | Open. A fix needs three parts: recovery levels of a drone's own, keeping that level until the drone holds the pad zone, and waiting points spread around the zone. A trial of the first part alone (local branch `fix-recovery-levels`) is not enough. A behaviour change, so its own change with its own tests. Until then, a run where this happens is a failed run: it is recorded in `docs/refactor/reference_checks.md` with the rest of the check's output, and a rerun that passes does not clear it. |
| The package only runs from a checkout of the repository. `pip install` builds a wheel, but it contains no mission or scene files, the default `--mission` and `--scenario` paths are relative to the repository root, and the default scene and output folders are found from the source layout. Outside the checkout the run commands need explicit `--mission`, `--scenario` and `--out`. | `pyproject.toml`, `atal/runtime/`, `atal/simulation/airsim.py`, `atal/core/paths.py` | Open. To decide: bundle the sample missions and scenes with a stable lookup, or keep them outside the package and require explicit paths. Needs an installed-package test from another folder before any source-root workflow is retired. |
| Some team tests fail at random even with two workers, because they assert on which drone swept which sector. `test_full_sweep_shares_the_grid_without_duplicates` failed in 7 of 40 single executions at the baseline commit, and in 7 of 40 after step 3. `test_without_cooperation_drones_sweep_the_same_sectors` and `test_swarm_lands_one_drone_at_a_time` also failed at the baseline. | `tests/test_sar_swarm.py`, `tests/test_phase3_completion.py` | Measured 2026-11-12. Same cause as the row on team repeatability. Rerun a failed team test before treating it as a regression. |
