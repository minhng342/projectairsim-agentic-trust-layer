# Restructuring baseline

The repository is being reorganised so the team can read, navigate and explain it. This
file records the starting point, so every later step can be checked against it. The reading
guide for the code as it stands is [../walkthrough.md](../walkthrough.md).

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

These must keep working, with the same defaults, through every step.

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

The check compares, per run:

- whether the victim was found, in every case;
- whether the run passed acceptance, in every case but the obstacle one;
- the time to find within a tolerance (15 % or 30 s, whichever is larger), only where the
  sweep order is fixed: one drone, and the static team split.

Why not exact numbers: the offline simulator is paced by the host clock, so the same seed
gives slightly different times on each execution. One drone repeats within about 2 %. A
cooperating team does not: the drones re-split the open sectors at every decision, so a
small timing difference changes who sweeps what. Between two executions of this same commit
the team time to find moved by up to 150 % (run 2 of the obstacle case: 313 s and 781 s).
Team times are therefore recorded but not compared.

| Case | What it exercises | Found | Passed | Time to find (s), runs 0 to 2 | Time compared |
|---|---|---|---|---|---|
| `one_fixed` | One drone, fixed sector order | 3/3 | 3/3 | 731, 204, 636 | yes |
| `one_llm` | One drone, scripted stand-in for the LLM, invariant guard on | 3/3 | 3/3 | 196, 1041, 805 | yes |
| `team_fixed` | Three drones cooperating over the bus | 3/3 | 3/3 | 367, 156, 307 | no |
| `team_llm` | Three drones, scripted stand-in for the LLM | 3/3 | 3/3 | 344, 495, 202 | no |
| `team_dropout` | One drone fails 200 s into the search | 3/3 | 3/3 | 523, 156, 425 | no |
| `team_static` | Fixed pre-mission split, no claims | 3/3 | 3/3 | 359, 150, 258 | yes |
| `team_obstacle` | An obstacle in the grid: impacts, recovery, shared blocked sectors | 3/3 | 2/3 | 426, 156, 313 | no |
| `team_self_only` | Peer clears are recorded but never accepted | 3/3 | 3/3 | 683, 156, 745 | no |

In the obstacle case, whether a drone touches the obstacle varies between executions (one
of three runs passed in one execution, two of three in the next), and any impact fails a
run by design. It is in the set because it exercises obstacle recovery and shared blocked
sectors, and the victim must still be found.

What this means for the restructuring: the reference cases catch a broken run path, a lost
find or a new failure. They cannot detect a small change in team search efficiency. The
557 tests are the main protection for behaviour; a team efficiency comparison needs a
20-run batch (`python run_sar_swarm.py --offline`, then `python -m evaluation.compare`).

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
| Team runs are not repeatable for a given seed: time to find varies widely between executions because allocation depends on task timing. | `agent/swarm.py`, offline simulator pacing | Affects how experiments are compared; 20-run batches hide it. Not a restructuring change. |
| The deck mission (60 m altitude) has never been flown live; the altitude is thought to be too low for the city scene. | `missions/sar/mission.yaml` | Open since Phase 3. |
| A live 20-run team baseline and a live one-drone run have not been done. | zeus-g | Open since Phase 3. |
| Three root scripts are named `test_*.py` but are live demos. `pytest.ini` works around it with `testpaths = tests`. | repo root | Resolved by the restructuring (they move to `legacy/`). |
| The test suite is timing-sensitive above two workers. | `tests/` | Open. |
