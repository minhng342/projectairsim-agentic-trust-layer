# Commands

Every run command, what it needs and what it checks. [setup.md](setup.md) covers
installation.


The commands below are written for the repo root; the `atal` command works from anywhere. On zeus-g prefix them with `pas` (runs them in the lab image).

## Installed use

`pip install .` (or a built wheel) installs the `atal` command, with the sample missions and scenes inside the package. From any folder:

```bash
atal swarm --offline --runs 1          # = python run_sar_swarm.py --offline --runs 1
atal mission --offline --runs 1 --mission bundled:missions/sar/mission_dev.yaml \
    --scenario bundled:missions/sar/scenario_dev.yaml
atal mission --offline --mission my/mission.yaml --scenario my/scenario.yaml   # your own files
atal compare|archive|plot|summarize ...  # the evaluation tools
```

`bundled:<path>` names a file inside `atal/configs/` (read-only). Any other path is your own file, relative to the folder you run in. A name that matches nothing is an error that lists the bundled files. A scene of your own needs its robot files next to it. Output goes to `./runs` unless `--out` says otherwise.

## What's where

| Phase | Entry point | Docs |
|---|---|---|
| 1. One drone, no LLM (SAR) | `run_sar_mission.py` | [history/phase1.md](history/phase1.md) |
| 2. One drone, LLM planner (SAR) | `run_sar_mission.py --planner llm` | [history/phase2.md](history/phase2.md) |
| 3. Three honest drones (SAR swarm) | `run_sar_swarm.py [--planner llm]` | [history/phase3.md](history/phase3.md) |
| 0. Corridor scenario (parked; Remote ID, conflict detection reused later) | `legacy/corridor/corridors_live.py` | [history/build_log.md](history/build_log.md) |
| Model check | `legacy/llm_pilot/check_llm.py` | [setup.md](setup.md) |
| Simulator bring-up checks, diagnostics, the earlier LLM pilot | `legacy/` | [setup.md](setup.md#archived-scripts) |

## Demo: Phases 1 and 2 live

Project AirSim (CityEnviron) and Ollama running. Same seed, so both search for the same victim.

```bash
# Phase 1: fixed serpentine order
python run_sar_mission.py --mission bundled:missions/sar/mission_dev.yaml --scenario bundled:missions/sar/scenario_dev.yaml --runs 1
# Phase 2: the LLM picks each next sector (invariant guard on by default)
python run_sar_mission.py --planner llm --model ollama:qwen3.5:9b --thinking off \
    --mission bundled:missions/sar/mission_dev.yaml --scenario bundled:missions/sar/scenario_dev.yaml --runs 1
```

Each run writes `runs/sar/<batch>/run_00.jsonl`, `.memory.json`, `.summary.json` and a top-down `.svg`.

## Current commands

| Command | Needs sim | What it checks |
|---|---|---|
| `python -m pytest` | no | The whole offline test suite (`tests/` and `legacy/tests/`); real model calls are blocked |
| `python run_sar_mission.py --offline [--mission bundled:missions/sar/mission_dev.yaml --scenario bundled:missions/sar/scenario_dev.yaml] [--runs N] [--full-sweep]` | no | Phase 1 search and rescue on the kinematic sim: one drone sweeps the 8 x 8 grid (Voronoi order, boustrophedon lanes) until it detects the victim; PASS = found, home, landed, no failures (`--full-sweep`: also every sector swept and 95% measured coverage) |
| `python run_sar_mission.py --offline --planner llm --model scripted[:stubborn\|broken]` | no | Phase 2 pipeline without a model server: guard refusals sent back to the model (`stubborn`), fallback to the Phase 1 order (`broken`); `--model test` uses Pydantic AI's TestModel |
| `python run_sar_mission.py [--planner llm --model SPEC --thinking off] [--mission ... --scenario ...]` | yes | The same missions in Project AirSim (scene `atal/configs/sim_config/scene_sar_one_drone.jsonc`); refuses to fly if the drone spawns > 5 m from the scenario start |
| `python run_sar_swarm.py --offline [--runs N] [--planner llm --model scripted] [--radio-range M] [--peer-clear-rule single\|corroborated\|self_only] [--allocation static] [--dropout Drone2@200] [--obstacle NAME:N0,N1,E0,E1,TOP] [--reference BATCH] [--full-sweep]` | no | Phase 3: three drones on the dev mission share the search over the bus (claims, clears, finds; Voronoi re-split at every decision); PASS = found by some drone, every drone home and landed, no failures; `batch_summary.json` has the baseline (time to find, coverage, duplicate sweeps, delivery) and, with `--reference`, the paired speedup with a 95% interval; every batch writes `manifest.json` (commit, config hashes, model, seeds, counts). Compare batches: `python -m atal.evaluation.compare REF BATCH...`; pack evidence: `python -m atal.evaluation.archive BATCH... --reference REF` |
| `python run_sar_swarm.py [--planner llm --model SPEC --thinking off] --runs 1` | yes | The same with three drones in Project AirSim (scene `atal/configs/sim_config/scene_sar_three_drones.jsonc`, layers 150 / 160 / 170 m) |
| `python -m legacy.llm_pilot.check_llm [--harness pydantic\|native] [--model SPEC]` | no | The model answers and calls tools correctly (against a fake drone); `--model test` needs no network |
| `python -m atal.evaluation.plot_tracks runs/<...>.jsonl` | no | Top-down SVG of a run: SAR grid with swept sectors, track, victim and detection; or corridors and tracks for Phase 0 runs |
| `python -m legacy.corridor.corridors_live [--mission legacy/corridor/missions/inspection_5.json] [--no-maneuvers]` | yes | Phase 0 base scenario: 5 drones fly crossing corridors (scene_inspection_5.jsonc), Remote ID between all of them, scripted mission agents; PASS = all corridors done, all landed, separation >= 2 m with coverage |
| `python -m legacy.corridor.corridors_live --mission legacy/corridor/missions/inspection_5_conflict.json --llm Drone2,Drone3` | yes | 2 LLM mission agents + 3 scripted drones that do not yield; two head-on encounters on shared layers |
| `python -m atal.evaluation.summarize_run runs/run_<id>.jsonl` | no | Per-drone observation counts and the chain observation -> proposal -> gate decision -> result -> feedback; works on incomplete runs |
