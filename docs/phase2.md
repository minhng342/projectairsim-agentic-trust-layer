# Phase 2: one LLM drone

Goal: the same single-drone search as [Phase 1](phase1.md), with an LLM choosing which sector
to search next. Everything else stays the same code: the climb, the action port, the sector
executor, the sensor, the memory, stop on detection or time limit, the return home, the run
log and the metrics. So Phase 1 and Phase 2 runs differ only in the planner's choices and
can be compared directly on time to find.

```bash
# live (zeus-g): LLM_MODEL in .env, or --model
python run_sar_mission.py --planner llm --mission missions/sar/mission_dev.yaml \
    --scenario missions/sar/scenario_dev.yaml --runs 1
python run_sar_mission.py --planner llm --model ollama:qwen3.5:9b --thinking off ...

# offline, no model server: the scripted stand-in model
python run_sar_mission.py --planner llm --model scripted --offline \
    --mission missions/sar/mission_dev.yaml --scenario missions/sar/scenario_dev.yaml
```

## What was built

| Piece | File | Notes |
|---|---|---|
| LLM planner | `agent/sar_llm_planner.py` (`LLMSectorPlanner`) | Pydantic AI, any model (`agent/model_config.py`); overrides one method of `FixedSectorPlanner` |
| Invariant guard | `gate/sar_guard.py` (`SarGuard`, `SarInvariantGate`) | the agent's output validator AND the port's gate slot |
| Scripted model | `agent/sar_scripted_model.py` | Pydantic AI `FunctionModel`: offline runs and tests |
| Planner loop | `agent/sector_planner.py` | the loop now asks `choose()` for each next action; Phase 1's `choose()` is "next in the fixed order" |
| Runner | `run_sar_mission.py` | `--planner llm`, `--model`, `--thinking`, `--decision-timeout` |
| Metrics | `evaluation/sar_metrics.py` | decisions, fallbacks, guard and gate refusals, decision latency |
| Tracing | `observability.py` | Logfire spans per run and per decision when `LOGFIRE_TOKEN` is set |

## One decision

```mermaid
flowchart LR
    B[LocalBelief<br/>memory]:::data --> S[situation + memory map]:::code
    O[own observation]:::code --> S
    M[mission, prior]:::data --> S
    S --> A[Pydantic AI run<br/>output SweepSector or ReturnHome]:::llm
    A --> V{guard<br/>output validator}:::code
    V -->|refused: code + message| A
    V -->|ok| P[ActionPort]:::code
    A -.->|no valid action, timeout, API error| F[fallback:<br/>Phase 1 choice]:::code
    F --> P
    P --> G{SarInvariantGate}:::code --> E[SectorExecutor]:::code

    classDef code fill:#dbeafe,stroke:#2563eb,color:#111
    classDef data fill:#fef9c3,stroke:#ca8a04,color:#111
    classDef llm fill:#fce7f3,stroke:#db2777,color:#111
```

- **Input.** Each decision is a fresh, short model run. The prompt is rebuilt from the drone's
  memory every time: own position, time used and left, a text map of the grid (`#` swept,
  `.` unsearched, `~` partly swept), the eight nearest unsearched sectors with distance and an
  estimated time, and, if the mission enables a prior, the most likely unsearched sectors.
  Memory is the belief file, not the chat history, so the prompt does not grow over a long
  search and what the model knows is exactly what `run_NN.memory.json` says.
- **Output.** A typed action, `SweepSector | ReturnHome` (`models/sar.py`). There are no tools:
  the model never calls the simulator.
- **Guard.** The output validator runs the guard. A refused choice goes back to the model with
  the rule's code and message (for example `already_swept: A1 is already swept (your memory)`),
  up to 2 retries.
- **Fallback.** No valid action after the retries, a timeout (`--decision-timeout`, 60 s) or an
  API error: the drone takes the Phase 1 choice (next unsearched sector in serpentine order)
  and the decision is logged with `fallback: true`. A failing model never leaves the drone
  without a next step.
- **Gate.** The same guard sits in the port's gate slot (`SarInvariantGate`, assessed: every
  decision is APPROVED or REJECTED, never BYPASSED). Nothing that breaks a rule reaches the
  executor even if a planner skipped the validator. The gate always lets a drone go home.

## Guard rules

| Code | Rule |
|---|---|
| `unknown_sector` | the sector id is not in the mission grid |
| `not_assigned` | the sector is outside the drone's Voronoi cells (all 64 with one drone) |
| `already_swept` | the drone's memory marks it swept |
| `not_enough_time` | flying there, sweeping it and flying home would overrun `max_mission_time_s` |
| `premature_return` | return home while sectors still fit in the time and the victim is not found |
| `wrong_altitude` | change altitude to anything but the mission altitude |
| `not_allowed` | any other action type |

Time estimate: straight transit + the sector's lanes + straight flight home, divided by the
flight speed, times 2 (live CityEnviron sweeps took about twice the straight-line time),
plus 30 s reserve.

This is a **validity** guard: it uses only the drone's own observation, memory and the
frozen mission. Trust checks on peers' claims and human approval belong to Phase 5.

## What the run log adds

Every decision is an `llm_decision` event: model, latency, the choice, the guard's refusals
(choice, code, message), whether the fallback was used and why, token usage, sim time, how
many sectors were open, and what the Phase 1 planner would have chosen
(`serpentine_choice`). Model-level retries and rate waits are logged as before
(`source: model`). The per-run summary and the batch table add `llm_decisions`,
`fallbacks`, `guard_rejections`, `gate_rejections` and decision latency (mean, max, total).

Note that **time to find includes thinking time**: the drone hovers while the model decides,
and sim time keeps running. With qwen3.5:9b at ~5-10 s per decision and thinking on, 30
decisions add several minutes; `--thinking off` is the first thing to try.

## Logfire

Set `LOGFIRE_TOKEN` in the repo's `.env` (create a write token in the Logfire project
settings). The runner then prints `Logfire tracing on` and traces:

- `sar run N`: one span per run (mission, planner, model);
- `sar decision N`: one span per decision (unsearched count, sim time), with the model
  request(s) and output validation nested under it by Pydantic AI's instrumentation.

Without a token nothing is sent and runs are unaffected. The JSONL run log stays the
experiment's source of truth; Logfire is for watching runs live and browsing prompts.

## Offline results (kinematic sim, scripted model, 2026-11-02)

Same 10 seeds (`mission_dev.yaml` at 150 m, random victim per run), speedup 100:

| Planner | Result | Time to find, mean (min-max), sim s | Fallbacks |
|---|---|---|---|
| `--planner fixed` (Phase 1) | 10 of 10 found and landed | 650 (175-1310) | |
| `--planner llm --model scripted` (greedy nearest) | 10 of 10 found and landed | 746 (210-1317) | 0 |
| `--model scripted:stubborn` (proposes a swept sector first) | guard refusals sent back, model fixes them | | 0 |
| `--model scripted:broken` (never answers) | found; every decision fell back to the Phase 1 order | | all |

The scripted model is a greedy "nearest unsearched sector" policy, not an LLM: these rows
check the pipeline (guard, retries, fallback, logs), not search quality. With no prior the
serpentine order is already a good coverage path; an LLM can only beat it with information
the fixed order does not use, such as the prior (`prior.enabled` in the mission file).

## Live checklist (zeus-g)

1. Ollama running (runbook), `LLM_MODEL=ollama:qwen3.5:9b` in `.env`.
2. `--runs 1 --thinking off` on `mission_dev.yaml` and watch the decisions print.
3. Compare with thinking on, and with qwen3.5:27b if the GPU allows.
4. For the research question, run the same seeds with `--planner fixed` and `--planner llm`
   and compare `time_to_find_s` in the two `batch_summary.json` files.

## Not in Phase 2 yet

- `VerifySector` and `ReportVictim` from the deck's action list: with one drone, detection is
  the drone's own sensor and stops the search by itself, so there is nothing to verify or
  report to. They come with peers in Phase 3.
- Human approval of consequential actions (Phase 5, with a scripted operator in experiments).
