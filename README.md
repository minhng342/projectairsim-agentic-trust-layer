# Project AirSim Agentic Trust Layer

Drone agents searching for a missing person in [Project AirSim](https://github.com/iamaisim/ProjectAirSim),
with a trust layer between what an agent proposes and what the drone does. Each drone's
agent (a fixed search order or a language model) sees only its own observations and its
teammates' radio messages. Deterministic checks refuse unsafe actions and decide which peer
claims may change a drone's beliefs. The research question is how fast a team finds the
victim, and what a dishonest teammate does to that.

Phases 0 to 3 are built: one drone without an LLM, one LLM drone, and three cooperating
honest drones. Phases 4 (an insider attack) and 5 (defenses) are paused while the code is
made easy to read and explain.

## Start here

1. [docs/architecture.md](docs/architecture.md): the folders, the two paths an action and a
   peer message take, the reading order, and who may see what.
2. [docs/walkthrough.md](docs/walkthrough.md): one real team run followed through the code,
   with function names and log lines. It answers eight questions, from "which command starts
   a run" to "who lands the drones after an error".
3. `atal/runtime/run_team.py`: the first file to open.

## Install and run

Python 3.12 (the Project AirSim client does not install on 3.13).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                       # or: pip install -r requirements.txt
python -m pytest                       # the offline test suite, no simulator needed
atal swarm --offline --runs 1          # three drones on the offline simulator
atal mission --offline --runs 1 --mission bundled:missions/sar/mission_dev.yaml \
    --scenario bundled:missions/sar/scenario_dev.yaml   # one drone, small grid
```

From a checkout, `python run_sar_swarm.py ...` and `python run_sar_mission.py ...` are the
same commands. The sample missions and scenes ship inside the package (`atal/configs/`)
and are named `bundled:missions/sar/<file>`; any other path is your own file. Output goes to
`./runs`. A live run needs Project AirSim running and the scene in
`atal/configs/sim_config/`.

## Documents

| Document | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Folders, reading order, dependency rules, ownership, conventions, every file |
| [docs/walkthrough.md](docs/walkthrough.md) | One run end to end through the code |
| [docs/commands.md](docs/commands.md) | Every command, what it needs and what it checks |
| [docs/setup.md](docs/setup.md) | Simulator, Python environment, language models, archived scripts |
| [docs/recovery/recovery_policy.md](docs/recovery/recovery_policy.md) | The managed airspace: obstacle recovery, traffic pauses, the landing queue |
| [docs/refactor/baseline.md](docs/refactor/baseline.md) | The restructuring: reference checks and known issues |
| [docs/refactor/moved.md](docs/refactor/moved.md) | Where every old file and name went |
| [docs/history/](docs/history/build_log.md) | Phases 1 to 3 as built, and the build log |

## What is trusted

The flight executor and the airspace transition manager use simulator truth on purpose.
The executor plays a reliable flight controller, so a misled agent's wrong command is still
flown accurately. The transition manager stands in for a UTM service that keeps vertical
moves apart, and runs the same way in every experiment condition. What it contributes to
separation is not evidence for the agents' trust layer. Sam's `search_and_rescue_stack.py`
prototype stays at the root, and only its import lines are ever edited.
