# Setup

How to install the simulator, the Python environment and the language models. For the lab
server, see the zeus-g runbook (Apptainer image, VNC display, Ollama on the second GPU).

## Windows

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

Scene and robot configs are vendored in `atal/configs/sim_config/` (copied from the
Project AirSim v1.0.1 samples), so no sibling checkout is required. They and the sample
missions in `atal/configs/missions/` ship with the package; see [commands.md](commands.md#installed-use).

## Archived scripts

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
