# Walkthrough: how a search and rescue run works

This is the reading guide for the code **as it is today**: the files are in their new
folders (restructuring step 3), the run is split into small files in `atal/runtime/`
(step 4), and every drone's planner is one search loop with a chooser and a role plugged in
(step 5). The guard, channels and flight code are unchanged from commit `5a06556`. "The
planner" below means a drone's `SearchLoop` with its parts. It answers eight questions with
the real file and function names, then follows
one real decision through a recorded run. It is updated with every restructuring step, so it
always describes the code in front of you. The folders are listed in the
[README](../README.md#layout).

Conventions used everywhere in the code:

- **Frame:** NED. `x` / `north_m` is north, `y` / `east_m` is east, altitude is positive up
  in the mission files and observations.
- **Units:** metres, seconds, metres per second.
- **Two clocks.** *Sim time* is the simulator's clock (`sim_time_ns`, `sim_time_s`); mission
  time limits, claim expiry and time to find are all in sim time. *Host time* is the
  computer's clock (`host_s`); it is used for observation age and for timeouts that protect
  against a frozen simulator. Offline, sim time runs 50 times faster than host time
  (`--speedup`).

## The two paths to keep in mind

**An action.** The planner proposes an action. The action port checks it and asks the gate.
An approved action reaches the executor, which flies the drone and returns a result. The
planner updates its memory from that result.

**A peer message.** A message arrives on the bus, is validated, and is judged by the
peer-claim rules. The drone's memory is updated, marked as peer-reported. That memory
informs the next choice.

## 1. Which command starts a run, and which function owns its lifetime?

```
python run_sar_swarm.py --offline            # three drones, offline simulator
python run_sar_mission.py --offline          # one drone
```

For the team run, in `atal/runtime/run_team.py`:

| Function | Job |
|---|---|
| `parse_args()` | Reads the command-line flags. |
| `main()` | Loads the mission and scenario YAML (`atal/core/mission.py: load_mission`, `atal/core/scenario.py: load_scenario`), loops over the seeded runs, writes `batch_summary.json` and `manifest.json`. |
| `run_one()` / `_run_one()` | **Owns one run from start to finish.** Everything below happens inside it. |

`_run_one()` is six numbered steps. Each step calls one small file in `atal/runtime/`:

| Step | What happens | Where |
|---|---|---|
| 1 | Open the simulator: `KinematicSim` (offline) or `ProjectAirSimAdapter` (live). Create the recorder and the flight executor. | `simulator.py: open_simulator` |
| 2 | Build what the team shares: the radio bus, the pad zone, the static split if that condition is on. | `run_team.py` |
| 3 | Build each drone: observation feed, memory, sector executor, guard, gate, action port, radio receiver, messenger, coordinator, planner, victim sensor, Remote ID broadcaster. | `build_drone.py: build_team_drone` |
| 4 | Start the background tasks: observation sampling, sensor, Remote ID broadcast, victim repeat, time-limit watch, truth record, and (live only) the sim-stall watch. | `run_team.py: start_background`, `watchers.py` |
| 5 | Fly every drone at once. Then always stop the background tasks. | `landing.py: fly_team_drone`, `watchers.py: stop_background` |
| 6 | Close the run log, compute the metrics from it, save them. | `run_team.py` |

Whether a finished run passed is decided by `team_run_failures()` in
`atal/runtime/acceptance.py`, which lists the reasons if it did not.

The one-drone run `atal/runtime/run_single.py` has the same six steps with one drone and no
bus (`build_single_drone`, `single_run_failures`). The rates and time limits used by both
are in `atal/runtime/rates.py`.

## 2. What does one drone receive, and what is withheld?

A drone's planner receives exactly three things:

- **Its own observations**, through its `ObservationFeed` (`atal/channels/observation_feed.py`).
  `_take()` asks the simulator adapter for a telemetry snapshot, strips the simulator's
  validation verdict, runs any tampering hooks, and stamps the observation with an id.
- **Feedback on its own actions**, as `ActionFeedback` (`atal/core/action.py`): a status and a
  code from a fixed list such as `sector_swept` or `gate_rejected`. No positions, no
  measured errors.
- **Peer messages and Remote ID position reports** that reached it on the bus.

Withheld from the planner:

- The victim's true position. Only `VictimSensor` (`atal/simulation/victim_sensor.py`) knows it, and it
  only reports a `Detection` once the drone's sensor footprint actually covers the victim.
- Other drones' true positions. A drone only knows what peers broadcast.
- The executor's full `CommandResult`, which holds ground-truth measurements. It goes to
  the recorder only.
- Whether a message was tampered with. That is recorded for evaluation, never delivered.

## 3. Where is the drone's memory, and who can change it?

The memory is one `LocalBelief` object per drone (`atal/core/belief.py`), saved to
`run_NN.<drone>.memory.json` on every change. It holds a status per sector, the peer claims
heard, and the drone's own detections.

Only three callers change it:

| Caller | Method | When |
|---|---|---|
| The search loop (`atal/agents/search_loop.py: SearchLoop._step`, `_after_sweep`) | `mark_started`, `mark_swept`, `mark_partial` | Before and after its own sweeps. These are marked `verified_by_self`. |
| The coordinator (`atal/agents/team_coordinator.py: SwarmCoordinator.on_message`) | `add_peer_claim`, `mark_peer_cleared` | When a peer message arrives. A cleared sector gets status `peer_cleared`, with the sender as source and `verified_by_self` false. |
| The `detected()` callback set up in `atal/runtime/build_drone.py` | `add_detection` | When this drone's own sensor sees the victim. |

The difference between "I swept it" and "a peer said it is clear" is kept in the memory and
shown to the LLM planner.

## 4. What does the fixed chooser or the LLM return?

Both return one of two typed actions defined in `atal/core/action.py`:

- `SweepSector(sector_id, reason)`
- `ReturnHome(reason)`

The choice is made by the drone's chooser, in a method called `choose(remaining)`:

- **Fixed:** `FixedChooser.choose()` in `atal/agents/fixed_chooser.py` returns the next
  sector in its coverage order that the guard allows.
- **LLM:** `LLMChooser.choose()` in `atal/agents/llm_chooser.py` builds a prompt from
  the drone's memory, runs a Pydantic AI agent whose output type is
  `[SweepSector, ReturnHome]`, and falls back to the fixed choice on a timeout or a model
  error.

The loop that calls `choose()` is `SearchLoop` in `atal/agents/search_loop.py`. Every drone
runs the same loop: `_start()` (plan, climb), then `_step()` repeated (stop?, what is left,
choose, submit, `_after_sweep()`), then `_finish()` (fly home). Two parts plug into it:

- the **chooser** (fixed or LLM): which of the remaining sectors comes next;
- the **role**: which sectors are this drone's and who hears about it. `SoloRole` (in
  `search_loop.py`) is the whole grid with no radio. `TeamRole`
  (`atal/agents/team_coordinator.py`) asks the coordinator for a fresh share of the open
  sectors at every step, claims a sector before sweeping it and clears it after.

Any chooser works with any role, so the four planners of the logs (`FixedSectorPlanner`,
`LLMSectorPlanner`, `SwarmFixedPlanner`, `SwarmLLMPlanner`) are four combinations of the
same three files.

## 5. Where can code refuse a proposal?

Three places, in the order a proposal meets them:

1. **The LLM's output validator** (`guard_check` inside `LLMChooser._build_agent`).
   It calls `SarGuard.check()`. A refused choice goes back to the model with the rule's
   code so it can choose again.
2. **The action port** (`atal/channels/action_port.py: ActionPort.submit`). It refuses a proposal
   that is malformed, is for another vehicle, is based on an unknown, altered, expired or
   superseded observation, repeats a request id, or arrives while another command is
   running.
3. **The gate** (`ActionPort._gate_and_execute` calls `gate.decide()`). In SAR runs the gate
   is `SarInvariantGate` (`atal/trust_layer/invariant_guard.py`), which calls the same `SarGuard.check()`. A
   gate that raises an error blocks the action.

`SarGuard.check()` holds the rules: the sector exists, is assigned to this drone, is not
already done, is not blocked by an obstacle, fits in the mission time left, and the search
is not already over. The full list with codes is at the top of `atal/trust_layer/invariant_guard.py`.

## 6. Who sends simulator commands, and how are completion and failure decided?

Every action a planner proposes reaches the simulator through the executors, and nothing
else flies the drone during the search. Two things outside the planner's reach also send
simulator commands directly: the run's setup (arm and take off, `atal/runtime/landing.py:
takeoff`) and `safe_shutdown()` (`atal/flight/shutdown.py`: hover, land, disarm).

- `SectorExecutor.execute()` (`atal/flight/sector_sweep.py`) handles the SAR actions. For a
  sweep it computes the lanes (`atal/search/sweep_lanes.py: sweep_path`) and flies them with
  a `PathFollower` (`atal/flight/path_follower.py`).
- `ProjectAirSimExecutor` (`atal/flight/executor.py`) handles the basic commands:
  rotate, change altitude, move to a position. Its `_run()` method is the command
  lifecycle: preflight check, dispatch, wait for the simulator task, wait until the drone
  has settled, return a result.

Completion is decided from ground-truth telemetry by the pure functions in
`atal/flight/completion.py` (for example `position_settled`, `altitude_settled`). Failure is a
timeout, a telemetry problem, or no progress along the path for 10 seconds
(`path_blocked`). On failure the executor tells the drone to hover.

Every executor returns a `CommandResult` with a `reason_code`. The action port translates
that into the limited `ActionFeedback` the planner sees
(`_EXECUTOR_FEEDBACK` in `atal/channels/action_port.py`).

## 7. What happens when a peer claims a sector is clear?

1. The sender's `PeerMessenger.send()` (`atal/channels/peer_link.py`) puts a `SectorCleared` message
   (`atal/core/peer_messages.py`) on the bus.
2. `RemoteIdBus.transmit()` (`atal/channels/radio.py`) delivers it to every other drone in
   radio range.
3. The receiver's `SwarmReceiver.receive_linked()` drops it if the claimed sender is not the
   real transmitter (when identity is authenticated). `receive()` then validates it as a
   typed message; an invalid one is dropped and recorded.
4. `SwarmCoordinator.on_message()` asks `PeerClaimRules.on_cleared()`
   (`atal/trust_layer/peer_rules.py`). With the default rule `single`, one peer's word is accepted.
   With `corroborated`, it needs a second peer. With `self_only`, it is never accepted.
5. If accepted, `LocalBelief.mark_peer_cleared()` sets the sector to `peer_cleared`. The
   verdict is recorded as a `peer_rule` event with the message ids behind it.
6. The next time the planner picks a sector, that one is no longer open.

A peer's `VictimDetected` follows the same route through `PeerClaimRules.on_victim()`. If
accepted, it ends this drone's search.

## 8. Who stops tasks, lands the drones and closes the record after an error?

- **Each drone** flies inside `fly_team_drone()` (`atal/runtime/landing.py`). Its `finally`
  block always calls `shut_down()`, which calls `safe_shutdown()`
  (`atal/flight/shutdown.py`): hover, land, and disarm only once the drone is confirmed on
  the ground. If landing cannot be confirmed, it does not disarm and reports the drone as
  unresolved.
- **The pads.** Before any descent, `clear_pads()` (`atal/runtime/landing.py`) makes sure
  the drone holds the pad zone or is outside it (`atal/flight/pad_zone.py`). The zone is
  released only after a confirmed landing. A timeout never lets a drone in.
- **Background tasks.** After every drone's flight has ended, the `finally` in step 5 of
  `_run_one()` calls `stop_background()` (`atal/runtime/watchers.py`). It sets the `stop`
  event, waits for every background task, and records any that failed as
  `background_error`.
- **The record.** The outermost `finally` calls `recorder.close(outcome)`
  (`atal/runtime/recorder.py`), so the run log is complete even after an error.
- **Watchdogs** that can end a run early: `watch_mission_time` (mission time limit),
  `sample_truth` (geofence breach or impact), `watch_sim_clock` (live simulator froze), all in
  `atal/runtime/watchers.py`.

## One real decision, followed end to end

From an offline team run (`python run_sar_swarm.py --offline`, run 1, seed 43). The events
below are from the run log `run_01.jsonl`, for Drone2 sweeping sector E1. Times are sim
seconds.

| Sim time | Log event | What happened | Code |
|---|---|---|---|
| 146.6 | `sar_allocation` | Drone2 recomputes who should sweep what from its own memory and the peers it hears. | `SwarmCoordinator.my_sectors` |
| 147.6 | `belief_update` | Drone2's memory marks E1 `in_progress`, verified by self. | `LocalBelief.mark_started` |
| 147.7 | `peer_message_sent` | Drone2 tells the team it is taking E1 (`sector_claimed`). | `SwarmCoordinator.claim` |
| 147.9 | `proposal` | The planner submits `sweep_sector E1`, naming the observation it decided from. | `SearchLoop.submit` |
| 147.9 | `gate_decision` | The gate `sar_invariants` approves. | `SarInvariantGate.decide` |
| 147.9 to 165.1 | `sector_started`, `follower_*` | The sector executor flies the lanes. | `SectorExecutor.execute` |
| 165.1 | `command_result` | The executor reports `sector_swept`. | `ActionPort._gate_and_execute` |
| 165.2 | `belief_update` | Drone2's memory marks E1 `swept`, verified by self. | `LocalBelief.mark_swept` |
| 165.2 | `peer_message_sent` | Drone2 tells the team E1 is clear (`sector_cleared`). | `SwarmCoordinator.cleared` |

While Drone2 was flying E1, a peer message arrived:

| Sim time | Log event | What happened | Code |
|---|---|---|---|
| 151.2 | `peer_message` | Drone1 says B1 is clear. | `SwarmReceiver.receive` |
| 151.2 | `peer_rule` | Accepted, code `single_peer`. | `PeerClaimRules.on_cleared` |
| 151.3 | `belief_update` | Drone2's memory marks B1 `peer_cleared`, source Drone1, not verified by self. | `LocalBelief.mark_peer_cleared` |

The proposal, the gate decision and the command result share one `request_id`, and the
proposal names its `observation_id`. That chain is what lets every action in a run be
traced back to what the drone knew when it decided.

To see this yourself:

```
python run_sar_swarm.py --offline --runs 2
python -m atal.evaluation.summarize_run runs/sar_swarm/<batch>/run_01.jsonl
```
