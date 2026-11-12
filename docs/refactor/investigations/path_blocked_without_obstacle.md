# Investigation: path_blocked with no obstacle in the mission (offline team runs)

Status: cause found 2026-11-12. Not fixed. The fix belongs to the offline simulator and
gets its own change. Found during the step 5 review, which asked that a known flaky test
not be treated as harmless.

## Symptom

Inside full-suite runs (`pytest -n 2`), a team test sometimes fails although the mission
has no obstacle. The run summary shows a recovery without an impact:

```
  -> found=True by Drone3 t_find=61.28 s, mission 447.44 s, 15/16 sectors (dup 30), coverage 99.8%,
     msgs 460 sent / 0 out of range; Drone3: 0 impact(s) (), 1/1 recovered; FAIL: 15/16 sectors swept
>       assert m["sectors_swept"] == 16
E       assert 15 == 16
```

The tests it hit: `test_without_cooperation_drones_sweep_the_same_sectors`,
`test_static_split_sends_no_claims_or_clears_and_keeps_to_fixed_parts`, and
`test_swarm_lands_one_drone_at_a_time`. In the last one it also led to a separation
failure: `['separation below 4.5 m in cruise (4.39 m)', '14/16 sectors swept']`.

## How often, on which code

| Code | Full-suite runs | Runs with this failure | Tests that failed |
|---|---|---|---|
| f729757 (before step 5) | 8 | 2 | no-cooperation, static split |
| aad88cd (step 5) | 8 | 2 | no-cooperation, lands-one-at-a-time |
| 3c2aca2, 8b720db (step 3 work, original planner) | 2 runs that failed earlier the same day | 2 | no-cooperation (both), lands-one-at-a-time (once); logs not kept |

Running the no-cooperation test alone never failed: 0 of 185 runs on each code, including
100 runs with four copies in parallel to add load. It needs the load of the whole suite.

## Mechanism (the same in all four kept logs, old and new code)

1. The host stalls for about 0.1 s. In the log this shows as **no events at all for 8 to
   11 sim seconds**: no truth samples, no observations, no follower updates. In the first
   log there is nothing between sim 128.05 s and 140.35 s, which is 123 ms of host time.
2. The offline simulator (`atal/simulation/kinematic_sim.py`) sets sim time to host time
   × speedup (×100), so the stall is 10 sim seconds or more.
3. Velocity commands last their `duration` in sim seconds and then the vehicle brakes to
   a hover. Nothing renewed them during the stall, so the drones stopped. In the first log,
   Drone1 moved 0.2 m in those 12 s.
4. When the path follower runs again, it sees more than `stuck_after_s` (10 s) without
   progress and reports `path_blocked` ("blocked: no progress along the path for 12 s").
5. The search loop handles that as an obstacle: it climbs 10 m and blocks the sector, so
   one sector is never swept. In a team, the 10 m climb puts the drone on a teammate's
   level. That is the known obstacle-recovery separation issue (baseline.md), and it
   explains the 4.39 m.

The planner code is not involved. It reacts correctly to the `path_blocked` it is given.
That is why the old and the new code fail the same way at about the same rate.

## Why the separation check did not fail these runs

Offline, the separation check allows a truth gap of `max_truth_gap_s + 0.25 × speedup` =
27 sim s, to absorb host jitter at ×100. The 8 to 11 s gaps are inside that allowance. A
close approach during such a gap would not be measured. This is recorded as a
measurement limit in the obstacle-recovery policy.

## Proposed fix (its own change, with tests)

The offline simulator should not advance its clock by more than a small step per host
tick. Then a host stall slows simulated time instead of skipping it, and commands do not
expire unseen. Test: a forced 0.2 s host stall during a sweep must not produce
`path_blocked`, and must not leave a truth gap longer than `max_truth_gap_s`.

## Evidence

The run folders of the four failures and the full pytest output are kept outside the
repository, in the project files (`restructure/evidence/`).
