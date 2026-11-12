# Reference check attempts

Every execution of `tools/reference_cases.py --check` during the restructuring and the
changes after it, in order, failures included. Up to attempt 5 the reference is the
original `docs/refactor/reference_baseline.json`, which is kept unchanged. After the
offline-clock fix a new reference was recorded beside it, in
`docs/refactor/reference_baseline_clock.json` (attempt 6); later behavioural changes are
checked against both. A passing check is smoke-test
evidence: 8 offline cases x 3 runs agree with the recorded baseline on the measures the
checker compares (finds, passes, health, failure reasons, and times only where runs repeat).
It is not proof of identical behaviour: team runs are not repeatable per seed, and three
runs per case cannot show a small change in a failure rate.

| # | Code checked | Checker | Result | Notes |
|---|---|---|---|---|
| 1 | step 3 (c11217d) | step 1-2 checker | SAME BEHAVIOUR | The step 3 review found this checker hid failures in the obstacle case; fixed in 7cb0f86. |
| 2 | step 3 (c11217d) | 7cb0f86 (health, exit status, identities) | SAME BEHAVIOUR | Baseline recorded again on d7af72e with this checker. |
| 3 | step 4 (ba6ff5f) | 7cb0f86 | DIFFERENT: 2 differences | Both were runs where drones came closer than the separation floor. The same kind of failure occurs on the step 3 code (0.93 m in cruise in the obstacle case, about 1 run in 12). Cause traced later to obstacle recovery (baseline.md, known issues). |
| 4 | step 5 (aad88cd) | 7cb0f86 | SAME BEHAVIOUR | team_obstacle passed 2 of 3, its third run failing only for impacts, which the checker allows in that case (as in the baseline). Full output below. |
| 5 | offline-clock fix (b254db2) | 7cb0f86 | SAME BEHAVIOUR | Checked against the original reference. team_obstacle passed 3 of 3. Full output below. |
| 6 | offline-clock fix (b254db2) | 7cb0f86, `--record` | recorded | New reference `reference_baseline_clock.json`. team_obstacle passed 2 of 3; the failed run (seed 42) failed for "2 impacts" only, with every drone landed. That failure is part of the recorded reference, not removed. |
| 7 | managed airspace wired in (f036aa1, branch claude/recovery-reservations) | 7cb0f86, original reference | DIFFERENT: 43 differences | Drones left airborne: landings waited at most 60 s (`max_hold_s`), less than one teammate's 150-170 m startup or landing through the shared pad column. |
| 8 | f036aa1 | 7cb0f86, clock reference | DIFFERENT: 43 differences | Same cause. An earlier attempt against the clock reference was stopped and not counted: the working tree changed while it ran. |
| 9 | landing queue (1fdf772) | 7cb0f86, original reference | DIFFERENT: 30 differences | A landing paused a teammate inside its own column and waited for it to its deadline (fixed in 2808999). |
| 10 | 1fdf772 | 7cb0f86, clock reference | DIFFERENT: 30 differences | Same cause. |
| 11 | pauses never freeze a drone in the column (2808999) | 7cb0f86, original reference | DIFFERENT: 2 expected timing differences (not "same behaviour") | Every case found 3/3 and passed 3/3 (team_obstacle included). Both differences are team_static times to find: 591.4 s vs 364.2 s, 376.8 s vs 260.5 s. Intentional, see below. |
| 12 | 2808999 | 7cb0f86, clock reference | DIFFERENT: 2 expected timing differences | The same two team_static times (591.0 vs 360.6 s, 376.3 vs 256.8 s). |

Attempts 7 to 12 are on the recovery branch, which is not merged. Both references are
unchanged, and no threshold was changed to make a case pass.

**Intentional differences from the recovery policy (attempts 11 and 12).** In the static
split each drone searches only its own cells, so the time to find depends on when the
victim's owner starts. Under the managed airspace (policy v3.4) startups are granted top
down and the 3 m pads share one column, so the drones take off one after another:
Drone2 waits about 120 s and Drone1 about 233 s for its startup reservation, and a drone
may also be paused while a teammate moves vertically. In a separate 3-run static batch on
2808999 the finder of the 587 s run was Drone1 (startup wait 233 s, paused 111 s), and the
finder of the 377 s run was Drone2 (startup wait 120 s), which matches the 227 s and 117 s
differences. Team cases without time comparison pass with the same finds.


Attempts before #3 are recorded from the PR descriptions of steps 3 and 4; from #4 on, the
full output is kept here.

## Attempts 5 and 6: offline-clock fix (b254db2), 2026-11-12

Attempt 5, `--check docs/refactor/reference_baseline.json`:

```
one_fixed: one drone, fixed sector order ...
    found 3/3, passed 3/3, 41 s
one_llm: one drone, scripted stand-in for the LLM, invariant guard on ...
    found 3/3, passed 3/3, 50 s
team_fixed: three drones cooperating over the bus ...
    found 3/3, passed 3/3, 45 s
team_llm: three drones, scripted stand-in for the LLM ...
    found 3/3, passed 3/3, 47 s
team_dropout: one drone fails 200 s into the search ...
    found 3/3, passed 3/3, 47 s
team_static: fixed pre-mission split, no claims ...
    found 3/3, passed 3/3, 41 s
team_obstacle: an obstacle in the grid: impacts, recovery, shared blocked sectors ...
    found 3/3, passed 3/3, 43 s
team_self_only: peer clears are recorded but never accepted ...
    found 3/3, passed 3/3, 56 s
RESULT: SAME BEHAVIOUR
check exit=0
```

Attempt 6, `--record docs/refactor/reference_baseline_clock.json`:

```
one_fixed: one drone, fixed sector order ...
    found 3/3, passed 3/3, 41 s
one_llm: one drone, scripted stand-in for the LLM, invariant guard on ...
    found 3/3, passed 3/3, 50 s
team_fixed: three drones cooperating over the bus ...
    found 3/3, passed 3/3, 44 s
team_llm: three drones, scripted stand-in for the LLM ...
    found 3/3, passed 3/3, 47 s
team_dropout: one drone fails 200 s into the search ...
    found 3/3, passed 3/3, 42 s
team_static: fixed pre-mission split, no claims ...
    found 3/3, passed 3/3, 41 s
team_obstacle: an obstacle in the grid: impacts, recovery, shared blocked sectors ...
    found 3/3, passed 2/3, 45 s
team_self_only: peer clears are recorded but never accepted ...
    found 3/3, passed 3/3, 54 s
recorded 8 case(s) in docs/refactor/reference_baseline_clock.json
exit=0
```

Commit ids: the clock fix was tested as 45d7e90 and b254db2 on top of 34e2980, then
rebased unchanged onto main (4387ece, whose tree is identical to 34e2980) as a9e133a and
c1aedf0. The trees of b254db2 and c1aedf0 are identical, so the results apply to both.

### Full test-suite runs on the clock fix

All with `pytest -q -n 2`, every run listed.

| Code | Runs | Result |
|---|---|---|
| 45d7e90 (first clock change: capped clock, physics still integrated on reads) | 1 to 5 | 573 passed each |
| 45d7e90 | 6 | stopped and not counted: the code was revised (b254db2) while it ran |
| b254db2 (only clock ticks move the world) | 1 to 8 | 578 passed each |

### The stall reproduction (`tests/test_offline_clock.py`)

| Code | What ran | Result |
|---|---|---|
| clock before the fix (34e2980) | the two end-to-end stall tests, 20 times | failed 20 of 20 (path_blocked, no obstacle) |
| 34e2980 | the whole new test file once | 9 failed, 1 passed (only the no-event-loop test passes, as intended) |
| 45d7e90 | the two end-to-end stall tests, 25 times | passed 25 of 25 |
| b254db2 | the two end-to-end stall tests, 20 times | passed 20 of 20 |

These are regression evidence for this stall pattern, not proof that no other timing
problem exists. Removing clock jumps does not make asynchronous team runs repeatable:
the order of events in the event loop still depends on the host.

## Attempt 4: step 5 (aad88cd), 2026-11-12

```
one_fixed: one drone, fixed sector order ...
    found 3/3, passed 3/3, 41 s
one_llm: one drone, scripted stand-in for the LLM, invariant guard on ...
    found 3/3, passed 3/3, 51 s
team_fixed: three drones cooperating over the bus ...
    found 3/3, passed 3/3, 47 s
team_llm: three drones, scripted stand-in for the LLM ...
    found 3/3, passed 3/3, 47 s
team_dropout: one drone fails 200 s into the search ...
    found 3/3, passed 3/3, 43 s
team_static: fixed pre-mission split, no claims ...
    found 3/3, passed 3/3, 41 s
team_obstacle: an obstacle in the grid: impacts, recovery, shared blocked sectors ...
    found 3/3, passed 2/3, 44 s
team_self_only: peer clears are recorded but never accepted ...
    found 3/3, passed 3/3, 54 s
RESULT: SAME BEHAVIOUR
exit=0
```
Full outputs of attempts 7 to 12: `docs/refactor/reference_outputs/`.
