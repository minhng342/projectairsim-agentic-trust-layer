# Reference check attempts

Every execution of `tools/reference_cases.py --check docs/refactor/reference_baseline.json`
during the restructuring, in order, failures included. A passing check is smoke-test
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

Attempts before #3 are recorded from the PR descriptions of steps 3 and 4; from #4 on, the
full output is kept here.

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
