# Obstacle recovery policy, version 3.4 (accepted basis for implementation)

Version 3 adds Matthew's decisions and tightened requirements from 2026-11-12 to version 2,
which he accepted as the implementation direction. Version 3.1 (17:01Z) makes four details
explicit, also at his request: startup exceptions (section 10), an atomic start (section 5),
the landing sequence (section 11) and separate outcome terms (section 14). Matthew accepted
v3 as the implementation direction, with no further broad redesign before coding.
Version 3.2 (17:43Z, Matthew's review of 3.1) lets a STUCK drone retry from its measured
altitude (sections 3 and 7) and corrects `r_stop` to 21.5 m, derived from the formula
(section 8). Version 3.3 (18:00Z, Matthew chose option A on the rule 5 question) adds
traffic pauses (section 8a) and states the assumption behind the deadlock rule
(section 4). Version 3.4 (18:54Z, Matthew chose a separate landing-queue timeout) adds
section 11a and the "not frozen in the column" rule of section 8a: startups and landings
wait on a budget derived from the team size and the
longest startup or landing move, with one cleanup deadline per drone; `max_hold_s` for the
recovery states is unchanged. v3.3 is kept as `recovery_policy_v3.3.md`. Earlier
versions are kept as `recovery_policy_v1.md` and `recovery_policy_v2.md`.

The starting design is still option A. After an obstacle, a drone climbs to a transit level
of its own, flies to its next sector up there, and comes back down to its search level
before it sweeps. The sensor rule is unchanged: it detects only within 5 m of the search
altitude, so nothing is ever searched from a transit level.

Order of work, as agreed: the offline-clock fix comes first, as its own change. This
recovery policy is implemented after it, as a separate change. Neither change weakens an
acceptance threshold, and every failed run stays recorded.

## 0. Decisions recorded (Matthew, 2026-11-12)

| Question | Decision |
|---|---|
| Central transition manager | Yes, as **explicitly trusted flight-safety infrastructure** for this simulation. It does not allocate search sectors, does not change any drone's beliefs, and does not expose simulator truth to the agents: agents get only grant, deny, stop and keep-out notices about their own requests and moves. The **same manager runs in every comparison condition** (dynamic, static, self-only, no radio, dropout, LLM or fixed, and also one-drone runs, where every grant is trivial). Its separation results are **not** evidence that the agents' trust layer maintains separation on its own. The thesis must say so. |
| Ceilings | 240 m (dev) and 150 m (deck) as **provisional simulation caps**, subject to configuration and scene checks. They are validated against the actual scenario's team size, not a fixed n = 3. A ceiling does not establish obstacle clearance, and it does not make the deck mission's 60 m search level suitable for CityEnviron (a known open issue). |
| Strict waiting rule | Acceptable for a first version. Waiting time, denied requests, completed recoveries and unresolved outcomes are all recorded (section 13). Tests must include recoveries that actually **succeed**. Zero separation violations achieved by refusing every recovery would not show a useful mechanism. |
| Hold timeout | 60 sim s, configurable. When it expires it never authorises a descent. It leads to a bounded UNRESOLVED outcome (section 9). |
| Return altitude | A drone already back at its search level goes home there, and does not climb just because it recovered earlier. Its return route still needs the checks that apply to it. |

## 1. Why the current behaviour fails

Team drones are kept apart only by altitude: drone *i* searches at `B + i·s`. Today a
recovery climbs a fixed 10 m, which is one layer spacing, so the drone lands on a
teammate's level. That climb does not clear a 165 m tower from 150 m. And a recovered drone
comes back down beside drones waiting for the pads, whose waiting points are 2 to 3 m
apart.

## 2. Altitudes, tolerances and the ceiling

| Name | Drone *i* of *n* (0-based) | Used for |
|---|---|---|
| Search level `S_i` | `B + i·s` | sweeping, the only level where the sensor detects |
| Transit level `T_i(k)` | `B + (k·n + i)·s`, k = 1 ... `max_transit_level` | flying after the k-th recovery, and the way home while still up there |

*n* is the size of the team in the scenario actually loaded.

**Tolerances are per drone, and pair allowances are the sum of two.**

| Symbol | Per drone | Pair allowance | Meaning |
|---|---|---|---|
| `e_z` | 1 m | `2·e_z` = 2 m | how far a holding drone may be off its altitude |
| `e_h` | 1.5 m | `2·e_h` = 3 m | how far a holding drone may be off its position |

**One horizontal margin** is used everywhere: `d_h = separation_cruise_m + 2·e_h`
(8 m on the dev mission). It is the minimum horizontal distance between two drones that
are both holding or moving only vertically. The grant rules, the waiting slots and the pad
geometry all use this one number, so a waiting drone placed correctly never blocks its
neighbour's landing by accident (section 11).

**Validated when the mission loads.** The mission is rejected with a message if any of
these fail:

- The layer spacing must be at least `separation_cruise_m + 2·e_z`: on dev, 10 ≥ 5 + 2.
- Every `S_i` and `T_i(k)` up to `max_transit_level`, plus `separation_cruise_m`, must be
  at or below `limits.altitude_ceiling_m`.
- The waiting slots of section 11 must fit inside the geofence, at least `d_h` apart and
  at least `d_h` from every pad.

**Ceiling.** `limits.altitude_ceiling_m` is required for team missions. Proposed values
are 240 m for dev and 150 m for deck, both provisional. With `max_transit_level` = 2 and
n = 3, the highest level is 230 m on dev and 140 m on deck. A transit level above the
ceiling does not exist; needing one is the "no suitable altitude" case of section 9.

## 3. States of one drone

| State | Altitude | Exclusive? | Leaves when |
|---|---|---|---|
| STARTUP | on the pad, then climbing | no | see section 10 |
| SEARCHING | `S_i` | yes | after an obstacle: HOLDING, then REQUEST_CLIMB. When the search is over: RETURNING |
| HOLDING | where it is | only once confirmed at one of its own levels | confirmed: the request it was holding for |
| REQUEST_CLIMB | at `S_i` or `T_i(k)`, **confirmed holding** | yes | granted and started: CLIMBING. Not granted within `max_hold_s`: UNRESOLVED |
| CLIMBING | between levels | no | target reached and settled: TRANSIT. Stopped early or overran: STUCK |
| TRANSIT | `T_i(k)` | yes | over the next sector's entry point: HOLDING, then REQUEST_DESCENT. Search over: RETURNING at `T_i(k)`. Hit something: HOLDING, then REQUEST_CLIMB to `T_i(k+1)` |
| REQUEST_DESCENT | `T_i(k)`, confirmed holding | yes | granted: DESCENDING. Not granted within `retarget_s`: TRANSIT to another sector. After `max_retargets`: RETURNING at `T_i(k)` |
| DESCENDING | between levels | no | settled at `S_i`: SEARCHING. Blocked or impact: STUCK, then a fresh REQUEST_CLIMB back to `T_i(k)` |
| STUCK | partway | no | section 7 |
| RETURNING | `S_i`, or `T_i(k)` if still up there | yes | at its slot, holding the pad zone, confirmed holding: REQUEST_LANDING |
| REQUEST_LANDING / LANDING | down to its pad | inside the held zone, under its reservation | touchdown confirmed (section 5): landed |
| UNRESOLVED | where it is | yes at a level, otherwise a keep-out | the run ends. It never lands without a grant and is never disarmed |
| LOST | where it is | as UNRESOLVED | today's drop-out. It may land only through REQUEST_LANDING at its position; otherwise it becomes UNRESOLVED |

**"Confirmed holding"** means the drone's executor has seen, for `confirm_hold_s` = 1 sim s
in a row: horizontal speed below 0.3 m/s, vertical speed below 0.2 m/s, and altitude within
`e_z` of an assigned level. A hover command alone is not enough. The manager checks the same
thing in its own surveillance before it grants a request; if it disagrees, it denies
`not_holding`.

**"Confirmed stationary"** is the variant for a STUCK drone, which is between levels and so
can never be near an assigned level. It is the same test (1 sim s in a row, horizontal
speed below 0.3 m/s, vertical speed below 0.2 m/s), but the altitude must stay within `e_z`
of the drone's **measured altitude at the start of that second**, not of a level. The
follow-up request's `from_m` is that measured altitude. This is used only for a drone in
STUCK. Every other request still needs confirmed holding at a level.

## 4. Transition manager and grant rules

The transition manager is a runner-owned stand-in for a UTM or vertiport service, like
today's `PadZone`. It handles requests one at a time, in arrival order. Requests that do not
conflict with an earlier queued request may go ahead of it. Nobody is ever asked to move
aside.

**Grant rules.** A request from drone *a* is granted only if all of these hold:

1. Every team member's surveillance is fresh (section 6).
2. *a* is confirmed holding at the request's start altitude, within `e_h` of its stated
   position.
3. **Other vertical movers:** any other active reservation or keep-out whose altitude span
   overlaps *a*'s span must be at least `d_h` away horizontally.
4. **Holding drones:** any drone that is confirmed holding (in a REQUEST_* state, or
   UNRESOLVED) at an altitude inside *a*'s span must be at least `d_h` away horizontally.
5. **Drones that can move horizontally:** any drone that is SEARCHING, TRANSIT or
   RETURNING at an altitude inside *a*'s span must be farther than the exclusion radius
   `r` of section 8.

A lower drone already climbing into the same altitude range is covered by rule 3: its
reservation is active, so the overlapping request waits.

**Deadlock.** A cycle such as *a* waiting on *b* and *b* waiting on *a* (rule 4 both ways)
is detected. The newer **descent** is refused at once, and that drone retargets to another
sector.

*Assumption, for the validated configuration only:* with levels at least
`separation_cruise_m + 2·e_z` apart (10 m on dev, checked when the mission loads), a climb's
span never contains a lower drone's level from below, so two climbs cannot wait on each
other through rule 4. Version 3 expected a two-climb cycle that ends UNRESOLVED; with this
spacing it does not arise, and the test instead shows two adjacent climbs resolving in
order. This is not a claim that deadlock is impossible for every configuration. Any other
waiting cycle still ends at `max_hold_s` with a bounded UNRESOLVED outcome, which is
tested.

**Queue order.** A later request may go ahead of an earlier one that waits for the later
requester's own drone to move (rule 4), since that wait could otherwise never end.

## 5. Reservation lifecycle

A grant is a **reservation**: a specific record, never a reusable flag. Its fields:

| Field | Meaning |
|---|---|
| `reservation_id` | unique, made by the manager |
| `request_id` | the requester's id for this request; one request, at most one reservation |
| `vehicle_id` | the only vehicle it is valid for |
| `kind` | `climb`, `descent`, `landing` or `startup` |
| `center_xy`, `radius` | the column (radius `d_h / 2` for its own position tolerance) |
| `from_m`, `to_m`, `span` | start and target altitude; span is `[min − f, max + f]`, with f = `separation_cruise_m` |
| `granted_at`, `start_by`, `valid_until` | sim times (`start_by` = granted + `start_window_s`, 2 s) |
| `state` | see below |

**States:**

```
REQUESTED --grant--> GRANTED --executor starts--> ACTIVE --target reached--> SETTLING --settled--> COMPLETED
    |                    |                          |                            |
    |                    |                          +--overran or stalled--------+--> KEEP_OUT --> (new request finishes or retreats)
    |                    +--not started by start_by--> LAPSED (released; the manager checks the drone did not move)
    +--request deadline passed--> EXPIRED      (any state before ACTIVE) --cancel--> CANCELLED
```

**Rules:**

- **Binding.** The executor uses only a GRANTED reservation whose `vehicle_id` is its own,
  whose `from_m` is within `e_z` of where it is, whose `center_xy` is within `e_h` of where
  it is, and whose `to_m` is the target it was asked for. Anything else is refused before
  any command is sent (`reservation_mismatch`). `safe_shutdown` takes a landing
  reservation, not a boolean.
- **Atomic start.** Before it sends any motion command, the executor asks the manager
  `start(reservation_id, request_id, vehicle_id)` and waits for the answer. In one step the
  manager checks that the reservation is still GRANTED (not CANCELLED, EXPIRED or LAPSED),
  that now ≤ `start_by`, and that its own surveillance shows the drone confirmed holding
  within `e_z` of `from_m` and `e_h` of `center_xy`. Only then does it move the reservation
  to ACTIVE and answer `started`. Any other answer (`lapsed`, `cancelled`, `expired`,
  `not_holding`, `mismatch`) means no command is sent, and the drone stays holding. A
  notice sent after the motion has begun does not count: motion seen while the
  reservation is not ACTIVE is `unauthorised_motion` (a failure). Offline the manager runs
  in the same event loop, so the check and the state change are one step with no await in
  between.
- **One outstanding request per vehicle.** A second request from the same vehicle while
  one is REQUESTED, GRANTED, ACTIVE, SETTLING or KEEP_OUT is denied (`already_pending`),
  unless it is the follow-up a STUCK drone is allowed to make (section 7).
- **Duplicates.** The same `request_id` sent again returns the same reservation (or the
  same denial). It never creates a second one.
- **Request deadline.** Each request carries `wait_until` (now + `max_hold_s`, or less). If
  it is not granted by then, it becomes EXPIRED and leaves the queue.
- **Cancellation.** The requester may cancel a request or a GRANTED reservation before it
  starts. An ACTIVE reservation cannot be cancelled. It ends only by completing or by
  becoming a KEEP_OUT.
- **Late answers.** A grant that reaches a drone after it stopped waiting is never used:
  the drone has already sent `cancel(request_id)`, or it simply does not start. The
  reservation lapses at `start_by`. The manager releases a lapsed reservation only after its
  surveillance shows the drone still holding at `from_m`. If the drone moved anyway, the
  reservation becomes a KEEP_OUT and the run records `unauthorised_motion`, a failure.
- **Completion.** Reaching the target altitude once is not enough. The reservation goes to
  SETTLING when the drone is within `e_z` of `to_m`. It is COMPLETED only after
  `settle_s` = 2 sim s in a row of confirmed holding there, seen in the manager's
  surveillance. If the drone drifts out of tolerance during SETTLING, the settle timer
  starts again.
- **Landing.** A landing reservation stays until touchdown is confirmed. That uses the same
  test as `safe_shutdown`: a valid snapshot with ground state GROUNDED, polled after the
  land command. Then the order is: reservation COMPLETED, disarm, and only after the disarm
  **pad zone released**. The drone **leaves the traffic roster** only once it is confirmed
  on the ground and disarmed. A LOST or UNRESOLVED drone stays on the roster, and keeps
  blocking, until the run ends.

## 6. Traffic information: freshness, missing drones, link loss

- **Source.** The manager tracks the team with its own surveillance: the simulator's
  positions sampled at 5 Hz, standing in for UTM surveillance with full coverage. It does
  not use the peer radio (150 m range on dev). That full-coverage assumption is part of the
  "trusted infrastructure" statement in section 0.
- **Freshness.** A position counts if it is less than `a_s` = 1.0 sim s old. While any team
  member's position is stale, nothing new is granted.
- **Missing team member.** The roster comes from the scenario. A member never seen, or no
  longer seen, blocks every grant until it is fresh again or confirmed landed and disarmed.
  "Not heard" never means "clear".
- **Drone-to-manager link.** This is the command link the runner already uses for each
  drone, not the peer radio. A request with no answer within 2 s counts as not granted, and
  the drone keeps holding. A move that has started continues to its target, and the
  reservation ends only when the manager sees it complete (section 5).
- **Surveillance lost during a move.** Active reservations stay in force and nothing new is
  granted. Any move ACTIVE while surveillance was stale gets a flag, and if the gap was
  longer than `a_s` the run fails as `separation_unverified`.

## 7. A drone stopped partway (STUCK)

A drone that stops between levels may be at another drone's level, so it is not safe by
default:

- At that instant its reservation becomes a **KEEP_OUT** with the same column and span. No
  new reservation is involved, so protection never has a gap.
- The stop-line rule of section 8 applies to the keep-out the whole time. Crossing traffic
  is stopped before it can reach it.
- The stuck drone may make one follow-up request: finish the move, or after a failed
  descent go back up to `T_i(k)`. It requests from its **measured current altitude** once
  it is confirmed stationary there (section 3), so being between levels never blocks the
  retry. That request goes through all the grant rules again (Matthew: a column used
  earlier is not necessarily still clear). Its own keep-out is not counted against it.
- **The keep-out stays until the follow-up is ACTIVE.** While the request waits, or if it
  is denied, expires or lapses, the keep-out protects the drone as before. At the atomic
  start (section 5) the manager replaces the keep-out by the new ACTIVE reservation in the
  same step, so there is no moment without protection. The new reservation's span covers
  the measured altitude and the target.
- After `max_stuck_retries` (default 2) it is UNRESOLVED, and the keep-out stays until the
  run ends.

## 8. Validity horizon, exclusion radius and stopping crossing traffic

The exclusion radius rests on stated bounds. Each is a configuration value with a source:

| Bound | Symbol | Dev value | Source / check |
|---|---|---|---|
| Horizontal speed of any drone | `v_h` | 6 m/s | flight speed 5 m/s × 1.2; the run fails `speed_bound_exceeded` if surveillance ever sees more |
| Vertical speed used for duration | `v_z` | 1.0 m/s | below the follower's 1.5 m/s maximum, to allow the slow final approach |
| Settling time | `settle_s` | 2 s | section 5 |
| Surveillance age allowance | `a_s` | 1 s | section 6 |
| Detection + command latency | `L` | 0.5 s | manager sampling (0.2 s) + command delivery; to be measured live |
| Braking | `b` | 4 m/s² | offline simulator's acceleration limit; to be measured live |

- **Validity time** of a move with span `h`: `V = h / v_z + settle_s + 5 s`. A 30 m climb
  gives V = 37 s.
- **Exclusion radius at grant (rule 5):** `r = separation_cruise_m + e_h·2 + v_h·(V + a_s + L) + v_h² / (2b)`.
  For the 30 m climb: 5 + 3 + 6 × 38.5 + 4.5 ≈ 244 m. That is more than the whole dev
  area, which is the strict rule Matthew accepted: in practice no teammate may be
  searching or transiting inside the span.
- **Stop-line, checked at every surveillance sample, for every reservation that is ACTIVE,
  SETTLING or KEEP_OUT:** any drone flying horizontally at an altitude inside the span and
  closer than `r_stop = separation_cruise_m + 2·e_h + v_h·(a_s + L) + v_h²/(2b)` is ordered
  to **stop**. With the dev values: 5 + 2 × 1.5 + 6 × (1 + 0.5) + 6² / (2 × 4) = 5 + 3 + 9 +
  4.5 = **21.5 m**. (Version 3 said 22.7 m, which was an arithmetic error.) The code
  computes `r_stop` from the configured values with this formula, and a test checks the
  21.5 m result. Its executor interrupts the leg and the drone hovers. It is then
  told it may not resume a leg that would come closer while the reservation or keep-out
  lasts. `r_stop` is exactly the distance it needs: it may be seen `a_s` late, the order
  takes `L` to arrive, and it then needs `v_h²/(2b)` to stop. Separation therefore holds
  however long the move takes. This is how protection stays continuous when a move overruns
  V and becomes a keep-out.
- **Legs that start later** are checked against all active reservations and keep-outs when
  they start. A leg that would come within `r_stop` at a level inside a span is refused, and
  the planner chooses again.
- Every stop order is logged (who, why, distance) and counted in the metrics. If stop
  orders ever happen, they show that the exclusion radius at grant time was not enough, and
  that is reported.

## 8a. Traffic pauses (option A, Matthew 2026-11-12 18:00Z)

The exclusion radius for a 30 m move (243.5 m) is larger than the dev area's diagonal
(226 m), so rule 5 alone would never let a lower drone recover while a teammate searches
inside its span. Instead the manager **pauses** those teammates.

- **When.** A queued request blocked by rule 5 (a drone that can move horizontally, inside
  the span, within the exclusion radius) orders that drone to pause. A stop-line order
  (section 8) is the same kind of pause.
- **What a pause is.** The drone's executor interrupts the leg it is flying and hovers. It is
  not asked to move aside. Surveillance, the victim sensor, the mission-time watchdog and
  the stall watchdog keep running. An interrupt for another reason (a detection, the time
  limit) still ends the leg during a pause.
- **Grant only on confirmed stationary and clear.** A paused drone stops counting under
  rule 5 only once the manager's own surveillance shows it **confirmed stationary**
  (section 3) and at least the configured `d_h` from the column. Until it is stationary it
  blocks (`pausing`). If it stops closer than `d_h`, it blocks (`holding`), and it is never
  moved aside: the request runs into its deadline and the bounded outcome applies.
- **Not frozen in the column (v3.4).** A drone that is already within `d_h` of the
  requested column when the request is queued is **not** paused by it: pausing it there
  would hold it exactly where it blocks the request, which then waits to its deadline (a
  deadlock the manager itself makes; seen on dev, a drone just airborne over its pad,
  2.8 m from a teammate's landing column, paused for 790 s). It is not moved aside either:
  it keeps flying its own plan and keeps blocking the request (as a mover, or as a holder
  within `d_h`), and it is paused by the request once it is outside `d_h`. If it stops
  inside `d_h` on its own, the request runs into its deadline as above. Each skipped
  pause is recorded (`traffic_pause_skipped`). Pauses it already has from other owners
  stay.
- **Ownership.** Each pause has an owner: the reservation that caused it. A drone may be
  paused by several owners at once. It resumes only when **none** is left.
- **When an owner ends.** An owner ends when its reservation is COMPLETED, EXPIRED,
  CANCELLED, DENIED or LAPSED. A reservation that becomes a KEEP_OUT keeps its pauses: a
  timeout or a STUCK move never releases traffic into an occupied column. When a keep-out
  is replaced by its follow-up (SUPERSEDED), its pauses move to the follow-up in the same
  step, so protection stays continuous. A keep-out of an UNRESOLVED drone keeps its pauses
  until the run ends.
- **Sweep progress.** A pause is not an obstacle. The leg does not end, so it never
  produces `path_blocked`, obstacle recovery, or a "sector swept" update. When the drone
  resumes, it flies the rest of the same path from where it stopped, and the progress
  recorded for the sector counts the part flown before the pause.
- **Recorded:** every pause and resume with its owner, the pause count and total paused
  time per drone, and the reason each request waited.
- **Same in every condition.** The pause rule is part of the trusted manager, so it runs in
  every comparison condition.

## 9. When there is no safe option: bounded outcomes

| Situation | Outcome |
|---|---|
| `T_i(k+1)` needed but above `max_transit_level` or the ceiling | LOST at its current level |
| More than `max_recoveries` recoveries | LOST |
| Climb not granted within `max_hold_s` (60 s) | UNRESOLVED at its current level |
| Descent retargeted `max_retargets` (2) times | RETURNING at its transit level |
| Landing not granted by the drone's cleanup deadline (section 11a) | UNRESOLVED |
| STUCK after `max_stuck_retries` | UNRESOLVED with a keep-out |
| A vertical command fails | STUCK, then as above |

**UNRESOLVED is bounded.** The drone hovers, broadcasts `mission_state: unresolved`, and
asks for nothing more. Once every drone is landed, LOST and holding, or UNRESOLVED, the
runner ends the run. The result is failed: `unresolved_airborne`, with its last position
and the reason. An expired timeout never authorises a descent.

## 10. Startup (takeoff and the first climb)

Startup is **no longer exempt** from reservations, with the explicit exceptions below. The team takes off from pads 3 m apart, and at the same
climb rate a higher-level drone passes a lower drone's level 3 m from it. Version 2 left
that under the pad separation floor of 1.5 m, which contradicts "every vertical move is
granted". Now:

- Each drone's takeoff and climb to `S_i` is one `startup` reservation, from the ground to
  `S_i`.
- The manager grants startup **top-down**: the drone with the highest level first, then the
  next. Each later climb ends at least `s` below the drones already up there, so it never
  crosses an occupied level (rule 4).
- A drone not yet granted waits on its pad, disarmed or idle. It counts as on the ground,
  not airborne, and it **stays on the traffic roster** the whole time: it is a ground
  occupant of its pad column, so no other drone gets a landing or low descent reservation
  through that column. The missing-member rule (section 6) applies to it as to any member.
- **Startup starts from confirmed ground state, not from a level.** A `startup` request
  is made from the pad: the drone is confirmed grounded (a valid snapshot with ground state
  GROUNDED) and within `e_h` of its pad. That replaces "confirmed holding at an assigned
  airborne level", which a grounded drone cannot meet. Its `from_m` is the ground altitude
  and its column is centred on the pad. The atomic start (section 5) checks this ground
  state instead of a hover.
- **A drone already on the ground does not need a descent reservation to disarm.** If
  `safe_shutdown` finds the drone confirmed grounded (takeoff never granted, or it had
  already landed), it disarms without asking the manager. It still releases its pad only
  after the disarm, and leaves the roster only then, as in section 5. A drone whose ground
  state is unknown (no valid snapshot) is not treated as grounded: it needs a landing
  reservation, or it is UNRESOLVED.
- Cost: startup takes up to *n* climbs in a row instead of one. The search timer of each
  drone starts at its own plan, as today. Time to find is measured from the first drone's
  plan, and the startup delay is recorded so the comparison with earlier Phase 3 numbers
  can account for it.
- `swarm.startup: pad_floor` keeps the old simultaneous climb, only to reproduce
  earlier results. Runs that use it are labelled as such and are not used for
  separation claims.

## 11. Pad zone and waiting slots

- **Slot geometry:** the slots are on the circle of radius `R_zone + d_h` around the zone
  centre. Each slot is at least `d_h` from every other slot and at least `d_h` from every
  pad, so a waiting drone never blocks a neighbour's landing column under rule 4. The
  mission is rejected if this does not fit inside the geofence. The dev pads are at the
  area's corner, so the slots go on the quarter circle inside it.
- A drone flies to its slot on its current level. Waiting drones are apart both
  horizontally (slots) and vertically (their own levels).
- **Landing sequence, in this order:**
  1. Obtain the pad zone (the existing pad-zone lock). Until then, wait at the slot on the
     current level.
  2. Fly in, on the current level, to directly above its own pad.
  3. Confirm holding there (section 3).
  4. Request a `landing` reservation. Its `center_xy` is the pad and its `from_m` is the
     current level, so the position binding matches where the descent actually begins
     (the slot is never the bound position for a landing over the pad).
  5. When it is granted, do the atomic start (section 5), then descend and land.
  6. Touchdown confirmed, then COMPLETED, disarm, pad zone released, roster removal
     (section 5).
  A drone that does not hold the zone never leaves its slot, and a drone above its pad
  without a granted landing reservation keeps holding there.
- A `pad_zone_timeout` leads to a landing request **at its slot**, with `center_xy` at the
  slot and `from_m` the current level, because that is where that descent would begin. It
  is subject to the same grant rules; if it is not granted, the drone is UNRESOLVED.

## 11a. Landing queue: how long startups and landings wait (v3.4)

Why: the dev pads are 3 m apart, inside `d_h`, so every startup and landing column overlaps
a teammate's. One startup or landing from 150 to 230 m holds that column for 157 to 237 s
(its validity time), far longer than `max_hold_s` = 60 s, so correct waits were ending
UNRESOLVED (3 of 20 dev tower runs, every reference case). The pad geometry is kept.

All of it is computed in one place, `atal/airspace/landing_queue.py`, and recorded in the
run's `airspace` event (`queue_budget`).

- **Top level.** For a landing: the highest altitude any team drone can start a landing
  from, which is its highest transit level below the ceiling (a drone that went up in a
  recovery and could not come down flies home and lands from there), or its search level
  if it has none. On dev this is `T_2(2)` = 230 m. For a startup: the highest search level
  (170 m on dev).
- **Budget.** `V_max = V(top level)` (section 8), `per_move = V_max + 30 s`,
  `queue = n x per_move`. Dev, n = 3: landing 3 x 267 = 801 s, startup 3 x 207 = 621 s.
  This is a **documented default, not a proven worst-case bound.** It covers what the
  queue order gives (a conflicting later request never overtakes a waiting one, except the
  drone the waiting one is held up by, section 4, so each other drone passes through the
  column at most once ahead of it, and each of those moves ends within its validity time
  or becomes a keep-out). A keep-out that never clears, a teammate holding in the column,
  or a stale track can make the wait longer; the request then expires and the drone is
  UNRESOLVED.
- **One cleanup deadline per drone.** Set once, the first time the drone waits for the
  pad zone or asks to land: `deadline = start + queue`. The pad-zone wait is
  `min(configured pad-zone wait, deadline - now - per_move)`, so the landing request keeps
  time for at least one teammate's move; every landing request's `wait_until` is the
  deadline itself. Polling, a retry after a "not holding yet" denial, and repeated requests
  never move it. The stages do not each get a full budget.
- **Unchanged.** `start_by`, active-reservation validity (a reserved move is now also cut
  off at its validity time and becomes STUCK, section 7), freshness, separation margins,
  and the UNRESOLVED outcome at the deadline. An expired request can never be started, so
  it never authorises a descent. `max_hold_s` for climbs in recovery stays 60 s.
- **Recorded separately.** Per drone: `queue_wait_s` (landing request to grant),
  `descent_s` (start to confirmed touchdown), `cleanup_total_s` (cleanup start to the end of
  shutdown), the deadline and the number of landing requests.

## 12. Cleanup and shutdown obey the same policy

- `safe_shutdown()` is the only place in `atal/` that commands a landing. It takes a
  landing reservation (section 5) and refuses to send `land` without a valid one. Without
  one it does not land and does not disarm: it hovers, keeps API control, and reports
  UNRESOLVED (`landing_not_cleared`).
- The normal path and every `finally:` path go through the same function. That includes an
  exception, a stalled simulator, a KeyboardInterrupt, a LOST drone and a single-drone run.
  There is no landing without a reservation.
- **Single-drone runs** use the same manager with a roster of one, so the same code path
  runs and the grant is immediate.

## 13. What is recorded

Per run and per drone:

- every request, with id, kind, span, outcome and the reason for any denial;
- waiting time per request and in total;
- reservations that lapsed, expired or were cancelled;
- STUCK and keep-out events;
- stop orders;
- **completed recoveries** (impact or blocked, then climb, transit, descent, and back to
  searching);
- UNRESOLVED and LOST outcomes with their reasons;
- the startup delay;
- the recovery outcome of each episode, apart from the run's acceptance (section 14).

These appear in the run summary and the batch table next to the existing separation
numbers.

## 14. How the result is judged

Three results are kept apart and reported separately, because they answer different
questions:

| Term | What it says | Values |
|---|---|---|
| **Recovery outcome** | one recovery episode, from the impact or blocked path to back at the search level (or home) | `completed`, `unresolved`, `lost` |
| **Reference-check result** | whether the reference cases behave like the recorded reference (`tools/reference_cases.py`) | same behaviour, or the differences listed |
| **Run acceptance** | whether the whole run passed the acceptance rules below | pass, or fail with every reason |

A recovery can be `completed` in a run that still **fails**, for example because of the
impact that started it. A completed recovery never turns a failed run into a pass, and a
passing reference check says nothing about any single run's acceptance. Run summaries and
the batch table show the three in separate columns.

- Every separation below the floor fails the run, even if a rerun passes. A rerun is a
  different run.
- Every impact fails the run, as today.
- These also fail the run: UNRESOLVED, LOST, `landing_not_cleared`,
  `unauthorised_motion`, `separation_unverified` and `speed_bound_exceeded`.
- No threshold is relaxed. The offline gap allowance of `max_truth_gap_s + 0.25 × speedup`
  is left as it is in this change. Whether to tighten it now that the clock no longer jumps
  is a separate decision, made with evidence.
- 40 or more passing offline runs are regression evidence, not proof that separation
  failures cannot happen.

## 15. Tests planned

| Area | Tests |
|---|---|
| Levels, tolerances, ceiling | levels distinct and at least `s` apart for n = 1 to 5 and for the loaded scenario's n; rejected when the spacing is below floor + `2·e_z`, when a level is above the ceiling, or when the slots do not fit at `d_h` |
| Reservation lifecycle | the binding check refuses the wrong vehicle, place, start altitude or target; duplicate `request_id`; a second request while one is outstanding; a request that expires; cancelling before start; trying to cancel an ACTIVE reservation; a late grant lapses and is released only once the drone is seen still holding; motion without a grant gives `unauthorised_motion` |
| Completion | touching the target and drifting off does not complete; settling restarts; a landing completes only on confirmed touchdown; the pad zone is released after disarm; the roster removal conditions |
| Grant rules | the overlapping-span, holding-drone and horizontal-mover rules; a lower drone already climbing blocks a higher request; requests in order; a later non-conflicting request goes ahead; deadlock refuses the newer descent; two adjacent climbs resolve in order (validated 10 m configuration, section 4; not a general no-deadlock claim); any other waiting cycle ends at its deadline as UNRESOLVED |
| Stale and missing reports | a stale member blocks; a member never seen blocks; a confirmed landed and disarmed drone stops blocking; surveillance lost during a move keeps the reservation and flags the run |
| Simultaneous requests | climb + climb, climb + descent, descent + descent and landing + climb in the same instant: exactly one granted when they conflict |
| Interrupted transitions | the link drops mid-move; a move overruns V and becomes a keep-out with no gap; the stop-line stops a crossing drone in time (checked against the floor); a stuck drone's follow-up is checked fresh |
| Confirmed holding | a request right after a hover command is denied `not_holding` until 1 s of confirmed hold |
| STUCK retry | a drone stuck between levels is confirmed stationary at its measured altitude and its follow-up is accepted from there; it is denied while it still drifts; its keep-out stays while the follow-up waits and when it is denied; at the follow-up's start the keep-out is replaced in the same step |
| Traffic pauses | pause and resume during a sweep with the progress kept and no `path_blocked`; a teammate stopped too close is not moved and the request ends at its deadline; two owners at once, resuming only when both end; a cancelled request releases its pauses; a request whose move becomes STUCK keeps its pauses, and they move to the follow-up |
| Bounds | `r_stop` from the configured values is 21.5 m on dev; `r` at grant for a 30 m climb is 243.5 m; both change when a bound changes |
| Atomic start | `start` refused after `start_by`, after a cancel, after expiry, or when the drone is not holding where the reservation is bound: no motion command is sent in each case; motion seen before `started` gives `unauthorised_motion` |
| Startup | top-down order; no level crossed while occupied; `pad_floor` labelled; a grounded drone awaiting startup stays on the roster and blocks its pad column; a startup request is accepted from confirmed ground state and refused when the ground state is unknown; an already-grounded shutdown disarms with no reservation, then releases the pad |
| Landing sequence | no landing request before the zone is held and the drone is confirmed holding above its pad; the reservation is bound to the pad, not the slot; a `pad_zone_timeout` landing is bound to the slot |
| Outcome terms | a run with an impact and a completed recovery reports recovery `completed` and run acceptance `fail`; the batch table shows the two in separate columns |
| Obstacles, failing | a tower above `T_i(1)`: hit in TRANSIT, then `T_i(2)`, then LOST at the limit; a roof under a sector entry: STUCK, then back up only with a fresh grant |
| Obstacles, **succeeding** | a tower the drone recovers from: it climbs, transits, descends, searches on, the victim is found and every drone lands. Several such runs, with completed recoveries counted. The obstacle case of the reference set should then pass for reasons other than "no impacts" |
| Failed escape paths | climb denied until `max_hold_s`: UNRESOLVED; descent retargeted twice, then home at the transit level; landing not granted: UNRESOLVED and not disarmed |
| Occupied pads | a recovered drone arrives while another holds the zone: it waits at its slot on its level and lands only with zone and reservation |
| Shutdown | no reservation, then no land and no disarm; an exception mid-flight, then the `finally:` path still asks for a reservation; a stalled simulator path the same |
| End to end | offline obstacle batches with every failed run kept, and the reference check recorded again, failures included |

## 16. Limits

- The manager is trusted infrastructure with full coverage (section 0). A Phase 4 insider
  cannot lie to it about its position, but it could flood it with requests or refuse to
  start a granted move. Those go into the threat model.
- Obstacles are known only by contact. A granted vertical move can still meet a building.
  Section 3 (STUCK) says what happens then; it is not prevented.
- The bounds of section 8 are offline values. Live, `v_h`, `L` and `b` must be measured,
  and a run that breaks a bound fails.

## Related: the offline-clock fix (first change, separate)

The `path_blocked` failures without an obstacle came from the offline simulator skipping
time during a host stall
(`docs/refactor/investigations/path_blocked_without_obstacle.md`). The clock fix is its
own change and comes before this one. It gives one simulation clock that never jumps more
than 0.5 sim s past its last tick, with a deterministic reproduction test. A new reference
is recorded after it, and the original evidence is kept.
