"""
search_and_rescue.py

Five drones split a 100 x 100 m area into five sectors and search for a
randomly placed target.

- Each drone owns one 20 x 100 m strip. It sweeps the strip in lanes; the
  lane order is random (a random pick among the nearest uncovered lanes), so
  the motion varies from run to run but the whole sector is always covered.
- A sensor loop runs at SENSOR_HZ for every drone and compares its position to
  the target on the north and east axes only (height is ignored). The target
  is detected when the drone is within +/- DETECTION_MARGIN_M of it on both
  axes, i.e. inside a 2.0 x 2.0 m box centered on the target.
- The first drone to detect the target stops and hovers, and every other drone
  that is still searching is sent to that north/east position.
- The drones stack over that position, 1.0 m apart in height: the finder at
  30 m, then the other four (in drone-number order) at 31, 32, 33 and 34 m.
- A drone that sweeps its whole sector without detecting the target lands.

Coordinates are world NED in meters (north, east); the area is centered on
AREA_CENTER, which is the scene origin by default.
"""
import asyncio
import contextlib
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass

from adapters.projectairsim_adapter import ProjectAirSimAdapter
from executor.projectairsim_executor import ProjectAirSimExecutor
from models.action import CommandStatus
from models.telemetry import GroundState, ValidationStatus
from legacy.fleet import run_all_or_cancel
from utils.flight_safety import safe_shutdown
from utils.separation import SeparationTracker, assess, format_report


# ---------------- Configuration ----------------
# Copy scene_five_drones.jsonc into your sim_config folder.
SCENE = "scene_five_drones.jsonc"
VEHICLES = [f"Drone{i}" for i in range(1, 6)]

AREA_SIZE_M = 100.0
AREA_CENTER = (0.0, 0.0)        # (north, east) of the area's center

DETECTION_MARGIN_M = 1.0        # detect within +/- this on north AND east (z ignored)
TRACK_TOLERANCE_M = 0.3         # sideways drift allowed while flying a lane
# Lanes are spaced so that a drone drifting TRACK_TOLERANCE_M off its lane
# still has every point between lanes inside the detection box.
LANE_SPACING_M = 2 * (DETECTION_MARGIN_M - TRACK_TOLERANCE_M)
END_INSET_M = 0.5               # lanes stop this far inside the sector's north/south edge
NEXT_LANE_CHOICES = 3           # pick randomly among this many nearest uncovered lanes

SPEED = 4.0
SEARCH_ALTITUDE_M = 30.0        # all drones search at this height (above the tallest block)

# Stack over the found position: finder at the base height, then +STACK_STEP_M each.
STACK_BASE_ALTITUDE_M = SEARCH_ALTITUDE_M
STACK_STEP_M = 1.0
STACK_HOLD_S = 10.0             # hover time once all drones are stacked
# Landing straight down would put the stacked drones on top of each other,
# so they spread out sideways first.
DISPERSE_BEFORE_LANDING = True
DISPERSE_SPACING_M = 4.0
SENSOR_HZ = 20
GEOFENCE_TOL_M = 1.5            # excursion beyond the sector that counts as a breach
MISSION_TIMEOUT_S = 900.0
MIN_SEPARATION_M = 2.0
SEPARATION_SAMPLE_HZ = 20
LAND_S = 10.0
DESCEND_ABOVE_LAUNCH = 1.5
RANDOM_SEED = None              # set an int to repeat the same target and search


ALTITUDES = {v: SEARCH_ALTITUDE_M for v in VEHICLES}


# ---------------- Geometry ----------------
@dataclass
class Sector:
    index: int
    n_min: float
    n_max: float
    e_min: float
    e_max: float

    def contains(self, n, e) -> bool:
        return (
            self.n_min <= n <= self.n_max and self.e_min <= e <= self.e_max
        )

    def excursion(self, n, e) -> float:
        """Distance outside the sector (0 if inside)."""
        dn = max(self.n_min - n, 0.0, n - self.n_max)
        de = max(self.e_min - e, 0.0, e - self.e_max)
        return math.hypot(dn, de)


def make_sectors(n_sectors) -> list:
    """Split the area into equal strips running north-south, west to east."""
    cn, ce = AREA_CENTER
    half = AREA_SIZE_M / 2
    width = AREA_SIZE_M / n_sectors
    return [
        Sector(
            index=i,
            n_min=cn - half,
            n_max=cn + half,
            e_min=ce - half + i * width,
            e_max=ce - half + (i + 1) * width,
        )
        for i in range(n_sectors)
    ]


def target_detected(n, e, target, margin=DETECTION_MARGIN_M) -> bool:
    """
    True if (n, e) is within +/- margin of the target on BOTH north and east.
    Altitude is deliberately not part of this check.
    """
    return abs(n - target[0]) <= margin and abs(e - target[1]) <= margin


def segment_box_hit(p0, p1, target, margin=DETECTION_MARGIN_M):
    """
    If the straight path from p0 to p1 (north, east) passes through the
    target's +/- margin box, return the point on the path in the middle of
    that crossing (so the detection position is inside the box even when the
    samples are far apart). Otherwise return None.
    Checking the path between consecutive samples means a fast drone cannot
    skip over the narrow box between two samples.
    """
    t_lo, t_hi = 0.0, 1.0

    for a, b, c in ((p0[0], p1[0], target[0]), (p0[1], p1[1], target[1])):
        lo, hi = c - margin, c + margin
        d = b - a

        if abs(d) < 1e-12:
            if a < lo or a > hi:
                return None
        else:
            ta, tb = (lo - a) / d, (hi - a) / d
            if ta > tb:
                ta, tb = tb, ta
            t_lo, t_hi = max(t_lo, ta), min(t_hi, tb)
            if t_lo > t_hi:
                return None

    t = (t_lo + t_hi) / 2
    return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))


def segment_hits_box(p0, p1, target, margin=DETECTION_MARGIN_M) -> bool:
    return segment_box_hit(p0, p1, target, margin) is not None


class SectorSearch:
    """Lanes running north-south across one sector, and which are covered."""

    def __init__(self, sector: Sector, rng: random.Random):
        self.sector = sector
        self.rng = rng

        width = sector.e_max - sector.e_min
        count = math.ceil(width / LANE_SPACING_M)
        spacing = width / count

        self.lane_e = [sector.e_min + (k + 0.5) * spacing for k in range(count)]
        self.n_lo = sector.n_min + END_INSET_M
        self.n_hi = sector.n_max - END_INSET_M
        self.covered = [False] * count

    @property
    def lanes(self) -> int:
        return len(self.lane_e)

    @property
    def remaining(self) -> int:
        return self.covered.count(False)

    @property
    def fraction(self) -> float:
        return 1.0 - self.remaining / self.lanes

    def mark_lane(self, k):
        self.covered[k] = True

    def pick_next(self, n, e):
        """
        Random choice among the nearest uncovered lanes.
        Returns (lane, start, end) where start is the end of the lane nearest
        the drone, so consecutive lanes are flown in alternating directions.
        """
        candidates = []

        for k, lane_e in enumerate(self.lane_e):
            if self.covered[k]:
                continue

            a = (self.n_lo, lane_e)
            b = (self.n_hi, lane_e)
            da = math.hypot(a[0] - n, a[1] - e)
            db = math.hypot(b[0] - n, b[1] - e)

            if da <= db:
                candidates.append((da, k, a, b))
            else:
                candidates.append((db, k, b, a))

        candidates.sort(key=lambda c: c[0])
        _, k, start, end = self.rng.choice(candidates[:NEXT_LANE_CHOICES])
        return k, start, end


def stack_altitude(index) -> float:
    """Height for position `index` in the stack (0 = the drone that found it)."""
    return STACK_BASE_ALTITUDE_M + STACK_STEP_M * index


def claim_find(vid, detection, states, sync, found) -> bool:
    """
    The first drone to detect the target becomes the finder and recalls the
    rest. Stack order: the finder first, then the drones that are still
    searching, in drone-number order. Drones that already finished their
    sector and landed are not part of the stack.
    """
    if found.is_set():
        return False

    states[vid]["detection"] = detection
    sync["finder"] = vid
    sync["rally"] = (detection["n"], detection["e"])
    sync["order"] = [vid] + [
        v for v in VEHICLES if v != vid and states[v]["outcome"] is None
    ]
    found.set()
    return True


# ---------------- Telemetry display ----------------
def row(s) -> str:
    return (
        f"  {s.vehicle_id:<8} "
        f"{s.validation_status.value:<7} "
        f"N={s.position_ned_m.x:7.1f} "
        f"E={s.position_ned_m.y:7.1f} "
        f"alt={s.altitude_local_m:5.1f} "
        f"hdg={s.heading_deg:5.1f} "
        f"gs={s.ground_speed_mps:4.1f} "
        f"{s.ground_state.value}"
    )


async def monitor(adapter, stop, phase, states):
    t0 = time.monotonic()

    while not stop.is_set():
        lines = []

        for vid in VEHICLES:
            try:
                lines.append(
                    row(adapter.get_snapshot(vid))
                    + f"  [{states[vid]['phase']}]"
                )
            except Exception as err:
                lines.append(f"  {vid}: snapshot error: {err}")

        print(f"--- t+{time.monotonic() - t0:6.1f}s phase: {phase['name']}")
        print("\n".join(lines))

        try:
            await asyncio.wait_for(stop.wait(), 2.0)
        except asyncio.TimeoutError:
            pass


async def sample_separation(adapter, tracker, stop, phase):
    while not stop.is_set():
        tracker.update(adapter.latest_poses(), phase=phase["name"])

        try:
            await asyncio.wait_for(
                stop.wait(), 1.0 / SEPARATION_SAMPLE_HZ
            )
        except asyncio.TimeoutError:
            pass


# ---------------- Sensor loop ----------------
async def sensor_loop(adapter, vid, sector, target, states, sync, found, stop):
    """
    Runs continuously for one drone:
      - checks the sector boundary (geofence) while searching
      - compares its north/east position to the target; the first drone within
        +/- DETECTION_MARGIN_M on both axes claims the find and raises `found`
    Height is never used.
    """
    st = states[vid]
    prev = None

    while not stop.is_set():
        try:
            s = adapter.get_snapshot(vid)
        except Exception:
            s = None

        if (
            s is not None
            and st["airborne"]
            and s.validation_status == ValidationStatus.VALID
        ):
            n, e = s.position_ned_m.x, s.position_ned_m.y
            cur = (n, e)
            if prev is None:
                prev = cur

            if st["phase"] == "transit" and sector.contains(n, e):
                st["phase"] = "search"

            if st["phase"] == "search":
                out = sector.excursion(n, e)
                if out > GEOFENCE_TOL_M:
                    st["breach_samples"] += 1
                    st["max_excursion_m"] = max(st["max_excursion_m"], out)

            hit = None if found.is_set() else segment_box_hit(prev, cur, target)
            if hit is not None:
                claim_find(
                    vid,
                    {
                        "at": time.monotonic(),
                        "n": hit[0],
                        "e": hit[1],
                        "dn": hit[0] - target[0],
                        "de": hit[1] - target[1],
                    },
                    states, sync, found,
                )

            prev = cur

        try:
            await asyncio.wait_for(stop.wait(), 1.0 / SENSOR_HZ)
        except asyncio.TimeoutError:
            pass


# ---------------- Flight helpers ----------------
async def takeoff_all(adapter):
    async def one(vid):
        drone = adapter.drone(vid)
        drone.enable_api_control()
        drone.arm()
        await (await drone.takeoff_async())

    await run_all_or_cancel(one(v) for v in VEHICLES)

    deadline = time.monotonic() + 15.0
    pending = set(VEHICLES)

    while pending and time.monotonic() < deadline:
        for vid in list(pending):
            snapshot = adapter.get_snapshot(vid)

            if (
                snapshot.validation_status == ValidationStatus.VALID
                and snapshot.ground_state == GroundState.AIRBORNE
            ):
                pending.discard(vid)

        await asyncio.sleep(0.1)

    if pending:
        raise RuntimeError(
            f"Drones did not become airborne: {sorted(pending)}"
        )


async def interruptible(coro, event):
    """
    Run coro until it finishes or event is set.
    Returns (result, interrupted). If interrupted, coro is cancelled.
    """
    task = asyncio.ensure_future(coro)
    waiter = asyncio.ensure_future(event.wait())

    try:
        done, _ = await asyncio.wait(
            {task, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        task.cancel()
        waiter.cancel()
        raise

    if task in done:
        waiter.cancel()
        return task.result(), False

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    return None, True


async def hold_position(adapter, executor, vid):
    """Stop and hover in place."""
    try:
        await (await adapter.drone(vid).hover_async())
    except Exception:
        # Fallback: command a move to the current position.
        s = adapter.get_snapshot(vid)
        await executor.move_to_position(
            vid, s.position_ned_m.x, s.position_ned_m.y,
            s.altitude_local_m, 1.0, reason="hold position",
        )


async def land_descent(adapter, executor, vid, launch_alt, results, phase):
    """Descend toward the ground over about LAND_S seconds."""
    target = round(launch_alt + DESCEND_ABOVE_LAUNCH, 2)
    current = adapter.get_snapshot(vid).altitude_local_m
    speed = max(0.3, (current - target) / LAND_S)

    phase["steps"][vid] = "descend"
    result = await executor.change_altitude(
        vid, target, speed_mps=speed, reason="landing descent",
    )
    results[vid].append(("descend", result))


# ---------------- Mission ----------------
async def stack_over_target(adapter, executor, vid, home, st, sync, results, phase):
    """
    Finder: stop, then center over the found point. Everyone else: go to their
    stack height first (so the drones are never at the same height), then fly
    to the found point. Once all are stacked they hover, spread out, and land.
    """
    order = sync["order"]
    idx = order.index(vid)
    altitude = stack_altitude(idx)
    rn, re_ = sync["rally"]
    is_finder = vid == sync["finder"]

    async def run(name, coro):
        phase["steps"][vid] = name
        result = await coro
        results[vid].append((name, result))

    try:
        if is_finder:
            st["phase"] = "found"
            phase["steps"][vid] = "target found - hovering"
            await hold_position(adapter, executor, vid)
            await run(
                "center over found point",
                executor.move_to_position(
                    vid, rn, re_, altitude, 1.5, reason="center over found point",
                ),
            )
        else:
            st["phase"] = "responding"
            await run(
                "to stack height",
                executor.change_altitude(
                    vid, altitude, speed_mps=2.0, reason="stack height",
                ),
            )
            await run(
                "to found point",
                executor.move_to_position(
                    vid, rn, re_, altitude, SPEED,
                    reason="respond to found target",
                ),
            )

        await hold_position(adapter, executor, vid)
        st["phase"] = "stacked"
    finally:
        # Always count in, so one failure cannot hold up the others.
        sync["arrived"] += 1
        if sync["arrived"] >= len(order):
            sync["stack_ready"].set()

    phase["steps"][vid] = f"stacked at {altitude:.1f} m"
    await sync["stack_ready"].wait()
    await asyncio.sleep(STACK_HOLD_S)

    if DISPERSE_BEFORE_LANDING:
        offset = DISPERSE_SPACING_M * (idx - (len(order) - 1) / 2)
        await run(
            "spread out to land",
            executor.move_to_position(
                vid, rn, re_ + offset, altitude, SPEED,
                reason="spread out before landing",
            ),
        )

    st["phase"] = "landing"
    await land_descent(adapter, executor, vid, home[2], results, phase)


async def search_mission(
    adapter, executor, vid, home, search, st, found, sync, results,
    phase, t_limit,
):
    launch_alt = home[2]
    altitude = ALTITUDES[vid]
    outcome = "covered"

    async def command(name, coro):
        """Returns (ok, interrupted)."""
        phase["steps"][vid] = name
        result, interrupted = await interruptible(coro, found)
        if interrupted:
            return False, True
        results[vid].append((name, result))
        return result.status == CommandStatus.SUCCEEDED, False

    try:
        ok, interrupted = await command(
            "climb",
            executor.change_altitude(
                vid, altitude, speed_mps=2.0, reason="search altitude",
            ),
        )
        if interrupted:
            outcome = "found"
        elif not ok:
            outcome = "climb failed"
        else:
            while True:
                if found.is_set():
                    outcome = "found"
                    break
                if search.remaining == 0:
                    outcome = "covered"
                    break
                if time.monotonic() > t_limit:
                    outcome = "timeout"
                    break

                snap = adapter.get_snapshot(vid)
                k, start, end = search.pick_next(
                    snap.position_ned_m.x, snap.position_ned_m.y
                )

                # Get to the near end of the lane, then sweep it end to end.
                ok, interrupted = await command(
                    f"lane {k} start",
                    executor.move_to_position(
                        vid, start[0], start[1], altitude, SPEED,
                        reason="move to lane start",
                    ),
                )
                if interrupted:
                    outcome = "found"
                    break
                if not ok:
                    outcome = "move failed"
                    break

                ok, interrupted = await command(
                    f"lane {k} sweep",
                    executor.move_to_position(
                        vid, end[0], end[1], altitude, SPEED,
                        reason="sweep lane",
                    ),
                )
                if interrupted:
                    outcome = "found"
                    break
                if not ok:
                    outcome = "move failed"
                    break

                search.mark_lane(k)
    finally:
        st["outcome"] = outcome

    if outcome == "found" and vid in sync["order"]:
        await stack_over_target(
            adapter, executor, vid, home, st, sync, results, phase
        )
    else:
        st["phase"] = "landing"
        await land_descent(adapter, executor, vid, launch_alt, results, phase)


async def logged(vid, coro):
    """Print the real exception from a mission before run_all_or_cancel hides it."""
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        print(f"\n[{vid}] MISSION EXCEPTION:")
        traceback.print_exc()
        raise


# ---------------- Main ----------------
async def main() -> int:
    rng = random.Random(RANDOM_SEED)

    sectors = make_sectors(len(VEHICLES))
    half = AREA_SIZE_M / 2
    target = (
        AREA_CENTER[0] + rng.uniform(-half, half),
        AREA_CENTER[1] + rng.uniform(-half, half),
    )
    owner = next(
        v for v, sec in zip(VEHICLES, sectors)
        if sec.contains(target[0], target[1])
    )

    searches = {
        v: SectorSearch(sec, random.Random(rng.random()))
        for v, sec in zip(VEHICLES, sectors)
    }
    states = {
        v: {
            "phase": "transit",
            "airborne": False,
            "outcome": None,
            "detection": None,
            "breach_samples": 0,
            "max_excursion_m": 0.0,
        }
        for v in VEHICLES
    }

    results = {v: [] for v in VEHICLES}
    phase = {"name": "connect", "steps": {}}

    print(f"Area: {AREA_SIZE_M:.0f} x {AREA_SIZE_M:.0f} m centered on {AREA_CENTER}")
    for v, sec in zip(VEHICLES, sectors):
        print(
            f"  {v}: N [{sec.n_min:.0f}, {sec.n_max:.0f}]  "
            f"E [{sec.e_min:.0f}, {sec.e_max:.0f}]  "
            f"{searches[v].lanes} lanes, alt {ALTITUDES[v]:.0f} m"
        )
    print(
        f"Target (hidden from drones): N={target[0]:.1f} E={target[1]:.1f} "
        f"(in {owner}'s sector), detected within +/- {DETECTION_MARGIN_M:.1f} m "
        "on N and E"
    )

    reports = []
    sensor_tasks = []

    with ProjectAirSimAdapter(
        vehicle_ids=VEHICLES,
        scene=SCENE,
    ) as adapter:
        await asyncio.sleep(0.5)

        executor = ProjectAirSimExecutor(adapter)
        stop = asyncio.Event()
        tracker = SeparationTracker(VEHICLES)

        mon = asyncio.create_task(monitor(adapter, stop, phase, states))
        sep = asyncio.create_task(
            sample_separation(adapter, tracker, stop, phase)
        )

        found = asyncio.Event()          # set by the first drone to detect
        sync = {
            "finder": None,
            "rally": None,
            "order": [],
            "arrived": 0,
            "stack_ready": asyncio.Event(),
        }
        t0 = time.monotonic()

        try:
            homes = {}
            for vid in VEHICLES:
                s = adapter.get_snapshot(vid)
                homes[vid] = (
                    s.position_ned_m.x,
                    s.position_ned_m.y,
                    s.altitude_local_m,
                )

            phase["name"] = "takeoff"
            await takeoff_all(adapter)

            for v in VEHICLES:
                states[v]["airborne"] = True

            sensor_tasks = [
                asyncio.create_task(
                    sensor_loop(
                        adapter, v, sec, target, states, sync, found, stop,
                    )
                )
                for v, sec in zip(VEHICLES, sectors)
            ]

            phase["name"] = "search"
            t0 = time.monotonic()
            t_limit = t0 + MISSION_TIMEOUT_S

            await run_all_or_cancel(
                [
                    logged(v, search_mission(
                        adapter, executor, v, homes[v], searches[v],
                        states[v], found, sync, results, phase,
                        t_limit,
                    ))
                    for v in VEHICLES
                ]
            )

        finally:
            phase["name"] = "safe shutdown"
            phase["steps"] = {}

            reports = await asyncio.gather(
                *(
                    safe_shutdown(
                        adapter.drone(v),
                        adapter=adapter,
                        vehicle_id=v,
                        log=lambda m, v=v: print(f"[{v}] {m}"),
                    )
                    for v in VEHICLES
                ),
                return_exceptions=True,
            )

            stop.set()
            await mon
            await sep
            if sensor_tasks:
                await asyncio.gather(*sensor_tasks, return_exceptions=True)

            tracker.finish()
            collisions = {
                v: adapter.collision_log(v) for v in VEHICLES
            }

    print("\n================ SEARCH RESULT ================")
    print(f"Target: N={target[0]:.1f} E={target[1]:.1f} (in {owner}'s sector)")

    if sync["finder"]:
        v = sync["finder"]
        d = states[v]["detection"]
        print(
            f"FOUND by {v} after {d['at'] - t0:.1f}s at "
            f"N={d['n']:.1f} E={d['e']:.1f} "
            f"(dN={d['dn']:+.2f} m, dE={d['de']:+.2f} m from target)"
        )
        print(
            f"Stack over N={sync['rally'][0]:.1f} E={sync['rally'][1]:.1f}: "
            + ", ".join(
                f"{u} {stack_altitude(i):.1f} m"
                for i, u in enumerate(sync["order"])
            )
        )
    else:
        print("Target NOT found. All drones covered their sectors and landed.")

    print("\n================ PER DRONE ================")
    for v in VEHICLES:
        st = states[v]
        done = searches[v].lanes - searches[v].remaining
        print(
            f"{v}: {st['outcome']}, lanes swept {done}/{searches[v].lanes} "
            f"({searches[v].fraction * 100:.0f}%), geofence breach samples "
            f"{st['breach_samples']} (max {st['max_excursion_m']:.1f} m outside)"
        )

    failures = [
        (v, name, r)
        for v in VEHICLES
        for name, r in results[v]
        if r.status != CommandStatus.SUCCEEDED
    ]
    if failures:
        print("\n================ FAILED COMMANDS ================")
        for v, name, r in failures:
            print(f"{v} {name}: {r.status.value} {r.reason}")

    print("\n================ SHUTDOWN ================")
    for vid, report in zip(VEHICLES, reports):
        if isinstance(report, Exception):
            print(f"{vid}: shutdown error: {report}")
        else:
            print(f"{vid}: {'disarmed' if report.disarmed else 'UNRESOLVED'}")

    print("\n================ COLLISIONS ================")
    for vid, events in collisions.items():
        print(f"{vid}: {events}")

    print("\n================ SEPARATION ================")
    print(format_report(assess(tracker, collisions, VEHICLES, MIN_SEPARATION_M)))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
