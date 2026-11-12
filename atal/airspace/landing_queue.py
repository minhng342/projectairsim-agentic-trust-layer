"""How long startups and landings may wait for their reservations (recovery policy section
11a). Every number of the landing queue's timing is computed here, from the team size,
the levels and the bounds, so the policy stays easy to explain.

Pads closer together than d_h share one vertical column, so a startup or a landing waits
for every teammate's startup or landing through that column. One such move from the top
level holds the column for at most its validity time V_max (V = h / v_z + settle_s +
margin_s, atal/airspace/bounds.py). The budget is therefore

    per_move_s = V_max + MARGIN_S
    queue_s    = n x per_move_s

TOP LEVEL. For a landing, the top level is the highest altitude any drone of the team can
start a landing from: its highest transit level (a drone that went up in an obstacle
recovery and could not come down flies home and lands from there), or its search level
when it has no transit level below the ceiling. For a startup, the top level is the
highest search level (startups end at a search level).

DEFAULT, NOT A PROVEN BOUND. n x (V_max + 30 s) covers the case the queue's order gives:
requests that conflict with a waiting one never overtake it (only the drone the waiting
one is held up by may go first, so it can move out of the way), so each of the other
n - 1 drones moves through the column at most once ahead of it, and each of those moves
ends within its own validity time or becomes a keep-out. It is not a worst-case bound
for everything: a keep-out that never clears, a teammate holding inside the column, or
a stale track can make any wait longer, and then the request expires and the drone ends
UNRESOLVED (it never lands without a reservation and is never disarmed in the air).

ONE CLEANUP DEADLINE. A drone's cleanup begins the first time it waits for the pad zone or
asks to land, and its deadline is then fixed once: cleanup_deadline = start + queue_s.
Both waits draw on it. The pad-zone wait is capped at the deadline minus one per_move_s
(and at the configured pad-zone wait), so the landing request always keeps time for one
teammate's move; the landing request's wait_until is the deadline itself. Polling,
retries after a "not holding yet" answer, and repeated requests never move it.

The fixed max_hold_s of the recovery states (60 s, policy section 7) is not changed by
any of this.
"""
from dataclasses import asdict, dataclass

from atal.airspace.bounds import Bounds, Levels

MARGIN_S = 30.0          # added to V_max per move ahead in the queue


def landing_top_level_m(levels: Levels) -> float:
    """The highest altitude any team drone can start a landing from (see TOP LEVEL)."""
    return max(max(levels.of(i)) for i in range(levels.team_size))


def startup_top_level_m(levels: Levels) -> float:
    """The highest search level: where the highest startup ends."""
    return max(levels.search(i) for i in range(levels.team_size))


@dataclass(frozen=True)
class QueueBudget:
    team_size: int
    landing_top_level_m: float
    landing_v_max_s: float
    landing_per_move_s: float
    landing_queue_s: float       # the cleanup budget: pad-zone wait + landing request wait
    startup_top_level_m: float
    startup_v_max_s: float
    startup_queue_s: float       # how long a startup request may wait
    pad_zone_s: float            # the configured pad-zone wait (sim s), capped by the above

    @classmethod
    def derive(cls, levels: Levels, bounds: Bounds, pad_zone_s: float) -> "QueueBudget":
        n = levels.team_size
        top_l, top_s = landing_top_level_m(levels), startup_top_level_m(levels)
        v_l, v_s = bounds.validity_s(top_l), bounds.validity_s(top_s)
        return cls(team_size=n, landing_top_level_m=top_l, landing_v_max_s=round(v_l, 2),
                   landing_per_move_s=round(v_l + MARGIN_S, 2),
                   landing_queue_s=round(n * (v_l + MARGIN_S), 2),
                   startup_top_level_m=top_s, startup_v_max_s=round(v_s, 2),
                   startup_queue_s=round(n * (v_s + MARGIN_S), 2), pad_zone_s=pad_zone_s)

    def record(self) -> dict:
        return asdict(self) | {"margin_s": MARGIN_S, "rule": "n x (V_max + margin); default "
                               "budget, not a proven worst-case bound"}


class CleanupDeadline:
    """One drone's overall cleanup deadline, fixed the first time start() is called."""

    def __init__(self, budget: QueueBudget, now):
        self.budget, self._now = budget, now
        self.started_at: float | None = None
        self.deadline: float | None = None

    def start(self) -> float:
        if self.deadline is None:
            self.started_at = self._now()
            self.deadline = self.started_at + self.budget.landing_queue_s
        return self.deadline

    def left_s(self) -> float:
        return max(0.0, self.start() - self._now())

    def pad_zone_wait_s(self) -> float:
        """Sim seconds the drone may wait for the pad zone now: never past the deadline
        minus one move, never more than the configured pad-zone wait."""
        return max(0.0, min(self.budget.pad_zone_s,
                            self.left_s() - self.budget.landing_per_move_s))
