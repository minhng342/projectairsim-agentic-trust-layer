"""
The numbers the transition manager's rules rest on (recovery policy sections 2 and 8),
and the altitude levels of a team.

    b = Bounds()                    # dev values
    b.d_h                           # 8.0 m: horizontal margin between two holding drones
    b.r_stop                        # 21.5 m: stop-line radius around a vertical move
    b.exclusion_radius(30.0)        # 243.5 m: radius at grant time for a 30 m move
    levels = Levels(base_m=150, spacing_m=10, team_size=3, max_transit_level=2,
                    ceiling_m=240, separation_m=5, e_z=1)
    levels.search(1), levels.transit(1, k=1)      # 160, 190

Every value is configuration with a stated source; the formulas are here once, so tests
can check them and a changed bound changes every radius that depends on it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Bounds:
    separation_m: float = 5.0        # separation_cruise_m, the cruise floor
    e_z: float = 1.0                 # per drone: altitude tolerance while holding
    e_h: float = 1.5                 # per drone: position tolerance while holding
    v_h: float = 6.0                 # horizontal speed bound (flight speed x 1.2)
    v_z: float = 1.0                 # vertical speed used for a move's duration
    settle_s: float = 2.0            # confirmed holding at the target before COMPLETED
    a_s: float = 1.0                 # surveillance age allowance
    latency_s: float = 0.5           # L: detection + command latency
    brake_mps2: float = 4.0          # b: braking
    margin_s: float = 5.0            # added to a move's validity time
    confirm_hold_s: float = 1.0      # how long "confirmed holding" must last
    hold_h_mps: float = 0.3          # horizontal speed below which a drone is still
    hold_v_mps: float = 0.2          # vertical speed below which a drone is still
    start_window_s: float = 2.0      # a grant must be started within this

    @property
    def d_h(self) -> float:
        """One horizontal margin between drones holding or moving only vertically."""
        return self.separation_m + 2 * self.e_h

    @property
    def stopping_m(self) -> float:
        return self.v_h ** 2 / (2 * self.brake_mps2)

    @property
    def r_stop(self) -> float:
        """Stop-line: separation + 2 e_h + v_h (a_s + L) + v_h^2 / 2b."""
        return self.d_h + self.v_h * (self.a_s + self.latency_s) + self.stopping_m

    def validity_s(self, span_m: float) -> float:
        """V = h / v_z + settle_s + margin."""
        return span_m / self.v_z + self.settle_s + self.margin_s

    def exclusion_radius(self, span_m: float) -> float:
        """r at grant: separation + 2 e_h + v_h (V + a_s + L) + v_h^2 / 2b."""
        return (self.d_h + self.v_h * (self.validity_s(span_m) + self.a_s + self.latency_s)
                + self.stopping_m)


@dataclass(frozen=True)
class Levels:
    """Search level S_i = base + i s; transit level T_i(k) = base + (k n + i) s."""
    base_m: float
    spacing_m: float
    team_size: int
    max_transit_level: int
    ceiling_m: float | None
    separation_m: float = 5.0
    e_z: float = 1.0

    def search(self, i: int) -> float:
        return self.base_m + i * self.spacing_m

    def transit(self, i: int, k: int) -> float | None:
        """T_i(k), or None when k is past max_transit_level or the level would be above
        the ceiling (the "no suitable altitude" outcome)."""
        if k < 1 or k > self.max_transit_level:
            return None
        alt = self.base_m + (k * self.team_size + i) * self.spacing_m
        if self.ceiling_m is not None and alt + self.separation_m > self.ceiling_m:
            return None
        return alt

    def of(self, i: int) -> list[float]:
        """Every level drone i may hold at."""
        out = [self.search(i)]
        for k in range(1, self.max_transit_level + 1):
            t = self.transit(i, k)
            if t is not None:
                out.append(t)
        return out

    def problems(self) -> list[str]:
        """Why these levels are not acceptable for this team (empty when they are)."""
        out = []
        if self.team_size > 1 and self.spacing_m < self.separation_m + 2 * self.e_z:
            out.append(f"layer spacing {self.spacing_m:g} m is below separation "
                       f"{self.separation_m:g} m + 2 x e_z {self.e_z:g} m")
        if self.ceiling_m is None:
            if self.team_size > 1:
                out.append("limits.altitude_ceiling_m is required for a team mission")
            return out
        top = self.base_m + ((self.max_transit_level * self.team_size)
                             + self.team_size - 1) * self.spacing_m
        if top + self.separation_m > self.ceiling_m:
            out.append(f"highest level {top:g} m + separation {self.separation_m:g} m is "
                       f"above the ceiling {self.ceiling_m:g} m for a team of "
                       f"{self.team_size}")
        return out
