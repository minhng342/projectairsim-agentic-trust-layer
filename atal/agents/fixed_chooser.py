"""
The fixed chooser (Phase 1, no LLM): the next sector is the first one still to search in the
role's order. With the invariant guard on, it is the first one the guard allows; if none
fits in the mission time left, the drone flies home.

    loop = SearchLoop("Drone1", mission, feed, port, belief, chooser=FixedChooser())

Every chooser has `kind` (the name it is logged under, see atal/agents/search_loop.py),
attach(loop) and choose(remaining). The LLM chooser (atal/agents/llm_chooser.py) is the
other one, and falls back to this same choice when the model fails.
"""
from atal.core.action import ReturnHome, SweepSector


class FixedChooser:
    kind = "fixed"

    def __init__(self):
        self.loop = None

    def attach(self, loop) -> None:
        self.loop = loop

    async def choose(self, remaining: list[str]) -> SweepSector | ReturnHome:
        """The next sector in the fixed order (with a guard: the next one the guard allows,
        else return home)."""
        loop = self.loop
        reason = f"next sector in the fixed {loop.mission.search.sector_order} order"
        if loop.guard is None:
            return SweepSector(sector_id=remaining[0], reason=reason)
        _, state = await loop.observe()
        if state is None:
            return SweepSector(sector_id=remaining[0], reason=reason)
        for sid in remaining:
            a = SweepSector(sector_id=sid, reason=reason)
            if loop.guard.check(a.to_proposed(loop.vehicle_id), state.x_m, state.y_m,
                                state.sim_time_s).ok:
                return a
        loop.return_outcome = "time_limit"
        return ReturnHome(reason="no unsearched sector fits in the mission time left")
