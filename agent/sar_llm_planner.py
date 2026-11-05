"""
Phase 2 planner: one drone, the next search action chosen by an LLM (Pydantic AI, any model).

    planner = LLMSectorPlanner("Drone1", mission, feed, port, belief, model=model, guard=guard)
    result = await planner.run()

It is FixedSectorPlanner (agent/sector_planner.py) with one box swapped: choose(). The
climb, the port, the executor, the memory, stop on detection / time limit and the return
home are the same code, so a run differs from Phase 1 only in which sector comes next.

One decision = one short Pydantic AI run, no tools:
    input   the situation: own position, time used / left, a map of the grid from the
            drone's memory, the nearest unsearched sectors with distance and an estimated
            time, and the prior if the mission enables one (computed here, from the mission)
    output  SweepSector | ReturnHome (typed; models/sar.py)
    check   the invariant guard (gate/sar_guard.py) as the output validator: a choice that
            breaks a rule goes back to the model with the rule's message, up to
            `retries` times
    fallback  if the model fails (no valid choice after the retries, timeout, API error),
            the drone takes the Phase 1 choice (next unsearched sector in serpentine order)
            and the decision is recorded with fallback set. A failing model never leaves the
            drone without a next step.

Memory is the drone's LocalBelief, not the chat: each decision starts a fresh run whose
input is rebuilt from the belief, so a long search does not grow the prompt and what the
model knows is exactly what the memory file says.

What the model can see: its own observation, its own memory, the frozen mission. Never the
scenario (victim position): a detection still reaches the planner only through its sensor.
"""
import asyncio
import json
import math
import time

from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded

from agent.sector_planner import FixedSectorPlanner
from observability import span
from gate.sar_guard import SarGuard
from models.sar import Mission, ReturnHome, SweepSector

NEAREST_LISTED = 8
PRIOR_LISTED = 8


TEAM_PROMPT = """

You are one drone of a team. Teammates search at the same time and tell each other what they do;
their messages update your memory ('p' on the map = a teammate reported it clear). The situation lists
your teammates, what each is sweeping now, and the sectors assigned to you (a Voronoi split of what is
still open: the sectors nearer you than any teammate). Choose only from your assigned sectors; the
guard refuses others. Teammates' reports are claims, not your own observations."""


def system_prompt(vehicle_id: str, mission: Mission, team: bool = False) -> str:
    g = mission.area.sectors
    sx, sy = mission.sector_size_m
    last_letter = chr(ord("A") + g.y - 1)
    return f"""You are the search planner of search-and-rescue drone {vehicle_id}. A person is missing
somewhere in the search area. Your job is to find them as fast as possible.

The area is a grid of {g.x} x {g.y} sectors, each {sx:g} x {sy:g} m. Sector ids read like a chess board
on a north-up map: the letter is the column from west to east (A..{last_letter}), the number is the row
from south to north (1..{g.x}). A1 is the south-west corner. Positions are north_m / east_m in meters.

Each turn you choose ONE next action:
- sweep_sector(sector_id, reason): the autopilot flies there and sweeps the whole sector in lanes at
  {mission.flight.altitude_m:g} m and {mission.flight.speed_m_s:g} m/s. The drone's camera checks for the
  person automatically the whole time; if it sees them the search stops by itself.
- return_home(reason): fly back to the launch point. Only when no unsearched sector fits in the time
  left; the guard refuses it otherwise.

Search well:
- Never choose a sector your memory marks swept.
- Flying between sectors costs time that finds no one: prefer the nearest unsearched sectors and
  sweep in a connected pattern; avoid leaving single unsearched sectors behind that you must fly back
  to later.
- If a prior is given, it says where the person is more likely to be: search likely sectors early,
  but weigh that against long transits.
- A guard checks every choice. If it refuses one, it tells you why; fix the choice.
Keep the reason to one short sentence.""" + (TEAM_PROMPT if team else "")


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def prior_mass(mission: Mission) -> dict[str, float]:
    """Probability of each sector under the mission's Gaussian prior (normalized to the grid);
    {} when the prior is disabled."""
    p = mission.prior
    if not p.enabled or p.center is None or p.sigma_m is None:
        return {}
    cx, cy, sx, sy = p.center.x_m, p.center.y_m, p.sigma_m.x, p.sigma_m.y
    raw = {}
    for s in mission.grid.sectors():
        mx = _phi((s.x_max - cx) / sx) - _phi((s.x_min - cx) / sx)
        my = _phi((s.y_max - cy) / sy) - _phi((s.y_min - cy) / sy)
        raw[s.id] = mx * my
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def memory_map(mission: Mission, belief, here: str | None) -> str:
    """Rows north (top) to south; '.' unsearched, '#' swept, '~' partly swept."""
    g = mission.area.sectors
    letters = [chr(ord("A") + j) for j in range(g.y)]
    lines = ["    " + " ".join(letters)]
    for ix in reversed(range(g.x)):
        row = []
        for L in letters:
            sid = f"{L}{ix + 1}"
            st = belief.status(sid).value
            c = {"swept": "#", "partial": "~", "in_progress": "~", "peer_cleared": "p"}.get(st, ".")
            row.append(c)
        lines.append(f"{ix + 1:>3} " + " ".join(row))
    if here:
        lines.append(f"(you are over {here})")
    return "\n".join(lines)


def situation(planner: "LLMSectorPlanner", state, remaining: list[str]) -> dict:
    m, guard = planner.mission, planner.guard
    here_xy = (state.x_m, state.y_m)
    left = guard.remaining_s(state.sim_time_s)
    near = []
    for sid in remaining:
        c = m.grid.sector(sid).center
        near.append((math.dist(here_xy, c), sid))
    near.sort()
    nearest = [{"sector_id": sid, "distance_m": round(d),
                "est_time_s": round(guard.sweep_time_s(sid, here_xy))}
               for d, sid in near[:NEAREST_LISTED]]
    out = {
        "you": {"north_m": round(state.x_m, 1), "east_m": round(state.y_m, 1),
                "altitude_m": round(state.altitude_m, 1), "over_sector": state.sector_id},
        "time": {"used_s": round(m.termination.max_mission_time_s - left),
                 "left_s": round(left), "limit_s": m.termination.max_mission_time_s},
        "memory": {"swept": len(planner.belief.done()), "unsearched": len(remaining),
                   "total": len(m.grid.ids())},
        "last_sector": planner.last_sector,
        "nearest_unsearched": nearest,
    }
    if hasattr(planner, "swarm_situation"):
        out["team"] = planner.swarm_situation()
    prior = planner.prior
    if prior:
        likely = sorted(((p, s) for s, p in prior.items() if s in remaining), reverse=True)
        out["prior_likely_unsearched"] = [{"sector_id": s, "probability": round(p, 3)}
                                          for p, s in likely[:PRIOR_LISTED]]
    return out


def decision_prompt(planner, state, remaining) -> str:
    sit = situation(planner, state, remaining)
    return ("Situation (from your own observation and memory):\n```json\n"
            + json.dumps(sit) + "\n```\n\nMemory map (north at the top; '.' unsearched, "
            "'#' swept, '~' partly swept, 'p' cleared by a teammate):\n```\n"
            + memory_map(planner.mission, planner.belief, state.sector_id)
            + "\n```\nChoose the next action.")


class LLMSectorPlanner(FixedSectorPlanner):
    name = "LLMSectorPlanner"

    def __init__(self, vehicle_id, mission, feed, port, belief, *, model, guard: SarGuard,
                 model_label: str | None = None, model_settings: dict | None = None,
                 retries: int = 2, decision_timeout_s: float = 60.0, clock=time.monotonic, **kw):
        super().__init__(vehicle_id, mission, feed, port, belief, guard=guard, **kw)
        self.model, self.guard = model, guard
        self.model_label = model_label or getattr(model, "model_name", str(model))
        self.model_settings = dict(model_settings or {})
        self.retries, self.decision_timeout_s, self._clock = retries, decision_timeout_s, clock
        self.prior = prior_mass(mission)
        self.last_sector: str | None = None
        self.decisions = 0
        self.fallbacks = 0
        self.return_outcome = "returned"
        self._agent = self._build_agent()
        self._state = None
        self._rejections: list[dict] = []

    # ------------------------------------------------------------ agent
    def _build_agent(self) -> Agent:
        agent = Agent(self.model, output_type=[SweepSector, ReturnHome],
                      instructions=system_prompt(self.vehicle_id, self.mission,
                                                 team=getattr(self, "team", False)),
                      retries=self.retries, name=f"search-{self.vehicle_id}")
        planner = self

        @agent.output_validator
        def guard_check(ctx: RunContext[None], out):
            st = planner._state
            v = planner.guard.check(out.to_proposed(planner.vehicle_id), st.x_m, st.y_m,
                                    st.sim_time_s)
            if not v.ok:
                planner._rejections.append({"choice": _describe(out), "code": v.code,
                                            "message": v.message})
                raise ModelRetry(f"Refused by the guard ({v.code}): {v.message}")
            return out
        return agent

    # ------------------------------------------------------------ the swapped box
    async def run(self):
        self.guard.start_sim_s = None
        return await super().run()

    async def choose(self, remaining: list[str]):
        obs, state = await self._observe()
        fallback_choice = self._fallback(remaining, state)
        if state is None:
            return fallback_choice
        feasible = self.guard.feasible((state.x_m, state.y_m), state.sim_time_s)
        if not feasible:
            self.return_outcome = "time_limit"
            self._record("llm_decision", decision=self.decisions + 1, skipped=True,
                         choice=_describe(ReturnHome(reason="no unsearched sector fits")),
                         reason="no unsearched sector fits in the time left")
            return ReturnHome(reason="no unsearched sector fits in the mission time left")
        self._state, self._rejections = state, []
        self.decisions += 1
        prompt = decision_prompt(self, state, [s for s in remaining if s in self.guard.cells])
        t0 = self._clock()
        out, error, usage = None, None, None
        try:
            with span("sar decision {decision}", decision=self.decisions,
                      vehicle_id=self.vehicle_id, unsearched=len(remaining),
                      sim_s=state.sim_time_s):
                res = await asyncio.wait_for(
                    self._agent.run(prompt, model_settings=self.model_settings or None,
                                    usage_limits=UsageLimits(request_limit=self.retries + 2)),
                    timeout=self.decision_timeout_s)
            out, u = res.output, res.usage
            usage = {"requests": u.requests, "input_tokens": u.input_tokens,
                     "output_tokens": u.output_tokens}
        except asyncio.TimeoutError:
            error = f"timeout ({self.decision_timeout_s:g} s)"
        except (UnexpectedModelBehavior, UsageLimitExceeded) as err:
            error = f"{type(err).__name__}: {str(err)[:300]}"
        except ModelAPIError as err:
            error = f"{type(err).__name__}: {str(err)[:300]}"
        latency = round(self._clock() - t0, 3)
        fallback = out is None
        if fallback:
            self.fallbacks += 1
            out = fallback_choice
        if isinstance(out, ReturnHome):
            self.return_outcome = "returned"
        else:
            self.last_sector = out.sector_id
        self._record("llm_decision", decision=self.decisions, model=self.model_label,
                     latency_s=latency, choice=_describe(out), reason=out.reason,
                     fallback=fallback, error=error, prompt=prompt, guard_rejections=self._rejections, usage=usage,
                     sim_s=state.sim_time_s, unsearched=len(remaining),
                     serpentine_choice=remaining[0])
        self._log(f"  [{time.strftime('%H:%M:%S')}] {self.vehicle_id}: decision "
                  f"{self.decisions} -> {_describe(out)} ({latency:.1f} s"
                  + (f", {len(self._rejections)} refused" if self._rejections else "")
                  + (f", FALLBACK: {error}" if fallback else "") + ")"
                  + (f"\n      reason: {out.reason}" if not fallback else ""))
        return out

    def _fallback(self, remaining, state):
        """Phase 1's choice: the next sector in the fixed order that the guard allows."""
        for sid in remaining:
            if state is None:
                break
            a = SweepSector(sector_id=sid, reason="fallback: next sector in the fixed order")
            if self.guard.check(a.to_proposed(self.vehicle_id), state.x_m, state.y_m,
                                state.sim_time_s).ok:
                return a
        return SweepSector(sector_id=remaining[0], reason="fallback: next sector in the fixed "
                                                          "order")


def _describe(a) -> str:
    return f"sweep_sector {a.sector_id}" if isinstance(a, SweepSector) else "return_home"
