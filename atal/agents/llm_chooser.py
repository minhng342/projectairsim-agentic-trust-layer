"""
The LLM chooser (Phase 2): an LLM picks the next search action (Pydantic AI, any model).

    loop = SearchLoop("Drone1", mission, feed, port, belief, guard=guard,
                      chooser=LLMChooser(model=model))
    result = await loop.run()

It plugs into the same search loop as the fixed chooser (atal/agents/search_loop.py). The
climb, the port, the executor, the memory, stop on detection / time limit and the return
home are the same code, so a run differs from Phase 1 only in which sector comes next.

One decision = one short Pydantic AI run, no tools:
    input   the situation: own position, time used / left, a map of the grid from the
            drone's memory, the nearest unsearched sectors with distance and an estimated
            time, and the prior if the mission enables one (computed here, from the mission)
    output  SweepSector | ReturnHome (typed; atal/core/action.py)
    check   the invariant guard (atal/trust_layer/invariant_guard.py) as the output validator: a choice that
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
scenario (victim position): a detection still reaches the loop only through its sensor.
"""
import asyncio
import json
import math
import time

from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded

from atal.core.action import ReturnHome, SweepSector
from atal.core.mission import Mission
from atal.observability import span

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


def situation(chooser, state, remaining: list[str]) -> dict:
    """The decision's input. `chooser` is anything with mission, belief, guard, last_sector
    and prior (the LLMChooser, or a stand-in in tests); blocked and team_situation() are
    used when it has them."""
    m, guard = chooser.mission, chooser.guard
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
        "memory": {"swept": len(chooser.belief.done()), "unsearched": len(remaining),
                   "total": len(m.grid.ids())},
        "last_sector": chooser.last_sector,
        "nearest_unsearched": nearest,
    }
    if getattr(chooser, "blocked", None):
        out["blocked_sectors"] = sorted(chooser.blocked)     # obstacle recovery: not offered
    team = chooser.team_situation() if hasattr(chooser, "team_situation") else None
    if team is not None:
        out["team"] = team
    prior = chooser.prior
    if prior:
        likely = sorted(((p, s) for s, p in prior.items() if s in remaining), reverse=True)
        out["prior_likely_unsearched"] = [{"sector_id": s, "probability": round(p, 3)}
                                          for p, s in likely[:PRIOR_LISTED]]
    return out


def decision_prompt(chooser, state, remaining) -> str:
    sit = situation(chooser, state, remaining)
    return ("Situation (from your own observation and memory):\n```json\n"
            + json.dumps(sit) + "\n```\n\nMemory map (north at the top; '.' unsearched, "
            "'#' swept, '~' partly swept, 'p' cleared by a teammate):\n```\n"
            + memory_map(chooser.mission, chooser.belief, state.sector_id)
            + "\n```\nChoose the next action.")


class LLMChooser:
    kind = "llm"

    def __init__(self, *, model, model_label: str | None = None,
                 model_settings: dict | None = None, retries: int = 2,
                 decision_timeout_s: float = 60.0, clock=time.monotonic):
        self.model = model
        self.model_label = model_label or getattr(model, "model_name", str(model))
        self.model_settings = dict(model_settings or {})
        self.retries, self.decision_timeout_s, self._clock = retries, decision_timeout_s, clock
        self.loop = None
        self.prior: dict[str, float] = {}
        self.last_sector: str | None = None
        self.decisions = 0
        self.fallbacks = 0
        self._agent = None
        self._state = None
        self._rejections: list[dict] = []

    def attach(self, loop) -> None:
        """Bind to the drone's search loop (it needs the loop's guard) and build the agent."""
        if loop.guard is None:
            raise ValueError("the LLM chooser needs the invariant guard (its output validator)")
        self.loop = loop
        self.prior = prior_mass(loop.mission)
        self._agent = self._build_agent()

    # ------------------------------------------------------------ what situation() reads
    @property
    def mission(self):
        return self.loop.mission

    @property
    def belief(self):
        return self.loop.belief

    @property
    def guard(self):
        return self.loop.guard

    @property
    def blocked(self):
        return self.loop.blocked

    def team_situation(self) -> dict | None:
        return self.loop.role.situation()

    # ------------------------------------------------------------ agent
    def _build_agent(self) -> Agent:
        loop = self.loop
        agent = Agent(self.model, output_type=[SweepSector, ReturnHome],
                      instructions=system_prompt(loop.vehicle_id, loop.mission,
                                                 team=loop.role.team),
                      retries=self.retries, name=f"search-{loop.vehicle_id}")
        chooser = self

        @agent.output_validator
        def guard_check(ctx: RunContext[None], out):
            st = chooser._state
            v = loop.guard.check(out.to_proposed(loop.vehicle_id), st.x_m, st.y_m,
                                 st.sim_time_s)
            if not v.ok:
                chooser._rejections.append({"choice": _describe(out), "code": v.code,
                                            "message": v.message})
                raise ModelRetry(f"Refused by the guard ({v.code}): {v.message}")
            return out
        return agent

    # ------------------------------------------------------------ one decision
    async def choose(self, remaining: list[str]):
        loop = self.loop
        obs, state = await loop.observe()
        fallback_choice = self._fallback(remaining, state)
        if state is None:
            return fallback_choice
        feasible = loop.guard.feasible((state.x_m, state.y_m), state.sim_time_s)
        if not feasible:
            loop.return_outcome = "time_limit"
            loop.record("llm_decision", decision=self.decisions + 1, skipped=True,
                        choice=_describe(ReturnHome(reason="no unsearched sector fits")),
                        reason="no unsearched sector fits in the time left")
            return ReturnHome(reason="no unsearched sector fits in the mission time left")
        self._state, self._rejections = state, []
        self.decisions += 1
        prompt = decision_prompt(self, state, [s for s in remaining if s in loop.guard.cells])
        t0 = self._clock()
        out, error, usage = None, None, None
        try:
            with span("sar decision {decision}", decision=self.decisions,
                      vehicle_id=loop.vehicle_id, unsearched=len(remaining),
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
            loop.return_outcome = "returned"
        else:
            self.last_sector = out.sector_id
        loop.record("llm_decision", decision=self.decisions, model=self.model_label,
                    latency_s=latency, choice=_describe(out), reason=out.reason,
                    fallback=fallback, error=error, prompt=prompt,
                    guard_rejections=self._rejections, usage=usage,
                    sim_s=state.sim_time_s, unsearched=len(remaining),
                    serpentine_choice=remaining[0])
        loop.log(f"  [{time.strftime('%H:%M:%S')}] {loop.vehicle_id}: decision "
                 f"{self.decisions} -> {_describe(out)} ({latency:.1f} s"
                 + (f", {len(self._rejections)} refused" if self._rejections else "")
                 + (f", FALLBACK: {error}" if fallback else "") + ")"
                 + (f"\n      reason: {out.reason}" if not fallback else ""))
        return out

    def _fallback(self, remaining, state):
        """Phase 1's choice: the next sector in the fixed order that the guard allows."""
        loop = self.loop
        for sid in remaining:
            if state is None:
                break
            a = SweepSector(sector_id=sid, reason="fallback: next sector in the fixed order")
            if loop.guard.check(a.to_proposed(loop.vehicle_id), state.x_m, state.y_m,
                                state.sim_time_s).ok:
                return a
        return SweepSector(sector_id=remaining[0], reason="fallback: next sector in the fixed "
                                                          "order")


def _describe(a) -> str:
    return f"sweep_sector {a.sector_id}" if isinstance(a, SweepSector) else "return_home"
