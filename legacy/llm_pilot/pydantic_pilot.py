"""
PydanticPilot: the pilot agent on Pydantic AI (any model: agent/model_config.py).

Same contract as LLMPilot (legacy/llm_pilot/llm_pilot.py), whose PilotCore it shares: the same
system prompt and task, the same two tools running the same code against the agent's
own feed and port, the same recording, limits and final comment. Only the model loop
is Pydantic AI's. So the trust layer (feed, port, gate) stays outside the framework:
nothing the framework does can bypass the port, and swapping the harness changes no
security property.

Tool calls in one model turn run one at a time, in order (sequential=True; Pydantic
AI's default is concurrent). Arguments that do not fit a tool's schema are answered
by Pydantic AI itself with the validation errors and never reach the port; they are
recorded as "tool_rejected" so malformed proposals still show up in the run log.

Observability: if Logfire is configured (observability.py), every model request and
tool call is traced there too; the JSONL run record is written either way.
"""
from typing import Literal

from pydantic_ai import (Agent, RunContext, RunTimedOut, UnexpectedModelBehavior,
                         UsageLimitExceeded, UsageLimits)
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart

from legacy.llm_pilot.llm_pilot import FlightRules, PilotCore, PilotLimits, PilotOutcome
from legacy.llm_pilot.providers.base import ToolCall


class PydanticPilot(PilotCore):
    def __init__(self, vehicle_id: str, feed, port, model, task: str, recorder=None,
                 limits: PilotLimits = PilotLimits(), rules: FlightRules = FlightRules(),
                 clock=None, on_step=None, model_settings: dict | None = None,
                 model_label: str | None = None, tool_retries: int = 3):
        super().__init__(vehicle_id, feed, port, task, recorder=recorder, limits=limits,
                         rules=rules, clock=clock, on_step=on_step)
        self._model = model
        self._model_settings = dict(model_settings or {})
        self._model_label = model_label or getattr(model, "model_name", str(model))
        self._tool_retries = tool_retries

    # ------------------------------------------------------------ agent
    def _build_agent(self, system: str) -> Agent:
        supported = tuple(self._port.supported_actions)
        ActionName = Literal[supported] if supported else str
        agent = Agent(self._model, instructions=system, output_type=str,
                      retries=self._tool_retries, name=f"pilot-{self.vehicle_id}")
        pilot = self

        @agent.tool(sequential=True)
        async def get_observation(ctx: RunContext[None]) -> dict:
            """Get the latest observation of your own drone (position, heading, speeds, ground
            state) with its observation_id. Call it before your first submit_action."""
            return (await pilot._run_tool(ToolCall(ctx.tool_call_id, "get_observation", {}))).result

        @agent.tool(sequential=True)
        async def submit_action(ctx: RunContext[None], observation_id: str, action_type: ActionName,
                                reason: str, heading_deg: float | None = None,
                                altitude_m: float | None = None, north_m: float | None = None,
                                east_m: float | None = None,
                                speed_mps: float | None = None) -> dict:
            """Propose ONE action for your drone. Blocks until the command finishes and returns
            feedback plus observation_after, a fresh observation for your next decision.

            Args:
                observation_id: id of the observation you decided from.
                action_type: rotate_to_heading (heading_deg), change_altitude (altitude_m,
                    optional speed_mps) or move_to_position (north_m, east_m, altitude_m,
                    optional speed_mps). Do not send parameters the action does not use.
                reason: short justification.
                heading_deg: 0 = north, 90 = east, in [0, 360).
                altitude_m: target altitude, same reference as observations.
                north_m: target north position.
                east_m: target east position.
                speed_mps: optional speed.
            """
            args = {"observation_id": observation_id, "action_type": action_type,
                    "reason": reason, "heading_deg": heading_deg, "altitude_m": altitude_m,
                    "north_m": north_m, "east_m": east_m, "speed_mps": speed_mps}
            args = {k: v for k, v in args.items() if v is not None}
            return (await pilot._run_tool(ToolCall(ctx.tool_call_id, "submit_action", args))).result

        return agent

    # ------------------------------------------------------------ run
    async def run(self) -> PilotOutcome:
        out, lim = self.outcome, self.limits
        system, intro, tools = self.prompts()
        self.record_session("pydantic-ai", self._model_label, system, intro, tools,
                            harness="pydantic-ai", model_settings=self._model_settings)
        agent = self._build_agent(system)
        settings = {"timeout": lim.model_timeout_s, **self._model_settings}
        try:
            t0 = self._clock()
            async with agent.iter(intro, usage_limits=UsageLimits(request_limit=lim.max_model_turns),
                                  model_settings=settings, timeout=lim.time_budget_s) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        self._record_rejected(node.request)
                        self._on_step(self.vehicle_id, "thinking")
                        t0 = self._clock()
                    elif Agent.is_call_tools_node(node):
                        self._record_turn(node.model_response, t0)
            out.comment = (run.result.output or "").strip() or None
            self._event("agent_comment", text=out.comment)
            return self._end("completed" if out.comment else "ended without a comment")
        except UsageLimitExceeded:
            return self._end(f"stopped: model turn limit ({lim.max_model_turns})")
        except RunTimedOut:
            return self._end(f"stopped: time budget ({lim.time_budget_s:g} s)")
        except ModelAPIError as err:     # ResilientModel already retried transient errors
            self._event("llm_error", attempt="final", error=type(err).__name__,
                        detail=str(err)[:500])
            return self._end(f"stopped: model unavailable ({type(err).__name__})")
        except UnexpectedModelBehavior as err:
            self._event("llm_error", attempt="final", error="UnexpectedModelBehavior",
                        detail=str(err)[:500])
            return self._end("stopped: model misbehaved (see llm_error)")
        except BaseException as err:
            self._interrupted(err)
            raise

    # ------------------------------------------------------------ recording
    def _record_turn(self, resp, t0: float) -> None:
        self.outcome.model_turns += 1
        usage = getattr(resp, "usage", None)
        self._event("llm_turn", turn=self.outcome.model_turns,
                    latency_s=round(self._clock() - t0, 3),
                    text="".join(p.content for p in resp.parts if isinstance(p, TextPart)),
                    tool_calls=[{"call_id": p.tool_call_id, "name": p.tool_name,
                                 "args": p.args_as_dict()}
                                for p in resp.parts if isinstance(p, ToolCallPart)],
                    usage={} if usage is None else {"input_tokens": usage.input_tokens,
                                                    "output_tokens": usage.output_tokens},
                    finish_reason=getattr(resp, "finish_reason", None))

    def _record_rejected(self, request) -> None:
        """Tool results the framework produced itself (schema errors, unknown tools)."""
        for p in getattr(request, "parts", ()):
            if isinstance(p, (ToolReturnPart, RetryPromptPart)) and p.tool_call_id \
                    and p.tool_call_id not in self._recorded_calls:
                self._recorded_calls.add(p.tool_call_id)
                self._event("tool_rejected", call_id=p.tool_call_id,
                            name=getattr(p, "tool_name", None), content=p.content)
