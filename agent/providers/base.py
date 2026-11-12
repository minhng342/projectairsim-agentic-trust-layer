"""
Model-provider interface for LLM agents: the ONLY provider-specific seam.

The pilot loop (agent/llm_pilot.py) talks to a ChatSession; each provider
(Gemini now; Ollama or an OpenClaw runtime later) implements it. A session keeps
its own native conversation history, because some providers attach opaque data
to their turns (e.g. Gemini thought signatures) that must be sent back verbatim.

    session = provider.start(system=..., tools=[ToolSpec, ...])
    turn = await session.send_user("...")                 # -> ModelTurn
    turn = await session.send_tool_results([ToolResult])  # -> ModelTurn

A ModelTurn has text and/or tool calls. Tool calls are executed by the pilot
loop, never by the provider (no automatic function calling): every call must go
through the agent's own feed and action port, and every call is recorded.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.rate_limit import RateLimiter  # noqa: F401  (moved; kept importable from here)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict           # JSON schema (type "object")


@dataclass(frozen=True)
class ToolCall:
    call_id: str               # provider's id, or one the provider adapter made up
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    result: dict


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)     # token counts, if reported
    finish_reason: str | None = None


class ChatSession(Protocol):
    # Optional: "async def wait_turn() -> float" reserves the next request slot under the
    # provider's rate limit and returns the wait. The pilot calls it BEFORE starting its
    # per-call timeout, so quota pacing is never mistaken for a slow or failed model.

    async def send_user(self, text: str) -> ModelTurn: ...

    async def send_tool_results(self, results: list[ToolResult]) -> ModelTurn: ...


class ModelProvider(Protocol):
    name: str                  # e.g. "gemini"
    model: str                 # e.g. the model id

    def start(self, system: str, tools: list[ToolSpec]) -> ChatSession: ...
