"""
A scripted stand-in for the LLM (Pydantic AI FunctionModel): offline runs and tests of the
Phase 2 pipeline with no model server.

    --model scripted            nearest unsearched sector (greedy), like a sensible model
    --model scripted:prior      most likely unsearched sector under the prior, else nearest
    --model scripted:stubborn   first proposes a swept sector (tests the guard's retry path)
    --model scripted:broken     never returns a valid action (tests the fallback path)

It reads only what a real model would: the decision prompt's situation JSON and the guard's
refusals. After a refusal it tries the next candidate.
"""
import json
import re

from pydantic_ai.messages import (ModelResponse, RetryPromptPart, TextPart, ToolCallPart,
                                  ToolReturnPart, UserPromptPart)
from pydantic_ai.models.function import AgentInfo, FunctionModel

_JSON = re.compile(r"```json\n(.*?)\n```", re.S)
POLICIES = ("nearest", "prior", "stubborn", "broken")


def _situation(messages) -> tuple[dict | None, int]:
    sit, retries = None, 0
    for m in messages:
        for p in getattr(m, "parts", ()):
            if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                hit = _JSON.search(p.content)
                if hit:
                    sit, retries = json.loads(hit.group(1)), 0
            elif isinstance(p, (RetryPromptPart, ToolReturnPart)):   # a refusal
                retries += 1
    return sit, retries


def _tool(info: AgentInfo, kind: str) -> str:
    for t in info.output_tools:
        if kind in t.name:
            return t.name
    raise RuntimeError(f"no output tool for {kind}: {[t.name for t in info.output_tools]}")


def scripted_search_model(policy: str = "nearest") -> FunctionModel:
    if policy not in POLICIES:
        raise ValueError(f"scripted policy must be one of {POLICIES}, not {policy!r}")

    def decide(messages, info: AgentInfo) -> ModelResponse:
        sit, retries = _situation(messages)
        if policy == "broken" or sit is None:
            return ModelResponse(parts=[TextPart("I am not sure what to do.")])
        cands = [c["sector_id"] for c in sit.get("nearest_unsearched", [])]
        if policy == "prior" and sit.get("prior_likely_unsearched"):
            cands = [c["sector_id"] for c in sit["prior_likely_unsearched"]] + cands
        if policy == "stubborn":
            cands = [sit.get("last_sector") or "A1"] + cands
        if retries < len(cands):
            return ModelResponse(parts=[ToolCallPart(_tool(info, "SweepSector"), {
                "sector_id": cands[retries], "reason": f"scripted {policy} choice"})])
        return ModelResponse(parts=[ToolCallPart(_tool(info, "ReturnHome"), {
            "reason": "scripted: nothing left to try"})])

    return FunctionModel(decide, model_name=f"scripted:{policy}")
