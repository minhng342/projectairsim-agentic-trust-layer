"""
Choose a harness and a model, check the model answers, then build pilots.

    setup = await prepare_pilot("pydantic", model_arg)    # or "native"
    setup.attach_recorder(recorder)
    pilot = setup.make(vehicle_id, feed, port, task, recorder, limits, rules, on_step)

pydantic  PydanticPilot on any Pydantic AI model string (LLM_MODEL / --model, e.g.
          "ollama:qwen3.5:9b" or "google:gemini-3.5-flash-lite"), paced and retried by
          ResilientModel (agent/model_config.py). The default.
native    LLMPilot on this repo's own Gemini provider (GEMINI_MODEL / --model, bare
          Gemini model names). Kept as the framework-free reference harness.

The preflight ping happens here, before anything connects to the simulator, so a dead
or overloaded model never leaves a drone airborne with no pilot.
"""
from dataclasses import dataclass, field
from typing import Callable

HARNESSES = ("pydantic", "native")


@dataclass
class PilotSetup:
    harness: str
    label: str                          # "provider:model" for printing and the run config
    preflight: list[dict]
    make: Callable
    config: dict = field(default_factory=dict)
    _model: object = None

    def attach_recorder(self, recorder) -> None:
        """Route model-level events (rate waits, retried errors) into the run record."""
        if hasattr(self._model, "on_event"):
            self._model.on_event = recorder.recorder_for(source="model")


async def prepare_pilot(harness: str, model_arg: str | None = None, log=print) -> PilotSetup:
    if harness == "pydantic":
        from agent.model_config import pick_model
        from legacy.llm_pilot.pydantic_pilot import PydanticPilot
        model, spec, tried = await pick_model(model_arg, log=log)

        def make(vid, feed, port, task, recorder=None, limits=None, rules=None, on_step=None):
            kw = {k: v for k, v in (("limits", limits), ("rules", rules)) if v is not None}
            return PydanticPilot(vid, feed, port, model, task=task, recorder=recorder,
                                 on_step=on_step, model_label=spec, **kw)
        return PilotSetup("pydantic", spec, tried, make,
                          {"harness": "pydantic-ai", "model": spec,
                           "rpm": getattr(getattr(model, "limiter", None), "rpm", None)}, model)
    if harness == "native":
        from legacy.llm_pilot.llm_pilot import LLMPilot
        from legacy.llm_pilot.providers import pick_gemini
        provider, tried = await pick_gemini(model_arg, log=log)

        def make(vid, feed, port, task, recorder=None, limits=None, rules=None, on_step=None):
            kw = {k: v for k, v in (("limits", limits), ("rules", rules)) if v is not None}
            return LLMPilot(vid, feed, port, provider, task=task, recorder=recorder,
                            on_step=on_step, **kw)
        label = f"{provider.name}:{provider.model}"
        return PilotSetup("native", label, tried, make,
                          {"harness": "native", "model": label,
                           "temperature": provider.temperature})
    raise ValueError(f"harness must be one of {HARNESSES}, not {harness!r}")
