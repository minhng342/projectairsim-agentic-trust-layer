"""Choosing and checking the LLM before a run starts."""
from atal.observability import configure_observability


async def prepare_llm(args) -> dict:
    """Pick and ping the model BEFORE anything connects to the simulator."""
    observed = configure_observability()
    print(f"Logfire tracing {'on' if observed else 'off (no LOGFIRE_TOKEN)'}")
    settings = {}
    spec = args.model or ""
    if args.thinking != "default":
        settings["thinking"] = args.thinking == "on"
    from atal.agents.llm.model_config import model_specs
    if args.thinking == "off" and model_specs(args.model)[0].startswith("ollama:"):
        # Pydantic AI does not translate `thinking` for Ollama; Ollama's OpenAI-compatible
        # endpoint takes reasoning_effort "none" (and its native API `think: false`).
        settings["extra_body"] = {"reasoning_effort": "none", "think": False}
    if spec == "test":
        # Pydantic AI's TestModel: no LLM, answers with schema-valid but meaningless actions
        # (sector "a"), so every decision exercises the guard's refusal and the fallback.
        from pydantic_ai.models.test import TestModel
        model = TestModel()
        return {"model": model, "label": "test", "settings": {},
                "config": {"model": "test", "harness": "pydantic-ai"}}
    if spec == "scripted" or spec.startswith("scripted:"):
        from atal.agents.llm.scripted_model import scripted_search_model
        policy = spec.split(":", 1)[1] if ":" in spec else "nearest"
        model = scripted_search_model(policy)
        return {"model": model, "label": model.model_name, "settings": {},
                "config": {"model": model.model_name, "harness": "pydantic-ai"}}
    from atal.agents.llm.model_config import pick_model
    model, label, tried = await pick_model(args.model)
    return {"model": model, "label": label, "settings": settings,
            "config": {"model": label, "harness": "pydantic-ai", "preflight": tried,
                       "model_settings": settings, "logfire": observed,
                       "rpm": getattr(getattr(model, "limiter", None), "rpm", None)}}
