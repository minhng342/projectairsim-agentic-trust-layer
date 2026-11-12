"""
LLM pilot against a FAKE drone: checks the model, its tool calling and the whole
agent loop WITHOUT Project AirSim. Run this before the live test.

    python -m legacy.llm_pilot.check_llm [--harness pydantic|native] [--model SPEC] [--slow-first SECONDS]
    python -m legacy.llm_pilot.check_llm --list-models [--model SPEC]

--model / LLM_MODEL (pydantic harness): a Pydantic AI model string, or a comma list
tried in order, e.g.  ollama:qwen3.5:9b   google:gemini-3.5-flash-lite
(the native harness takes bare Gemini names from GEMINI_MODEL instead).

The fake drone (tests/test_executor.py fakes) starts airborne at N=-1, E=8, altitude
5 and flies every command exactly. The real ActionPort, executor logic, recorder and
summarizer are used, so the run log in runs/ looks like a live one.

--slow-first N   holds the model's FIRST submit_action for N seconds after it was
                 decided (N > 10 expires its observation): checks that the model
                 handles observation_expired by observing again.
--list-models    lists what the server behind --model / LLM_MODEL offers (Ollama /
                 OpenAI-compatible: GET <base_url>/models; Google: the Gemini model list).
"""
import argparse
import asyncio
import os
import sys
import time

from atal.channels.action_port import ActionPort
from atal.channels.observation_feed import ObservationFeed
from atal.evaluation.summarize_run import summarize
from atal.flight.executor import ProjectAirSimExecutor
from atal.observability import configure_observability
from atal.runtime.recorder import EvaluationRecorder
from legacy.llm_pilot.llm_pilot import FlightRules, PilotLimits
from legacy.llm_pilot.pilot_setup import HARNESSES, prepare_pilot
from tests.test_executor import FAST, MultiAdapter

TASK = ("Climb to 3 m above your current altitude, fly to a point 10 m north of the launch "
        "point at that altitude, face east, then fly back above the launch point at the same "
        "altitude. Then give your final comment.")


def hold_first_submit(port, delay_s):
    """Wrap port.submit: the first submission waits delay_s before reaching the port."""
    original, state = port.submit, {"done": False}

    async def submit(action, based_on):
        if not state["done"]:
            state["done"] = True
            print(f"  (holding the first submit_action for {delay_s:g} s)")
            await asyncio.sleep(delay_s)
        return await original(action, based_on)
    port.submit = submit


async def main(args) -> int:
    traced = configure_observability()
    print(f"harness {args.harness}; checking model availability...")
    try:
        setup = await prepare_pilot(args.harness, args.model)
    except RuntimeError as err:
        print(f"\nNO MODEL AVAILABLE: {err}")
        return 2
    print(f"model {setup.label}" + ("  (Logfire tracing on)" if traced else ""))
    rec = EvaluationRecorder(run_config={
        "script": "check_llm.py", "fake_drone": True, "model": {**setup.config,
                                                                 "preflight": setup.preflight},
        "slow_first_s": args.slow_first,
        "pipeline": {"observation_source": "FAKE drone (no simulator)", "attack_hooks": [],
                     "trust_assessor": "none", "action_gate": "none",
                     "agents": f"{setup.harness} pilot ({setup.label})"}})
    setup.attach_recorder(rec)
    adapter = MultiAdapter(["Drone1"])
    ex = ProjectAirSimExecutor(adapter, config=FAST, log=lambda *_: None,
                               on_event=rec.recorder_for(source="executor"))
    feed = ObservationFeed(adapter, "Drone1", recorder=rec)
    port = ActionPort("Drone1", ex, feed, recorder=rec)
    if args.slow_first:
        hold_first_submit(port, args.slow_first)

    def on_step(vid, label):
        print(f"  [{time.strftime('%H:%M:%S')}] {vid}: {label}")
    pilot = setup.make("Drone1", feed, port, TASK, recorder=rec,
                       limits=PilotLimits(time_budget_s=300), rules=FlightRules(), on_step=on_step)
    outcome = "aborted"
    try:
        pilot.capture_home()
        stop = asyncio.Event()
        sampler = asyncio.create_task(feed.sample(5.0, stop))
        try:
            out = await pilot.run()
        finally:
            stop.set()
            await sampler
        print(f"\nended: {out.ended}; {out.model_turns} model turns; "
              f"{out.succeeded_actions}/{len(out.submissions)} actions succeeded")
        for f in out.submissions:
            print(f"  {f['action_type']}: {f['code']}")
        print(f"\nfinal comment:\n{out.comment}")
        outcome = "pass" if out.ended == "completed" and out.succeeded_actions >= 1 else "fail"
    finally:
        rec.close(outcome)
        print("\n================ RUN LOG SUMMARY ================")
        print(summarize(rec.path))
    print("\nRESULT:", outcome.upper())
    return 0 if outcome == "pass" else 1


def list_models(model_arg) -> int:
    from atal.agents.llm.model_config import DEFAULT_OLLAMA_URL, model_specs
    spec = model_specs(model_arg)[0]
    kind = spec.split(":", 1)[0]
    if kind in ("google", "google-cloud"):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        for m in genai.Client(api_key=key).models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                print("google:" + m.name.removeprefix("models/"))
        return 0
    base = (os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL) if kind == "ollama"
            else os.environ.get("OPENAI_BASE_URL"))
    if not base:
        print(f"don't know where to list models for {spec!r} (set OPENAI_BASE_URL)")
        return 2
    import httpx
    r = httpx.get(base.rstrip("/") + "/models", timeout=10.0,
                  headers={"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY', 'none')}"})
    r.raise_for_status()
    for m in r.json().get("data", []):
        print(f"{kind}:{m.get('id')}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=HARNESSES, default="pydantic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--slow-first", type=float, default=0.0)
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()
    sys.exit(list_models(a.model) if a.list_models else asyncio.run(main(a)))
