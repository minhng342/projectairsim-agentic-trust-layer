"""
Choosing the language model: one setting, any backend (Pydantic AI model strings).

    LLM_MODEL=ollama:qwen3.5:9b                        local Ollama (OLLAMA_BASE_URL, default
                                                       http://localhost:11434/v1)
    LLM_MODEL=google:gemini-3.5-flash-lite             Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
    LLM_MODEL=openai-chat:<model>                      any OpenAI-compatible server (vLLM,
                                                       LM Studio, llama.cpp): OPENAI_BASE_URL
    LLM_MODEL=ollama:qwen3.5:9b,google:gemini-3.5-flash-lite   preference list; the first that
                                                       answers the preflight ping is used

Set it in the repo's .env (or pass --model). Every model is wrapped in ResilientModel:
    - pacing: at most LLM_RPM requests per minute, evenly spaced (default: ~80% of the
      Gemini free tier for google:* models, unlimited for local models), shared by every
      agent that uses the same model object;
    - retries: overload / rate limit / server / connection errors are retried with
      exponential backoff (or the provider's retry_after); a rejected request (unknown
      model, bad key) is not.
Both happen below the agent framework, so they apply to any harness using the model.
"""
import asyncio
import os
import time

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelHTTPError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    ModelUnavailableError,
)
from pydantic_ai.models import infer_model
from pydantic_ai.models.wrapper import WrapperModel

from atal.agents.llm.rate_limit import RateLimiter, rpm_for
from atal.env import load_env_file

DEFAULT_MODEL = "google:gemini-3.5-flash-lite"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
_TRANSIENT = (ModelUnavailableError, ModelRateLimitError, ModelServerError, ModelConnectionError)


def is_transient(err: BaseException) -> bool:
    """Worth retrying: overload, rate limit, server or connection trouble."""
    if isinstance(err, ModelRequestRejectedError):
        return False
    if isinstance(err, _TRANSIENT):
        return True
    if isinstance(err, ModelQuotaExceededError):          # per-minute quota: retry_after given
        return getattr(err, "retry_after", None) is not None
    if isinstance(err, ModelHTTPError):
        return err.status_code == 429 or err.status_code >= 500
    return False


class ResilientModel(WrapperModel):
    """Any Pydantic AI model + client-side pacing + retries of transient errors.
    on_event(kind, fields) reports "llm_rate_wait" and "llm_error" (for the run record)."""

    def __init__(self, wrapped, *, rpm: float | None = None, retries: int = 5,
                 backoff_s: float = 2.0, backoff_max_s: float = 20.0, on_event=None,
                 sleep=asyncio.sleep, clock=time.monotonic):
        super().__init__(wrapped)
        self.limiter = RateLimiter(rpm, clock=clock, sleep=sleep) if rpm else None
        self.retries, self.backoff_s, self.backoff_max_s = retries, backoff_s, backoff_max_s
        self.on_event = on_event
        self._sleep, self._clock = sleep, clock

    async def request(self, messages, model_settings, model_request_parameters):
        for attempt in range(self.retries + 1):
            if self.limiter is not None:
                waited = await self.limiter.acquire()
                if waited > 0:
                    self._emit("llm_rate_wait", waited_s=round(waited, 3))
            t0 = self._clock()
            try:
                return await super().request(messages, model_settings, model_request_parameters)
            except ModelAPIError as err:
                self._emit("llm_error", attempt=attempt + 1, error=type(err).__name__,
                           status=getattr(err, "status_code", None), detail=str(err)[:500],
                           latency_s=round(self._clock() - t0, 3))
                if attempt >= self.retries or not is_transient(err):
                    raise
                wait = getattr(err, "retry_after", None) or min(
                    self.backoff_s * 2 ** attempt, self.backoff_max_s)
                await self._sleep(min(wait, 60.0))
        raise AssertionError("unreachable")

    def _emit(self, kind, **fields):
        if self.on_event is not None:
            try:
                self.on_event(kind, {"model": self.model_name, **fields})
            except Exception:
                pass


def model_specs(spec: str | None = None) -> list[str]:
    """Preference list from --model or LLM_MODEL (comma-separated). Loads .env first."""
    load_env_file()
    raw = spec or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    return [s.strip() for s in raw.split(",") if s.strip()]


def default_rpm(spec: str) -> float | None:
    """LLM_RPM if set; the Gemini free-tier pace for google:* models; else no pacing."""
    env = os.environ.get("LLM_RPM")
    if env:
        return float(env) if float(env) > 0 else None
    if spec.startswith(("google:", "google-cloud:")):
        return rpm_for(spec.split(":", 1)[1])
    return None


def make_model(spec: str, on_event=None, **kw) -> ResilientModel:
    if spec.startswith("ollama:"):
        os.environ.setdefault("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    return ResilientModel(infer_model(spec), rpm=default_rpm(spec), on_event=on_event, **kw)


async def ping(model) -> float:
    """One tiny request. Returns latency; raises on failure. Paced like any request, but
    without ResilientModel's retries: pick_model does its own, shorter, retrying."""
    from pydantic_ai import Agent, UsageLimits
    if isinstance(model, ResilientModel):
        if model.limiter is not None:
            await model.limiter.acquire()
        model = model.wrapped
    t0 = time.monotonic()
    await Agent(model).run("Reply with the single word OK.",
                           usage_limits=UsageLimits(request_limit=1))
    return time.monotonic() - t0


async def pick_model(spec: str | None = None, *, attempts: int = 3, backoff_s: float = 3.0,
                     log=print, make=make_model, ping_fn=ping, sleep=asyncio.sleep, **make_kw):
    """Preflight BEFORE anything flies: try each preferred model in order and return
    (model, spec, tried) for the first that answers. Raises RuntimeError if none does."""
    tried = []
    for s in model_specs(spec):
        try:
            model = make(s, **make_kw)
        except Exception as err:          # bad spec, missing SDK extra or key
            tried.append({"model": s, "attempt": 1, "ok": False,
                          "error": f"{type(err).__name__}: {str(err)[:160]}"})
            log(f"  {s}: cannot create ({type(err).__name__}: {str(err)[:160]})")
            continue
        for attempt in range(attempts):
            try:
                latency = await ping_fn(model)
            except Exception as err:
                msg = f"{type(err).__name__}: {str(err)[:160]}"
                tried.append({"model": s, "attempt": attempt + 1, "ok": False, "error": msg})
                log(f"  {s}: attempt {attempt + 1} failed: {msg}")
                if not is_transient(err):
                    break
                if attempt + 1 < attempts:
                    await sleep(backoff_s * 2 ** attempt)
                continue
            tried.append({"model": s, "attempt": attempt + 1, "ok": True,
                          "latency_s": round(latency, 3)})
            log(f"  {s}: available ({latency:.1f} s)")
            return model, s, tried
    raise RuntimeError("no model answered: "
                       + "; ".join(f"{t['model']}#{t['attempt']} {t['error']}" for t in tried))
