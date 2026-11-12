"""Model providers for LLM agents. See base.py for the interface."""
import asyncio
import os

from agent.rate_limit import (FREE_TIER_RPM, RPM_HEADROOM,  # noqa: F401  (moved)
                              UNKNOWN_MODEL_RPM, rpm_for)

DEFAULT_GEMINI_MODELS = ["gemini-3.5-flash-lite"]


# HTTP codes that will not fix themselves by retrying (bad key, no access, unknown model)
_PERMANENT = {400, 401, 403, 404}


def gemini_models_from_env(model: str | None = None) -> list[str]:
    """Preference list: `model` (CLI) or GEMINI_MODEL, comma-separated, e.g.
    "gemini-3.5-flash-lite,gemini-3.6-flash". Loads the repo's .env first."""
    from utils.env import load_env_file
    load_env_file()
    raw = model or os.environ.get("GEMINI_MODEL") or ",".join(DEFAULT_GEMINI_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def gemini_from_env(model: str | None = None, temperature: float | None = 0.2):
    """GeminiProvider for the FIRST preferred model, without checking it is available."""
    from legacy.llm_pilot.providers.gemini import GeminiProvider
    m = gemini_models_from_env(model)[0]
    return GeminiProvider(model=m, temperature=temperature, rpm=rpm_for(m))


async def pick_gemini(model: str | None = None, temperature: float | None = 0.2,
                      attempts: int = 3, backoff_s: float = 3.0, log=print,
                      make=None, sleep=asyncio.sleep):
    """Preflight, BEFORE anything flies: ping each preferred model in order and return
    (provider, tried) for the first that answers. `tried` lists every attempt (for the
    run record). Raises RuntimeError if none answers.

    Overload (503/429) and network errors are retried `attempts` times with backoff;
    a permanent error (bad key, no access, unknown model) moves on at once."""
    if make is None:
        from legacy.llm_pilot.providers.gemini import GeminiProvider

        def make(m):
            return GeminiProvider(model=m, temperature=temperature, rpm=rpm_for(m))
    tried = []
    for m in gemini_models_from_env(model):
        try:
            provider = make(m)
        except Exception as err:            # e.g. SDK missing, no key: same for every model
            raise RuntimeError(f"cannot create the Gemini client: {err}") from err
        for attempt in range(attempts):
            try:
                latency = await provider.ping()
            except Exception as err:
                code = getattr(err, "code", None)
                msg = f"{type(err).__name__}" + (f" {code}" if code else "") + f": {str(err)[:160]}"
                tried.append({"model": m, "attempt": attempt + 1, "ok": False, "error": msg})
                log(f"  {m}: attempt {attempt + 1} failed: {msg}")
                if code in _PERMANENT:
                    break
                if attempt + 1 < attempts:
                    await sleep(backoff_s * 2 ** attempt)
                continue
            tried.append({"model": m, "attempt": attempt + 1, "ok": True,
                          "latency_s": round(latency, 3)})
            log(f"  {m}: available ({latency:.1f} s)")
            return provider, tried
    raise RuntimeError("no Gemini model answered: "
                       + "; ".join(f"{t['model']}#{t['attempt']} {t['error']}" for t in tried))
