"""
Optional Logfire tracing of every model request and tool call (Pydantic AI agents).

    from atal.observability import configure_observability
    configure_observability()        # call once at startup, before running agents

On only if LOGFIRE_TOKEN is set (environment or the repo's .env); otherwise this does
nothing and runs are unaffected. The JSONL run record (atal/evaluation/recorder.py) is the
experiment's source of truth either way; Logfire is for watching and browsing runs.
"""
import os

from atal.env import load_env_file


def configure_observability(service_name: str = "atal-agentic-trust-layer") -> bool:
    """Returns True if Logfire tracing was turned on."""
    load_env_file()
    if not os.environ.get("LOGFIRE_TOKEN"):
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
        return False
    import logfire
    logfire.configure(service_name=service_name, send_to_logfire="if-token-present",
                      console=False)
    logfire.instrument_pydantic_ai()
    global _ENABLED
    _ENABLED = True
    return True


_ENABLED = False


def span(name: str, **attributes):
    """A Logfire span when tracing is on (configure_observability returned True), else a
    no-op context manager. Model requests made inside it nest under it."""
    if not _ENABLED:
        import contextlib
        return contextlib.nullcontext()
    import logfire
    return logfire.span(name, **attributes)
