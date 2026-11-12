"""
Gemini provider (Google Gen AI SDK, `pip install google-genai`).

    provider = GeminiProvider(model="gemini-...")      # key: GEMINI_API_KEY (or GOOGLE_API_KEY)
    session = provider.start(system=..., tools=[...])

- Automatic function calling is OFF: the pilot loop executes every tool call itself.
- The session keeps Gemini's own Content objects as history and sends them back
  unchanged, which preserves thought signatures on thinking models.
- History only grows when a call succeeds, so a retried call resends the same turn.
- An empty or blocked response raises GeminiEmptyResponse (the pilot retries it).
- Function-call ids: Gemini may or may not send one. When it doesn't, the session
  makes one up for the record and answers that call without an id, by name.

The SDK is imported lazily, so the rest of the system (and the offline tests) do
not need it installed.
"""
import os
import time

from legacy.llm_pilot.providers.base import ModelTurn, RateLimiter, ToolCall, ToolResult, ToolSpec


class GeminiEmptyResponse(RuntimeError):
    pass


def api_key_from_env() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY (environment variable or the repo's .env file)")
    return key


def _upper_types(schema):
    """JSON-schema-ish dict -> the SDK's Schema dialect (type names upper case)."""
    if isinstance(schema, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else _upper_types(v))
                for k, v in schema.items()}
    if isinstance(schema, list):
        return [_upper_types(x) for x in schema]
    return schema


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None, temperature: float | None = 0.2,
                 client=None, types_module=None, rpm: float | None = None):
        if client is None or types_module is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as err:
                raise RuntimeError("the Gemini SDK is not installed: pip install google-genai") from err
            client = client or genai.Client(api_key=api_key or api_key_from_env())
            types_module = types_module or types
        self.model = model
        self.temperature = temperature
        # one limiter per provider: every session (pilot) on it shares the model's quota
        self.limiter = RateLimiter(rpm) if rpm else None
        self._client = client
        self._types = types_module

    async def ping(self) -> float:
        """One tiny request to check the key, the model and current availability.
        Returns the latency in seconds; raises the SDK's error (with .code) on failure."""
        if self.limiter is not None:
            await self.limiter.acquire()
        t0 = time.monotonic()
        resp = await self._client.aio.models.generate_content(
            model=self.model, contents="Reply with the single word OK.")
        if not getattr(resp, "candidates", None):
            raise GeminiEmptyResponse(f"empty response from {self.model}")
        return time.monotonic() - t0

    def start(self, system: str, tools: list[ToolSpec]) -> "GeminiSession":
        return GeminiSession(self._client, self._types, self.model, system, tools, self.temperature,
                             self.limiter)


class GeminiSession:
    def __init__(self, client, types, model, system, tools, temperature, limiter=None):
        self._client, self._t, self._model = client, types, model
        cfg = {"system_instruction": system,
               "tools": [types.Tool(function_declarations=[self._declare(t) for t in tools])],
               "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)}
        if temperature is not None:
            cfg["temperature"] = temperature
        self._config = types.GenerateContentConfig(**cfg)
        self._history: list = []
        self._made_up_ids: set[str] = set()
        self._n_calls = 0
        self._limiter = limiter
        self._reserved = False

    async def wait_turn(self) -> float:
        """Reserve the next request slot (see ChatSession). Returns the wait in seconds."""
        if self._limiter is None or self._reserved:
            return 0.0
        waited = await self._limiter.acquire()
        self._reserved = True
        return waited

    def _declare(self, spec: ToolSpec):
        t = self._t
        if not spec.parameters.get("properties"):
            return t.FunctionDeclaration(name=spec.name, description=spec.description)
        try:
            return t.FunctionDeclaration(name=spec.name, description=spec.description,
                                         parameters_json_schema=spec.parameters)
        except Exception:      # older SDKs: Schema dialect only
            return t.FunctionDeclaration(name=spec.name, description=spec.description,
                                         parameters=_upper_types(spec.parameters))

    async def send_user(self, text: str) -> ModelTurn:
        t = self._t
        return await self._send(t.Content(role="user", parts=[t.Part(text=text)]))

    async def send_tool_results(self, results: list[ToolResult]) -> ModelTurn:
        t = self._t
        parts = [t.Part(function_response=t.FunctionResponse(
                     id=None if r.call_id in self._made_up_ids else r.call_id,
                     name=r.name, response=r.result))
                 for r in results]
        return await self._send(t.Content(role="user", parts=parts))

    async def _send(self, content) -> ModelTurn:
        if self._limiter is not None and not self._reserved:
            await self._limiter.acquire()
        self._reserved = False
        contents = self._history + [content]
        resp = await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=self._config)
        cands = getattr(resp, "candidates", None) or []
        cand = cands[0] if cands else None
        parts = (getattr(getattr(cand, "content", None), "parts", None) or []) if cand else []
        if not parts:
            why = getattr(cand, "finish_reason", None) if cand else getattr(resp, "prompt_feedback", None)
            raise GeminiEmptyResponse(f"empty response from {self._model} ({why})")
        self._history = contents + [cand.content]

        calls, texts = [], []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                self._n_calls += 1
                cid = getattr(fc, "id", None)
                if not cid:
                    cid = f"gemini-call-{self._n_calls}"
                    self._made_up_ids.add(cid)
                calls.append(ToolCall(call_id=cid, name=fc.name, args=dict(fc.args or {})))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                texts.append(p.text)
        um = getattr(resp, "usage_metadata", None)
        usage = {} if um is None else {
            k: getattr(um, k) for k in ("prompt_token_count", "candidates_token_count",
                                        "thoughts_token_count", "total_token_count")
            if getattr(um, k, None) is not None}
        fr = getattr(cand, "finish_reason", None)
        return ModelTurn(text="".join(texts), tool_calls=calls, usage=usage,
                         finish_reason=getattr(fr, "value", None) or (str(fr) if fr else None))
