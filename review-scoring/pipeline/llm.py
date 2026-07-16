"""LLM wrapper with structured JSON output and a persistent SQLite cache.

Two backends:
  provider: anthropic   -> official Anthropic SDK (AsyncAnthropic,
                           ANTHROPIC_API_KEY)
  provider: openrouter  -> OpenRouter chat/completions over aiohttp
                           (OPENROUTER_API_KEY), model slugs like
                           "anthropic/claude-sonnet-4.5"

The implementation is async (json_call_async) so the pipeline passes can
gather() many provider calls concurrently; json_call is a sync facade for
one-off callers. Total in-flight calls across ALL LLM instances are capped
by one shared per-event-loop semaphore (config.yaml:
max_concurrent_requests) so concurrent passes don't trip provider rate
limits; 429/5xx still get exponential backoff on top.

Every call is cached by a hash of (provider, model, system, user, schema)
in the product's scoring.db (llm_cache table), so reruns after a crash or
a tweak elsewhere in the pipeline are free. Several LLM instances
(extract/group/consolidate) share one database; INSERT OR IGNORE makes
concurrent writers safe.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openrouter": "anthropic/claude-sonnet-4.5",
}

# $ per 1M tokens (input, output). Used to price the direct Anthropic API
# backend, where the response carries no cost field (unlike OpenRouter,
# which reports real billed cost in usage.cost). Sonnet 5 price is the
# intro rate in effect through 2026-08-31 (standard rate is $3/$15).
ANTHROPIC_PRICING = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _lookup_pricing(model: str) -> tuple[float, float] | None:
    """Best-effort $ / 1M token (input, output) lookup for a model id,
    tolerating an OpenRouter-style "anthropic/" prefix and dotted
    version suffixes like "claude-sonnet-4.5"."""
    name = model.split("/")[-1]
    if name in ANTHROPIC_PRICING:
        return ANTHROPIC_PRICING[name]
    name = re.sub(r"\.(\d)", r"-\1", name)  # "4.5" -> "4-5"
    return ANTHROPIC_PRICING.get(name)


def load_env() -> None:
    """Load KEY=value pairs from the project's .env into os.environ
    (existing environment variables win)."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and not os.environ.get(key):
            os.environ[key] = value


load_env()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------- async runtime shared by every LLM instance ----------

# Cap on TOTAL in-flight provider calls per event loop, across all LLM
# instances (extraction/grouping/verify/... draw from one budget).
# Overridden from config.yaml (max_concurrent_requests) by run.py / app.py /
# calibrate.py before any pass runs.
_MAX_CONCURRENT = 10

# per-loop shared resources: {"sem": Semaphore, "anthropic": AsyncAnthropic,
# "session": aiohttp.ClientSession}. Keyed by the loop object; run_async pops
# and closes the entry when its loop finishes, so nothing leaks across the
# one-loop-per-pipeline-pass lifecycle.
_LOOP_RES: dict[asyncio.AbstractEventLoop, dict] = {}


def set_max_concurrency(n) -> None:
    """Install the concurrency cap (config.yaml: max_concurrent_requests).
    Takes effect for event loops started after the call; 1 = sequential."""
    global _MAX_CONCURRENT
    _MAX_CONCURRENT = max(1, int(n or 1))


def _res() -> dict:
    loop = asyncio.get_running_loop()
    r = _LOOP_RES.get(loop)
    if r is None:
        r = {"sem": asyncio.Semaphore(_MAX_CONCURRENT)}
        _LOOP_RES[loop] = r
    return r


async def _close_res() -> None:
    r = _LOOP_RES.pop(asyncio.get_running_loop(), None)
    if not r:
        return
    client = r.get("anthropic")
    if client is not None:
        await client.close()
    session = r.get("session")
    if session is not None:
        await session.close()


def run_async(coro):
    """asyncio.run plus cleanup of the per-loop LLM resources (the shared
    aiohttp session / AsyncAnthropic client). Every sync-facing pipeline
    pass funnels its gathered calls through this."""
    async def _main():
        try:
            return await coro
        finally:
            await _close_res()
    return asyncio.run(_main())


def _parse_json_loose(text: str) -> dict:
    """Parse JSON that may be wrapped in code fences or prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLM:
    def __init__(self, model: str | None = None, cache=None,
                 effort: str = "medium", provider: str = "anthropic"):
        """`cache` is a storage.DB (or anything with llm_cache_get /
        llm_cache_put); None disables caching."""
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
        self.effort = effort
        self.cache = cache
        self._or_schema_ok: bool | None = None  # does the model accept json_schema?
        self._or_reasoning_ok: bool | None = None  # does it accept reasoning?
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.cost_usd = 0.0
        self.cost_known = True  # False once we hit a call we can't price

    # ---------- public ----------

    async def json_call_async(self, system: str, user: str, schema: dict,
                              max_tokens: int = 16000) -> dict:
        """Cache lookup, then the provider call under the shared per-loop
        concurrency semaphore. The sqlite cache get/put are sub-ms and run
        directly on the loop (the DB layer is lock-guarded)."""
        key = self._key(system, user, schema)
        if self.cache is not None:
            cached = self.cache.llm_cache_get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached

        async with _res()["sem"]:
            if self.provider == "openrouter":
                result = await self._acall_openrouter(system, user, schema,
                                                      max_tokens)
            else:
                result = await self._acall_anthropic(system, user, schema,
                                                     max_tokens)
        self.calls += 1

        if self.cache is not None:
            self.cache.llm_cache_put(key, self.provider, self.model,
                                     self.effort, result)
        return result

    def json_call(self, system: str, user: str, schema: dict,
                  max_tokens: int = 16000) -> dict:
        """Sync facade over json_call_async for one-off callers (quote
        repair, calibrate). Must not be called from inside a running event
        loop — the concurrent passes await json_call_async directly."""
        return run_async(self.json_call_async(system, user, schema,
                                              max_tokens))

    def usage_report(self) -> str:
        def fmt(n: int) -> str:
            return f"{n:,}".replace(",", " ")

        total = self.input_tokens + self.output_tokens
        line = (f"LLM: {self.calls} викликів ({self.cache_hits} з кешу), "
                f"{fmt(total)} токенів (вхід {fmt(self.input_tokens)} + "
                f"вихід {fmt(self.output_tokens)})")
        if self.cache_read_tokens or self.cache_write_tokens:
            line += (f", prompt cache: {fmt(self.cache_read_tokens)} read / "
                     f"{fmt(self.cache_write_tokens)} write")
        if self.cost_known:
            line += f" | вартість: ${self.cost_usd:.4f}"
        else:
            line += " | вартість: невідома (модель без відомої ціни)"
        return line

    def usage_dict(self) -> dict:
        """Machine-readable snapshot of this instance's usage, for logging."""
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": self.cost_usd,
            "cost_known": self.cost_known,
        }

    # ---------- combined report across LLM instances (e.g. a separate
    # extraction model) ----------

    @staticmethod
    def combined_usage_report(llms: list["LLM"]) -> str:
        def fmt(n: int) -> str:
            return f"{n:,}".replace(",", " ")

        calls = sum(l.calls for l in llms)
        hits = sum(l.cache_hits for l in llms)
        in_tok = sum(l.input_tokens for l in llms)
        out_tok = sum(l.output_tokens for l in llms)
        cache_read = sum(l.cache_read_tokens for l in llms)
        cache_write = sum(l.cache_write_tokens for l in llms)
        cost = sum(l.cost_usd for l in llms)
        total = in_tok + out_tok
        line = (f"LLM разом: {calls} викликів ({hits} з кешу), "
                f"{fmt(total)} токенів (вхід {fmt(in_tok)} + вихід {fmt(out_tok)})")
        if cache_read or cache_write:
            line += (f", prompt cache: {fmt(cache_read)} read / "
                     f"{fmt(cache_write)} write")
        if all(l.cost_known for l in llms):
            line += f" | вартість: ${cost:.4f}"
        else:
            line += " | вартість: невідома (є модель без відомої ціни)"
        return line

    # ---------- anthropic backend ----------

    async def _acall_anthropic(self, system: str, user: str, schema: dict,
                               max_tokens: int) -> dict:
        import anthropic
        res = _res()
        client = res.get("anthropic")
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY не заданий. Задайте змінну середовища\n"
                    "або переключіть provider: openrouter у config.yaml."
                )
            client = anthropic.AsyncAnthropic()
            res["anthropic"] = client
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Модель відхилила запит (refusal).")
        u = response.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += cache_write
        pricing = _lookup_pricing(self.model)
        if pricing:
            in_price, out_price = pricing
            self.cost_usd += (
                u.input_tokens * in_price
                + cache_read * in_price * 0.1       # cache read ≈ 0.1x
                + cache_write * in_price * 1.25      # cache write ≈ 1.25x
                + u.output_tokens * out_price
            ) / 1_000_000
        else:
            self.cost_known = False
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    # ---------- openrouter backend ----------

    async def _acall_openrouter(self, system: str, user: str, schema: dict,
                                max_tokens: int) -> dict:
        import aiohttp
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY не заданий. Впишіть ключ у файл .env "
                "(рядок OPENROUTER_API_KEY=sk-or-...)."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.review-scoring",
            "X-Title": "review-scoring",
        }

        def body(with_schema: bool, with_reasoning: bool) -> dict:
            system_content = system
            if "claude" in self.model.lower():
                # OpenRouter forwards cache_control through to Anthropic
                # models. The system prompt is identical on every call of a
                # given step (extraction/grouping/...), so caching it turns
                # a repeated ~2-3k token system prompt into a ~0.1x-priced
                # cache read after the first call within the TTL window.
                system_content = [{"type": "text", "text": system,
                                   "cache_control": {"type": "ephemeral"}}]
            messages = [{"role": "system", "content": system_content}]
            if with_schema:
                messages.append({"role": "user", "content": user})
            else:
                messages.append({"role": "user", "content":
                    user + "\n\nRespond with ONLY a JSON object matching this "
                    "JSON Schema (no prose, no code fences):\n"
                    + json.dumps(schema, ensure_ascii=False)})
            b = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if with_schema:
                b["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True,
                                    "schema": schema},
                }
            if with_reasoning:
                # unified OpenRouter reasoning parameter (extended thinking)
                b["reasoning"] = {"effort": self.effort}
            return b

        res = _res()
        session = res.get("session")
        if session is None:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600))
            res["session"] = session

        attempts = [True, False] if self._or_schema_ok is None else \
                   [self._or_schema_ok]
        last_err: Exception | None = None
        for with_schema in attempts:
            for retry in range(6):
                with_reasoning = bool(self.effort) and \
                                 self._or_reasoning_ok is not False
                try:
                    async with session.post(
                            OPENROUTER_URL, headers=headers,
                            json=body(with_schema, with_reasoning)) as resp:
                        status = resp.status
                        text = await resp.text()
                    if status == 429 or status >= 500:
                        last_err = RuntimeError(
                            f"OpenRouter {status}: {text[:300]}")
                        await asyncio.sleep(min(60, 2 ** retry * 2))
                        continue
                    data = json.loads(text)
                    if status != 200 or "error" in data:
                        err = data.get("error", {})
                        msg = str(err.get("message", text[:300]))
                        # upstream hiccups arrive as HTTP 200 + error body —
                        # retry them with backoff instead of failing the run
                        transient = (err.get("code") in (429, 500, 502, 503, 529)
                                     or "overload" in msg.lower()
                                     or "rate limit" in msg.lower()
                                     or "timed out" in msg.lower())
                        if transient:
                            last_err = RuntimeError(
                                f"OpenRouter {status}: {msg}")
                            await asyncio.sleep(min(60, 2 ** retry * 2))
                            continue
                        raise RuntimeError(
                            f"OpenRouter {status}: {msg}")
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        reasoning = data["choices"][0]["message"].get("reasoning")
                        raise RuntimeError(
                            "OpenRouter повернув порожній content "
                            f"(finish_reason={data['choices'][0].get('finish_reason')}, "
                            f"reasoning_len={len(reasoning or '')}) — модель "
                            "не підтримує цей формат виводу належним чином.")
                    result = _parse_json_loose(content)
                    self._or_schema_ok = with_schema
                    if with_reasoning:
                        self._or_reasoning_ok = True
                    usage = data.get("usage") or {}
                    self.input_tokens += usage.get("prompt_tokens", 0) or 0
                    self.output_tokens += usage.get("completion_tokens", 0) or 0
                    self.cache_read_tokens += (
                        usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0)
                    self.cache_write_tokens += (
                        usage.get("prompt_tokens_details", {}).get("cache_write_tokens", 0) or 0)
                    if "cost" in usage:
                        self.cost_usd += usage["cost"] or 0.0
                    else:
                        self.cost_known = False
                    return result
                except (RuntimeError, json.JSONDecodeError,
                        aiohttp.ClientError, asyncio.TimeoutError,
                        KeyError) as e:
                    last_err = e
                    if isinstance(e, (aiohttp.ClientError,
                                      asyncio.TimeoutError)):
                        await asyncio.sleep(min(60, 2 ** retry * 2))
                        continue
                    # the model may reject the reasoning param — drop it
                    # once and retry the same mode before switching modes
                    if with_reasoning and self._or_reasoning_ok is None:
                        self._or_reasoning_ok = False
                        continue
                    break  # schema rejected or bad JSON -> try next mode
        raise RuntimeError(f"OpenRouter виклик не вдався: {last_err}")

    # ---------- cache ----------

    def _key(self, system: str, user: str, schema: dict) -> str:
        # openrouter calls now run with reasoning enabled — the marker keeps
        # old no-reasoning cache entries from being reused for them
        variant = "or-reasoning" if self.provider == "openrouter" and self.effort else ""
        payload = json.dumps(
            [self.provider, self.model, self.effort, variant,
             system, user, schema],
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
