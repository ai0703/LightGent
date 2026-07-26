"""LightGent — a lightweight Claygent clone: web-research enrichment agent.

Adapted from local-enrichos/services/enrich-service (agent.py / tools.py /
config.py). Same battle-tested loop: OpenAI-compatible endpoint via env-driven
base URL, parallel tool dispatch, <tool_call> tag fallback for vLLM servers
without a tool parser, force-conclude near the iteration cap, and
bracket-balanced JSON extraction from messy OSS-model output.

Difference vs enrich-service: the task, context, and output fields are
request-driven (like a Claygent column) instead of hardcoded employee research.

Run:  uvicorn lightgent_service:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI
from openai import (AsyncOpenAI, APIConnectionError, BadRequestError,
                    InternalServerError, RateLimitError)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lightgent")


# ── Config ────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI-compatible endpoint — point at the Colab Cloudflare tunnel
    # (https://xxxx.trycloudflare.com/v1), RunPod, Ollama (/v1), anything.
    llm_base_url: str = ""
    # Pool of endpoints for multi-Colab parallelism (comma-separated, each
    # ending in /v1). When set, the batch runner round-robins rows across them
    # for N-times throughput. Falls back to llm_base_url when empty.
    llm_base_urls: str = ""
    llm_api_key: str = "none"
    llm_model: str = ""

    searxng_url: str = ""
    # Pool of SearXNG endpoints (comma-separated). With many Colab sessions you
    # have many search backends — spread search load and survive one dying.
    # Falls back to searxng_url.
    searxng_urls: str = ""
    searxng_token: str = ""
    # Search fallback tiers (used only when SearXNG errors/saturates):
    # tier 2 = OmniRoute's hosted search (/v1/search, duckduckgo-free — free);
    # tier 3 = Apify search-scraper actor (only if apify_token is set).
    omniroute_search_url: str = "http://localhost:20128/v1/search"
    apify_token: str = ""
    apify_search_actor: str = "apify~google-search-scraper"

    # Optional comma-separated SOCKS5 proxies for Jina fetches (empty = direct)
    jina_proxies: str = ""
    # Paid Jina Reader key. Used ONLY as a fallback when the free proxy path
    # fails or is rate-limited — keeps paid spend to the overflow only.
    jina_api_key: str = ""

    max_iterations: int = 12
    llm_timeout: int = 180
    tool_timeout: int = 45
    # A single Colab GPU chokes fast — keep in-flight LLM calls low.
    max_concurrent: int = 3

    fetch_truncate: int = 2500
    # Endurance: when the LLM backend is down/rate-limited mid-company, hold the
    # in-progress conversation and retry the SAME step rather than discarding it.
    # Give up on a company only after this many cumulative seconds of waiting.
    max_cap_wait: int = 1200

    def endpoints(self) -> list[str]:
        """All LLM endpoints to spread work across (pool, else the single URL)."""
        pool = [u.strip() for u in self.llm_base_urls.split(",") if u.strip()]
        return pool or ([self.llm_base_url] if self.llm_base_url else [])

    def searxng_endpoints(self) -> list[str]:
        """All SearXNG endpoints (pool, else the single URL)."""
        pool = [u.strip().rstrip("/") for u in self.searxng_urls.split(",") if u.strip()]
        return pool or ([self.searxng_url.rstrip("/")] if self.searxng_url else [])


settings = Settings()
llm_semaphore = asyncio.Semaphore(settings.max_concurrent)


# ── Schemas ───────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    task: str = Field(..., description="What to research, in plain English")
    context: dict[str, Any] = Field(default_factory=dict,
                                    description="The row being enriched, e.g. {company, domain, city}")
    output_fields: dict[str, str] = Field(default_factory=dict,
                                          description="field name -> description of what to put there")
    max_iterations: int | None = None


class ResearchResponse(BaseModel):
    status: Literal["success", "parse_error", "error"]
    data: Any = None
    iterations: int = 0
    tool_calls: int = 0


# ── Tools ─────────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a list of results (url, title, snippet).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the full content of a webpage as readable text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL to fetch"}},
                "required": ["url"],
            },
        },
    },
]


def _fmt_results(rows: list[dict]) -> str:
    """Normalise any provider's rows to the agent-facing shape."""
    return json.dumps([
        {"url": r.get("url"),
         "title": (r.get("title") or "")[:120],
         "snippet": (r.get("content") or r.get("snippet") or r.get("description") or "")[:240]}
        for r in rows[:6]
    ])


async def _search_searxng(query: str, http: httpx.AsyncClient) -> list[dict]:
    """Tier 1: self-hosted SearXNG pool. Raises on error/saturation."""
    pool = settings.searxng_endpoints()
    if not pool:
        return []
    base = random.choice(pool)
    headers = {"Authorization": f"Bearer {settings.searxng_token}"} if settings.searxng_token else {}
    resp = await http.get(f"{base}/search", params={"q": query, "format": "json"}, headers=headers)
    resp.raise_for_status()
    return resp.json().get("results", [])


async def _search_omniroute(query: str, http: httpx.AsyncClient) -> list[dict]:
    """Tier 2: OmniRoute hosted search (duckduckgo-free) — the SearXNG backup."""
    resp = await http.post(settings.omniroute_search_url, json={"query": query},
                           timeout=settings.tool_timeout)
    resp.raise_for_status()
    return resp.json().get("results", [])


async def _search_apify(query: str, http: httpx.AsyncClient) -> list[dict]:
    """Tier 3: Apify search-scraper (only when apify_token is set)."""
    if not settings.apify_token:
        return []
    url = (f"https://api.apify.com/v2/acts/{settings.apify_search_actor}"
           f"/run-sync-get-dataset-items?token={settings.apify_token}")
    resp = await http.post(url, json={"queries": query, "resultsPerPage": 6,
                                      "maxPagesPerQuery": 1}, timeout=90)
    resp.raise_for_status()
    items = resp.json()
    rows: list[dict] = []
    for it in items if isinstance(items, list) else []:
        rows.extend(it.get("organicResults", []) or [])
    return rows


async def web_search(query: str, http: httpx.AsyncClient) -> str:
    """SearXNG first; on error/saturation/empty fall to OmniRoute DuckDuckGo,
    then to Apify (if configured). Each tier is tried only when the one above
    it fails — the paid/heavier tiers stay idle until they're needed."""
    # Tier 1 — SearXNG
    try:
        rows = await _search_searxng(query, http)
        if rows:
            return _fmt_results(rows)
        log.warning("searxng empty q=%r -> fallback", query[:70])
    except Exception as e:
        log.warning("searxng failed (%s) q=%r -> fallback", type(e).__name__, query[:70])
    # Tier 2 — OmniRoute DuckDuckGo
    try:
        rows = await _search_omniroute(query, http)
        if rows:
            log.info("search via OmniRoute-ddg (searxng backup) q=%r", query[:60])
            return _fmt_results(rows)
    except Exception as e:
        log.warning("omniroute search failed (%s) q=%r", type(e).__name__, query[:70])
    # Tier 3 — Apify
    try:
        rows = await _search_apify(query, http)
        if rows:
            log.info("search via Apify (tier 3) q=%r", query[:60])
            return _fmt_results(rows)
    except Exception as e:
        log.warning("apify search failed (%s) q=%r", type(e).__name__, query[:70])
    return json.dumps([])


async def web_fetch(url: str) -> str:
    """Fetch a page as readable text via Jina reader.

    Order (cheapest first): free SOCKS5 proxy path → paid Jina key (fallback only,
    used when the free path fails/rate-limits) → unauthenticated direct.
    Headers ask Jina to drop images and return plain text, which cuts the tokens
    Jina bills per fetch (page content varies ~800-8000 tokens)."""
    if "r.jina.ai/" in url:
        url = url.split("r.jina.ai/", 1)[-1]
    jina_url = f"https://r.jina.ai/{url}"
    # X-Retain-Images:none + text format => fewer billed tokens, no accuracy loss
    headers = {"Accept": "text/plain", "X-Return-Format": "text", "X-Retain-Images": "none"}

    # 1) FREE: proxy path (no key). Any non-200 falls through to the paid fallback.
    proxies = [p.strip() for p in settings.jina_proxies.split(",") if p.strip()]
    for proxy in proxies:
        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=settings.tool_timeout) as client:
                resp = await client.get(jina_url, headers=headers)
                if resp.status_code == 200:
                    return resp.text[: settings.fetch_truncate]
                log.warning("jina free-proxy status %s url=%s", resp.status_code, url[:80])
        except Exception as e:
            log.warning("jina via proxy failed: %s url=%s", e, url[:80])
            continue

    # 2) PAID FALLBACK: direct with the API key (only reached when free path fails)
    if settings.jina_api_key:
        try:
            paid_headers = {**headers, "Authorization": f"Bearer {settings.jina_api_key}"}
            async with httpx.AsyncClient(timeout=settings.tool_timeout) as client:
                resp = await client.get(jina_url, headers=paid_headers)
                if resp.status_code == 200:
                    return resp.text[: settings.fetch_truncate]
                log.warning("jina paid status %s url=%s", resp.status_code, url[:80])
        except Exception as e:
            log.warning("jina paid fetch failed: %s url=%s", e, url[:80])

    # 3) LAST RESORT: unauthenticated direct
    async with httpx.AsyncClient(timeout=settings.tool_timeout) as client:
        resp = await client.get(jina_url, headers=headers)
        resp.raise_for_status()
        return resp.text[: settings.fetch_truncate]


async def _dispatch(tc: dict, http: httpx.AsyncClient) -> str:
    name = tc["function"]["name"]
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    try:
        if name == "web_search":
            return await web_search(args.get("query", ""), http)
        if name == "web_fetch":
            return await web_fetch(args.get("url", ""))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


# ── Parsing helpers (same tricks as enrich-service) ───────────────────────

def _content_to_text(content: Any) -> str:
    """Coerce an LLM message's content to plain text. Some providers return
    content as a LIST of {type,text} blocks instead of a string; regex helpers
    crash on that (TypeError: expected string, got 'list'). Normalise here."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                out.append(p.get("text") or p.get("content") or "")
        return "".join(out)
    return str(content)


def _parse_tag_tool_calls(content: Any) -> list[dict]:
    """Parse <tool_call>...</tool_call> tags emitted when vLLM tool parser is missing."""
    content = _content_to_text(content)
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    result = []
    for i, m in enumerate(matches):
        try:
            data = json.loads(m.strip())
            result.append({
                "id": f"tag_{i}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", data.get("parameters", {}))),
                },
            })
        except json.JSONDecodeError:
            pass
    return result


def _normalise_tool_calls(msg: Any) -> list[dict]:
    if not msg.tool_calls:
        return []
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in msg.tool_calls
    ]


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    """Walk back from the last close_ch balancing brackets to the matching open_ch."""
    end = text.rfind(close_ch)
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        ch = text[i]
        if ch == close_ch:
            depth += 1
        elif ch == open_ch:
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
    return None


def _extract_json(content: Any) -> Any:
    """Extract the last balanced JSON object or array from model output.

    Handles markdown fences, explanatory prose before/after, and nested
    structures. Returns None when nothing parseable is found."""
    content = _content_to_text(content)
    if not content:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", content)
    cleaned = re.sub(r"\s*```", "", cleaned)
    candidates: list[tuple[int, Any]] = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        s = _balanced_slice(cleaned, open_ch, close_ch)
        if not s:
            continue
        try:
            candidates.append((cleaned.rfind(close_ch), json.loads(s)))
        except (json.JSONDecodeError, ValueError):
            pass
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]


# ── Prompt ────────────────────────────────────────────────────────────────

# The general research methodology. Task-agnostic on purpose: it works for ANY
# goal the user's prompt states (what a company sells, B2B vs B2C, tech stack,
# a person's role, pricing, headcount, a LinkedIn URL, …). Because the method
# lives here, the per-row task prompt can stay short — the user should NOT have
# to re-explain how to search; they just say WHAT they want.
RESEARCH_METHOD = """You are LightGent, a general web-research agent. You are \
given a SUBJECT (a row of data) and a TASK, and you fill the requested OUTPUT \
FIELDS with accurate, verified facts gathered from the web using the \
web_search and web_fetch tools.

## How to research (applies to any task)
1. PLAN per field. For each output field, decide what evidence answers it and \
where that evidence most likely lives — the subject's own website (home, \
/about, /products, /services, /pricing, /solutions, /customers, /contact, \
/team), an official registry, a store/marketplace listing, news, a directory, \
or a profile site. Different fields need different sources.
2. PRIMARY SOURCE FIRST. If a website/domain is known, fetch the most relevant \
pages for the task IN PARALLEL before searching. The company's own site is the \
best evidence for what it does, sells, and offers.
3. SEARCH for what the site didn't answer. Use plain keywords — DO NOT use \
quotation marks (the search backend ignores quotes, so a quoted phrase returns \
unrelated results). Use site: to constrain a source (e.g. site:linkedin.com/in \
for a person's profile). Read the result snippets before fetching a page.
4. SEARCH THE SPECIFIC THING, not a generic category. To find an attribute of \
a named entity, search that entity by name — e.g. once you learn a person's \
name, search THAT NAME + company (not "company + job title", which surfaces the \
wrong people); to find a product's price, search the product name + pricing.
5. CORROBORATE. Prefer primary/official sources. For a claim not stated by the \
subject itself, confirm it with a second source when you can.

## Accuracy (hard rules — these matter more than completeness)
- NEVER fabricate. If a field is not supported by something you actually read, \
set it to null. Do not guess.
- Only output URLs, names, emails, numbers, and facts that appeared in a tool \
result. Never construct or guess a URL or an email from a pattern.
- Each field must contain EXACTLY what it asks for. A near-miss substitute is \
wrong — use null instead (e.g. if a personal profile is asked for, a company \
page is not an acceptable substitute).
- Before setting a field to null, run at least one targeted search for it. Null \
is only allowed after a genuine search came up empty.

## Efficiency
- PARALLEL TOOL CALLS: request independent lookups together in ONE turn.
- STOP EARLY: as soon as every field is filled (or confirmed unfindable), answer.
- You have a hard tool-call budget — do not loop or repeat searches."""


# ── Location playbooks ────────────────────────────────────────────────────
# Per-country research packs (registries, local search terms, niche directories)
# live as separate files in playbooks/. Only the pack matching the subject's
# country is injected into the prompt — rows from other countries pay zero
# tokens for it. Files are plain markdown so they can be edited without code.
PLAYBOOKS_DIR = Path(__file__).resolve().parent / "playbooks"
_COUNTRY_TO_PACK = {
    "netherlands": "nl", "nederland": "nl", "holland": "nl", "nl": "nl",
    "ireland": "ie", "ie": "ie",
    "united kingdom": "uk", "uk": "uk", "gb": "uk", "great britain": "uk",
    "england": "uk", "scotland": "uk", "wales": "uk",
    "germany": "dach", "deutschland": "dach", "de": "dach",
    "austria": "dach", "at": "dach", "switzerland": "dach", "ch": "dach",
    "sweden": "nordics", "se": "nordics", "finland": "nordics", "fi": "nordics",
    "united states": "us", "usa": "us", "us": "us", "america": "us",
}
_pack_cache: dict[str, str] = {}


def _load_playbook(context: dict[str, Any]) -> str:
    """Return the playbook section for the subject's country, or '' if none."""
    country = str(context.get("country", "")).strip().lower()
    pack = _COUNTRY_TO_PACK.get(country)
    if not pack:
        return ""
    if pack not in _pack_cache:
        path = PLAYBOOKS_DIR / f"{pack}.md"
        try:
            _pack_cache[pack] = path.read_text(encoding="utf-8").strip()
        except OSError:
            _pack_cache[pack] = ""
    text = _pack_cache[pack]
    return f"\n\n## Local research playbook (use these sources/terms first)\n{text}" if text else ""


def build_system_prompt(task: str, context: dict[str, Any],
                        output_fields: dict[str, str]) -> str:
    ctx_lines = "\n".join(f"- {k}: {v}" for k, v in context.items() if v) or "- (none provided)"
    if output_fields:
        field_lines = ",\n".join(f'  "{k}": <{v}, or null if not verifiable>'
                                 for k, v in output_fields.items())
    else:
        field_lines = '  "answer": <the answer to the task, or null if not verifiable>'
    return f"""{RESEARCH_METHOD}{_load_playbook(context)}

## Task
{task}

## Subject (the row being researched)
{ctx_lines}

## Output
When done, output ONLY a raw JSON object — no markdown fences, no prose:
{{
{field_lines},
  "confidence": "high" | "medium" | "low",
  "sources": ["url of each source actually used", ...]
}}"""


def _strip_fabricated_urls(data: Any, seen_text: str) -> Any:
    """Null out any http(s) URL in the output that never appeared in a tool
    result. Small models guess plausible URLs (e.g. linkedin.com/in/first-last)
    — this ensures every returned URL was actually seen on the web."""
    def clean(v: Any) -> Any:
        if isinstance(v, str) and v.startswith(("http://", "https://")):
            # match on the path portion so http/https + trailing-slash differences don't matter
            core = v.split("://", 1)[-1].rstrip("/")
            return v if core and core in seen_text else None
        if isinstance(v, list):
            cleaned = [clean(x) for x in v]
            # drop URLs that got nulled (keeps e.g. a "sources" list tidy)
            return [x for x in cleaned if x is not None] if any(
                isinstance(x, str) and x.startswith("http") for x in v) else cleaned
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v
    return clean(data)


# ── Agent loop ────────────────────────────────────────────────────────────

async def run_agent(req: ResearchRequest, http: httpx.AsyncClient) -> ResearchResponse:
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "none")
    messages: list[Any] = [
        {"role": "system", "content": build_system_prompt(req.task, req.context, req.output_fields)},
        {"role": "user", "content": "Begin research now."},
    ]

    max_iterations = req.max_iterations or settings.max_iterations
    force_conclude_at = max(max_iterations - 3, 1)
    use_tools = True  # flips to tag mode if the server lacks a tool-call parser
    total_tool_calls = 0
    seen_text = ""  # everything the tools returned — used to catch fabricated URLs

    for iteration in range(max_iterations):
        if iteration == force_conclude_at:
            messages.append({
                "role": "user",
                "content": (
                    "You have used many tool calls. Based on everything gathered so far, "
                    "output your FINAL JSON object now. Unverified fields are null. "
                    "No explanation — only the raw JSON object."
                ),
            })
        kwargs: dict[str, Any] = dict(
            model=settings.llm_model,
            messages=messages,
            timeout=settings.llm_timeout,
        )
        if use_tools:
            kwargs["tools"] = TOOLS_SCHEMA
            kwargs["tool_choice"] = "auto"

        # Backend-aware LLM call. If the broker/provider is down or rate-limited,
        # HOLD the (fully populated) `messages` and retry the SAME step — the
        # gathered searches/fetches are never discarded by a mid-run cooldown.
        response = None
        cap_waited = 0.0
        while response is None:
            try:
                async with llm_semaphore:
                    response = await client.chat.completions.create(**kwargs)
            except (InternalServerError, BadRequestError, RateLimitError,
                    APIConnectionError) as exc:
                msg_text = str(exc).lower()
                status = getattr(exc, "status_code", None)
                if "maximum context length" in msg_text:
                    log.warning("context overflow after %d iters; bailing", iteration + 1)
                    return ResearchResponse(status="parse_error", iterations=iteration + 1,
                                            tool_calls=total_tool_calls)
                backend_down = (
                    status in (429, 502, 503, 504)
                    or isinstance(exc, (RateLimitError, APIConnectionError))
                    or "exhausted" in msg_text or "rate limit" in msg_text
                    or "unavailable" in msg_text or "overloaded" in msg_text
                )
                if backend_down:
                    if cap_waited >= settings.max_cap_wait:
                        log.warning("backend down > max_cap_wait (%ds); giving up company",
                                    settings.max_cap_wait)
                        return ResearchResponse(status="parse_error", iterations=iteration + 1,
                                                tool_calls=total_tool_calls)
                    wait = min(120.0, 15.0 + cap_waited / 4)
                    log.warning("backend down (%s) — holding %ds, retrying same step "
                                "(iter %d, %d msgs preserved)",
                                status or type(exc).__name__, int(wait), iteration + 1,
                                len(messages))
                    await asyncio.sleep(wait)
                    cap_waited += wait
                    continue
                if use_tools:
                    log.warning("server rejected tools (%s) — switching to tag mode",
                                type(exc).__name__)
                    use_tools = False
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    continue
                raise

        msg = response.choices[0].message
        tool_calls = _normalise_tool_calls(msg)
        if not tool_calls and msg.content:
            tool_calls = _parse_tag_tool_calls(msg.content)

        if not tool_calls:
            data = _extract_json(msg.content or "")
            if data is None:
                log.warning("parse_error; final content (first 600): %r",
                            (msg.content or "")[:600])
            else:
                data = _strip_fabricated_urls(data, seen_text)
            return ResearchResponse(
                status="success" if data is not None else "parse_error",
                data=data, iterations=iteration + 1, tool_calls=total_tool_calls,
            )

        total_tool_calls += len(tool_calls)
        log.info("iteration %d: %d tool call(s)", iteration + 1, len(tool_calls))
        results = await asyncio.gather(*[_dispatch(tc, http) for tc in tool_calls])
        seen_text += "\n".join(results)

        if use_tools:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [{"id": tc["id"], "type": "function", "function": tc["function"]}
                               for tc in tool_calls],
            })
            for tc, result in zip(tool_calls, results):
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        else:
            messages.append({"role": "assistant", "content": msg.content})
            messages.append({"role": "user", "content": "\n\n".join(
                f"Result of {tc['function']['name']}:\n{r}"
                for tc, r in zip(tool_calls, results)
            )})

    log.warning("max iterations reached")
    return ResearchResponse(status="parse_error", iterations=max_iterations,
                            tool_calls=total_tool_calls)


# ── FastAPI app ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from lane_queue import get_queue
    app.state.http = httpx.AsyncClient(timeout=settings.tool_timeout)
    broker = settings.llm_base_url
    if broker:
        broker = broker.removesuffix("/v1").rstrip("/")
    app.state.queue = get_queue(broker_url=broker, api_key=settings.llm_api_key)
    await app.state.queue.start()
    yield
    await app.state.queue.stop()
    await app.state.http.aclose()


app = FastAPI(title="LightGent", lifespan=lifespan)


@app.get("/health")
async def health():
    q = app.state.queue
    return {
        "ok": True,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "queue_depth": q.queue_depth,
        "active_requests": q.active_count,
        "lanes_available": q._lanes_available,
    }


@app.get("/queue/status")
async def queue_status():
    """Check queue and lane status without making a request."""
    q = app.state.queue
    active, parked = await q._check_lanes()
    return {
        "lanes_active": active,
        "lanes_parked": parked,
        "queue_depth": q.queue_depth,
        "active_requests": q.active_count,
        "lanes_available": q._lanes_available,
    }


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """Direct research (no queue). Fails if backend is down."""
    if not settings.llm_base_url or not settings.llm_model:
        return ResearchResponse(status="error",
                                data={"error": "Set LLM_BASE_URL and LLM_MODEL in .env"})
    try:
        return await run_agent(req, app.state.http)
    except Exception as e:
        log.exception("agent crashed")
        return ResearchResponse(status="error", data={"error": str(e)})


# ── enrichos adapter ───────────────────────────────────────────────────────
# Drop-in replacement for the old enrichos enrich-service. The enrichos worker
# POSTs {company, domain, city, limit, roles} and expects {employees:[...]}
# where each item has the keys insert_employee() reads: name, title,
# linkedin_url, source, confidence. This runs LightGent's research agent on the
# free balance broker instead of the retired RunPod pod.

class EnrichCompanyRequest(BaseModel):
    company: str
    domain: str = ""
    city: str = ""
    limit: int = 3
    roles: list[str] = Field(default_factory=list)


def _enrich_task(req: "EnrichCompanyRequest") -> str:
    roles = ", ".join([r for r in req.roles if r]) or \
        "Owner, CEO, Founder, Managing Director, Director"
    where = f" (website {req.domain})" if req.domain else ""
    loc = f", located in {req.city}" if req.city else ""
    return (
        f'Find up to {req.limit} senior decision-makers at the company "{req.company}"{where}{loc}. '
        f"Only include people whose role is one of: {roles}. "
        "For each person capture: full name, exact job title, LinkedIn profile URL if you find "
        "one, the source URL where you found them, and a confidence of high/medium/low. "
        "Do not invent people or titles. If you cannot find anyone in those roles, return an "
        "empty list."
    )


def _coerce_employees(data: Any) -> list[dict]:
    """Normalise run_agent output into the worker's expected employee dicts."""
    if isinstance(data, dict):
        emps = data.get("employees")
    elif isinstance(data, list):
        emps = data
    else:
        emps = None
    if not isinstance(emps, list):
        return []
    out = []
    for e in emps:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or e.get("full_name") or "").strip()
        if not name or name.lower() in ("unknown", "null", "none"):
            continue
        out.append({
            "name": name,
            "title": e.get("title") or e.get("role") or "",
            "linkedin_url": e.get("linkedin_url") or e.get("linkedin") or None,
            "source": e.get("source") or e.get("source_url") or "",
            "confidence": e.get("confidence") or "low",
        })
    return out


@app.post("/enrich-company")
async def enrich_company(req: EnrichCompanyRequest):
    if not settings.llm_base_url or not settings.llm_model:
        return {"company": req.company, "domain": req.domain, "status": "error",
                "employees": [], "error": "Set LLM_BASE_URL and LLM_MODEL"}
    rr = ResearchRequest(
        task=_enrich_task(req),
        context={"company": req.company, "domain": req.domain, "city": req.city},
        output_fields={
            "employees": ("JSON array of the people found. Each item is an object with keys "
                          "name, title, linkedin_url, source, confidence. Return [] if none.")
        },
    )
    try:
        res = await run_agent(rr, app.state.http)
    except Exception as e:
        log.exception("enrich-company crashed")
        return {"company": req.company, "domain": req.domain, "status": "error",
                "employees": [], "error": str(e)}
    employees = _coerce_employees(res.data)
    if employees:
        status = "success"
    else:
        status = "parse_error" if res.status == "success" else res.status
    return {"company": req.company, "domain": req.domain, "status": status,
            "employees": employees, "iterations": res.iterations,
            "tool_calls": res.tool_calls}


@app.post("/research/queued", response_model=ResearchResponse)
async def research_queued(req: ResearchRequest):
    """Queue-aware research. Waits for a lane to become available (even days).

    Use this endpoint when you want guaranteed completion -- no 503 errors.
    The request sits in a queue and is dispatched as soon as a lane opens.
    """
    if not settings.llm_base_url or not settings.llm_model:
        return ResearchResponse(status="error",
                                data={"error": "Set LLM_BASE_URL and LLM_MODEL in .env"})
    try:
        from lane_queue import get_queue
        queue = get_queue()
        result = await queue.enqueue(req.model_dump())
        return ResearchResponse(**result)
    except Exception as e:
        log.exception("queued research failed")
        return ResearchResponse(status="error", data={"error": str(e)})


@app.post("/research/batch")
async def research_batch(requests: list[ResearchRequest]):
    """Process multiple requests through the queue.

    Returns results in the same order as the input requests.
    All requests wait for lanes -- none fail due to exhaustion.
    """
    if not settings.llm_base_url or not settings.llm_model:
        return [{"status": "error", "data": {"error": "Set LLM_BASE_URL and LLM_MODEL in .env"}}
                for _ in requests]
    from lane_queue import get_queue
    queue = get_queue()
    tasks = [queue.enqueue(req.model_dump()) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append({"status": "error", "data": {"error": str(r)}, "iterations": 0, "tool_calls": 0})
        else:
            out.append(r)
    return out
