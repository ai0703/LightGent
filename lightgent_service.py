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
import os
import random
import re
import unicodedata
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI
from openai import (AsyncOpenAI, APIConnectionError, BadRequestError,
                    InternalServerError, RateLimitError)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from proxy_balancer import ProxyBalancer
from searxng_balancer import SearxngBalancer

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
    trajectory_log_dir: str = ""

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
    # tier 0 = Serper (Google). Leads the chain when set: SearXNG upstreams
    # get rate-limited and silently return irrelevant results.
    serper_api_key: str = ""
    serper_country: str = "nl"
    serper_lang: str = "nl"
    apify_token: str = ""
    apify_search_actor: str = "apify~google-search-scraper"

    # Optional comma-separated SOCKS5 proxies for Jina fetches (empty = direct)
    jina_proxies: str = ""
    # File of datacenter proxies, one `ip:port:user:pass` per line, gitignored.
    # Measured 2026-07-26: 100 such IPs sustain 2,393 fetches/min through the
    # balancer with 0 pct waste, versus 39/min from a single direct IP that 429s
    # on 68 pct of calls. So when this file exists it is the PRIMARY fetch lane.
    proxy_list_file: str = "proxies.txt"
    # Per-attempt timeout for ONE proxy lane, not for the whole fetch. With 100
    # lanes, abandoning a silent lane fast beats waiting on it: p95 through the
    # pool is 1.3s, so 12s is already ~9x the slowest healthy response.
    fetch_lane_timeout: int = 12
    # Paid Jina Reader key. Used ONLY as a fallback when the free proxy path
    # fails or is rate-limited — keeps paid spend to the overflow only.
    jina_api_key: str = ""

    max_iterations: int = 12
    # Floor for the answer-or-search decision: below this many tool calls an
    # answer is only accepted when the values it reports are actually present
    # in the evidence gathered so far. Off by default because LightGent serves
    # caller-defined tasks, some of which are legitimately answerable without
    # research. Set MIN_TOOL_CALLS=2 for enrichment runs, where an eager null
    # is the expensive failure.
    min_tool_calls: int = 0
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

    def datacenter_proxies(self) -> list[str]:
        """Proxy URLs from proxy_list_file, or [] when the file is absent."""
        path = Path(self.proxy_list_file)
        if not self.proxy_list_file or not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            f = line.strip().split(":")
            if len(f) >= 4:
                out.append(f"http://{f[2]}:{f[3]}@{f[0]}:{f[1]}")
            elif len(f) == 2:
                out.append(f"http://{f[0]}:{f[1]}")
        return out


settings = Settings()
llm_semaphore = asyncio.Semaphore(settings.max_concurrent)

# Lazily started balancers, shared for the process lifetime. Both are async
# context managers; we enter them once under a lock rather than per request,
# because their whole value is the state they accumulate across calls (per-IP
# pacing, per-shard engine health).
_fetch_pool: "ProxyBalancer | None" = None
_search_pool: "SearxngBalancer | None" = None
_pool_lock = asyncio.Lock()


async def _get_fetch_pool():
    """ProxyBalancer over the datacenter list, or None when not configured."""
    global _fetch_pool
    if _fetch_pool is not None:
        return _fetch_pool
    async with _pool_lock:
        if _fetch_pool is None:
            proxies = settings.datacenter_proxies()
            if not proxies:
                return None
            # Short per-attempt timeout on purpose. Measured p50 0.55s and p95
            # 1.3s through this pool, so a lane that has not answered in 12s is
            # dead, and waiting tool_timeout (45s) on it just to retry elsewhere
            # turned one fetch into 85s in the first wired run. Failing fast and
            # switching lanes is strictly better when there are 100 lanes.
            pool = ProxyBalancer(proxies, per_lane_connections=4,
                                 timeout=settings.fetch_lane_timeout)
            await pool.__aenter__()
            _fetch_pool = pool
            log.info("fetch balancer started with %d proxy lanes", len(proxies))
    return _fetch_pool


async def _get_search_pool():
    """SearxngBalancer over the shard pool, or None when fewer than 1 endpoint."""
    global _search_pool
    if _search_pool is not None:
        return _search_pool
    async with _pool_lock:
        if _search_pool is None:
            urls = settings.searxng_endpoints()
            if not urls:
                return None
            pool = SearxngBalancer(urls, timeout=settings.tool_timeout)
            await pool.__aenter__()
            _search_pool = pool
            log.info("search balancer started with %d shard(s)", len(urls))
    return _search_pool
trajectory_log_lock = asyncio.Lock()


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
    """Normalise any provider's rows to the agent-facing shape.

    ensure_ascii=False is load-bearing, not cosmetic. json.dumps defaults to
    ASCII escaping, so every non-ASCII character reached the model as a literal
    backslash-u sequence: "Wikstrøm" arrived as "Wikstr\\u00f8m". The model then
    had to read mangled text, and any grounding check comparing its answer back
    against the evidence failed on the same characters.

    Measured on the Brreg set, five CORRECT Nordic answers looked ungrounded
    for exactly this reason (Wikstrøm, Strøm, Grønlie, Røvik, Måle). It also
    degrades every accented language, not just Norwegian: Dutch ë, German
    umlauts, French accents all hit it.
    """
    return json.dumps([
        {"url": r.get("url"),
         "title": (r.get("title") or "")[:120],
         "snippet": (r.get("content") or r.get("snippet") or r.get("description") or "")[:240]}
        for r in rows[:6]
    ], ensure_ascii=False)


async def _search_serper(query: str, http: httpx.AsyncClient) -> list[dict]:
    """Tier 0: Serper (Google). Real results, unlike a rate-limited SearXNG.

    Measured 2026-07-26: with both SearXNG upstreams throttled (duckduckgo
    timeout, google cse "too many requests"), a Dutch company-owner query
    returned Jeffrey Epstein articles. The same query through Serper returned
    the company LinkedIn page, its KVK record and its team page.
    """
    key = settings.serper_api_key
    if not key:
        return []
    resp = await http.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "gl": settings.serper_country, "hl": settings.serper_lang,
              "num": 8},
        timeout=settings.tool_timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    rows = []
    for item in (payload.get("organic") or []):
        rows.append({
            "url": item.get("link"),
            "title": item.get("title"),
            "content": item.get("snippet"),
        })
    # Knowledge panel often names the founder or CEO outright.
    kg = payload.get("knowledgeGraph") or {}
    if kg.get("title"):
        attrs = "; ".join(f"{k}: {v}" for k, v in (kg.get("attributes") or {}).items())
        rows.insert(0, {
            "url": kg.get("website") or kg.get("descriptionLink") or "",
            "title": f"[knowledge graph] {kg.get('title')}",
            "content": " ".join(x for x in (kg.get("description"), attrs) if x),
        })
    return rows


async def _search_searxng(query: str, http: httpx.AsyncClient) -> list[dict]:
    """Tier 1: self-hosted SearXNG pool, routed by engine health.

    The balancer reads `unresponsive_engines` from each answer and steers queries
    away from a shard whose Google is suspended, while still using that shard for
    Bing and DuckDuckGo. Measured 2026-07-26: one instance held 86 pct Google,
    three shards round-robin 94 pct, because a suspension is per instance and
    takes Google off EVERY query through it for the whole cooldown.
    """
    pool = await _get_search_pool()
    if pool is None:
        return []
    payload, _shard = await pool.search(query, prefer_engine="google")
    if payload is None:
        raise RuntimeError("all searxng shards failed")
    return payload.get("results", [])


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
    """Serper first when a key is set, then SearXNG, then OmniRoute
    DuckDuckGo, then Apify. Each tier is tried only when the one above it
    fails. Serper leads because result QUALITY, not speed, was the binding
    constraint measured on 2026-07-26."""
    # Tier 0 — Serper (Google)
    if settings.serper_api_key:
        try:
            rows = await _search_serper(query, http)
            if rows:
                return _fmt_results(rows)
            log.warning("serper empty q=%r -> fallback", query[:70])
        except Exception as e:
            log.warning("serper failed (%s) q=%r -> fallback", type(e).__name__, query[:70])
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

    # 1) PRIMARY: the datacenter proxy pool, paced per IP by the balancer.
    # Jina's limit is PER IP (measured: 125 of 184 calls 429ed from one IP,
    # zero 429s across rotating IPs), so the pool is what makes fetch scale.
    pool = await _get_fetch_pool()
    if pool is not None:
        status, text, _lane = await pool.get(jina_url, headers=headers)
        if status == 200:
            return text[: settings.fetch_truncate]
        log.warning("jina via proxy pool status=%s url=%s", status, url[:80])
        # 1b) Straight to unauthenticated direct. Measured 2026-07-26: the SOCKS5
        # lane returns 451 and the paid key 402, so trying them after a pool miss
        # is a guaranteed ~3s tax before the lane that actually works. Direct is
        # 1.1-1.6s and rate limits only above ~39/min, which the pool absorbs.
        try:
            async with httpx.AsyncClient(timeout=settings.fetch_lane_timeout) as client:
                resp = await client.get(jina_url, headers=headers)
            if resp.status_code == 200:
                return resp.text[: settings.fetch_truncate]
            log.warning("jina direct status %s url=%s", resp.status_code, url[:80])
        except Exception as e:
            log.warning("jina direct failed: %s url=%s", e, url[:80])

    # 2) SOCKS5 proxies, when configured. Kept for the residential lane.
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


_TUSSENVOEGSELS = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la"}

# NFKD DELETES characters it cannot decompose, so "Oyvind" and "Øyvind" folded
# to different strings and a correct answer looked ungrounded. Transliterate
# first, the way each language romanises itself.
_TRANSLITERATE = {
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "å": "a", "Å": "a",
    "ð": "d", "Ð": "d", "þ": "th", "Þ": "th", "ß": "ss",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "İ": "i",
}


_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})")


def _fold(text: str) -> str:
    # Decode literal backslash-u sequences first. Tool output used to carry
    # them verbatim (see _fmt_results), and any already-banked trajectory or
    # double-encoding provider still does, so grounding must see through it.
    out = _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), text or "")
    out = "".join(_TRANSLITERATE.get(ch, ch) for ch in out)
    return unicodedata.normalize("NFKD", out).encode("ascii", "ignore").decode().lower()


def _grounded_in(value: str, seen_text: str) -> bool:
    """Is this reported value actually present in what the tools returned?

    Deliberately task-agnostic: LightGent fills caller-defined output fields,
    so this cannot key on owner_name. For person names the last significant
    token (the surname, skipping Dutch tussenvoegsels) is the discriminating
    part; for other values the whole string is checked.
    """
    haystack = _fold(seen_text)
    if not haystack:
        return False
    folded = _fold(value)
    if folded and folded in haystack:
        return True
    parts = [p for p in re.sub(r"[^a-z ]", " ", folded).split()
             if p not in _TUSSENVOEGSELS and len(p) > 1]
    return bool(parts) and parts[-1] in haystack


def _answer_is_grounded(data: Any, seen_text: str) -> bool:
    """True when the answer asserts at least one substantive value that the
    evidence supports. An all-null answer is never 'grounded', which is
    exactly the eager-abstention case the floor exists to catch."""
    if not isinstance(data, dict):
        return False
    return any(
        isinstance(v, str) and v.strip() and _grounded_in(v, seen_text)
        for v in data.values()
    )


_PERSON_FIELDS = ("owner_name", "name", "full_name", "contact_name", "person_name")


def _strip_ungrounded_person(data: Any, seen_text: str) -> Any:
    """Null out a reported PERSON whose surname never appeared in evidence.

    Measured on the Brreg statutory set: 2 of 120 answers named someone absent
    from the gathered text entirely. adnav.com returned "Anders Nilsen" with
    neither token anywhere in 11k characters, a generically plausible Norwegian
    name invented from nothing. 5-pluss.no returned "Rune Gjelsteen Johansson"
    where the first two tokens WERE present but the surname was not.

    Fabrication was 0 of 107 on Dutch companies and 2 of 120 on Norwegian ones,
    so the risk rises when the name space is unfamiliar. For a lead product a
    fabricated contact is worse than no contact: it burns a send and can cost a
    sender domain. This converts that failure into an honest null.

    Only person fields are touched. Titles and free text legitimately paraphrase
    what a page said, so holding them to a substring test would delete good data.
    """
    if not isinstance(data, dict):
        return data
    if not (seen_text or "").strip():
        # No research happened, so there is nothing to check a claim against
        # and a substring test against "" would null every field
        # unconditionally. Answering with no evidence is a different failure,
        # and min_tool_calls is the guard for it.
        return data
    out = dict(data)
    for field in _PERSON_FIELDS:
        value = out.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        if _grounded_in(value, seen_text):
            continue
        log.warning("ungrounded person %r in %s - nulling", value[:60], field)
        out[field] = None
        # The dependent fields described that person, so they go too.
        for dependent in ("title", "linkedin_url"):
            if dependent in out:
                out[dependent] = None
        if "confidence" in out:
            out["confidence"] = "low"
    return out


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

def _response_lane(response: Any) -> Any:
    extra = getattr(response, "model_extra", None)
    if not isinstance(extra, dict):
        extra = getattr(response, "__pydantic_extra__", None)
    broker = extra.get("_broker") if isinstance(extra, dict) else None
    return broker.get("lane") if isinstance(broker, dict) else None


async def _write_trajectory(record: dict[str, Any], trajectory_dir: str | Path | None = None) -> None:
    log_dir = trajectory_dir if trajectory_dir is not None else settings.trajectory_log_dir
    if not log_dir:
        return
    try:
        directory = Path(log_dir)
        async with trajectory_log_lock:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"trajectories-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                # Durability is process-local: one service process and this module
                # asyncio.Lock. Multi-process writers are out of scope.
                os.fsync(handle.fileno())
    except Exception:
        log.exception("failed to write trajectory log")


async def run_agent(
    req: ResearchRequest,
    http: httpx.AsyncClient,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    trajectory_dir: str | Path | None = None,
) -> ResearchResponse:
    resolved_base_url = settings.llm_base_url if base_url is None else base_url
    resolved_model = settings.llm_model if model is None else model
    resolved_api_key = settings.llm_api_key if api_key is None else api_key
    started = time.monotonic()
    state: dict[str, Any] = {}
    result: ResearchResponse | None = None
    try:
        result = await _run_agent(
            req, http, state, resolved_base_url, resolved_model, resolved_api_key
        )
        return result
    finally:
        resolved_trajectory_dir = (
            settings.trajectory_log_dir if trajectory_dir is None else trajectory_dir
        )
        if resolved_trajectory_dir:
            now = datetime.now(timezone.utc)
            await _write_trajectory({
                "ts": now.isoformat(),
                "subject": req.context,
                "task": req.task,
                "fields_or_schema": req.output_fields,
                "model": resolved_model,
                "lane": state.get("lane"),
                "lanes": state.get("lanes", []),
                "status": result.status if result is not None else "error",
                "iterations": result.iterations if result is not None else state.get("iterations", 0),
                "tool_calls": result.tool_calls if result is not None else state.get("tool_calls", 0),
                "duration_sec": time.monotonic() - started,
                "messages": state.get("messages", []),
                "final_content": state.get("final_content"),
                "data": result.data if result is not None else None,
            }, resolved_trajectory_dir)


async def _run_agent(req: ResearchRequest, http: httpx.AsyncClient,
                     state: dict[str, Any], base_url: str, model: str,
                     api_key: str) -> ResearchResponse:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or "none")
    messages: list[Any] = [
        {"role": "system", "content": build_system_prompt(req.task, req.context, req.output_fields)},
        {"role": "user", "content": "Begin research now."},
    ]
    state["messages"] = messages

    max_iterations = req.max_iterations or settings.max_iterations
    force_conclude_at = max(max_iterations - 3, 1)
    use_tools = True  # flips to tag mode if the server lacks a tool-call parser
    total_tool_calls = 0
    seen_text = ""  # everything the tools returned — used to catch fabricated URLs

    for iteration in range(max_iterations):
        state["iterations"] = iteration + 1
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
            model=model,
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
        response_lane = _response_lane(response)
        state.setdefault("lanes", []).append(response_lane)
        if response_lane is not None:
            state["lane"] = response_lane
        tool_calls = _normalise_tool_calls(msg)
        if not tool_calls and msg.content:
            tool_calls = _parse_tag_tool_calls(msg.content)

        if not tool_calls:
            # FLOOR. Nothing used to stop the model answering on turn one with
            # no evidence at all, so any eagerness (from training or from a
            # confident base model) produced instant nulls that the loop
            # accepted as successes. An answer is only admissible once real
            # evidence exists, or if it is a grounded non-null answer.
            data = _extract_json(msg.content or "")
            if (total_tool_calls < settings.min_tool_calls
                    and not _answer_is_grounded(data, seen_text)):
                log.warning("premature answer after %d tool calls - pushing back",
                            total_tool_calls)
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({"role": "user", "content": (
                    "You have not gathered enough evidence yet. Do NOT answer "
                    "now. Call the research tools first, then answer."
                )})
                continue

            state["final_content"] = msg.content
            messages.append({"role": "assistant", "content": msg.content})
            if data is None:
                log.warning("parse_error; final content (first 600): %r",
                            (msg.content or "")[:600])
            else:
                data = _strip_fabricated_urls(data, seen_text)
                data = _strip_ungrounded_person(data, seen_text)
            return ResearchResponse(
                status="success" if data is not None else "parse_error",
                data=data, iterations=iteration + 1, tool_calls=total_tool_calls,
            )

        total_tool_calls += len(tool_calls)
        state["tool_calls"] = total_tool_calls
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

    # FORCED FINALISATION. Running out of iterations used to return nothing,
    # even when the evidence already named the person. Measured 2026-07-26:
    # 9 of 9 companies had the correct name in the tool results and every one
    # returned null, because the loop never asked the model to conclude. One
    # extra turn with no tools offered turns those into real answers.
    log.warning("max iterations reached - forcing a final answer")
    try:
        messages.append({"role": "user", "content": (
            "You have run out of research steps. Do NOT call any more tools. "
            "Using ONLY the evidence already gathered above, output the final "
            "JSON object now. Fill every field you can support with evidence "
            "you actually saw, and use null for anything you genuinely could "
            "not find."
        )})
        # CEILING. The old nudge asked the model to stop calling tools while
        # still OFFERING them, and it answered with a plain-text <tool_call>
        # block every time, which nothing parsed. Measured 2026-07-26 over
        # n=20: 17 companies had the owner name in hand, 1 emitted the JSON.
        # Withdrawing the tools is what makes the instruction enforceable.
        final_resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
            tools=TOOLS_SCHEMA, tool_choice="none",
            timeout=settings.llm_timeout,
        )
        final_msg = final_resp.choices[0].message
        state["final_content"] = final_msg.content
        if final_msg.content:
            data = _extract_json(final_msg.content)
            if data is not None:
                data = _strip_fabricated_urls(data, seen_text)
                data = _strip_ungrounded_person(data, seen_text)
                log.info("forced finalisation produced an answer")
                return ResearchResponse(status="success", data=data,
                                        iterations=max_iterations,
                                        tool_calls=total_tool_calls)
            log.warning("forced finalisation did not parse: %r",
                        (final_msg.content or "")[:300])
    except Exception as exc:  # noqa: BLE001 - never let this mask the timeout
        log.warning("forced finalisation failed (%s)", type(exc).__name__)

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
