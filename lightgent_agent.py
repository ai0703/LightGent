"""LightGent orchestration layer — OpenAI Agents SDK edition.

Architecture:
  Colab (vLLM + model, Cloudflare tunnel) exposes ONE OpenAI-compatible API.
  This file runs anywhere (your PC / the droplet): the Agents SDK drives the
  loop against that endpoint, and the tools execute HERE — SearXNG for web
  search, Jina reader (hosted API, r.jina.ai) for scraping.

Reads the same .env as lightgent_service.py (LLM_BASE_URL / LLM_API_KEY /
LLM_MODEL / SEARXNG_URL / ...).

CLI:
  python lightgent_agent.py "Find the CEO and their LinkedIn" ^
      --context company="Bakkerij Holtkamp" --context city=Amsterdam ^
      --field ceo_name:"full name" --field linkedin:"profile URL"

Note: the Agents SDK requires NATIVE tool calling from the server (our vLLM
runs --tool-call-parser hermes, so this works). It has no tag-mode fallback —
if a model can't do native tool calls, use lightgent_service.py instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from contextvars import ContextVar
from typing import Any

import httpx
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.exceptions import MaxTurnsExceeded
from openai import AsyncOpenAI

from lightgent_service import (
    RESEARCH_METHOD,
    Settings,
    _extract_json,
    _strip_fabricated_urls,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lightgent.agent")

settings = Settings()
set_tracing_disabled(True)  # tracing would try to upload runs to OpenAI's platform


# ── Tools (execute locally, not on Colab) ─────────────────────────────────

# Per-run record of everything the tools returned, so we can reject any URL the
# model outputs that it never actually saw (small models fabricate plausible
# slugs). A ContextVar (not a global list) so concurrent rows stay isolated:
# each run_research task sets its own fresh list, siblings don't cross-pollute.
_seen: ContextVar[list[str]] = ContextVar("seen")


def _record(text: str) -> None:
    try:
        _seen.get().append(text)
    except LookupError:
        pass  # tool called outside a run_research context (e.g. a bare test)


@function_tool
async def web_search(query: str) -> str:
    """Search the web and return a JSON list of results (url, title, snippet)."""
    headers = {}
    if settings.searxng_token:
        headers["Authorization"] = f"Bearer {settings.searxng_token}"
    pool = settings.searxng_endpoints()
    base = random.choice(pool) if pool else ""
    async with httpx.AsyncClient(timeout=settings.tool_timeout) as http:
        resp = await http.get(f"{base}/search",
                              params={"q": query, "format": "json"}, headers=headers)
        resp.raise_for_status()
    results = resp.json().get("results", [])
    out = json.dumps([
        {
            "url": r.get("url"),
            "title": (r.get("title") or "")[:120],
            "snippet": (r.get("content") or "")[:240],
        }
        for r in results[:6]
    ])
    _record(out)
    return out


@function_tool
async def web_fetch(url: str) -> str:
    """Fetch the full content of a webpage as readable text."""
    if "r.jina.ai/" in url:
        url = url.split("r.jina.ai/", 1)[-1]
    async with httpx.AsyncClient(timeout=settings.tool_timeout) as http:
        resp = await http.get(f"https://r.jina.ai/{url}", headers={"Accept": "text/plain"})
        resp.raise_for_status()
    text = resp.text[: settings.fetch_truncate]
    _record(text)
    return text


# ── Agent ─────────────────────────────────────────────────────────────────

INSTRUCTIONS = f"""{RESEARCH_METHOD}

## Output
The user message states the task, the subject row, and the output fields. When \
done, output ONLY a raw JSON object — no markdown fences, no prose — with \
exactly the requested fields, plus:
  "confidence": "high" | "medium" | "low",
  "sources": ["url of each source actually used", ...]"""


def build_agent(base_url: str | None = None, model: str | None = None) -> Agent:
    client = AsyncOpenAI(base_url=base_url or settings.llm_base_url,
                         api_key=settings.llm_api_key or "none")
    return Agent(
        name="LightGent",
        instructions=INSTRUCTIONS,
        tools=[web_search, web_fetch],
        model=OpenAIChatCompletionsModel(model=model or settings.llm_model,
                                         openai_client=client),
        model_settings=ModelSettings(temperature=0.2, parallel_tool_calls=True),
    )


def _build_input(task: str, context: dict[str, Any],
                 output_fields: dict[str, str] | None,
                 json_schema: dict | None) -> str:
    ctx = "\n".join(f"- {k}: {v}" for k, v in context.items() if v) or "- (none)"
    head = f"## Task\n{task}\n\n## Subject (the row being researched)\n{ctx}\n\n"
    if json_schema is not None:
        # User fully defined the output shape — honor it exactly, add nothing.
        return (head + "## Output — return ONLY a raw JSON object conforming to "
                "this JSON Schema (no markdown, no prose):\n"
                + json.dumps(json_schema, indent=2))
    if output_fields:
        skeleton = ",\n".join(f'  "{k}": <{v}, or null if not verifiable>'
                              for k, v in output_fields.items())
    else:
        skeleton = '  "answer": <the answer to the task, or null if not verifiable>'
    return (head + "## Output — fill this exact JSON object (keep every key):\n"
            f"{{\n{skeleton},\n"
            '  "confidence": "high" | "medium" | "low",\n'
            '  "sources": ["url of each source actually used", ...]\n}')


async def run_research(task: str, context: dict[str, Any] | None = None,
                       output_fields: dict[str, str] | None = None,
                       json_schema: dict | None = None,
                       base_url: str | None = None, model: str | None = None) -> dict:
    """Run one enrichment. Returns {status, data, ...} like lightgent_service.

    base_url/model override the .env endpoint — the batch runner passes one per
    row to spread load across a pool of Colab tunnels."""
    _seen.set([])  # fresh per-run URL record (isolated across concurrent rows)
    agent = build_agent(base_url, model)
    try:
        result = await Runner.run(
            agent,
            _build_input(task, context or {}, output_fields, json_schema),
            max_turns=settings.max_iterations,
        )
    except MaxTurnsExceeded:
        return {"status": "parse_error", "data": None, "error": "max turns exceeded"}
    except Exception as e:
        log.warning("run failed on %s: %s", base_url or settings.llm_base_url, e)
        return {"status": "error", "data": None, "error": str(e)}
    data = _extract_json(str(result.final_output or ""))
    if data is None:
        log.warning("parse_error; final output (first 600): %r",
                    str(result.final_output)[:600])
    else:
        data = _strip_fabricated_urls(data, "\n".join(_seen.get([])))
    return {"status": "success" if data is not None else "parse_error", "data": data}


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", help="research task in plain English")
    ap.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--field", action="append", default=[], metavar="NAME:DESC")
    ap.add_argument("--schema", default=None, metavar="FILE.json",
                    help="path to a JSON Schema file for the output (overrides --field)")
    args = ap.parse_args()

    if not settings.endpoints() or not settings.llm_model:
        raise SystemExit("Set LLM_BASE_URL (or LLM_BASE_URLS) and LLM_MODEL in "
                         ".env (printed by the Colab notebook)")

    context = dict(c.split("=", 1) for c in args.context if "=" in c)
    schema = None
    fields = {}
    if args.schema:
        with open(args.schema, encoding="utf-8") as fh:
            schema = json.load(fh)
    else:
        for f in args.field:
            name, _, desc = f.partition(":")
            fields[name.strip()] = desc.strip() or name.strip()

    out = asyncio.run(run_research(args.task, context, fields, json_schema=schema))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
