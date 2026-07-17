"""LightGent batch runner — Clay-style CSV enrichment, in-process.

Reads a CSV, runs each row through the LightGent agent, writes the answers
back as new columns to <input>_enriched.csv.

Features matching the Clay "Use AI" column:
- Variable prompt with {{Column}} placeholders, filled per row from the CSV.
- Output as simple Fields (--field) OR a full JSON Schema (--schema file).
- Multi-Colab: spreads rows across every endpoint in LLM_BASE_URLS (round
  robin) for N-times throughput. Falls back to LLM_BASE_URL.

Runs the agent in-process (imports run_research) — no separate server needed,
so Claude Code can drive it directly.

Examples:
  # Fields output, prompt references the {{Domain}} column
  python batch_enrich.py leads.csv \
    --prompt "Research the company at {{Domain}} and say what they sell." \
    --field what_they_sell:"one sentence" --field business_model:"B2B or B2C"

  # JSON Schema output
  python batch_enrich.py leads.csv \
    --prompt "Profile the company at {{Domain}}." --schema schema.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

import httpx

from lightgent_agent import run_research
from lightgent_service import settings

PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def load_endpoints(path: str) -> tuple[list[str], list[str]] | None:
    """Load a notebook registry: a JSON list of Colab sessions, each an
    object {"llm": ".../v1", "searxng": "..."} (searxng optional) or a bare
    LLM URL string. Returns (llm_urls, searxng_urls) or None if no file.

    This is how you scale: run the notebook in N Colab tabs and append one
    line per session here — no .env surgery."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    llm, sx = [], []
    for e in data:
        if isinstance(e, str):
            llm.append(e)
        elif isinstance(e, dict):
            if e.get("llm"):
                llm.append(e["llm"])
            if e.get("searxng"):
                sx.append(e["searxng"])
    return llm, sx


async def live_endpoints(endpoints: list[str]) -> list[str]:
    """Ping each LLM endpoint's /models; keep only the ones that answer. With
    many ephemeral Colab tunnels some are always dead — drop them up front so
    rows aren't wasted on a broken box."""
    headers = {"Authorization": f"Bearer {settings.llm_api_key or 'none'}"}

    async def ok(ep: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(f"{ep.rstrip('/')}/models", headers=headers)
                return r.status_code == 200
        except Exception:
            return False

    checks = await asyncio.gather(*[ok(e) for e in endpoints])
    alive = [e for e, good in zip(endpoints, checks) if good]
    for e, good in zip(endpoints, checks):
        if not good:
            print(f"  [dead] skipping: {e}")
    return alive


def fill_prompt(template: str, row: dict) -> str:
    """Replace {{Column}} with the row's value (case-insensitive on column name)."""
    lower = {k.lower(): v for k, v in row.items()}
    def sub(m: re.Match) -> str:
        key = m.group(1).strip().lower()
        return str(lower.get(key, "")).strip()
    return PLACEHOLDER.sub(sub, template)


def schema_columns(schema: dict) -> list[str]:
    props = (schema or {}).get("properties", {})
    return list(props.keys())


async def enrich_row(sem: asyncio.Semaphore, prompt_tmpl: str,
                     fields: dict[str, str] | None, schema: dict | None,
                     endpoint: str, row: dict, idx: int, total: int) -> dict:
    task = fill_prompt(prompt_tmpl, row)
    context = {k: v for k, v in row.items() if v and not k.startswith("lg_")}
    async with sem:
        res = await run_research(task, context=context, output_fields=fields,
                                 json_schema=schema, base_url=endpoint)
    data = res.get("data")
    if not isinstance(data, dict):
        data = {"answer": data}
    out = dict(row)
    cols = schema_columns(schema) if schema else list((fields or {"answer": ""}).keys())
    for c in cols:
        v = data.get(c)
        out[c] = (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                  else (v if v is not None else ""))
    if not schema:  # field mode adds these meta columns
        out["lg_confidence"] = data.get("confidence", "")
        src = data.get("sources") or []
        out["lg_sources"] = "; ".join(src) if isinstance(src, list) else str(src)
    out["lg_status"] = res.get("status", "error")
    print(f"[{idx+1}/{total}] {res.get('status'):11} "
          f"via {endpoint.split('//')[-1][:30]:30} | {next(iter(context.values()), '')}")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_file")
    ap.add_argument("--prompt", required=True,
                    help="task prompt; use {{Column}} to insert CSV values")
    ap.add_argument("--field", action="append", default=[], metavar="NAME:DESC",
                    help="output column, repeatable (ignored if --schema given)")
    ap.add_argument("--schema", default=None, metavar="FILE.json",
                    help="JSON Schema file defining the output object")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="in-flight rows (default: 3 per endpoint)")
    ap.add_argument("--endpoints", default="endpoints.json", metavar="FILE.json",
                    help="notebook registry (one entry per Colab session); "
                         "overrides .env when present")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    reg = load_endpoints(args.endpoints)
    if reg:
        endpoints, sx_pool = reg
        if sx_pool:
            settings.searxng_urls = ",".join(sx_pool)  # tools read this pool
        print(f"registry {args.endpoints}: {len(endpoints)} LLM, {len(sx_pool)} SearXNG")
    else:
        endpoints = settings.endpoints()
    if not endpoints or not settings.llm_model:
        sys.exit("No endpoints. Add endpoints.json (see endpoints.example.json) "
                 "or set LLM_BASE_URL(S) + LLM_MODEL in .env")
    print(f"health-checking {len(endpoints)} endpoint(s)...")
    endpoints = await live_endpoints(endpoints)
    if not endpoints:
        sys.exit("no live LLM endpoints — start a Colab session and update .env")

    schema = None
    fields: dict[str, str] = {}
    if args.schema:
        with open(args.schema, encoding="utf-8") as fh:
            schema = json.load(fh)
    else:
        for f in args.field:
            name, _, desc = f.partition(":")
            fields[name.strip()] = desc.strip() or name.strip()
        if not fields:
            sys.exit("Provide --field NAME:DESC (one or more) or --schema FILE.json")

    src = Path(args.csv_file)
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    if not rows:
        sys.exit("empty CSV")

    concurrency = args.concurrency or (len(endpoints) * 3)
    sem = asyncio.Semaphore(concurrency)
    print(f"{len(rows)} rows | {len(endpoints)} endpoint(s) | concurrency {concurrency} | "
          f"output: {'schema' if schema else ', '.join(fields)}")

    results = await asyncio.gather(*[
        enrich_row(sem, args.prompt, fields or None, schema,
                   endpoints[i % len(endpoints)], row, i, len(rows))
        for i, row in enumerate(rows)
    ])

    out_path = Path(args.out) if args.out else src.with_name(src.stem + "_enriched.csv")
    fieldnames: list[str] = []
    for r in results:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    ok = sum(1 for r in results if r.get("lg_status") == "success")
    print(f"\ndone: {ok}/{len(results)} success -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
