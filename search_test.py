"""Isolated SearXNG diagnostic — test the search backend ALONE, no LLM.

Reads SEARXNG_URL from .env and probes each engine separately, plus the
site:/quoted queries LightGent depends on, so we can see exactly which
engines respond through the proxy and whether operators are honored.

Usage:
  python search_test.py                       # full diagnostic
  python search_test.py "any query here"      # one ad-hoc query, all engines
"""

from __future__ import annotations

import sys

import httpx
from lightgent_service import Settings

settings = Settings()
BASE = settings.searxng_url.rstrip("/")


def probe(label: str, params: dict) -> int:
    try:
        r = httpx.get(f"{BASE}/search", params={**params, "format": "json"}, timeout=90)
        d = r.json()
    except Exception as e:
        print(f"\n=== {label} ===\n  ERROR {type(e).__name__}: {e}")
        return 0
    res = d.get("results", [])
    engs: dict[str, int] = {}
    for x in res:
        for e in x.get("engines", []):
            engs[e] = engs.get(e, 0) + 1
    print(f"\n=== {label} ===")
    print(f"  http {r.status_code} | results {len(res)} | by engine {engs}")
    unr = d.get("unresponsive_engines")
    if unr:
        print(f"  unresponsive: {unr}")
    for x in res[:4]:
        print("   -", x.get("url"))
    return len(res)


def main() -> None:
    print("SEARXNG_URL:", BASE)
    if len(sys.argv) > 1:
        probe(f"ad-hoc: {sys.argv[1]!r}", {"q": sys.argv[1]})
        return

    # 1. Each engine alone on a trivial query — who responds through the proxy?
    for eng in ["google", "bing", "duckduckgo", "brave"]:
        probe(f"plain 'openai' via {eng} only", {"q": "openai", "engines": eng})

    # 2. The operator queries LightGent actually needs
    probe("site: via google only", {"q": "site:linkedin.com/in WUA Amsterdam CEO", "engines": "google"})
    probe("quoted+site: via google only", {"q": '"Henk Kroezen" site:linkedin.com/in', "engines": "google"})
    probe("site: via bing only", {"q": "site:linkedin.com/in WUA Amsterdam CEO", "engines": "bing"})

    # 3. Default (all engines) — what the agent gets today
    probe("DEFAULT all-engines site: query", {"q": '"Henk Kroezen" site:linkedin.com/in'})


if __name__ == "__main__":
    main()
