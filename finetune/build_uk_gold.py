"""Build UK statutory gold from Companies House, via its PUBLIC website.

The API returns 401 without a key, but find-and-update.company-information
.service.gov.uk is free to fetch, so no signup is needed.

Two design points, both learned the hard way today:

1. GOLD COMES FROM THE REGISTER, NOT FROM APOLLO. Haiku agents were asked to
   verify Apollo's claimed owner and only ~11 pct confirmed, but that number is
   confounded and must not be quoted as "Apollo is 11 pct accurate": several
   matches landed on DISSOLVED shells with similar names (M RESTAURANTS LTD is
   dissolved), and several source rows have junk in the person field
   ("Justicia Divina", "Hristo B."). What the agents produced that IS reliable
   is the officer list for each matched company. So Apollo's claim is discarded
   and the register's active officers become the label.

2. ANY ACTIVE OFFICER COUNTS. A company with four directors has four correct
   answers. Scoring one arbitrary pick as the only truth is what made the
   campaign set read 52 pct strict against 75 pct crediting alternates. Gold is
   therefore a LIST of surnames.

Dissolved and liquidating companies are dropped: their officer lists are stale,
and the agent should not be expected to find a current owner for a dead entity.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import re
from pathlib import Path

import httpx

from finetune.namefold import fold, surname

BASE = "https://find-and-update.company-information.service.gov.uk"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; lightgent-eval/1.0)"}

DEAD = re.compile(r"dissolved|liquidation|in administration|converted / closed", re.I)


def officer_surname(name: str) -> str:
    """Companies House writes 'SURNAME, Firstname'."""
    part = name.split(",")[0] if "," in name else name
    return surname(part)


async def company_status(client: httpx.AsyncClient, number: str) -> str | None:
    try:
        resp = await client.get(f"{BASE}/company/{number}", timeout=30)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    html = resp.text
    match = re.search(r'id="company-status"[^>]*>\s*([^<]{3,60})', html)
    if match:
        return match.group(1).strip()
    return "Dissolved" if re.search(r">\s*Dissolved\s*<", html) else "Active"


async def main_async(args) -> None:
    rows = []
    for path in sorted(glob.glob(str(Path(args.in_dir) / "uk_verified_*.json"))):
        try:
            rows += json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

    candidates = [
        r for r in rows
        if r.get("company_number") and (r.get("active_officers") or [])
    ]
    print(f"merged rows: {len(rows)} | with register match + officers: {len(candidates)}")

    sem = asyncio.Semaphore(8)

    async def check(row):
        async with sem:
            return row, await company_status(client, str(row["company_number"]))

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        results = await asyncio.gather(*(check(r) for r in candidates))

    gold, dropped = [], {"dead": 0, "no_surname": 0, "unknown_status": 0}
    seen: set[str] = set()
    for row, status in results:
        if status is None:
            dropped["unknown_status"] += 1
            continue
        if DEAD.search(status):
            dropped["dead"] += 1
            continue
        surnames = sorted({
            officer_surname(o.get("name", "")) for o in row["active_officers"]
            if officer_surname(o.get("name", ""))
        })
        if not surnames:
            dropped["no_surname"] += 1
            continue
        domain = fold(row["domain"]).strip().removeprefix("www.")
        if domain in seen:
            continue
        seen.add(domain)
        gold.append({
            "domain": domain,
            "company_name": row.get("company", ""),
            "gold": surnames,
            "officers": [o.get("name") for o in row["active_officers"]],
            "company_number": row["company_number"],
            "status": status,
            "source": "companies-house-active-officers",
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_uk.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["company_name", "domain", "city", "country"])
        writer.writeheader()
        for row in gold:
            writer.writerow({"company_name": row["company_name"], "domain": row["domain"],
                             "city": "", "country": "United Kingdom"})
    with open(out_dir / "gold_uk.jsonl", "w", encoding="utf-8") as fh:
        for row in gold:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    multi = sum(1 for g in gold if len(g["gold"]) > 1)
    print(f"dropped: {dropped}")
    print(f"\nUK statutory gold: {len(gold)} companies")
    print(f"  with more than one active officer: {multi} "
          f"(any of them scores as correct)")
    print(f"  {out_dir/'eval_uk.csv'}\n  {out_dir/'gold_uk.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="finetune/data")
    parser.add_argument("--out-dir", default="finetune/data/uk_gold")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
