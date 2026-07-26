"""Build owner gold from a STATUTORY register, not from scraped contact data.

Every label problem today traces to the same root: our gold came from scraped
sources. Apollo columns list staff rather than owners and disagreed with our
enrichment 12 times out of 12 with our answer right every time; gold.jsonl
turned out to be partly LightGent's own earlier output, making the sealed
score partly a measure of agreement with ourselves; one company carried an
empty gold list and silently scored a plausible answer as a miss; and matching
on surname substrings counted a city ("aoc Helmond") and a business name
("Camping Koops Koeienpad") as people.

The Brønnøysund Register (Norway) fixes the root cause. It is the statutory
company register, it is free, and it needs NO API KEY:

    GET /enhetsregisteret/api/enheter?...        name, org number, hjemmeside
    GET /enhetsregisteret/api/enheter/{org}/roller   Daglig leder, board

"Daglig leder" is the managing director, which for a small company is the
decision maker this task asks for. The label is a legal filing, so it cannot
be a scraped staff member and cannot be our own output.

Trade-off worth stating: these are Norwegian companies, so the eval also
changes language and web conventions. That makes it a harder, more honest test
than the Dutch agri set, where 30 percent of companies handed the model the
owner's surname inside the domain.

Selection rules:
  - must publish a website (the agent needs something to research)
  - must have exactly ONE Daglig leder, so the gold is unambiguous
  - skip companies whose daglig leder surname is inside the domain, which
    would reintroduce the eponymous freebie we found in the agri set
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

import httpx

BASE = "https://data.brreg.no/enhetsregisteret/api"
TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la", "het"}


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()


def surname(value: str) -> str:
    """Last significant name token.

    A tussenvoegsel is only a prefix when another name FOLLOWS it. Stripping
    them unconditionally turned "Quoc Trung Le" into "trung", because the
    Vietnamese surname Le collides with the Dutch preposition le.
    """
    tokens = [p for p in re.sub(r"[^a-z ]", " ", fold(value)).split() if len(p) > 1]
    if not tokens:
        return ""
    kept = [t for i, t in enumerate(tokens)
            if not (t in TUSSEN and i < len(tokens) - 1)]
    return kept[-1] if kept else tokens[-1]


def norm_domain(value: str) -> str:
    value = fold(value).strip()
    value = re.sub(r"^https?://", "", value).removeprefix("www.").split("/")[0]
    return value


def daglig_leder(client: httpx.Client, org: str) -> tuple[str, int] | None:
    """Return (full name, how many people hold the role)."""
    try:
        resp = client.get(f"{BASE}/enheter/{org}/roller", timeout=30)
        if resp.status_code != 200:
            return None
        groups = resp.json().get("rollegrupper", [])
    except (httpx.HTTPError, json.JSONDecodeError):
        return None
    names = []
    for group in groups:
        if "daglig leder" not in fold(group.get("type", {}).get("beskrivelse", "")):
            continue
        for role in group.get("roller", []):
            if role.get("fratraadt"):
                continue  # resigned; the register keeps history
            person = (role.get("person") or {}).get("navn") or {}
            full = " ".join(x for x in (person.get("fornavn"), person.get("mellomnavn"),
                                        person.get("etternavn")) if x)
            if full.strip():
                names.append(full.strip())
    if not names:
        return None
    return names[0], len(set(names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--pages", type=int, default=40)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--out-dir", default="finetune/data/registry_gold")
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen_domains: set[str] = set()

    with httpx.Client(headers={"Accept": "application/json"}) as client:
        for page in range(args.pages):
            if len(rows) >= args.target:
                break
            try:
                resp = client.get(
                    f"{BASE}/enheter",
                    params={
                        "size": args.size,
                        "page": page,
                        # Small firms: the daglig leder really is the decision
                        # maker, which is what the task asks for.
                        "fraAntallAnsatte": 5,
                        "tilAntallAnsatte": 50,
                    },
                    timeout=40,
                )
                units = resp.json().get("_embedded", {}).get("enheter", [])
            except (httpx.HTTPError, json.JSONDecodeError):
                continue
            if not units:
                break

            for unit in units:
                if len(rows) >= args.target:
                    break
                site = unit.get("hjemmeside")
                if not site:
                    continue
                domain = norm_domain(site)
                if not domain or "." not in domain or domain in seen_domains:
                    continue
                if unit.get("konkurs") or unit.get("underAvvikling"):
                    continue

                found = daglig_leder(client, unit["organisasjonsnummer"])
                time.sleep(args.sleep)
                if not found:
                    continue
                name, count = found
                if count != 1:
                    continue  # ambiguous, skip
                if surname(name) and surname(name) in domain:
                    continue  # eponymous freebie, exactly what flattered the agri set

                seen_domains.add(domain)
                rows.append({
                    "company_name": unit.get("navn", "").strip(),
                    "domain": domain,
                    "gold": [surname(name)],
                    "full_name": name,
                    "org": unit["organisasjonsnummer"],
                })
            print(f"  page {page}: {len(rows)} usable so far")

    csv_path = out_dir / "eval_registry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["company_name", "domain", "city", "country"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"company_name": row["company_name"], "domain": row["domain"],
                             "city": "", "country": "Norway"})
    gold_path = out_dir / "gold_registry.jsonl"
    with open(gold_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"domain": row["domain"], "gold": row["gold"],
                                 "full_name": row["full_name"], "org": row["org"],
                                 "source": "brreg-daglig-leder"}, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} companies with STATUTORY owner gold")
    print(f"  {csv_path}\n  {gold_path}")


if __name__ == "__main__":
    main()
