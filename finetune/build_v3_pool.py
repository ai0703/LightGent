"""Build the v3 banking pool and its gold, with hard exclusion of eval data.

v3 uses rejection sampling: bank with v2 as its own teacher, then keep only
trajectories whose answer is independently corroborated. That is only sound if
the pool shares nothing with anything the model has trained on or will be
measured against, so exclusion is computed from three places at once:

  - the frozen dev and sealed domain lists
  - every domain in every banked trajectory on disk
  - both campaign eval tiers (48 replied, 200 delivered)

The 200-company delivered tier is the v3 holdout. It must never appear here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import unicodedata
from pathlib import Path

FREE_MAIL = {
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "live.nl",
    "ziggo.nl", "kpnmail.nl", "icloud.com", "me.com", "hotmail.nl",
    "protonmail.com", "googlemail.com",
}

DECISION_MAKER = re.compile(
    r"\b(owner|co-owner|founder|co-founder|founding partner|ceo|chief executive"
    r"|managing partner|managing director|algemeen directeur|eigenaar|oprichter"
    r"|dga|president|proprietor)\b",
    re.I,
)

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la", "het"}


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()


def surname(value: str) -> str:
    parts = [p for p in re.sub(r"[^a-z ]", " ", fold(value)).split()
             if p not in TUSSEN and len(p) > 1]
    return parts[-1] if parts else ""


def norm_domain(value: str) -> str:
    value = fold(value).strip()
    value = re.sub(r"^https?://", "", value).removeprefix("www.").split("/")[0]
    return value


def excluded_domains(data_dir: Path) -> set[str]:
    out: set[str] = set()
    for name in ("test_domains.txt", "dev_domains.txt"):
        path = data_dir / name
        if path.exists():
            out |= {norm_domain(l) for l in open(path, encoding="utf-8") if l.strip()}
    for path in glob.glob(str(data_dir / "**" / "trajectories-*.jsonl"), recursive=True):
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            domain = (row.get("subject") or {}).get("domain")
            if domain:
                out.add(norm_domain(domain))
    for tag in ("replied", "delivered"):
        path = data_dir / "campaign_eval" / f"eval_{tag}.csv"
        if path.exists():
            for row in csv.DictReader(open(path, encoding="utf-8")):
                out.add(norm_domain(row["domain"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", default=r"C:\Users\hi\Desktop\Soony prjoject\NL_leads_valid_deduped.csv")
    parser.add_argument("--data-dir", default="finetune/data")
    parser.add_argument("--out-dir", default="finetune/data/v3")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    excluded = excluded_domains(data_dir)

    pool: dict[str, dict] = {}
    for row in csv.DictReader(open(args.leads, encoding="utf-8-sig")):
        title = (row.get("Title") or "").strip()
        if not DECISION_MAKER.search(title):
            continue
        email = (row.get("Email Business") or "").strip().lower()
        domain = norm_domain(row.get("Company Website") or (email.split("@")[-1] if "@" in email else ""))
        if not domain or domain in FREE_MAIL or domain in excluded:
            continue
        last = (row.get("Last Name") or "").strip()
        if not surname(last):
            continue
        pool.setdefault(domain, {
            "domain": domain,
            "company_name": (row.get("Company Name") or "").strip(),
            "gold": [surname(last)],
            "title": title,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bank_pool.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["company_name", "domain", "city", "country"])
        writer.writeheader()
        for row in pool.values():
            writer.writerow({
                "company_name": row["company_name"],
                "domain": row["domain"],
                "city": "",
                "country": "Netherlands",
            })
    gold_path = out_dir / "gold_pool.jsonl"
    with open(gold_path, "w", encoding="utf-8") as fh:
        for row in pool.values():
            fh.write(json.dumps({"domain": row["domain"], "gold": row["gold"],
                                 "title": row["title"]}, ensure_ascii=False) + "\n")

    leak = {d for d in pool} & excluded
    print(f"excluded (trained on or reserved for eval): {len(excluded)}")
    print(f"v3 banking pool: {len(pool)} companies with decision-maker gold")
    print(f"leak check (must be 0): {len(leak)}")
    print(f"  {csv_path}")
    print(f"  {gold_path}")


if __name__ == "__main__":
    main()
