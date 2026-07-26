"""Build an owner-finding eval set from REAL campaign outcomes.

The gold labels used so far came from Apollo-style scrapes, and they have
repeatedly been wrong: one rescued file disagreed with our enrichment 12 times
out of 12 and our answer was right every time, and the sealed run contained at
least one company whose gold list was empty, which silently scored a plausible
answer as a miss.

Campaign outcomes are stronger evidence. If a cold email DELIVERED to
jan@company.nl, that mailbox exists at that company. If the person REPLIED,
a real human at that company answered. Neither fact can be scraped wrong.
Combining that with a decision-maker job title gives a label whose weakest
link is the title, not the person-company link.

Two tiers are emitted:

    replied    person answered the email. Strongest available label.
    delivered  mail reached the mailbox (COMPLETED/REPLIED sequence status).

Deliberate properties:
  - Free-mail domains are dropped: the eval needs a COMPANY domain to research.
  - One row per domain, so a company cannot be scored twice.
  - Only owner/founder/CEO/managing-partner style titles, matching the task.
  - Gold stores the SURNAME only, never the email, to limit PII spread. The
    output lives under finetune/data/, which is gitignored.

These companies are marketing, accounting, legal and HR firms, NOT the Dutch
agri set every previous eval used, so this also measures whether the model
generalises past the sector it was trained on.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

FREE_MAIL = {
    "gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "live.nl",
    "ziggo.nl", "kpnmail.nl", "icloud.com", "me.com", "protonmail.com",
    "hotmail.nl", "yahoo.co.uk", "googlemail.com",
}

DECISION_MAKER = re.compile(
    r"\b(owner|co-owner|founder|co-founder|founding partner|ceo|chief executive"
    r"|managing partner|managing director|algemeen directeur|eigenaar|oprichter"
    r"|dga|president|proprietor)\b",
    re.I,
)

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la", "het"}


def surname(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    parts = [p for p in re.sub(r"[^a-z ]", " ", text).split() if p not in TUSSEN and len(p) > 1]
    return parts[-1] if parts else ""


def domain_of(email: str) -> str:
    email = (email or "").strip().lower()
    return email.split("@")[-1] if "@" in email else ""


def load_titles(project: Path) -> dict[str, dict]:
    """email -> source row, from the lists the campaign was built from."""
    out: dict[str, dict] = {}
    for name in ("marketing.csv", "accounting.csv", "legal-big.csv", "legal-hr.csv"):
        path = project / name
        if not path.exists():
            continue
        for row in csv.DictReader(open(path, encoding="utf-8-sig")):
            email = (row.get("email") or "").strip().lower()
            if email:
                out.setdefault(email, row)
    return out


def build(project: Path, statuses: set[str], titles: dict[str, dict]) -> list[dict]:
    leads = list(csv.DictReader(open(project / "campaigns" / "all-leads.csv", encoding="utf-8-sig")))
    by_domain: dict[str, dict] = {}
    for row in leads:
        if row.get("status") not in statuses:
            continue
        email = (row.get("email") or "").strip().lower()
        domain = domain_of(email)
        if not domain or domain in FREE_MAIL:
            continue
        source = titles.get(email) or {}
        title = (source.get("job_title") or "").strip()
        if not DECISION_MAKER.search(title):
            continue
        first = (row.get("first_name") or source.get("first_name") or "").strip()
        last = (row.get("last_name") or source.get("last_name") or "").strip()
        if not (first and last) or not surname(last):
            continue
        by_domain.setdefault(domain, {
            "domain": domain,
            "company_name": (row.get("company_name") or source.get("company_name") or "").strip(),
            "full_name": f"{first} {last}",
            "title": title,
            "gold": [surname(last)],
            "evidence": "replied" if row.get("status") == "REPLIED" else "delivered",
        })
    return list(by_domain.values())


def write(rows: list[dict], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["company_name", "domain", "city", "country"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "company_name": row["company_name"],
                "domain": row["domain"],
                "city": "",
                "country": "Netherlands",
            })
    gold_path = out_dir / f"gold_{tag}.jsonl"
    with open(gold_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({
                "domain": row["domain"],
                "gold": row["gold"],
                "title": row["title"],
                "evidence": row["evidence"],
            }, ensure_ascii=False) + "\n")
    print(f"  {csv_path}  ({len(rows)} companies)")
    print(f"  {gold_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=r"C:\Users\hi\Desktop\Soony prjoject")
    parser.add_argument("--out-dir", default="finetune/data/campaign_eval")
    args = parser.parse_args()

    project = Path(args.project)
    titles = load_titles(project)
    out_dir = Path(args.out_dir)

    replied = build(project, {"REPLIED"}, titles)
    delivered = build(project, {"REPLIED", "COMPLETED"}, titles)
    # Keep the tiers disjoint so a holdout is never also a dev row.
    replied_domains = {r["domain"] for r in replied}
    delivered_only = [r for r in delivered if r["domain"] not in replied_domains]

    print(f"titles loaded from source lists: {len(titles)}")
    print("REPLIED tier (strongest label):")
    write(replied, out_dir, "replied")
    print("DELIVERED-only tier (dev):")
    write(delivered_only, out_dir, "delivered")


if __name__ == "__main__":
    main()
