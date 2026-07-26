"""Explain every eval miss instead of just counting it.

A raw accuracy number cannot tell these apart, and they demand opposite
responses:

  ALT-OWNER      the model named a DIFFERENT real decision maker. Campaign gold
                 records only the person we emailed, but agencies and
                 partnerships have several owners, so this is usually a correct
                 answer scored wrong. Fix the label, not the model.
  FORMAT         right person, unusable string ("Hans E." against Ekhart).
                 Fix the output contract.
  MODEL-MISSED   the gold name WAS in the gathered evidence and the model still
                 abstained or picked someone else. The genuine trainable defect.
  NO-EVIDENCE    search never surfaced the gold name. A retrieval problem; more
                 training data will not touch it.
  UNGROUNDED     the model asserted a name that appears NOWHERE in the evidence.
                 The only category that is actually dangerous, because it is
                 fabrication rather than error.

Every judgement is made against the trajectory's own tool output, so it is
evidence-based rather than a second opinion from another model.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la", "het"}

OWNER_WORDS = re.compile(
    r"eigenaar|oprichter|founder|owner|directeur|director|ceo|partner|"
    r"managing|bestuurder|dga|oprichtster|mede-eigenaar",
    re.I,
)


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()


def surname(value: str) -> str:
    parts = [p for p in re.sub(r"[^a-z ]", " ", fold(value)).split()
             if p not in TUSSEN and len(p) > 1]
    return parts[-1] if parts else ""


def first_name(value: str) -> str:
    parts = [p for p in re.sub(r"[^a-z ]", " ", fold(value)).split() if len(p) > 1]
    return parts[0] if parts else ""


def company_tokens(domain: str) -> list[str]:
    """Distinctive words from the domain, used to tie a title to THIS company."""
    stem = re.sub(r"\.(nl|com|eu|co\.uk|net|org|de|be)$", "", fold(domain))
    parts = [p for p in re.split(r"[^a-z0-9]+", stem) if len(p) > 3]
    return parts or [stem]


def near_owner_word(evidence: str, name: str, domain: str, window: int = 220) -> bool:
    """Is the model's name described as an owner OF THIS COMPANY?

    The earlier version only asked whether an owner-word sat near the name,
    which credited raypack.nl's "raymond kleine schaars - dga @ studium
    academy" as a Raypack owner. It names a real DGA of an entirely different
    firm. The company token must appear in the same window as the title.
    """
    key = surname(name)
    if not key:
        return False
    tokens = company_tokens(domain)
    for match in re.finditer(re.escape(key), evidence):
        chunk = evidence[max(0, match.start() - window): match.start() + window]
        if OWNER_WORDS.search(chunk) and any(t in chunk for t in tokens):
            return True
    return False


def classify(answer: str | None, gold: list[str], evidence: str, domain: str) -> tuple[str, str]:
    gold_in = any(g and g in evidence for g in gold)
    if not answer:
        if gold_in:
            return "MODEL-MISSED", "abstained though the gold name was in evidence"
        return "NO-EVIDENCE", "abstained and search never surfaced the gold name"

    ans_sur = surname(answer)
    if any(g == ans_sur or g in fold(answer) for g in gold):
        return "CORRECT", ""

    # Right person, mangled string: first name matches and the surname was
    # abbreviated away ("Hans E.").
    tokens = [t for t in re.sub(r"[^a-z ]", " ", fold(answer)).split()]
    truncated = len(re.sub(r"[^A-Za-z]", "", str(answer).split()[-1])) <= 1 if str(answer).split() else False
    if gold and first_name(answer) and any(
        first_name(answer) == first_name(g) or truncated for g in gold
    ) and truncated:
        return "FORMAT", "right first name, surname abbreviated to an initial"

    ans_in = bool(ans_sur) and ans_sur in evidence
    if not ans_in:
        return "UNGROUNDED", "named a person who appears nowhere in the evidence"
    if near_owner_word(evidence, answer, domain):
        if gold_in:
            return "ALT-OWNER", "named a different person the evidence calls an owner/director (gold also present)"
        return "ALT-OWNER", "named a different person the evidence calls an owner/director; gold absent from evidence"
    if gold_in:
        return "MODEL-MISSED", "picked a grounded person NOT tied to this company, while gold was in evidence"
    return "WRONG-COMPANY", "named a real person the evidence does not tie to this company"


def load_gold(path: str) -> dict[str, list[str]]:
    gold: dict[str, list[str]] = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        gold[row["domain"]] = [fold(g) for g in row["gold"]]
    return gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    rows = []
    for path in sorted(glob.glob(str(Path(args.traj_dir) / "trajectories-*.jsonl"))):
        rows += [json.loads(line) for line in open(path, encoding="utf-8")]

    buckets: Counter = Counter()
    detail: list[tuple] = []
    for row in rows:
        domain = (row.get("subject") or {}).get("domain")
        want = gold.get(domain)
        if want is None:
            continue
        data = row.get("data") or {}
        answer = data.get("owner_name") if isinstance(data, dict) else None
        evidence = fold("\n".join(
            m.get("content") or "" for m in row.get("messages", []) if m.get("role") == "tool"
        ))
        kind, why = classify(answer, want, evidence, domain)
        buckets[kind] += 1
        if kind != "CORRECT":
            detail.append((domain, answer, want, kind, why))

    total = sum(buckets.values())
    correct = buckets["CORRECT"]
    print(f"\n=== {args.label}: {total} companies scored ===\n")
    print(f"{'category':14} {'n':>3}  {'%':>5}   meaning")
    print("-" * 78)
    order = ["CORRECT", "ALT-OWNER", "FORMAT", "MODEL-MISSED", "WRONG-COMPANY", "NO-EVIDENCE", "UNGROUNDED"]
    meaning = {
        "CORRECT": "matched gold",
        "ALT-OWNER": "different REAL owner; label names only who we emailed",
        "FORMAT": "right person, unusable string",
        "MODEL-MISSED": "gold was in evidence, model failed to commit  <-- TRAINABLE",
        "WRONG-COMPANY": "real person, but evidence ties them elsewhere  <-- TRAINABLE",
        "NO-EVIDENCE": "search never surfaced the gold name  <-- RETRIEVAL",
        "UNGROUNDED": "fabricated: name absent from evidence  <-- SERIOUS",
    }
    for key in order:
        n = buckets.get(key, 0)
        print(f"{key:14} {n:3}  {100*n/total:5.1f}   {meaning[key]}")

    generous = correct + buckets.get("ALT-OWNER", 0) + buckets.get("FORMAT", 0)
    print(f"\nstrict accuracy      {correct}/{total} = {100*correct/total:.1f} pct")
    print(f"credit ALT-OWNER+FORMAT as right: {generous}/{total} = {100*generous/total:.1f} pct")
    print(f"fabrication rate     {buckets.get('UNGROUNDED',0)}/{total} = "
          f"{100*buckets.get('UNGROUNDED',0)/total:.1f} pct")

    print("\n--- every miss, with its reason ---")
    for domain, answer, want, kind, why in sorted(detail, key=lambda r: r[3]):
        print(f"  [{kind:12}] {domain[:26]:26} said={str(answer)[:22]:22} gold={str(want)[:14]:14} {why}")


if __name__ == "__main__":
    main()
