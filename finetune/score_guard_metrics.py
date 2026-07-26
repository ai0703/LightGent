"""Score an enrichment run on all FOUR guard metrics, never accuracy alone.

Accuracy on its own cannot tell the two failure modes apart. An eager model
that answers null instantly scores "20/20 answered, 5/20 correct"; a dithering
model that never concludes scores "1/20 answered, 5/20 correct". Same
accuracy, opposite diseases, and the broken one looks like progress. So every
report carries:

    accuracy            correct owner / companies scored
    premature rate      answered with fewer than 3 tool results
    null rate           owner_name null
    searches per correct   the actual cost line, and the number to optimise

Usage:
    python -m finetune.score_guard_metrics --traj-dir finetune/data/eval_v2 \
        --label "v2 adapter"
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import unicodedata
from pathlib import Path

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la"}


def surname(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    parts = re.sub(r"[^a-z ]", " ", text).split()
    core = [p for p in parts if p not in TUSSEN and len(p) > 1]
    return core[-1] if core else (parts[-1] if parts else "")


def load_gold(path: str) -> dict[str, list[str]]:
    gold: dict[str, list[str]] = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        gold[row["domain"]] = [g.lower() for g in row["gold"]]
    return gold


def score(traj_dir: str, gold: dict[str, list[str]], label: str) -> dict:
    rows = []
    for path in sorted(glob.glob(str(Path(traj_dir) / "trajectories-*.jsonl"))):
        rows += [json.loads(line) for line in open(path, encoding="utf-8")]

    scored = answered = correct = premature = nulls = 0
    searches = 0
    detail = []
    for row in rows:
        subject = row.get("subject") or {}
        domain = subject.get("domain")
        want = gold.get(domain)
        data = row.get("data") or {}
        name = data.get("owner_name") if isinstance(data, dict) else None
        tool_results = sum(1 for m in row.get("messages", []) if m.get("role") == "tool")
        searches += row.get("tool_calls") or tool_results

        if name:
            answered += 1
            if tool_results < 3:
                premature += 1
        else:
            nulls += 1

        if want is None:
            continue
        scored += 1
        hit = bool(name) and any(w == surname(name) or w in str(name).lower() for w in want)
        correct += hit
        detail.append((domain, name, want, tool_results, hit))

    total = len(rows) or 1
    out = {
        "label": label,
        "companies": len(rows),
        "scored_against_gold": scored,
        "accuracy": round(correct / scored * 100, 1) if scored else 0.0,
        "answer_rate": round(answered / total * 100, 1),
        "premature_rate": round(premature / total * 100, 1),
        "null_rate": round(nulls / total * 100, 1),
        "searches_per_correct": round(searches / correct, 1) if correct else None,
        "total_searches": searches,
        "correct": correct,
    }
    return out, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--gold", default="finetune/data/gold.jsonl")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    gold = load_gold(args.gold)
    labels = args.label or [Path(d).name for d in args.traj_dir]
    results = []
    for traj_dir, label in zip(args.traj_dir, labels):
        result, detail = score(traj_dir, gold, label)
        results.append(result)
        if args.detail:
            print(f"\n--- {label} ---")
            print(f"{'domain':26} {'answered':24} {'gold':14} tools hit")
            for domain, name, want, tools, hit in detail:
                print(f"{domain[:26]:26} {str(name)[:24]:24} {str(want)[:14]:14} "
                      f"{tools:5} {'YES' if hit else 'no'}")

    header = f"\n{'metric':22}" + "".join(f"{r['label'][:16]:>18}" for r in results)
    print(header)
    print("-" * len(header))
    for key, nice in (
        ("accuracy", "accuracy %"),
        ("answer_rate", "answer rate %"),
        ("premature_rate", "premature % (<3)"),
        ("null_rate", "null rate %"),
        ("searches_per_correct", "searches / correct"),
        ("companies", "companies"),
    ):
        row = f"{nice:22}" + "".join(f"{str(r[key]):>18}" for r in results)
        print(row)
    print("\nsearches per correct is the number to optimise: dithering inflates")
    print("it, eagerness shrinks the numerator, so neither can hide.")


if __name__ == "__main__":
    main()
