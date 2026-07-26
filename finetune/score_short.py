"""Score the short-cap run by the NAME the model surfaced, not by whether it
managed to emit final JSON.

The n=20 run at MAX_ITERATIONS=6 answered 1/20, but reading the final turns
shows the model had already identified a person for nearly every company and
was spending its last step hunting the optional linkedin_url. This scores the
knowledge, separating "found the owner" from "emitted the JSON".
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la"}


def surname(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    parts = re.sub(r"[^a-z ]", " ", text).split()
    core = [p for p in parts if p not in TUSSEN and len(p) > 1]
    return core[-1] if core else (parts[-1] if parts else "")


NOISE = re.compile(r"site:linkedin\.com\S*|linkedin|profile|url|photo|com|www|nl|https?", re.I)


def candidate_name(final_content: str) -> str | None:
    """The person the model named in its last turn, however it phrased it."""
    direct = re.search(r'"owner_name"\s*:\s*"([^"]+)"', final_content)
    if direct:
        return direct.group(1)
    arg = re.search(r'"(?:query|url)"\s*:\s*"(.*?)"', final_content)
    if not arg:
        return None
    text = re.sub(r"https?://\S*?/in/", " ", arg.group(1))
    text = re.sub(r"[-/_\\]", " ", text)
    text = NOISE.sub(" ", text)
    words = [w for w in text.split() if not any(c.isdigit() for c in w) and len(w) > 1]
    return " ".join(words)[:40] or None


def main() -> None:
    gold = {}
    for line in open("finetune/data/gold.jsonl", encoding="utf-8"):
        row = json.loads(line)
        gold[row["domain"]] = [g.lower() for g in row["gold"]]

    rows = [
        json.loads(line)
        for line in open("finetune/data/traj_short/trajectories-20260726.jsonl", encoding="utf-8")
    ]

    print(f"{'domain':26} {'name surfaced':26} {'gold':14} hit")
    print("-" * 76)
    named = hits = 0
    for row in rows:
        domain = row["subject"]["domain"]
        cand = candidate_name(row.get("final_content") or "")
        want = gold.get(domain) or []
        if cand:
            named += 1
        hit = bool(cand) and any(w == surname(cand) or w in cand.lower() for w in want)
        hits += hit
        print(f"{domain[:26]:26} {str(cand)[:26]:26} {str(want)[:14]:14} {'YES' if hit else 'no'}")

    print()
    print(f"surfaced a person:      {named}/{len(rows)}")
    print(f"that person is correct: {hits}/{len(rows)}")


if __name__ == "__main__":
    main()
