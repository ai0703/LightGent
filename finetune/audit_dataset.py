"""Teacher-mode audit of the SFT data before rebuilding it.

The trained model refuses to conclude and burns its last step hunting
linkedin_url. Before changing the training recipe, check whether the data
taught it that: if every gold answer carries a populated linkedin_url, then
"never answer without a LinkedIn URL" is exactly the rule we demonstrated 81
times.

Also measures how early the answer was actually available, which decides
whether the trajectories can be truncated to teach stopping sooner.
"""
from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la"}
FIELDS = ("owner_name", "title", "linkedin_url", "confidence", "sources")


def surname(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    parts = re.sub(r"[^a-z ]", " ", text).split()
    core = [p for p in parts if p not in TUSSEN and len(p) > 1]
    return core[-1] if core else (parts[-1] if parts else "")


def first_json_object(text: str) -> dict | None:
    """Answers are wrapped in prose or fences, so scan for a decodable object."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def load_rows() -> list[dict]:
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        path = Path("finetune/data") / name
        if path.exists():
            rows += [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows


def final_answer(row: dict) -> dict | None:
    finals = [
        m for m in row["messages"]
        if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls")
    ]
    return first_json_object(finals[-1]["content"]) if finals else None


def main() -> None:
    rows = load_rows()
    filled: Counter = Counter()
    parsed = 0
    earliest: list[int] = []
    total_tools: list[int] = []
    wasted: list[int] = []

    for row in rows:
        answer = final_answer(row)
        if answer is None:
            continue
        parsed += 1
        for field in FIELDS:
            value = answer.get(field)
            empty = value in (None, "", "null", [], {})
            filled[(field, "null" if empty else "filled")] += 1

        # How early was the owner name actually visible in the tool results?
        name = answer.get("owner_name")
        tool_msgs = [m for m in row["messages"] if m["role"] == "tool"]
        total_tools.append(len(tool_msgs))
        if name:
            want = surname(name)
            hit = next(
                (i for i, m in enumerate(tool_msgs, 1)
                 if want and want in surname_haystack(m.get("content") or "")),
                None,
            )
            if hit:
                earliest.append(hit)
                wasted.append(len(tool_msgs) - hit)

    print(f"rows with a parseable final answer: {parsed}/{len(rows)}\n")
    print("FIELD COMPLETENESS IN THE GOLD ANSWERS")
    for field in FIELDS:
        f = filled[(field, "filled")]
        n = filled[(field, "null")]
        pct = round(f / parsed * 100) if parsed else 0
        flag = "  <-- never null: teaches 'always find this'" if n == 0 and parsed else ""
        print(f"  {field:14} filled {f:3}  null {n:3}   {pct:3} pct filled{flag}")

    if earliest:
        print("\nWHEN THE ANSWER FIRST APPEARED (tool-result index)")
        print(f"  median {statistics.median(earliest):.0f} of "
              f"{statistics.median(total_tools):.0f} tool results")
        print(f"  searches made AFTER the answer was already visible: "
              f"median {statistics.median(wasted):.0f}, total {sum(wasted)}")
        print(f"  -> {round(sum(wasted) / sum(total_tools) * 100)} pct of all tool calls "
              f"in the training data were unnecessary")


def surname_haystack(text: str) -> str:
    """Normalise a tool result so surname matching works against it."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


if __name__ == "__main__":
    main()
