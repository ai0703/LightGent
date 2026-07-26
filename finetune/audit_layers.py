"""Which research LAYER actually produced the answer, and how much do
trajectories legitimately differ?

A single hard-coded opening sequence would treat every company the same. This
measures whether that is defensible: how many distinct source types win, how
often, and how wide the spread of "steps needed" really is. If the spread is
wide, the right design is a portfolio of strategies the model chooses between,
not one script.
"""
from __future__ import annotations

import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from finetune.curate_dataset import first_json_object, norm, split_groups, surname

REGISTRY = ("kvk.nl", "companieshouse", "handelsregister", "rechtspraak", "opencorporates")
SOCIAL = ("linkedin.", "facebook.", "twitter.", "x.com", "instagram.")
NEWS = ("nieuws", "news", "press", "krant", "fd.nl", "agf.nl", "boerderij")
DIRECTORY = ("bedrijven", "telefoonboek", "yellowpages", "kompass", "dnb.com", "bizz")


def layer_of(url: str, own_domain: str) -> str:
    host = (urlsplit(url).netloc or url).lower().removeprefix("www.")
    if own_domain and own_domain.removeprefix("www.") in host:
        return "own-site"
    if any(k in host for k in REGISTRY):
        return "registry"
    if any(k in host for k in SOCIAL):
        return "social"
    if any(k in host for k in NEWS):
        return "news"
    if any(k in host for k in DIRECTORY):
        return "directory"
    return "other-web"


def urls_in(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\"'<>\)\]]+", text or "")


def main() -> None:
    rows = [json.loads(line) for line in open("finetune/data/train.jsonl", encoding="utf-8")]
    rows += [json.loads(line) for line in open("finetune/data/val.jsonl", encoding="utf-8")]

    winning_layer: Counter = Counter()
    first_tool: Counter = Counter()
    discovery_steps: list[int] = []
    per_layer_steps: dict[str, list[int]] = {}

    for row in rows:
        preamble, groups, final = split_groups(row["messages"])
        if final is None or not groups:
            continue
        answer, _, _ = first_json_object(final.get("content") or "")
        if not answer or not answer.get("owner_name"):
            continue
        want = surname(answer["owner_name"])
        domain = (row.get("subject") or {}).get("domain", "")

        for i, group in enumerate(groups, 1):
            hit = None
            for msg in group:
                if msg["role"] != "tool":
                    continue
                body = msg.get("content") or ""
                if want and want in norm(body):
                    # Attribute to the URL nearest the name in the tool output.
                    idx = norm(body).find(want)
                    before = body[:idx]
                    found = urls_in(before)
                    hit = layer_of(found[-1] if found else "", domain)
                    break
            if hit:
                winning_layer[hit] += 1
                discovery_steps.append(i)
                per_layer_steps.setdefault(hit, []).append(i)
                break

        calls = groups[0][0].get("tool_calls") or []
        if calls:
            first_tool[calls[0]["function"]["name"]] += 1

    total = sum(winning_layer.values())
    print(f"companies whose answer was traced to a source: {total}\n")
    print("WHICH LAYER PRODUCED THE ANSWER")
    for layer, n in winning_layer.most_common():
        steps = per_layer_steps[layer]
        print(f"  {layer:12} {n:3}  ({round(n / total * 100):3} pct)   "
              f"found at step median {statistics.median(steps):.0f}, "
              f"range {min(steps)}-{max(steps)}")

    print("\nSTEPS NEEDED BEFORE THE ANSWER APPEARED")
    spread = Counter(discovery_steps)
    for step in sorted(spread):
        bar = "#" * spread[step]
        print(f"  step {step:2}: {spread[step]:3} {bar}")
    print(f"  median {statistics.median(discovery_steps):.0f}, "
          f"p90 {statistics.quantiles(discovery_steps, n=10)[8]:.0f}, "
          f"max {max(discovery_steps)}")
    one_shot = sum(1 for s in discovery_steps if s == 1)
    print(f"  solved on the FIRST step: {one_shot}/{total} "
          f"({round(one_shot / total * 100)} pct)")
    print(f"  needed 4+ steps: {sum(1 for s in discovery_steps if s >= 4)}/{total}")


if __name__ == "__main__":
    main()
