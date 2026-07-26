"""Teacher-mode rebuild of the SFT data: stop demonstrating the dithering.

The audit found the answer was visible at tool result 2 of 15 (median) and the
teacher kept searching for ten more steps anyway, so 71 pct of every tool call
we trained on was unnecessary. The student learned exactly that: it searches
past the answer and never concludes.

This rebuilds each trajectory to end where the evidence actually arrived:

  positive rows (an owner was found)
      truncated to the assistant/tool group that first revealed the name, then
      the final answer. Every search that was NEEDED is kept, only the ones
      after the discovery are dropped, so the model still learns to search
      until it finds something rather than to answer eagerly.

  negative rows (owner_name is null)
      kept at FULL length. These demonstrate exhausting the search before
      giving up, and they are the contrast that stops the model concluding on
      thin evidence.

After truncation the answer is re-verified against the evidence that survived:
any linkedin_url or source URL that only appeared in a deleted step is
stripped, because keeping it would teach the model to cite pages it never saw.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

TUSSEN = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la"}


def surname(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    parts = re.sub(r"[^a-z ]", " ", text).split()
    core = [p for p in parts if p not in TUSSEN and len(p) > 1]
    return core[-1] if core else (parts[-1] if parts else "")


def norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def canonical_url(url: str) -> str:
    """Host+path only, matching the runtime anti-fabrication check."""
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return ""
    host = (parts.netloc or "").lower().removeprefix("www.")
    path = (parts.path or "").rstrip("/").lower()
    return f"{host}{path}" if host else ""


def first_json_object(text: str) -> tuple[dict | None, int, int]:
    """Return the first decodable object plus its span, so we can rewrite it."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, match.start(), match.start() + end
    return None, -1, -1


def split_groups(messages: list[dict]) -> tuple[list[dict], list[list[dict]], dict | None]:
    """Split into (preamble, research groups, final answer).

    A research group is one assistant tool-call turn plus the tool results it
    produced, so truncation never orphans a tool message from its call.
    """
    preamble: list[dict] = []
    groups: list[list[dict]] = []
    final: dict | None = None
    current: list[dict] | None = None

    for msg in messages:
        role = msg["role"]
        if role in ("system", "user") and current is None and not groups:
            preamble.append(msg)
            continue
        if role == "assistant" and msg.get("tool_calls"):
            current = [msg]
            groups.append(current)
            continue
        if role == "tool" and current is not None:
            current.append(msg)
            continue
        if role == "assistant" and not msg.get("tool_calls") and msg.get("content"):
            final = msg
            current = None
            continue
        # Tag-mode nudges and mid-run user prompts ride along with the group.
        if current is not None:
            current.append(msg)
        else:
            preamble.append(msg)
    return preamble, groups, final


def discovery_index(groups: list[list[dict]], name: str) -> int | None:
    """Index of the first group whose tool output contains the owner surname."""
    want = surname(name)
    if not want:
        return None
    for i, group in enumerate(groups):
        for msg in group:
            if msg["role"] == "tool" and want in norm(msg.get("content") or ""):
                return i
    return None


def evidence_text(preamble: list[dict], groups: list[list[dict]]) -> str:
    parts = [m.get("content") or "" for m in preamble]
    for group in groups:
        parts += [m.get("content") or "" for m in group]
    return norm("\n".join(parts))


def reground(answer: dict, evidence: str) -> dict:
    """Drop any URL that did not survive truncation."""
    seen = evidence
    out = dict(answer)
    link = out.get("linkedin_url")
    if link and canonical_url(link) and canonical_url(link) not in seen:
        out["linkedin_url"] = None
    sources = out.get("sources")
    if isinstance(sources, list):
        kept = [s for s in sources if canonical_url(s) and canonical_url(s) in seen]
        out["sources"] = kept
    return out


def rewrite_final(final: dict, answer: dict) -> dict:
    """Replace the JSON inside the assistant turn, preserving any wrapper."""
    content = final.get("content") or ""
    _, start, end = first_json_object(content)
    payload = json.dumps(answer, ensure_ascii=False, indent=2)
    if start < 0:
        return {**final, "content": payload}
    return {**final, "content": content[:start] + payload + content[end:]}


def curate(row: dict, keep_confirm: int) -> tuple[dict | None, dict]:
    messages = row["messages"]
    preamble, groups, final = split_groups(messages)
    stats = {"groups": len(groups), "kept": len(groups), "kind": "unchanged"}
    if final is None or not groups:
        return row, stats

    answer, _, _ = first_json_object(final.get("content") or "")
    if answer is None:
        return row, {**stats, "kind": "unparseable"}

    name = answer.get("owner_name")
    if not name:
        # Negative example: keep the full exhaustive search.
        stats["kind"] = "negative-kept-full"
        return row, stats

    found = discovery_index(groups, name)
    if found is None:
        # The name never appears in the evidence. Training on this teaches
        # the model to invent a name, which is the one thing we cannot allow.
        stats["kind"] = "dropped-unsupported"
        return None, stats

    keep = min(len(groups), found + 1 + keep_confirm)
    kept_groups = groups[:keep]
    grounded = reground(answer, evidence_text(preamble, kept_groups))
    new_messages = list(preamble)
    for group in kept_groups:
        new_messages.extend(group)
    new_messages.append(rewrite_final(final, grounded))

    stats.update(kept=keep, kind="truncated", discovered_at=found + 1)
    return {**row, "messages": new_messages}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="finetune/data")
    parser.add_argument("--out-dir", default="finetune/data/curated")
    parser.add_argument("--keep-confirm", type=int, default=1,
                        help="extra research groups kept after the discovery step")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for split in ("train", "val"):
        src = Path(args.in_dir) / f"{split}.jsonl"
        if not src.exists():
            continue
        rows = [json.loads(line) for line in open(src, encoding="utf-8")]
        kept_rows, kinds, before, after = [], {}, [], []
        for row in rows:
            new_row, stats = curate(row, args.keep_confirm)
            kinds[stats["kind"]] = kinds.get(stats["kind"], 0) + 1
            before.append(stats["groups"])
            if new_row is not None:
                kept_rows.append(new_row)
                after.append(stats["kept"])

        dst = out_dir / f"{split}.jsonl"
        with open(dst, "w", encoding="utf-8") as fh:
            for row in kept_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        chars_ans = chars_tot = 0
        for row in kept_rows:
            blob = json.dumps(row["messages"], ensure_ascii=False)
            chars_tot += len(blob)
            finals = [m for m in row["messages"]
                      if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls")]
            if finals:
                chars_ans += len(finals[-1]["content"])

        report[split] = {
            "rows_in": len(rows),
            "rows_out": len(kept_rows),
            "kinds": kinds,
            "median_groups_before": statistics.median(before) if before else 0,
            "median_groups_after": statistics.median(after) if after else 0,
            "answer_pct_of_chars": round(chars_ans / chars_tot * 100, 2) if chars_tot else 0,
        }
        print(f"{split}: {len(rows)} -> {len(kept_rows)} rows   {kinds}")
        print(f"  research steps  median {report[split]['median_groups_before']:.0f} "
              f"-> {report[split]['median_groups_after']:.0f}")
        print(f"  answer share of characters: {report[split]['answer_pct_of_chars']} pct "
              f"(was 4.65 pct)")

    (out_dir / "curation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir}/train.jsonl, val.jsonl, curation_report.json")


if __name__ == "__main__":
    main()
