"""Synthesise abstention examples (T5, T6) by ablating positives.

Natural "owner not findable" cases are only ~4.5 percent of banked companies,
so harvesting 70 would mean banking ~1,500. Instead take a positive
trajectory, remove the ONE tool result that carried the owner name, and
rewrite the answer to null. The trajectory then honestly shows a full search
that turned up nothing, and the correct terminal action is an honest null.

  T5 exhausted-negative  surviving evidence names nobody
  T6 distractor-negative surviving evidence still names OTHER people
                         (staff, authors, unrelated profiles) and the answer
                         is STILL null. This is the direct antidote to
                         "a name is present, therefore answer it".

Safety: EVERY tool result carrying the owner surname is ablated, never just
one. Leaving a carrier behind would teach the model to ignore evidence it can
plainly see, which is worse than the disease. Rows where the name is spread
across more than four results are skipped, because blanking most of the
transcript stops it resembling a real search.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

from finetune.curate_dataset import (
    first_json_object,
    norm,
    rewrite_final,
    split_groups,
    surname,
)

# What a search that found nothing actually looks like coming back from the
# real tools, so the ablated step stays in-distribution.
BARREN_SEARCH = "[]"
BARREN_FETCH = (
    "Tool error (web_fetch): Client error '404 Not Found' for url '{url}'\n"
    "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"
)

NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+(?:van\s+|de\s+|der\s+|den\s+|ter\s+)?([A-Z][a-z]{2,})\b")

# Capitalised Dutch/English words that start headings, nav items and company
# names. Without these the detector calls "Onze Diensten" a person.
NOT_A_NAME = {
    "onze", "ons", "over", "onder", "meer", "alle", "deze", "voor", "door",
    "met", "van", "het", "een", "hier", "welkom", "contact", "home", "nieuws",
    "bekijk", "lees", "neem", "vraag", "wij", "zij", "onze", "agro", "agri",
    "boerderij", "kwekerij", "advies", "groep", "group", "holding", "beheer",
    "the", "our", "read", "more", "view", "learn", "about", "team", "privacy",
    "cookie", "algemene", "voorwaarden", "januari", "februari", "maart",
    "april", "juni", "juli", "augustus", "september", "oktober", "november",
    "december", "maandag", "dinsdag", "woensdag", "donderdag", "vrijdag",
    "tool", "error", "client", "found", "url", "http", "https", "for", "more",
    "information", "check", "web", "search", "fetch", "not", "page",
}


def other_person_names(text: str, exclude: str, company: str = "") -> set[str]:
    """Two-word capitalised sequences that plausibly name a PERSON.

    Deliberately conservative: a false positive here mislabels a T5 row as T6,
    which distorts the balance we are trying to fix.
    """
    company_words = {w for w in re.split(r"\W+", company.lower()) if len(w) > 2}
    out = set()
    for match in NAME_RE.finditer(text or ""):
        first, last = match.group(1), match.group(2)
        if first.lower() in NOT_A_NAME or last.lower() in NOT_A_NAME:
            continue
        if first.lower() in company_words or last.lower() in company_words:
            continue
        full = f"{first} {last}"
        if surname(full) and surname(full) != exclude:
            out.add(full)
    return out


def tool_results(groups: list[list[dict]]) -> list[tuple[int, int, dict]]:
    """(group index, message index within group, message) for tool results."""
    out = []
    for gi, group in enumerate(groups):
        for mi, msg in enumerate(group):
            if msg["role"] == "tool":
                out.append((gi, mi, msg))
    return out


def call_for(group: list[dict], tool_call_id: str) -> dict | None:
    for msg in group:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                if call.get("id") == tool_call_id:
                    return call
    return None


def ablate(row: dict) -> tuple[dict | None, str, int]:
    """Return (ablated row, archetype, other-name count) or (None, reason, 0)."""
    preamble, groups, final = split_groups(row["messages"])
    if final is None or not groups:
        return None, "no-final", 0
    answer, _, _ = first_json_object(final.get("content") or "")
    if not answer or not answer.get("owner_name"):
        return None, "already-negative", 0
    want = surname(answer["owner_name"])
    if not want:
        return None, "no-surname", 0

    carriers = [(gi, mi, m) for gi, mi, m in tool_results(groups)
                if want in norm(m.get("content") or "")]
    if not carriers:
        return None, "surname-not-in-evidence", 0
    if len(carriers) > 4:
        # Ablating most of the transcript leaves a trajectory that no longer
        # resembles a real search.
        return None, f"surname-in-{len(carriers)}-results", 0

    # Ablate EVERY carrier. Leaving one behind would teach the model to ignore
    # evidence it can plainly see, which is worse than the disease.
    new_groups = [list(g) for g in groups]
    for gi, mi, victim in carriers:
        call = call_for(groups[gi], victim.get("tool_call_id", ""))
        fn = (call or {}).get("function", {}).get("name", "web_search")
        if fn == "web_fetch":
            try:
                url = json.loads((call or {})["function"]["arguments"]).get("url", "")
            except Exception:  # noqa: BLE001 - malformed args are not fatal here
                url = ""
            # Keep the origin only. Fetch URLs frequently contain the person's
            # name in the path (/team/jan-de-vries), which would smuggle the
            # answer straight back into an example whose answer must be null.
            parts = urlsplit(url)
            url = f"{parts.scheme}://{parts.netloc}/" if parts.netloc else url
            replacement = BARREN_FETCH.format(url=url)
        else:
            replacement = BARREN_SEARCH
        new_groups[gi][mi] = {**victim, "content": replacement}

    survivors = "\n".join(
        m.get("content") or ""
        for g in new_groups for m in g if m["role"] == "tool"
    )
    if want in norm(survivors):
        return None, "surname-survived", 0

    # The teacher's own QUERIES leak the answer once it has seen it:
    # web_search "Asperges van de Eng eigenaar oprichter Han te Ronde".
    # A trajectory that searches for a name and then concludes null is
    # incoherent, so drop every group from the first such query onward.
    def mentions_owner(group: list[dict]) -> bool:
        for msg in group:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    if want in norm(call.get("function", {}).get("arguments", "")):
                        return True
        return False

    cut = next((i for i, g in enumerate(new_groups) if mentions_owner(g)), None)
    if cut is not None:
        new_groups = new_groups[:cut]
    if not new_groups:
        return None, "owner-named-in-first-query", 0

    survivors = "\n".join(
        m.get("content") or ""
        for g in new_groups for m in g if m["role"] == "tool"
    )

    company = ""
    for msg in preamble:
        if msg["role"] == "system":
            hit = re.search(r"company[_ ]?name['\":\s]+([^\n\"',}]+)",
                            msg.get("content") or "")
            if hit:
                company = hit.group(1)
            break
    others = other_person_names(survivors, want, company)
    archetype = "T6" if others else "T5"

    null_answer = {
        "owner_name": None,
        "title": None,
        "linkedin_url": None,
        "confidence": "low",
        "sources": [],
    }
    # Replace the final turn WHOLESALE, not just its JSON. The teacher's
    # surrounding prose still says "I have clear evidence that Han te Ronde
    # is...", which would sit directly above an answer of null.
    messages = list(preamble)
    for group in new_groups:
        messages.extend(group)
    messages.append({
        "role": "assistant",
        "content": json.dumps(null_answer, ensure_ascii=False, indent=2),
    })

    # Final safety sweep: the surname must not appear ANYWHERE in the row.
    blob = norm(json.dumps(messages, ensure_ascii=False))
    if want in blob:
        return None, "surname-leaked-somewhere", 0
    return {**row, "messages": messages}, archetype, len(others)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", action="append", required=True,
                        help="curated positive jsonl; repeatable")
    parser.add_argument("--out", default="finetune/data/curated/negatives.jsonl")
    parser.add_argument("--target-t5", type=int, default=45)
    parser.add_argument("--target-t6", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for path in args.in_file:
        rows += [json.loads(line) for line in open(path, encoding="utf-8")]

    made: dict[str, list[dict]] = {"T5": [], "T6": []}
    skipped: dict[str, int] = {}
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    for row in rows:
        out, kind, others = ablate(row)
        if out is None:
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        # T6 is the scarcer, more valuable class, so never let T5 crowd it out.
        cap = args.target_t6 if kind == "T6" else args.target_t5
        if len(made[kind]) < cap:
            out.setdefault("meta", {})
            out["meta"] = {**(out.get("meta") or {}), "archetype": kind,
                           "other_names_visible": others, "synthetic": "ablation"}
            made[kind].append(out)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for kind in ("T5", "T6"):
            for row in made[kind]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"eligible positives: {len(rows)}")
    print(f"  T5 exhausted-negative : {len(made['T5'])}")
    print(f"  T6 distractor-negative: {len(made['T6'])}")
    print(f"  skipped: {skipped}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
