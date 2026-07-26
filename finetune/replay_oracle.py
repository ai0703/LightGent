"""Isolate MODEL quality from SEARCH quality.

Tonight's evals scored the whole pipeline, and the search layer was returning
junk (Jeffrey Epstein articles for a Dutch agri query), so the models were
graded on evidence they never received. This script hands the model the
evidence a WORKING search would have returned, in the same conversation shape
it was trained on, and asks only: can you extract the right answer from it?

Usage:
    python -m finetune.replay_oracle --base-url URL --api-key KEY --model eval-model
"""
from __future__ import annotations

import argparse
import glob
import json
import unicodedata
import re
from pathlib import Path

import httpx

SYSTEM = """You are LightGent, a web-research agent. Given a SUBJECT (a company) and a TASK, you fill the requested OUTPUT FIELDS with accurate, verified facts gathered from tool results.

Accuracy (hard rules):
- Only report a fact you actually saw in a tool result. Never guess.
- If the evidence does not name the person, return null. An honest null beats a guess.
- Only cite URLs that appeared in tool results.

Output ONLY a JSON object with keys: owner_name, title, linkedin_url, confidence, sources."""


def surname(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    parts = re.sub(r"[^a-z ]", " ", text).split()
    tussen = {"van", "der", "den", "de", "ter", "ten", "te", "du", "le", "la"}
    core = [p for p in parts if p not in tussen and len(p) > 1]
    return core[-1] if core else (parts[-1] if parts else "")


def real_system_prompt(company: str, domain: str) -> str:
    """The exact prompt the model was trained on, not a paraphrase.

    Using a hand-written short prompt puts the model out of distribution and
    makes it answer null even when the evidence is unambiguous.
    """
    import lightgent_service as svc

    return svc.build_system_prompt(
        task="Find the owner or top decision maker of this company.",
        context={"company_name": company, "domain": domain, "country": "Netherlands"},
        output_fields={
            "owner_name": "full name of the owner or top decision maker",
            "title": "their job title",
            "linkedin_url": "their personal LinkedIn profile URL",
        },
    )


def build_messages(record: dict) -> list[dict]:
    """The trained shape: system, user, assistant tool call, tool result."""
    company = record.get("company") or record["domain"]
    evidence = "\n\n".join(
        f"[source: {url}]\n{snippet}"
        for url, snippet in zip(
            record.get("sources") or [],
            record.get("evidence_snippets") or [],
        )
    ) or "\n\n".join(record.get("evidence_snippets") or [])
    if not evidence:
        evidence = "(no results)"
    return [
        {"role": "system", "content": real_system_prompt(company, record["domain"])},
        {
            "role": "user",
            "content": (
                f"## Subject\ncompany: {company}\ndomain: {record['domain']}\n"
                f"country: Netherlands\n\n## Task\nFind the owner or top decision "
                f"maker.\n\n## Output fields\nowner_name, title, linkedin_url\n\n"
                f"Begin research now."
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": f"{company} eigenaar directeur"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": evidence},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="eval-model")
    parser.add_argument("--oracle-dir", default="finetune/data/oracle")
    args = parser.parse_args()

    records = []
    for path in sorted(glob.glob(str(Path(args.oracle_dir) / "*.json"))):
        records.extend(json.load(open(path, encoding="utf-8")))

    print(f"{'domain':22} {'model answer':24} {'oracle truth':24} match")
    print("-" * 80)
    correct = total = 0
    with httpx.Client(timeout=180) as client:
        for record in records:
            truth = record.get("owner_name")
            messages = build_messages(record)
            response = client.post(
                f"{args.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {args.api_key}"},
                json={
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": 400,
                    "temperature": 0.0,
                },
            )
            answer = None
            if response.status_code == 200:
                content = (response.json()["choices"][0]["message"].get("content") or "")
                match = re.search(r"\{.*\}", content, re.S)
                if match:
                    try:
                        answer = json.loads(match.group(0)).get("owner_name")
                    except json.JSONDecodeError:
                        answer = f"(unparseable) {content[:40]}"
                else:
                    answer = f"(no json) {content[:40]}"
            else:
                answer = f"(http {response.status_code})"

            total += 1
            if truth:
                hit = bool(answer) and isinstance(answer, str) and surname(answer) == surname(truth)
            else:
                hit = answer in (None, "null", "")
            correct += hit
            print(
                f"{record['domain'][:22]:22} {str(answer)[:24]:24} "
                f"{str(truth)[:24]:24} {'YES' if hit else 'no'}"
            )
    print(f"\nEXTRACTION ACCURACY WITH GOOD EVIDENCE: {correct}/{total}")
    print("(a null truth counts as correct only if the model also returns null)")


if __name__ == "__main__":
    main()
