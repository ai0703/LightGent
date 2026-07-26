#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible endpoint on LightGent owner enrichment."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI

# Imported canonical URLs intentionally strip ports.
from finetune.build_dataset import canonical_urls, normalize_domain
from lightgent_service import ResearchRequest, run_agent


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "finetune" / "data"
RESCUED_DIR = ROOT / "finetune" / "rescued-evals"
NEGATIVES_PATH = RESCUED_DIR / "negatives_to_verify.json"
OVERNIGHT_PATH = RESCUED_DIR / "overnight_results.jsonl"
OWNER_TASK = "Find the owner or principal decision-maker of this company."
OWNER_FIELDS = {
    "owner_name": "Full name of the owner or principal decision-maker, or null",
    "title": "Their current title, or null",
    "linkedin_url": "Their LinkedIn profile URL, or null",
}
PROBE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return results.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]
PARTICLES = {
    ("van",), ("de",), ("den",), ("ter",), ("ten",), ("te",),
    ("van", "der"), ("van", "den"), ("van", "de"),
}


def read_domains(path: Path) -> set[str]:
    if not path.is_file():
        raise ValueError(f"required domain file is missing: {path}")
    return {
        domain for line in path.read_text(encoding="utf-8").splitlines()
        if (domain := normalize_domain(line))
    }


def assert_disjoint(chosen: Path, other: Path, train: Path) -> None:
    groups = [(chosen, read_domains(chosen)), (other, read_domains(other)),
              (train, read_domains(train))]
    for index, (left_path, left) in enumerate(groups):
        for right_path, right in groups[index + 1:]:
            overlap = sorted(left & right)
            if overlap:
                raise ValueError(
                    f"domain files must be pairwise disjoint: {left_path} and "
                    f"{right_path} overlap on {', '.join(overlap[:5])}"
                )


def _plain_words(name: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return ["".join(ch for ch in word if ch.isalnum()) for word in text.split()
            if any(ch.isalnum() for ch in word)]


def surname_phrase(name: Any) -> str:
    words = _plain_words(name)
    if not words:
        return ""
    start = len(words) - 1
    for size in (3, 2, 1):
        candidate = tuple(words[max(0, len(words) - 1 - size):len(words) - 1])
        if candidate in PARTICLES:
            start = len(words) - 1 - size
            break
    return " ".join(words[start:])


def surname_agrees(answered: Any, banked: Any) -> bool:
    left, right = surname_phrase(answered), surname_phrase(banked)
    return bool(left and right and left == right)


def wilson(successes: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"count": 0, "total": 0, "rate": 0.0, "ci95": [0.0, 0.0]}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return {
        "count": successes, "total": total, "rate": p,
        "ci95": [max(0.0, centre - margin / denominator),
                 min(1.0, centre + margin / denominator)],
    }


def linkedin_host(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    host = (urlsplit(value if "://" in value else f"//{value}").hostname or "").lower()
    return host == "linkedin.com" or host.endswith(".linkedin.com")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required evaluation file is missing: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return rows


def _trajectory_rows(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in directory.glob("trajectories-*.jsonl"):
        rows.extend(load_jsonl(path))
    return rows


def fabricated_urls(trajectory: dict[str, Any]) -> list[str]:
    claimed = canonical_urls(str(trajectory.get("final_content") or ""))
    for value in _strings(trajectory.get("data")):
        claimed.update(canonical_urls(value))
    evidence: set[str] = set()
    for message in trajectory.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "tool":
            for value in _strings(message.get("content")):
                evidence.update(canonical_urls(value))
    return sorted(claimed - evidence)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _result_data(result: Any) -> dict[str, Any]:
    return result.data if isinstance(result.data, dict) else {}


async def run_rows(
    rows: list[dict[str, Any]], base_url: str, model: str, api_key: str,
    trajectory_dir: Path, concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as http:
        async def one(row: dict[str, Any]) -> dict[str, Any]:
            req = ResearchRequest(
                task=OWNER_TASK,
                context={"company": row.get("company"), "domain": row["domain"]},
                output_fields=OWNER_FIELDS,
            )
            started = time.perf_counter()
            async with semaphore:
                result = await run_agent(
                    req, http, base_url=base_url, model=model, api_key=api_key,
                    trajectory_dir=trajectory_dir,
                )
            return {"source": row, "result": result,
                    "sec": time.perf_counter() - started}
        return await asyncio.gather(*(one(row) for row in rows))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "count": count,
        "mean_iterations": (
            sum(row["iterations"] for row in rows) / count if count else 0.0
        ),
        "mean_seconds": sum(row["sec"] for row in rows) / count if count else 0.0,
        "tool_call_error_count": sum(row["status"] != "success" for row in rows),
        "rows": rows,
    }


async def admission_probes(base_url: str, model: str, api_key: str) -> dict[str, Any]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or "none")
    try:
        first = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Use web_search to find Anthropic's CEO."}],
            tools=PROBE_TOOLS, tool_choice="auto", max_tokens=200,
        )
        message = first.choices[0].message
        calls = message.tool_calls or []
        native = bool(
            calls and calls[0].id and calls[0].function.name == "web_search"
            and isinstance(json.loads(calls[0].function.arguments or "{}"), dict)
        )
        if not native:
            return {"native_tool_call": {"pass": False},
                    "two_turn_loop": {"pass": False}}
        call = calls[0]
        messages = [
            {"role": "user", "content": "Use web_search to find Anthropic's CEO."},
            {"role": "assistant", "content": message.content, "tool_calls": [{
                "id": call.id, "type": "function", "function": {
                    "name": call.function.name, "arguments": call.function.arguments,
                },
            }]},
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps([{
                "url": "https://anthropic.com/company",
                "snippet": "Dario Amodei is the CEO of Anthropic.",
            }])},
        ]
        second = await client.chat.completions.create(
            model=model, messages=messages, tools=PROBE_TOOLS,
            tool_choice="auto", max_tokens=200,
        )
        answer = second.choices[0].message.content or ""
        return {
            "native_tool_call": {"pass": True},
            "two_turn_loop": {"pass": "amodei" in answer.lower(), "answer": answer},
        }
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        return {"native_tool_call": {"pass": False, "error": note},
                "two_turn_loop": {"pass": False, "error": note}}


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    chosen = DATA_DIR / f"{args.split}_domains.txt"
    other = DATA_DIR / ("test_domains.txt" if args.split == "dev" else "dev_domains.txt")
    assert_disjoint(chosen, other, args.train_domains)
    selected_domains = read_domains(chosen)
    requested_sets = {item.strip().lower() for item in args.sets.split(",") if item.strip()}
    if not requested_sets or not requested_sets <= {"a", "b"}:
        raise ValueError("--sets must be a comma-separated subset of a,b")
    report: dict[str, Any] = {
        "configuration": {"base_url": args.base_url, "model": args.model,
                          "split": args.split, "sets": sorted(requested_sets)},
        "sets": {},
    }
    adjudication_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lightgent-eval-") as temp:
        trajectory_dir = Path(temp)
        if "a" in requested_sets:
            if not NEGATIVES_PATH.is_file():
                raise ValueError(f"required evaluation file is missing: {NEGATIVES_PATH}")
            sources = json.loads(NEGATIVES_PATH.read_text(encoding="utf-8"))
            if args.limit is not None:
                sources = sources[:args.limit]
            completed = await run_rows(
                sources, args.base_url, args.model, args.api_key,
                trajectory_dir / "a", args.concurrency,
            )
            trajectories = _trajectory_rows(trajectory_dir / "a")
            trajectories_by_domain = {
                normalize_domain((row.get("subject") or {}).get("domain")): row
                for row in trajectories
                if isinstance(row.get("subject"), dict)
            }
            raw = []
            for item in completed:
                source, result = item["source"], item["result"]
                data = _result_data(result)
                trajectory = trajectories_by_domain.get(normalize_domain(source["domain"]))
                fake = fabricated_urls(trajectory) if trajectory is not None else []
                raw.append({
                    "company": source.get("company"), "domain": source["domain"],
                    "expected": {"gt_lastnames": source.get("gt_lastnames", [])},
                    "got": data, "agree": {}, "fabricated_urls": fake,
                    "status": result.status, "iterations": result.iterations,
                    "sec": item["sec"],
                })
            section = _aggregate(raw)
            section["fabricated_url_count"] = sum(len(row["fabricated_urls"]) for row in raw)
            section["companies_with_fabricated_urls"] = wilson(
                sum(bool(row["fabricated_urls"]) for row in raw), len(raw)
            )
            report["sets"]["a"] = section
        if "b" in requested_sets:
            banked = {}
            for row in load_jsonl(OVERNIGHT_PATH):
                domain = normalize_domain(row.get("domain"))
                if domain in selected_domains and row.get("status") == "success":
                    banked.setdefault(domain, row)
            sources = list(banked.values())
            if args.limit is not None:
                sources = sources[:args.limit]
            completed = await run_rows(
                sources, args.base_url, args.model, args.api_key,
                trajectory_dir / "b", args.concurrency,
            )
            raw = []
            for item in completed:
                source, result = item["source"], item["result"]
                data = _result_data(result)
                got_name = data.get("owner_name")
                title = data.get("title")
                linkedin = data.get("linkedin_url")
                flags = {
                    "surname_agreement": surname_agrees(got_name, source.get("got")),
                    "title_nonempty": bool(isinstance(title, str) and title.strip()),
                    "linkedin_reported": bool(isinstance(linkedin, str) and linkedin.strip()),
                    "linkedin_host": linkedin_host(linkedin),
                }
                raw.append({
                    "company": source.get("company"), "domain": normalize_domain(source["domain"]),
                    "expected": {"owner_name": source.get("got"), "title": source.get("title"),
                                 "linkedin_url": source.get("linkedin")},
                    "got": data, "agree": flags, "fabricated_urls": [],
                    "status": result.status, "iterations": result.iterations,
                    "sec": item["sec"],
                })
                adjudication_rows.append({
                    "domain": normalize_domain(source["domain"]),
                    "answered_owner_name": got_name, "title": title,
                    "linkedin_url": linkedin,
                    "banked_silver": {"owner_name": source.get("got"),
                                      "title": source.get("title"),
                                      "linkedin_url": source.get("linkedin")},
                    "raw_agreement_flags": flags,
                })
            section = _aggregate(raw)
            section["metrics"] = {
                key: wilson(sum(bool(row["agree"][key]) for row in raw), len(raw))
                for key in ("surname_agreement", "title_nonempty",
                            "linkedin_reported", "linkedin_host")
            }
            report["sets"]["b"] = section
    adjudication_path = args.out.parent / "adjudication.jsonl"
    adjudication_path.parent.mkdir(parents=True, exist_ok=True)
    adjudication_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in adjudication_rows),
        encoding="utf-8",
    )
    report["adjudication_path"] = str(adjudication_path)
    report["probes"] = await admission_probes(args.base_url, args.model, args.api_key)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--out", type=Path, default=Path("report.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--train-domains", type=Path,
                        default=DATA_DIR / "train_domains.txt")
    parser.add_argument("--sets", default="a,b")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def print_summary(report: dict[str, Any]) -> None:
    print(f"{'SET':<4} {'N':>4} {'ITERS':>8} {'SEC':>8} {'ERRORS':>7}")
    for name, section in report["sets"].items():
        print(f"{name.upper():<4} {section['count']:>4} "
              f"{section['mean_iterations']:>8.2f} {section['mean_seconds']:>8.2f} "
              f"{section['tool_call_error_count']:>7}")
    probes = report["probes"]
    print("probes: native={} two_turn={}".format(
        "PASS" if probes["native_tool_call"]["pass"] else "FAIL",
        "PASS" if probes["two_turn_loop"]["pass"] else "FAIL",
    ))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = asyncio.run(evaluate(args))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
