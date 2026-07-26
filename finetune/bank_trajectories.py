"""Run LightGent over a CSV and bank resumable training trajectories."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from lightgent_service import ResearchRequest, run_agent


DEFAULT_INPUT = (
    r"C:\Users\hi\Desktop\local-enrichos\exports"
    r"\run1-agricultural-friesland-20260516-2010.csv"
)
DEFAULT_HOLDOUTS = (
    "finetune/data/dev_domains.txt",
    "finetune/data/test_domains.txt",
)
OWNER_TASK = (
    "Find the owner or most senior owner-like decision-maker for this company. "
    "Return the person's full name, exact title, and personal LinkedIn profile URL "
    "when verifiable. Do not guess. Use null for any field you cannot verify."
)
OUTPUT_FIELDS = {
    "owner_name": "Full name of the owner or most senior owner-like decision-maker",
    "title": "Their exact job title",
    "linkedin_url": "Their personal LinkedIn profile URL",
}


def normalize_domain(value: str | None) -> str:
    """Normalize a domain-like value for deduplication and exclusion."""
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def load_companies(path: Path) -> list[dict[str, str]]:
    seen: set[str] = set()
    companies: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            domain = normalize_domain(row.get("domain"))
            if not domain or domain in seen:
                continue
            seen.add(domain)
            companies.append({
                "company": (row.get("company_name") or "").strip(),
                "domain": domain,
                "city": (row.get("city") or "").strip(),
                "country": (row.get("country") or "").strip(),
            })
    return companies


def load_holdouts(paths: list[Path]) -> set[str]:
    domains: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                domain = normalize_domain(line.strip())
                if domain:
                    domains.add(domain)
    return domains


def load_successes(path: Path) -> set[str]:
    successes: set[str] = set()
    if not path.exists():
        return successes
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if record.get("status") == "success":
                domain = normalize_domain(record.get("domain"))
                if domain:
                    successes.add(domain)
    return successes


async def append_record(path: Path, record: dict[str, Any], lock: asyncio.Lock) -> None:
    async with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


async def bank(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    holdout_paths = [Path(item) for item in args.holdout]
    if not input_path.is_file():
        raise FileNotFoundError(f"missing input file: {input_path}")
    missing = [path for path in holdout_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing holdout file: {missing[0]}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.target_successes < 0 or args.limit < 0:
        raise ValueError("--target-successes and --limit cannot be negative")

    state_path = Path(args.state)
    trajectory_dir = Path(args.trajectory_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    successes = load_successes(state_path)
    excluded = load_holdouts(holdout_paths) | successes
    candidates = [
        row for row in load_companies(input_path) if row["domain"] not in excluded
    ]
    if args.limit:
        candidates = candidates[: args.limit]

    attempted = 0
    started = time.monotonic()
    write_lock = asyncio.Lock()
    stop_requested = asyncio.Event()

    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_requested.set)
            installed_signals.append(sig)
        except (NotImplementedError, RuntimeError):
            pass

    async def process(row: dict[str, str], client: httpx.AsyncClient) -> dict[str, Any]:
        began = time.monotonic()
        status = "error"
        iterations = 0
        tool_calls = 0
        error: str | None = None
        request = ResearchRequest(
            task=OWNER_TASK,
            context={
                "company_name": row["company"],
                "domain": row["domain"],
                "city": row["city"],
                "country": row["country"],
            },
            output_fields=OUTPUT_FIELDS,
        )
        overrides = {
            key: value
            for key, value in {
                "base_url": args.base_url,
                "model": args.model,
                "api_key": args.api_key,
            }.items()
            if value is not None
        }
        try:
            response = await run_agent(
                request,
                client,
                trajectory_dir=trajectory_dir,
                **overrides,
            )
            status = response.status
            iterations = response.iterations
            tool_calls = response.tool_calls
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "domain": row["domain"],
            "company": row["company"],
            "status": status,
            "iterations": iterations,
            "tool_calls": tool_calls,
            "sec": round(time.monotonic() - began, 3),
        }
        if error:
            record["error"] = error
        await append_record(state_path, record, write_lock)
        return record

    iterator = iter(candidates)
    pending: set[asyncio.Task[dict[str, Any]]] = set()
    async with httpx.AsyncClient() as client:
        try:
            while len(successes) < args.target_successes and not stop_requested.is_set():
                while len(pending) < args.concurrency and not stop_requested.is_set():
                    try:
                        row = next(iterator)
                    except StopIteration:
                        break
                    pending.add(asyncio.create_task(process(row, client)))
                if not pending:
                    break
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    record = task.result()
                    attempted += 1
                    if record["status"] == "success":
                        successes.add(record["domain"])
                    if attempted % 10 == 0:
                        elapsed = max(time.monotonic() - started, 0.001)
                        print(
                            f"attempted={attempted} successes={len(successes)} "
                            f"rate_per_min={attempted * 60 / elapsed:.1f}",
                            flush=True,
                        )
            if pending:
                done = await asyncio.gather(*pending)
                for record in done:
                    attempted += 1
                    if record["status"] == "success":
                        successes.add(record["domain"])
        finally:
            for sig in installed_signals:
                loop.remove_signal_handler(sig)

    elapsed = time.monotonic() - started
    summary = {
        "attempted": attempted,
        "successes": len(successes),
        "sec": round(elapsed, 3),
        "stopped": "interrupt" if stop_requested.is_set() else (
            "target" if len(successes) >= args.target_successes else "exhausted"
        ),
    }
    print(
        f"done attempted={attempted} successes={len(successes)} "
        f"sec={elapsed:.1f} reason={summary['stopped']}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--holdout", action="append", default=None)
    parser.add_argument("--state", default="finetune/data/banking/banking_results.jsonl")
    parser.add_argument("--trajectory-dir", default="finetune/data/trajectories")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--target-successes", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.holdout is None:
        args.holdout = list(DEFAULT_HOLDOUTS)
    try:
        asyncio.run(bank(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"fatal: {exc}", flush=True)
        return 2
    except KeyboardInterrupt:
        print("fatal: interrupted before graceful shutdown could complete", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
