#!/usr/bin/env python3
"""Convert logged Lightgent trajectories into chat SFT JSONL files."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


URL_RE = re.compile(r"""https?://[^\s<>{}'"]+""", re.IGNORECASE)
URL_TRAILING = "),.;:]'\""
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{1,79}$")
# Never rewritten when stripping unverified URLs: these carry the answer itself.
PROTECTED_FIELDS = ("owner_name", "title")
SPLIT_RULE = "SHA-256 of normalized domain modulo the hash space; values below val_frac go to val"


class DatasetMinimumError(ValueError):
    """Raised when a dataset does not meet the configured minimums."""


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def normalize_domain(value: Any) -> str:
    text = str(value).strip().lower()
    parsed = urlsplit(text if "://" in text else f"//{text}")
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host


def domain_for(row: dict[str, Any]) -> str:
    subject = row.get("subject")
    if isinstance(subject, dict) and subject.get("domain"):
        domain = normalize_domain(subject["domain"])
        if domain:
            return domain
    encoded = json.dumps(subject, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "subject-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_urls(value: str) -> set[str]:
    result = set()
    for raw in URL_RE.findall(value):
        parsed = urlsplit(raw.rstrip(URL_TRAILING))
        if not parsed.hostname:
            continue
        # Canonical URLs intentionally strip ports.
        host = parsed.hostname.lower()
        path = (parsed.path or "").rstrip("/")
        result.add((host + path).lower())
    return result


def tool_evidence(row: dict[str, Any]) -> set[str]:
    evidence: set[str] = set()
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "tool":
            for value in strings(message.get("content")):
                evidence.update(canonical_urls(value))
    return evidence


def claimed_urls(row: dict[str, Any]) -> set[str]:
    claimed = canonical_urls(str(row.get("final_content") or ""))
    for value in strings(row.get("data")):
        claimed.update(canonical_urls(value))
    return claimed


def has_supported_urls(row: dict[str, Any]) -> bool:
    return claimed_urls(row) <= tool_evidence(row)


def primary_url(row: dict[str, Any]) -> str | None:
    """The person link the answer stands on. Fabricating it disqualifies the row."""
    data = row.get("data")
    if isinstance(data, dict):
        value = data.get("linkedin_url")
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in [*strings(data), str(row.get("final_content") or "")]:
        for raw in URL_RE.findall(value):
            if "linkedin.com" in raw.lower():
                return raw.rstrip(URL_TRAILING)
    return None


def unverified_raw_urls(text: str, evidence: set[str]) -> list[str]:
    found = []
    for raw in URL_RE.findall(text):
        trimmed = raw.rstrip(URL_TRAILING)
        if canonical_urls(trimmed) - evidence:
            found.append(trimmed)
    return found


def strip_unverified(value: Any, evidence: set[str], keep: str | None) -> Any:
    """Remove URLs with no tool-result backing, keeping the primary URL as-is."""
    if isinstance(value, str):
        unverified = [u for u in unverified_raw_urls(value, evidence) if u != keep]
        if not unverified:
            return value
        if canonical_urls(value) and value.strip() in unverified:
            return None
        cleaned = value
        for url in unverified:
            cleaned = cleaned.replace(url, "")
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\(\s*\)|\[\s*\]", "", cleaned)
        return cleaned.strip(" ,;") or None
    if isinstance(value, list):
        result = []
        for item in value:
            cleaned = strip_unverified(item, evidence, keep)
            if cleaned not in (None, ""):
                result.append(cleaned)
        return result
    if isinstance(value, dict):
        return {
            key: (item if key in PROTECTED_FIELDS else strip_unverified(item, evidence, keep))
            for key, item in value.items()
        }
    return value


def answered_name(data: Any) -> str | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "owner_name" and isinstance(value, str):
                return value
    return next((value for value in strings(data) if NAME_RE.fullmatch(value.strip())), None)


def load_gold(path: Path | None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if path is None:
        return result
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                result[normalize_domain(item["domain"])] = [
                    str(name).lower() for name in item["gold"]
                ]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid gold row") from exc
    return result


def load_holdouts(paths: list[Path]) -> set[str]:
    domains = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    domain = normalize_domain(line)
                    if domain:
                        domains.add(domain)
    return domains


def lane_allowed(row: dict[str, Any], prefixes: tuple[str, ...] | None) -> bool:
    if prefixes is None:
        return True
    row_lanes = row.get("lanes")
    candidates = row_lanes if isinstance(row_lanes, list) and row_lanes else [row.get("lane")]
    return all(
        isinstance(lane, str) and any(lane.startswith(prefix) for prefix in prefixes)
        for lane in candidates
    )


def output_messages(row: dict[str, Any], final_override: Any = None) -> list[dict[str, Any]]:
    messages = copy.deepcopy(row.get("messages", []))
    final_content = row.get("final_content") if final_override is None else final_override
    for message in reversed(messages):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            message["content"] = final_content
            return messages
    messages.append({"role": "assistant", "content": final_content})
    return messages


def in_validation(domain: str, fraction: float) -> bool:
    bucket = int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[:8], "big")
    return bucket / 2**64 < fraction


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent / 100 * len(ordered)) - 1)]


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(matches)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("no input files matched")
    return unique


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def timestamp_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    if isinstance(value, str):
        return (1, value)
    return (2, "")


def build(
    inputs: list[str],
    out_dir: Path,
    lane_spec: str = "mistral",
    val_frac: float = 0.1,
    gold_path: Path | None = None,
    holdout_paths: list[Path] | None = None,
    min_train_rows: int = 600,
    min_val_rows: int = 60,
    min_train_domains: int = 400,
    max_rows_per_domain: int = 3,
    strip_mode: bool = True,
) -> dict[str, Any]:
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError("--val-frac must be between 0 and 1")
    if min(min_train_rows, min_val_rows, min_train_domains) < 0 or max_rows_per_domain < 1:
        raise ValueError("minimums must be nonnegative and --max-rows-per-domain must be positive")
    prefixes = None if lane_spec.strip() == "*" else tuple(
        part.strip() for part in lane_spec.split(",") if part.strip()
    )
    paths = expand_inputs(inputs)
    gold = load_gold(gold_path)
    holdouts = load_holdouts(holdout_paths or [])
    drops = {
        "status": 0,
        "anti_fabrication": 0,
        "primary_url_unverified": 0,
        "lanes": 0,
        "gold": 0,
    }
    kept: list[tuple[str, Any, int, dict[str, Any]]] = []
    rows_in = 0
    rows_url_stripped = 0

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                rows_in += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if row.get("status") != "success":
                    drops["status"] += 1
                    continue
                final_override = None
                if not has_supported_urls(row):
                    if not strip_mode:
                        drops["anti_fabrication"] += 1
                        continue
                    evidence = tool_evidence(row)
                    primary = primary_url(row)
                    # A fabricated person link is disqualifying; auxiliary
                    # citations are stripped so the model learns to cite only
                    # what it actually saw (mirrors _strip_fabricated_urls).
                    if primary is not None and canonical_urls(primary) - evidence:
                        drops["primary_url_unverified"] += 1
                        continue
                    final_override = strip_unverified(
                        row.get("final_content"), evidence, primary
                    )
                    if final_override is None:
                        final_override = ""
                    rows_url_stripped += 1
                if not lane_allowed(row, prefixes):
                    drops["lanes"] += 1
                    continue
                domain = domain_for(row)
                if domain in gold:
                    name = answered_name(row.get("data"))
                    if name is None or not any(last in name.lower() for last in gold[domain]):
                        drops["gold"] += 1
                        continue
                kept.append(
                    (
                        domain,
                        row.get("ts"),
                        rows_in,
                        {"messages": output_messages(row, final_override)},
                    )
                )

    holdout_exclusions = sum(domain in holdouts for domain, _, _, _ in kept)
    kept = [item for item in kept if item[0] not in holdouts]
    by_domain: dict[str, list[tuple[str, Any, int, dict[str, Any]]]] = {}
    for item in kept:
        by_domain.setdefault(item[0], []).append(item)
    dedup_drops = 0
    capped = []
    for domain in sorted(by_domain):
        items = sorted(by_domain[domain], key=lambda item: (timestamp_key(item[1]), item[2]))
        capped.extend(items[:max_rows_per_domain])
        dedup_drops += max(0, len(items) - max_rows_per_domain)
    capped.sort(key=lambda item: item[2])

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    domains: dict[str, set[str]] = {"train": set(), "val": set()}
    lengths: list[float] = []
    for domain, _, _, record in capped:
        split = "val" if in_validation(domain, val_frac) else "train"
        splits[split].append(record)
        domains[split].add(domain)
        lengths.append(len(json.dumps(record, ensure_ascii=False)) / 4)

    failures = []
    if len(splits["train"]) < min_train_rows:
        failures.append(f"train rows {len(splits['train'])} < {min_train_rows}")
    if len(splits["val"]) < min_val_rows:
        failures.append(f"val rows {len(splits['val'])} < {min_val_rows}")
    if len(domains["train"]) < min_train_domains:
        failures.append(f"train domains {len(domains['train'])} < {min_train_domains}")
    if failures:
        raise DatasetMinimumError("dataset minimum gate failed: " + "; ".join(failures))

    emitted: dict[str, bytes] = {}
    for split in ("train", "val"):
        emitted[f"{split}.jsonl"] = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in splits[split]
        ).encode("utf-8")
        emitted[f"{split}_domains.txt"] = "".join(
            domain + "\n" for domain in sorted(domains[split])
        ).encode("utf-8")
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    manifest = {
        "inputs": [{"path": str(path), "sha256": sha256_bytes(path.read_bytes())} for path in paths],
        "rows_in": rows_in,
        "drops": drops,
        "holdout_exclusions": holdout_exclusions,
        "per_domain_dedup_drops": dedup_drops,
        "rows_url_stripped": rows_url_stripped,
        "strip_unverified": strip_mode,
        "rows_out": {split: len(splits[split]) for split in splits},
        "unique_domains": {split: len(domains[split]) for split in domains},
        "val_frac": val_frac,
        "split_rule": SPLIT_RULE,
        "git_head": git_head,
        "output_sha256": {
            f"{split}.jsonl": sha256_bytes(emitted[f"{split}.jsonl"])
            for split in ("train", "val")
        },
    }
    emitted["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in emitted.items():
        (out_dir / filename).write_bytes(content)

    return {
        "rows_in": rows_in, "drops": drops, "holdout_exclusions": holdout_exclusions,
        "rows_url_stripped": rows_url_stripped,
        "per_domain_dedup_drops": dedup_drops, "train": len(splits["train"]),
        "val": len(splits["val"]), "percentiles": {
            p: percentile(lengths, p) for p in (50, 90, 99)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="input JSONL paths or glob patterns")
    parser.add_argument("--out-dir", type=Path, default=Path("finetune/data"))
    parser.add_argument("--lanes", default="mistral", help="comma-separated lane prefixes, or *")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--holdout", action="append", type=Path, default=[])
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--min-train-rows", type=int, default=600)
    parser.add_argument("--min-val-rows", type=int, default=60)
    parser.add_argument("--min-train-domains", type=int, default=400)
    parser.add_argument("--max-rows-per-domain", type=int, default=3)
    parser.add_argument(
        "--no-strip-unverified",
        dest="strip_unverified",
        action="store_false",
        help="drop rows with any unverified URL instead of stripping auxiliary ones",
    )
    parser.set_defaults(strip_unverified=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        stats = build(
            args.inputs, args.out_dir, args.lanes, args.val_frac, args.gold, args.holdout,
            args.min_train_rows, args.min_val_rows, args.min_train_domains,
            args.max_rows_per_domain, args.strip_unverified,
        )
    except DatasetMinimumError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"rows in: {stats['rows_in']}")
    for name, count in stats["drops"].items():
        print(f"dropped {name}: {count}")
    print(f"rows with unverified urls stripped: {stats['rows_url_stripped']}")
    print(f"excluded holdout: {stats['holdout_exclusions']}")
    print(f"dropped per-domain cap: {stats['per_domain_dedup_drops']}")
    print(f"rows out train: {stats['train']}")
    print(f"rows out val: {stats['val']}")
    values = stats["percentiles"]
    print(f"approx tokens: p50={values[50]:.0f} p90={values[90]:.0f} p99={values[99]:.0f}")


if __name__ == "__main__":
    main()
