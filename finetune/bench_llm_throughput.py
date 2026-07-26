"""Measure maximum sustainable LLM throughput for the enrichment workload.

Not a generic tokens/sec benchmark: it replays REAL banked trajectories so the
prompt sizes, tool schemas and output shapes match production. Reports, per
concurrency level, requests/sec, prompt and generation tokens/sec, latency
percentiles, and the implied companies/hour once search is no longer the wall.

Usage:
    python -m finetune.bench_llm_throughput --base-url URL --api-key KEY \
        --model eval-model --levels 1,4,8,16,32,64 --requests-per-level 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

TRAJ = Path("finetune/data/trajectories/trajectories-20260726.jsonl")
# Median trajectory is ~10 LLM calls; that is the multiplier from one model
# call to one finished company.
CALLS_PER_COMPANY = 10


def load_prompts(limit: int = 64) -> list[list[dict]]:
    """Real message prefixes from banked runs, truncated before a tool call."""
    prompts = []
    if not TRAJ.exists():
        raise SystemExit(f"no trajectories at {TRAJ}")
    with TRAJ.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            # Cut at a mid-run point so the prompt carries real accumulated
            # tool output, which is what makes these prompts expensive.
            cut = max(2, min(len(messages) - 1, 8))
            prefix = [
                {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")}
                for m in messages[:cut]
                if m.get("role") in ("system", "user", "assistant", "tool")
            ]
            if len(prefix) >= 3:
                prompts.append(prefix)
            if len(prompts) >= limit:
                break
    if not prompts:
        raise SystemExit("no usable prompts found")
    return prompts


async def one_call(client, base_url, api_key, model, messages, max_tokens):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        if response.status_code != 200:
            return {"ok": False, "sec": elapsed, "status": response.status_code}
        payload = response.json()
        usage = payload.get("usage") or {}
        return {
            "ok": True,
            "sec": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as exc:  # noqa: BLE001 - a failed call is a data point
        return {"ok": False, "sec": time.perf_counter() - start, "error": type(exc).__name__}


async def run_level(base_url, api_key, model, prompts, level, total, max_tokens):
    semaphore = asyncio.Semaphore(level)
    rng = random.Random(1234)

    async with httpx.AsyncClient() as client:
        async def worker(index):
            async with semaphore:
                return await one_call(
                    client, base_url, api_key, model,
                    prompts[rng.randrange(len(prompts))], max_tokens,
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(worker(i) for i in range(total)))
        wall = time.perf_counter() - started

    ok = [r for r in results if r["ok"]]
    if not ok:
        return {"level": level, "ok": 0, "failed": len(results), "wall": wall}
    latencies = sorted(r["sec"] for r in ok)
    return {
        "level": level,
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall": wall,
        "rps": len(ok) / wall,
        "prompt_tps": sum(r["prompt_tokens"] for r in ok) / wall,
        "gen_tps": sum(r["completion_tokens"] for r in ok) / wall,
        "p50": statistics.median(latencies),
        "p95": latencies[max(0, int(len(latencies) * 0.95) - 1)],
        "mean_prompt": statistics.mean(r["prompt_tokens"] for r in ok),
    }


async def main_async(args):
    prompts = load_prompts()
    print(f"replaying {len(prompts)} real trajectory prefixes\n")
    header = f"{'C':>4} {'ok':>4} {'fail':>5} {'req/s':>7} {'prompt tok/s':>13} {'gen tok/s':>10} {'p50 s':>7} {'p95 s':>7} {'companies/hr':>13}"
    print(header)
    print("-" * len(header))
    best = None
    for level in args.levels:
        stats = await run_level(
            args.base_url, args.api_key, args.model, prompts,
            level, args.requests_per_level, args.max_tokens,
        )
        if not stats.get("rps"):
            print(f"{level:>4} {stats['ok']:>4} {stats['failed']:>5}   all requests failed")
            continue
        companies = stats["rps"] * 3600 / CALLS_PER_COMPANY
        print(
            f"{level:>4} {stats['ok']:>4} {stats['failed']:>5} {stats['rps']:>7.2f} "
            f"{stats['prompt_tps']:>13.0f} {stats['gen_tps']:>10.0f} "
            f"{stats['p50']:>7.1f} {stats['p95']:>7.1f} {companies:>13.0f}"
        )
        if best is None or stats["rps"] > best[1]:
            best = (level, stats["rps"], companies, stats["p95"])
    if best:
        print(
            f"\nBEST: concurrency {best[0]} -> {best[1]:.2f} req/s "
            f"= {best[2]:.0f} companies/hr (p95 {best[3]:.1f}s per call)"
        )
        print(f"assumes {CALLS_PER_COMPANY} LLM calls per company (measured median)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="eval-model")
    parser.add_argument("--levels", default="1,4,8,16,32,64")
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()
    args.levels = [int(x) for x in args.levels.split(",") if x.strip()]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
