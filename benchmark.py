"""Benchmark the LightGent LLM endpoint: tok/s and req/s through the tunnel.

Measures the numbers that decide MAX_CONCURRENT and whether the model is
"fast enough": single-stream generation speed, then aggregate throughput at
increasing concurrency, plus a tool-calling round-trip.

Example:
  python benchmark.py --url https://xxxx.trycloudflare.com/v1 \
      --key lightgent-change-me --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

PROMPT = ("Write a 150-word explanation of why cold email deliverability "
          "depends on domain reputation and inbox warmup.")

TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


async def one_completion(client: AsyncOpenAI, model: str, max_tokens: int = 200):
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    dt = time.perf_counter() - t0
    toks = r.usage.completion_tokens if r.usage else max_tokens
    return dt, toks


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="LLM_BASE_URL (ends in /v1)")
    ap.add_argument("--key", default="none")
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="2,4,8", help="comma-separated concurrency levels")
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=args.url, api_key=args.key, timeout=600)

    print("warmup...")
    await one_completion(client, args.model, max_tokens=16)

    # 1. single-stream speed (3 runs)
    runs = [await one_completion(client, args.model) for _ in range(3)]
    speeds = [t / d for d, t in runs]
    print(f"\nsingle stream: {statistics.mean(speeds):.1f} tok/s "
          f"(runs: {', '.join(f'{s:.1f}' for s in speeds)})")

    # 2. aggregate throughput at each concurrency level
    print("\nconcurrency sweep (200-token completions):")
    for n in [int(x) for x in args.levels.split(",") if x.strip()]:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[one_completion(client, args.model) for _ in range(n)],
            return_exceptions=True,
        )
        wall = time.perf_counter() - t0
        ok = [r for r in results if not isinstance(r, Exception)]
        errs = len(results) - len(ok)
        total_toks = sum(t for _, t in ok)
        lat = statistics.mean(d for d, _ in ok) if ok else 0
        print(f"  n={n:>2}: {len(ok)/wall:.2f} req/s | {total_toks/wall:>6.0f} tok/s aggregate "
              f"| avg latency {lat:.1f}s" + (f" | {errs} ERRORS" if errs else ""))

    # 3. tool-calling round trip (what the agent actually does)
    t0 = time.perf_counter()
    try:
        r = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user",
                       "content": "Find the CEO of 'Bakkerij Holtkamp' in Amsterdam. Use the tool."}],
            tools=[TOOL], tool_choice="auto", max_tokens=150,
        )
        dt = time.perf_counter() - t0
        tc = r.choices[0].message.tool_calls
        print(f"\ntool call: {'OK' if tc else 'NOT USED (check model/parser)'} in {dt:.1f}s"
              + (f" -> {tc[0].function.name}({tc[0].function.arguments})" if tc else ""))
    except Exception as e:
        print(f"\ntool call FAILED ({e}) — agent will rely on tag-mode fallback")

    print("\nRule of thumb: set MAX_CONCURRENT to the highest n where req/s "
          "still improved, minus one step.")


if __name__ == "__main__":
    asyncio.run(main())
