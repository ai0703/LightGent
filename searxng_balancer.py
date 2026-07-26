"""SearXNG shard balancer: engine-aware routing across instances.

Why this is not just the proxy balancer again. SearXNG suspends a failing engine
per INSTANCE, in memory, and reports it in the JSON as `unresponsive_engines`.
Measured 2026-07-26 across three shards sharing one 100 proxy pool: all three got
exactly 915 queries, but Google was healthy on 100 and 98 percent of shard C and
A while shard B sat at 83 percent. On a single instance that same suspension
removes Google from EVERY query; sharded, the damage is confined to one shard.

So the win is not spreading load evenly, it is reading `unresponsive_engines` and
steering queries that need Google to a shard whose Google still answers, while
that shard keeps serving Bing and DuckDuckGo traffic. Round-robin cannot do that.

Numbers this replaces: 1 instance 244 q/min at 86 percent Google; 3 shards
round-robin 281 q/min at 94 percent.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

START_RATE_PER_MIN = 120.0        # shards are ours, so open generously
MIN_RATE_PER_MIN = 10.0
MAX_RATE_PER_MIN = 600.0
INCREASE_PER_SUCCESS = 1.0
DECREASE_FACTOR = 0.5
# SearXNG's own 429 suspension is configured to 60s server side, so the client
# should stop steering that engine here for slightly longer than that.
ENGINE_COOLDOWN_SEC = 75.0
LANE_COOLDOWN_SEC = 15.0
LANE_COOLDOWN_MAX_SEC = 300.0
THIN_RESULT_COUNT = 5             # below this, treat the answer as degraded


@dataclass
class Shard:
    """One SearXNG instance and its per-engine health."""
    url: str
    rate_per_min: float = START_RATE_PER_MIN
    next_allowed_at: float = 0.0
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    ok: int = 0
    failed: int = 0
    thin: int = 0
    engine_down_until: dict[str, float] = field(default_factory=dict)
    engine_suspensions: dict[str, int] = field(default_factory=dict)
    client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def interval(self) -> float:
        return 60.0 / max(self.rate_per_min, 0.01)

    def available_at(self, now: float) -> float:
        return max(self.next_allowed_at, self.cooldown_until, now)

    def engine_healthy(self, engine: str, now: float) -> bool:
        return self.engine_down_until.get(engine, 0.0) <= now

    def note_unresponsive(self, engines, now: float) -> None:
        """Record engines SearXNG says are down on this shard."""
        for name in engines:
            key = _engine_key(name)
            self.engine_down_until[key] = now + ENGINE_COOLDOWN_SEC
            self.engine_suspensions[key] = self.engine_suspensions.get(key, 0) + 1

    def on_success(self, thin: bool) -> None:
        self.ok += 1
        self.consecutive_failures = 0
        if thin:
            self.thin += 1
        self.rate_per_min = min(MAX_RATE_PER_MIN,
                                self.rate_per_min + INCREASE_PER_SUCCESS)

    def on_failure(self, now: float) -> None:
        self.failed += 1
        self.consecutive_failures += 1
        self.rate_per_min = max(MIN_RATE_PER_MIN,
                                self.rate_per_min * DECREASE_FACTOR)
        self.cooldown_until = now + min(
            LANE_COOLDOWN_MAX_SEC,
            LANE_COOLDOWN_SEC * (2 ** (self.consecutive_failures - 1)))


def _engine_key(name) -> str:
    """Normalise 'google cse' / ('google cse', 'too many requests') -> 'google'."""
    if isinstance(name, (list, tuple)):
        name = name[0] if name else ""
    n = str(name).lower()
    for known in ("google", "bing", "duckduckgo", "brave", "startpage",
                  "wikipedia", "mojeek", "qwant"):
        if known in n:
            return known
    return n.strip()


class SearxngBalancer:
    """Route queries across SearXNG shards, preferring shards whose engine is up.

        async with SearxngBalancer(urls) as pool:
            results, shard = await pool.search("owner of X", prefer_engine="google")
    """

    def __init__(self, urls, timeout: float = 90.0, max_attempts: int = 3,
                 per_shard_connections: int = 16):
        self.shards = [Shard(url=u.rstrip("/")) for u in urls]
        if not self.shards:
            raise ValueError("SearxngBalancer needs at least one shard URL")
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._per_shard_connections = per_shard_connections
        self._lock = asyncio.Lock()
        now = time.monotonic()
        for i, s in enumerate(self.shards):
            s.next_allowed_at = now + (i / len(self.shards)) * s.interval

    async def __aenter__(self):
        limits = httpx.Limits(
            max_connections=self._per_shard_connections,
            max_keepalive_connections=self._per_shard_connections)
        for s in self.shards:
            s.client = httpx.AsyncClient(timeout=self._timeout, limits=limits)
        return self

    async def __aexit__(self, *exc):
        await asyncio.gather(*(s.client.aclose() for s in self.shards
                               if s.client), return_exceptions=True)

    async def _acquire(self, prefer_engine: str | None,
                       exclude: set[str]) -> Shard:
        """Pick a due shard, preferring ones where prefer_engine is healthy.

        Falls back to any shard rather than stalling: a Bing-only answer beats
        no answer, and the caller can see which engines actually replied.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                pool = [s for s in self.shards if s.url not in exclude] \
                    or self.shards
                if prefer_engine:
                    healthy = [s for s in pool
                               if s.engine_healthy(prefer_engine, now)]
                    pool = healthy or pool
                best = min(pool,
                           key=lambda s: (s.available_at(now), random.random()))
                due = best.available_at(now)
                if due <= now:
                    best.next_allowed_at = now + best.interval
                    return best
                delay = due - now
            await asyncio.sleep(min(delay, 0.5))

    async def search(self, query: str, prefer_engine: str | None = "google",
                     params: dict | None = None):
        """Return (payload, shard). payload is the parsed SearXNG JSON.

        Retries on a different shard when a request fails, or when the answer
        comes back thin AND the preferred engine was reported unresponsive,
        because that is the signature of the degradation that silently returns
        irrelevant results instead of an error.
        """
        exclude: set[str] = set()
        last = (None, None)
        for _ in range(self._max_attempts):
            shard = await self._acquire(prefer_engine, exclude)
            now = time.monotonic()
            q = {"q": query, "format": "json"}
            if params:
                q.update(params)
            try:
                resp = await shard.client.get(f"{shard.url}/search", params=q)
                payload = resp.json()
            except Exception:
                shard.on_failure(now)
                exclude.add(shard.url)
                continue
            if resp.status_code != 200:
                shard.on_failure(now)
                exclude.add(shard.url)
                last = (None, shard)
                continue

            unresponsive = payload.get("unresponsive_engines") or []
            shard.note_unresponsive(unresponsive, now)
            results = payload.get("results") or []
            thin = len(results) < THIN_RESULT_COUNT
            shard.on_success(thin)

            engine_missing = prefer_engine is not None and not any(
                _engine_key(e) == prefer_engine
                for r in results for e in (r.get("engines") or []))
            # Only worth another shard if the answer is BOTH thin and missing
            # the engine we wanted. A full answer without Google is still fine.
            if thin and engine_missing and len(exclude) + 1 < len(self.shards):
                exclude.add(shard.url)
                last = (payload, shard)
                continue
            return payload, shard
        return last

    def engine_health(self) -> dict:
        now = time.monotonic()
        return {s.url.split("//")[-1][:40]:
                {e: round(max(0.0, t - now), 1)
                 for e, t in s.engine_down_until.items() if t > now}
                for s in self.shards}

    def stats(self) -> dict:
        ok = sum(s.ok for s in self.shards)
        return {
            "shards": len(self.shards),
            "ok": ok,
            "failed": sum(s.failed for s in self.shards),
            "thin": sum(s.thin for s in self.shards),
            "thin_pct": 100.0 * sum(s.thin for s in self.shards) / max(ok, 1),
            "engine_suspensions": {
                e: sum(s.engine_suspensions.get(e, 0) for s in self.shards)
                for e in sorted({e for s in self.shards
                                 for e in s.engine_suspensions})},
            "per_shard_ok": [s.ok for s in self.shards],
            "currently_down": self.engine_health(),
        }
