"""Proxy pool balancer: pace to each lane's measured limit, park before 429.

Same design as the balance broker used for LLM lanes, applied to proxies. The
2026-07-26 measurement that motivated it: blasting 40 workers across 105 proxies
delivered 2,069 fetches/min but 48 percent of calls came back 429, because every
lane was pushed past its limit continuously. Minute one had ZERO 429s, which is
the tell: the capacity is real, the pacing was not.

Each lane keeps its own rate estimate and adapts AIMD style, the way congestion
control does:
  success        -> additively raise the allowed rate, up to a ceiling
  429 / 451 /401 -> multiplicatively cut the rate AND park the lane for a cooldown

Selection prefers the lane that has been idle longest relative to its own
allowance, so load spreads evenly instead of hammering whichever lane answers
first. No credentials live in this file; pass them in.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

# Tunables. Deliberately module level so they can be adjusted without editing
# logic, and so tests can shrink the cooldowns.
START_RATE_PER_MIN = 20.0     # conservative opening bid per lane
MIN_RATE_PER_MIN = 2.0
MAX_RATE_PER_MIN = 60.0
INCREASE_PER_SUCCESS = 0.25   # additive increase
DECREASE_FACTOR = 0.5         # multiplicative decrease on a limit signal
COOLDOWN_BASE_SEC = 20.0
COOLDOWN_MAX_SEC = 600.0
LIMIT_STATUSES = frozenset({429, 401, 403, 451, 503})


@dataclass
class Lane:
    """One proxy endpoint plus its measured state."""
    url: str
    kind: str = "proxy"
    rate_per_min: float = START_RATE_PER_MIN
    next_allowed_at: float = 0.0
    cooldown_until: float = 0.0
    consecutive_limits: int = 0
    ok: int = 0
    limited: int = 0
    errors: int = 0
    client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def interval(self) -> float:
        return 60.0 / max(self.rate_per_min, 0.01)

    def available_at(self, now: float) -> float:
        return max(self.next_allowed_at, self.cooldown_until, now)

    def on_success(self) -> None:
        self.ok += 1
        self.consecutive_limits = 0
        self.rate_per_min = min(MAX_RATE_PER_MIN,
                                self.rate_per_min + INCREASE_PER_SUCCESS)

    def on_limited(self, now: float) -> None:
        self.limited += 1
        self.consecutive_limits += 1
        self.rate_per_min = max(MIN_RATE_PER_MIN,
                                self.rate_per_min * DECREASE_FACTOR)
        # Exponential backoff, so a genuinely dead lane stops being retried often.
        wait = min(COOLDOWN_MAX_SEC,
                   COOLDOWN_BASE_SEC * (2 ** (self.consecutive_limits - 1)))
        self.cooldown_until = now + wait

    def on_error(self, now: float) -> None:
        """Transport failure. Softer than a limit signal: brief pause, no rate cut."""
        self.errors += 1
        self.cooldown_until = now + COOLDOWN_BASE_SEC / 4


class ProxyBalancer:
    """Round-robin across lanes, paced to each lane's own measured rate.

    Usage:
        async with ProxyBalancer(urls) as pool:
            status, text = await pool.get(url, headers=...)
    """

    def __init__(self, proxies, per_lane_connections: int = 4,
                 timeout: float = 45.0, max_attempts: int = 3):
        self.lanes: list[Lane] = []
        for p in proxies:
            kind, url = p if isinstance(p, tuple) else ("proxy", p)
            self.lanes.append(Lane(url=url, kind=kind))
        if not self.lanes:
            raise ValueError("ProxyBalancer needs at least one proxy")
        self._per_lane_connections = per_lane_connections
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._lock = asyncio.Lock()
        # Stagger lane start times so the whole pool does not fire in lockstep.
        now = time.monotonic()
        for i, lane in enumerate(self.lanes):
            lane.next_allowed_at = now + (i / len(self.lanes)) * lane.interval

    async def __aenter__(self):
        limits = httpx.Limits(
            max_connections=self._per_lane_connections,
            max_keepalive_connections=self._per_lane_connections)
        for lane in self.lanes:
            lane.client = httpx.AsyncClient(proxy=lane.url, timeout=self._timeout,
                                            limits=limits)
        return self

    async def __aexit__(self, *exc):
        await asyncio.gather(*(lane.client.aclose() for lane in self.lanes
                               if lane.client), return_exceptions=True)

    async def _acquire(self) -> Lane:
        """Pick the lane that is free soonest, and wait until it is due.

        Waiting rather than firing immediately is the whole point: it converts
        would-be 429s into slightly later successes.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                # Earliest-available wins; ties broken randomly so identical
                # lanes do not always resolve in list order.
                best = min(self.lanes,
                           key=lambda l: (l.available_at(now), random.random()))
                due = best.available_at(now)
                if due <= now:
                    best.next_allowed_at = now + best.interval
                    return best
                delay = due - now
            await asyncio.sleep(min(delay, 1.0))

    async def request(self, method: str, url: str, **kw):
        """Send via a paced lane, retrying on a different lane when limited.

        Returns (status_or_exc_name, text, lane). status is an int on an HTTP
        response, or the exception class name when every attempt failed.
        """
        last = ("no_attempt", "", None)
        for _ in range(self._max_attempts):
            lane = await self._acquire()
            now = time.monotonic()
            try:
                resp = await lane.client.request(method, url, **kw)
            except Exception as exc:
                lane.on_error(now)
                last = (type(exc).__name__, "", lane)
                continue
            if resp.status_code in LIMIT_STATUSES:
                lane.on_limited(now)
                last = (resp.status_code, "", lane)
                continue
            lane.on_success()
            return resp.status_code, resp.text, lane
        return last

    async def get(self, url: str, **kw):
        return await self.request("GET", url, **kw)

    def stats(self) -> dict:
        ok = sum(l.ok for l in self.lanes)
        limited = sum(l.limited for l in self.lanes)
        errors = sum(l.errors for l in self.lanes)
        now = time.monotonic()
        return {
            "lanes": len(self.lanes),
            "ok": ok,
            "limited": limited,
            "errors": errors,
            "waste_pct": 100.0 * (limited + errors) / max(ok + limited + errors, 1),
            "parked": sum(1 for l in self.lanes if l.cooldown_until > now),
            "mean_rate_per_min": sum(l.rate_per_min for l in self.lanes)
            / len(self.lanes),
            "busiest_lane_share_pct": 100.0 * max((l.ok for l in self.lanes),
                                                  default=0) / max(ok, 1),
        }
