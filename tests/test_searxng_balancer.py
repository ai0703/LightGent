"""Tests for the SearXNG shard balancer. No network: shard clients are stubbed."""
import asyncio
import time

import pytest

import searxng_balancer as sb
from searxng_balancer import Shard, SearxngBalancer, _engine_key


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeClient:
    """Replays a scripted list of payloads / status codes / exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def get(self, url, params=None):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int):
            return FakeResponse({}, status_code=item)
        return FakeResponse(item)

    async def aclose(self):
        pass


def payload(engines=("google",), n=10, unresponsive=()):
    return {
        "results": [{"engines": list(engines)} for _ in range(n)],
        "unresponsive_engines": list(unresponsive),
    }


def attach(bal, scripts, ordered=False):
    """Stub each shard's client.

    ordered=True forces shard 0 to be tried first, which failover tests need.
    _acquire breaks ties with random.random(), and available_at() clamps to now,
    so staggering next_allowed_at into the past does nothing. Pushing the LATER
    shards a few ms into the future is what actually orders them.
    """
    now = time.monotonic()
    for i, (shard, script) in enumerate(zip(bal.shards, scripts)):
        shard.client = FakeClient(script)
        shard.next_allowed_at = 0.0
        shard.cooldown_until = (now + 0.02 * i) if ordered else 0.0
        shard.rate_per_min = sb.MAX_RATE_PER_MIN * 100  # never pace in tests


def test_engine_key_normalises_searxng_names():
    assert _engine_key("google cse") == "google"
    assert _engine_key(["google cse", "too many requests"]) == "google"
    assert _engine_key("duckduckgo") == "duckduckgo"
    assert _engine_key(("bing", "")) == "bing"


def test_unresponsive_marks_engine_down_on_that_shard_only():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        attach(bal, [[payload(unresponsive=[["google cse", "too many requests"]])],
                     [payload()]])
        now = time.monotonic()
        bal.shards[0].note_unresponsive([["google cse", "x"]], now)
        assert not bal.shards[0].engine_healthy("google", now)
        assert bal.shards[1].engine_healthy("google", now)
    asyncio.run(go())


def test_routes_away_from_shard_whose_engine_is_suspended():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        attach(bal, [[payload()], [payload()]])
        # Bench google on shard a.
        bal.shards[0].note_unresponsive([["google cse", "too many requests"]],
                                        time.monotonic())
        for _ in range(6):
            await bal.search("q", prefer_engine="google")
        assert bal.shards[0].ok == 0, "queries should avoid the suspended shard"
        assert bal.shards[1].ok == 6
    asyncio.run(go())


def test_suspended_shard_still_used_when_engine_not_required():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        attach(bal, [[payload(engines=("bing",))], [payload(engines=("bing",))]])
        bal.shards[0].note_unresponsive([["google cse", "x"]], time.monotonic())
        for _ in range(8):
            await bal.search("q", prefer_engine=None)
        # With no engine preference, both shards should carry traffic.
        assert bal.shards[0].ok > 0 and bal.shards[1].ok > 0
    asyncio.run(go())


def test_falls_back_to_any_shard_when_all_are_suspended():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        attach(bal, [[payload()], [payload()]])
        now = time.monotonic()
        for s in bal.shards:
            s.note_unresponsive([["google cse", "x"]], now)
        out, shard = await bal.search("q", prefer_engine="google")
        assert out is not None, "must still answer rather than stall"
        assert shard is not None
    asyncio.run(go())


def test_thin_answer_missing_engine_retries_elsewhere():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        # a: thin AND no google -> retry; b: full with google -> accept
        attach(bal, [[payload(engines=("bing",), n=2)],
                     [payload(engines=("google",), n=10)]])
        out, shard = await bal.search("q", prefer_engine="google")
        assert shard.url == "http://b"
        assert len(out["results"]) == 10
    asyncio.run(go())


def test_full_answer_without_preferred_engine_is_accepted():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        # Plenty of results, just no google. Should NOT waste a retry.
        attach(bal, [[payload(engines=("bing",), n=20)],
                     [payload(engines=("google",), n=20)]])
        out, _ = await bal.search("q", prefer_engine="google")
        assert len(out["results"]) == 20
        assert sum(s.client.calls for s in bal.shards) == 1
    asyncio.run(go())


def test_http_error_fails_over_to_next_shard():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        attach(bal, [[502], [payload()]], ordered=True)
        out, shard = await bal.search("q")
        assert shard.url == "http://b"
        assert bal.shards[0].failed == 1
    asyncio.run(go())


def test_exception_fails_over_and_backs_off():
    async def go():
        bal = SearxngBalancer(["http://a", "http://b"])
        attach(bal, [[RuntimeError("boom")], [payload()]], ordered=True)
        out, shard = await bal.search("q")
        assert shard.url == "http://b"
        a = bal.shards[0]
        assert a.failed == 1 and a.cooldown_until > time.monotonic()
    asyncio.run(go())


def test_failure_halves_rate_with_a_floor():
    shard = Shard(url="http://a", rate_per_min=100.0)
    shard.on_failure(time.monotonic())
    assert shard.rate_per_min == pytest.approx(50.0)
    for _ in range(20):
        shard.on_failure(time.monotonic())
    assert shard.rate_per_min >= sb.MIN_RATE_PER_MIN


def test_success_raises_rate_to_a_ceiling():
    shard = Shard(url="http://a", rate_per_min=sb.MAX_RATE_PER_MIN)
    for _ in range(5):
        shard.on_success(thin=False)
    assert shard.rate_per_min <= sb.MAX_RATE_PER_MIN


def test_empty_url_list_rejected():
    with pytest.raises(ValueError):
        SearxngBalancer([])


def test_stats_counts_thin_answers_and_suspensions():
    async def go():
        bal = SearxngBalancer(["http://a"])
        attach(bal, [[payload(engines=("bing",), n=1,
                              unresponsive=[["google cse", "too many requests"]])]])
        await bal.search("q", prefer_engine="google")
        s = bal.stats()
        assert s["shards"] == 1
        assert s["ok"] == 1
        assert s["thin"] == 1
        assert s["engine_suspensions"].get("google") == 1
        assert "google" in next(iter(s["currently_down"].values()))
    asyncio.run(go())
