"""Tests for the proxy pool balancer. No network: lane clients are stubbed."""
import asyncio
import time

import pytest

import proxy_balancer as pb
from proxy_balancer import Lane, ProxyBalancer


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class FakeClient:
    """Returns a scripted sequence of statuses, or raises when given an Exception."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def request(self, method, url, **kw):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    async def aclose(self):
        pass


def attach(balancer, scripts, ordered=False):
    """Stub each lane's client.

    ordered=True forces lane 0 to be selected first, which the failover tests
    need. _acquire breaks ties with random.random(), and available_at() clamps to
    now, so the only way to order lanes is to push the later ones slightly into
    the future.
    """
    now = time.monotonic()
    for i, (lane, script) in enumerate(zip(balancer.lanes, scripts)):
        lane.client = FakeClient(script)
        # Make every lane immediately due so tests do not wait on pacing.
        lane.next_allowed_at = 0.0
        lane.cooldown_until = (now + 0.02 * i) if ordered else 0.0


def test_success_raises_rate_additively():
    lane = Lane(url="http://x", rate_per_min=20.0)
    lane.on_success()
    assert lane.rate_per_min == pytest.approx(20.0 + pb.INCREASE_PER_SUCCESS)
    assert lane.consecutive_limits == 0


def test_limit_halves_rate_and_parks_lane():
    lane = Lane(url="http://x", rate_per_min=20.0)
    now = time.monotonic()
    lane.on_limited(now)
    assert lane.rate_per_min == pytest.approx(10.0)
    assert lane.cooldown_until >= now + pb.COOLDOWN_BASE_SEC


def test_cooldown_backs_off_exponentially():
    lane = Lane(url="http://x")
    now = time.monotonic()
    lane.on_limited(now)
    first = lane.cooldown_until - now
    lane.on_limited(now)
    second = lane.cooldown_until - now
    assert second > first


def test_rate_floor_and_ceiling_hold():
    lane = Lane(url="http://x", rate_per_min=pb.MIN_RATE_PER_MIN)
    for _ in range(10):
        lane.on_limited(time.monotonic())
    assert lane.rate_per_min >= pb.MIN_RATE_PER_MIN
    lane.rate_per_min = pb.MAX_RATE_PER_MIN
    for _ in range(10):
        lane.on_success()
    assert lane.rate_per_min <= pb.MAX_RATE_PER_MIN


def test_success_resets_consecutive_limits():
    lane = Lane(url="http://x")
    lane.on_limited(time.monotonic())
    assert lane.consecutive_limits == 1
    lane.on_success()
    assert lane.consecutive_limits == 0


def test_error_pauses_briefly_without_cutting_rate():
    lane = Lane(url="http://x", rate_per_min=20.0)
    now = time.monotonic()
    lane.on_error(now)
    assert lane.rate_per_min == pytest.approx(20.0)
    assert lane.cooldown_until > now
    assert lane.errors == 1


def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        ProxyBalancer([])


def test_limited_lane_is_retried_on_a_different_lane():
    async def go():
        bal = ProxyBalancer(["http://a", "http://b"], max_attempts=3)
        # Lane a always 429s; lane b always succeeds.
        attach(bal, [[429], [200]], ordered=True)
        status, _, lane = await bal.request("GET", "http://target")
        assert status == 200
        assert lane.url == "http://b"
        a = next(l for l in bal.lanes if l.url == "http://a")
        assert a.limited >= 1
    asyncio.run(go())


def test_all_lanes_limited_returns_the_limit_status():
    async def go():
        bal = ProxyBalancer(["http://a", "http://b"], max_attempts=2)
        attach(bal, [[429], [429]])
        status, _, _ = await bal.request("GET", "http://target")
        assert status == 429
    asyncio.run(go())


def test_transport_exception_is_reported_by_name():
    async def go():
        bal = ProxyBalancer(["http://a"], max_attempts=1)
        attach(bal, [[RuntimeError("boom")]])
        status, _, _ = await bal.request("GET", "http://target")
        assert status == "RuntimeError"
        assert bal.lanes[0].errors == 1
    asyncio.run(go())


def test_load_spreads_across_lanes_rather_than_hammering_one():
    async def go():
        n = 8
        bal = ProxyBalancer([f"http://p{i}" for i in range(n)])
        attach(bal, [[200]] * n)
        # Give every lane a huge allowance so pacing never blocks the test.
        for lane in bal.lanes:
            lane.rate_per_min = pb.MAX_RATE_PER_MIN * 100
        for _ in range(n * 4):
            await bal.request("GET", "http://target")
        used = [l.ok for l in bal.lanes]
        assert all(u > 0 for u in used), f"some lane never used: {used}"
        # No lane should carry more than double an even share.
        assert max(used) <= (sum(used) / n) * 2
    asyncio.run(go())


def test_stats_reports_waste_and_parked_counts():
    async def go():
        bal = ProxyBalancer(["http://a", "http://b"], max_attempts=2)
        attach(bal, [[429], [200]], ordered=True)
        await bal.request("GET", "http://target")
        s = bal.stats()
        assert s["lanes"] == 2
        assert s["ok"] == 1
        assert s["limited"] == 1
        assert 0 < s["waste_pct"] < 100
        assert s["parked"] >= 1
    asyncio.run(go())
