"""Lane-aware request queue for LightGent.

Monitors the Balance Broker status and holds requests until a lane is available.
No request ever fails due to lane exhaustion -- it waits however long it takes.

Usage:
    from lane_queue import LaneQueue, get_queue
    queue = get_queue()
    result = await queue.enqueue(request_data)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("lane_queue")


@dataclass
class QueuedRequest:
    """A request waiting for an available lane."""
    request_data: dict
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    enqueued_at: float = field(default_factory=time.time)
    attempts: int = 0


class LaneQueue:
    """Holds requests and dispatches them when a lane becomes available.

    Polls the broker /status endpoint to detect when lanes recover.
    If all lanes are parked, requests wait indefinitely (even days).
    """

    def __init__(self, broker_url: str, poll_interval: int = 15, max_concurrent: int = 2,
                 api_key: str = ""):
        self.broker_url = broker_url.rstrip("/")
        self.poll_interval = poll_interval  # seconds between status checks
        self.max_concurrent = max_concurrent
        # Bearer token for the (now authenticated) broker /status endpoint.
        self._auth_headers = ({"Authorization": f"Bearer {api_key}"}
                              if api_key and api_key.lower() != "none" else {})
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()
        self._active = 0  # currently in-flight requests
        self._lanes_available = False
        self._monitor_task: asyncio.Task | None = None
        self._processor_task: asyncio.Task | None = None

    async def start(self):
        """Start the background monitor and processor."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._processor_task = asyncio.create_task(self._process_loop())
        log.info("lane queue started (poll=%ds, max_concurrent=%d)",
                 self.poll_interval, self.max_concurrent)

    async def stop(self):
        """Stop background tasks."""
        for task in (self._monitor_task, self._processor_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        log.info("lane queue stopped")

    async def enqueue(self, request_data: dict) -> dict:
        """Add a request to the queue and wait for the result.

        Returns the agent response dict. Waits indefinitely if all lanes are parked.
        """
        req = QueuedRequest(request_data=request_data)
        await self._queue.put(req)
        log.info("request enqueued (queue depth=%d)", self._queue.qsize())
        return await req.future

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        return self._active

    async def _check_lanes(self) -> tuple[int, int]:
        """Poll broker status. Returns (active_count, parked_count)."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.broker_url}/status", headers=self._auth_headers)
                r.raise_for_status()
                lanes = r.json().get("lanes", [])
                active = sum(1 for l in lanes if l.get("status") == "active")
                parked = sum(1 for l in lanes if l.get("status") == "parked")
                log.info("broker check: %d active, %d parked (url=%s)", active, parked, self.broker_url)
                return active, parked
        except Exception as e:
            log.warning("broker status check failed: %s (url=%s)", e, self.broker_url)
            return 0, 0

    async def _monitor_loop(self):
        """Poll broker status and update lane availability."""
        while True:
            try:
                active, parked = await self._check_lanes()
                was_available = self._lanes_available
                self._lanes_available = active > 0

                if self._lanes_available and not was_available:
                    log.info("lanes recovered: %d active, %d parked -- resuming queue", active, parked)
                elif not self._lanes_available and was_available:
                    log.warning("all lanes parked (%d) -- queue holding requests", parked)
                elif not self._lanes_available:
                    log.debug("still waiting: 0 active, %d parked", parked)

            except Exception as e:
                log.error("monitor error: %s", e)

            await asyncio.sleep(self.poll_interval)

    async def _process_loop(self):
        """Dispatch queued requests when lanes are available and under concurrency limit."""
        while True:
            try:
                # Wait until lanes are available
                if not self._lanes_available:
                    await asyncio.sleep(2)
                    continue

                # Wait until we have capacity
                if self._active >= self.max_concurrent:
                    await asyncio.sleep(1)
                    continue

                # Get next request (wait if queue is empty)
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=5)
                except asyncio.TimeoutError:
                    continue

                # Dispatch in background
                self._active += 1
                asyncio.create_task(self._dispatch(req))

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("processor error: %s", e)
                await asyncio.sleep(5)

    async def _dispatch(self, req: QueuedRequest):
        """Send a request to LightGent and resolve the future."""
        try:
            req.attempts += 1
            waited = time.time() - req.enqueued_at
            log.info("dispatching request (waited %.0fs, attempt %d)", waited, req.attempts)

            # Import here to avoid circular imports
            from lightgent_service import run_agent, ResearchRequest, settings
            import httpx as _httpx

            # Create the request object
            research_req = ResearchRequest(**req.request_data)

            # Run the agent
            async with _httpx.AsyncClient(timeout=settings.tool_timeout) as http:
                result = await run_agent(research_req, http)

            req.future.set_result(result.model_dump())

        except Exception as e:
            log.exception("dispatch failed")
            # On failure, re-queue with exponential backoff (max 5 retries)
            if req.attempts < 5:
                backoff = min(300, 30 * (2 ** (req.attempts - 1)))
                log.warning("re-queuing request in %ds (attempt %d)", backoff, req.attempts)
                await asyncio.sleep(backoff)
                await self._queue.put(req)
                return
            req.future.set_exception(e)
        finally:
            self._active -= 1


# Singleton
_queue: LaneQueue | None = None


def get_queue(broker_url: str | None = None, **kwargs) -> LaneQueue:
    """Get or create the global lane queue.

    Defaults to the LOCAL broker. The queue is normally created in the service
    lifespan with the configured broker_url + api_key, so this fallback only
    matters if get_queue() is called before startup wiring."""
    global _queue
    if _queue is None:
        url = broker_url or "http://127.0.0.1:8902"
        _queue = LaneQueue(url, **kwargs)
    return _queue
