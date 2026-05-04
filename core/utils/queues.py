"""Aura Zenith: Backpressured Queue Utility."""
import asyncio
import ast
import logging
from typing import Any, Optional, Tuple

import threading
import queue

logger = logging.getLogger(__name__)

USER_FACING_ORIGINS = frozenset({
    "user",
    "voice",
    "admin",
    "external",
    "gui",
    "api",
    "websocket",
    "direct",
    "embodied_motor_reflex",
    "embodied_sensory_feed",
})


def unpack_priority_message(item: Any) -> Tuple[Any, Optional[str]]:
    """Unpack queue items across legacy and current tuple formats.

    Aura currently emits queue items in more than one shape:
    `(priority, timestamp, payload)`,
    `(priority, timestamp, seq, payload)`,
    `(priority, timestamp, seq, payload, origin)`,
    and sometimes raw payloads.

    Returning both payload and origin keeps background/internal stimuli from
    being misclassified as user turns when they move through older consumers.
    """
    payload = item
    origin: Optional[str] = None

    if isinstance(item, tuple):
        if len(item) >= 5:
            _, _, _, payload, origin = item[:5]
        elif len(item) == 4:
            _, _, _, payload = item
        elif len(item) == 3:
            _, _, payload = item

    if isinstance(payload, dict) and payload.get("origin"):
        origin = str(payload["origin"])

    return payload, origin


def decode_stringified_priority_message(text: Any) -> Tuple[Any, Optional[str], bool]:
    """Decode queue payloads that were accidentally persisted as tuple strings."""
    if not isinstance(text, str):
        return text, None, False

    stripped = text.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        return text, None, False

    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return text, None, False

    if not isinstance(parsed, tuple):
        return text, None, False

    payload, origin = unpack_priority_message(parsed)
    return payload, origin, True


def role_for_origin(origin: Optional[str]) -> str:
    """Map queue origins to working-memory roles."""
    return "user" if origin in USER_FACING_ORIGINS else "system"

class LoopAgnosticQueue:
    """A thread-safe queue bridge that can be used across different event loops.
    Uses a threading.Queue internally and provides async put/get via to_thread.
    
    This is the permanent fix for 'bound to a different event loop' errors
    when the Web API thread interacts with the Orchestrator thread.
    """
    def __init__(self, maxsize: int = 1000):
        self._q = queue.Queue(maxsize=maxsize)

    async def put(self, item, timeout: Optional[float] = None):
        """Put an item into the queue asynchronously."""
        try:
            return await asyncio.to_thread(self._q.put, item, timeout=timeout)
        except queue.Full:
            raise asyncio.QueueFull

    async def get(self, timeout: Optional[float] = None):
        """Get an item from the queue asynchronously."""
        try:
            return await asyncio.to_thread(self._q.get, timeout=timeout)
        except queue.Empty:
            raise asyncio.QueueEmpty

    def put_nowait(self, item):
        """Put an item into the queue without blocking."""
        try:
            self._q.put_nowait(item)
        except queue.Full:
            raise asyncio.QueueFull

    def get_nowait(self):
        """Get an item from the queue without blocking."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            raise asyncio.QueueEmpty

    def empty(self):
        """Check if the queue is empty."""
        return self._q.empty()

    def qsize(self):
        """Return the approximate size of the queue."""
        return self._q.qsize()

    def full(self):
        """Check if the queue is full."""
        return self._q.full()

    def task_done(self):
        """Unused bridge for task tracking."""
        self._q.task_done()

class BackpressuredQueue:
    """Async queue that tracks dropped items and supports backpressure circuit breaking.
    """
    def __init__(self, maxsize: int = 1000):
        self._q: Any = asyncio.Queue(maxsize=maxsize)
        self.dropped_count = 0
        self.maxsize = maxsize

    async def put(self, item, timeout: float = 5.0):
        """Put item with timeout. If queue is full after timeout, increment dropped count.
        """
        try:
            await asyncio.wait_for(self._q.put(item), timeout=timeout)
        except asyncio.TimeoutError:
            self.dropped_count += 1
            logger.warning(f"⚠️ Queue Backpressure: Dropped item. Total dropped: {self.dropped_count}")

    def put_nowait(self, item):
        """Standard put_nowait with drop tracking."""
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_count += 1
            logger.warning(f"⚠️ Queue Full (nowait): Dropped item {type(item).__name__}")

    async def get(self):
        return await self._q.get()

    def task_done(self):
        self._q.task_done()

    def qsize(self):
        return self._q.qsize()

    def full(self):
        return self._q.full()

    def empty(self):
        return self._q.empty()

    def get_nowait(self):
        """Standard get_nowait wrapper."""
        return self._q.get_nowait()


class PriorityBackpressuredQueue(BackpressuredQueue):
    """Priority-aware version of the backpressured queue.
    Items should be (priority, timestamp, payload) tuples.
    
    [HARDENING] Added a sequence counter to prevent payload comparisons 
    if priority and timestamp are identical.
    """
    def __init__(self, maxsize: int = 1000):
        import itertools
        self._q: Any = asyncio.PriorityQueue(maxsize=maxsize)
        self._count = itertools.count()
        self.dropped_count = 0
        self.maxsize = maxsize

    async def put(self, item, timeout: float = 5.0):
        """Standardize item to (priority, timestamp, sequence, payload)."""
        if isinstance(item, tuple) and len(item) == 3:
            p, t, payload = item
            standardized = (p, t, next(self._count), payload)
        else:
            standardized = item
        await super().put(standardized, timeout=timeout)

    def put_nowait(self, item):
        """Standardize item to (priority, timestamp, sequence, payload)."""
        if isinstance(item, tuple) and len(item) == 3:
            p, t, payload = item
            standardized = (p, t, next(self._count), payload)
        else:
            standardized = item
        super().put_nowait(standardized)

    async def get(self):
        """Unpack the standardized item."""
        item = await super().get()
        if isinstance(item, tuple) and len(item) == 4:
            return (item[0], item[1], item[3])
        return item

    def get_nowait(self):
        """Unpack the standardized item."""
        item = self._q.get_nowait()
        if isinstance(item, tuple) and len(item) == 4:
            return (item[0], item[1], item[3])
        return item
