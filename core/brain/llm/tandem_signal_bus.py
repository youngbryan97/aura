"""
core/brain/llm/tandem_signal_bus.py — asyncio pubsub for tandem oracle signals.
Slow lane publishes; fast lane drains. Multi-subscriber. Priority-ordered.
Order:  retract > handoff > correction > refine > continue
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
from typing import Any, AsyncIterator, List, Optional, Tuple

_PRIORITY = {"retract": 0, "handoff": 1, "correction": 2, "refine": 3, "continue": 4}


def signal_priority(kind: str) -> int:
    return _PRIORITY.get(kind, 99)


class _Subscriber:
    def __init__(self, capacity: int):
        self._heap: List[Tuple[int, int, Any]] = []
        self._capacity = capacity
        self._event = asyncio.Event()
        self._closed = False

    def push(self, prio: int, seq: int, signal: Any) -> None:
        if self._closed: return  # noqa: E701
        if len(self._heap) >= self._capacity:
            self._heap.sort(); self._heap.pop()
        heapq.heappush(self._heap, (prio, seq, signal))
        self._event.set()

    def pop_nowait(self) -> Optional[Any]:
        if not self._heap:
            self._event.clear(); return None
        _, _, sig = heapq.heappop(self._heap)
        if not self._heap: self._event.clear()  # noqa: E701
        return sig

    async def wait(self, timeout: Optional[float] = None) -> bool:
        if self._heap: return True  # noqa: E701
        if self._closed: return False  # noqa: E701
        try:
            if timeout is None: await self._event.wait()  # noqa: E701
            else: await asyncio.wait_for(self._event.wait(), timeout=timeout)  # noqa: E701
        except asyncio.TimeoutError:
            return False
        return bool(self._heap) or self._closed

    def close(self) -> None:
        self._closed = True
        self._event.set()


class TandemSignalBus:
    def __init__(self, *, subscriber_capacity: int = 64):
        self._subs: List[_Subscriber] = []
        self._cap = subscriber_capacity
        self._lock = asyncio.Lock()
        self._closed = False
        self._counter = itertools.count()

    async def publish(self, signal: Any) -> None:
        if self._closed: return  # noqa: E701
        prio = signal_priority(getattr(signal, "kind", "continue"))
        async with self._lock:
            subs = list(self._subs)
        seq = next(self._counter)
        for s in subs: s.push(prio, seq, signal)  # noqa: E701

    def subscribe(self) -> "TandemSubscription":
        sub = _Subscriber(self._cap)
        self._subs.append(sub)
        return TandemSubscription(self, sub)

    async def _unsubscribe(self, sub: _Subscriber) -> None:
        async with self._lock:
            try: self._subs.remove(sub)
            except ValueError: pass  # noqa: E701
        sub.close()

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for s in subs: s.close()  # noqa: E701


class TandemSubscription:
    """Handle for the fast lane: poll, drain, wait, async-iterate."""
    def __init__(self, bus: TandemSignalBus, sub: _Subscriber):
        self._bus = bus
        self._sub = sub

    def poll(self) -> Optional[Any]:
        return self._sub.pop_nowait()

    def drain(self) -> List[Any]:
        out: List[Any] = []
        while True:
            s = self._sub.pop_nowait()
            if s is None: break  # noqa: E701
            out.append(s)
        return out

    async def wait_for(self, timeout: Optional[float] = None) -> Optional[Any]:
        if not await self._sub.wait(timeout=timeout): return None  # noqa: E701
        return self._sub.pop_nowait()

    async def __aiter__(self) -> AsyncIterator[Any]:
        while True:
            ready = await self._sub.wait()
            if not ready and self._sub._closed: return  # noqa: E701
            sig = self._sub.pop_nowait()
            if sig is not None: yield sig  # noqa: E701

    async def aclose(self) -> None:
        await self._bus._unsubscribe(self._sub)
