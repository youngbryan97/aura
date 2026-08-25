"""
Request-scoped reply delivery for Aura.

This replaces the old "shared reply queue + global flush" pattern with a queue
that preserves reply ownership across overlapping user, voice, and autonomous
flows. It remains compatible with the plain queue interface most of Aura uses.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.utils.queues import LoopAgnosticQueue

logger = logging.getLogger("Aura.TaggedReplyQueue")

_reply_origin_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_reply_origin",
    default="",
)
_reply_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_reply_session_id",
    default="",
)


def current_reply_origin(default: str = "system") -> str:
    """Return the current request-scoped reply origin."""
    return str(_reply_origin_var.get() or default)


def current_reply_session_id(default: str = "") -> str:
    """Return the current request-scoped reply session id."""
    return str(_reply_session_id_var.get() or default)


@contextmanager
def reply_delivery_scope(origin: str, session_id: str = ""):
    """Scope reply emissions to a specific logical request.

    ContextVars propagate across awaits and newly created asyncio tasks, which
    lets late output still find the correct waiting caller.
    """
    resolved_origin = str(origin or "system")
    resolved_session_id = str(session_id or uuid.uuid4())[:12]
    origin_token = _reply_origin_var.set(resolved_origin)
    session_token = _reply_session_id_var.set(resolved_session_id)
    try:
        yield resolved_session_id
    finally:
        _reply_origin_var.reset(origin_token)
        _reply_session_id_var.reset(session_token)


@dataclass
class TaggedReply:
    content: Any
    origin: str
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def is_for(self, origin: str, session_id: str = "") -> bool:
        if self.origin != origin:
            return False
        if session_id:
            return self.session_id == session_id
        return True


class TaggedReplyQueue:
    """Compatibility queue with tagged delivery semantics."""

    STALE_AFTER_SECONDS = 30.0

    def __init__(self, maxsize: int = 50):
        self.maxsize = maxsize
        self._queue = LoopAgnosticQueue(maxsize=maxsize)
        self._deferred: deque[TaggedReply] = deque()
        self._lock = checked_lock(
            f"tagged_reply_queue.{id(self):x}",
            rank=LockRank.RESOURCE,
            reentrant=True,
        )
        self._waiters: dict[
            tuple[str, str],
            deque[asyncio.Future[TaggedReply]],
        ] = defaultdict(deque)
        logger.debug("TaggedReplyQueue initialized (maxsize=%d)", maxsize)

    def _coerce_reply(
        self,
        content: Any,
        origin: str | None = None,
        session_id: str | None = None,
    ) -> TaggedReply:
        if isinstance(content, TaggedReply):
            return content
        return TaggedReply(
            content=content,
            origin=str(origin or current_reply_origin("system") or "system"),
            session_id=str(
                session_id if session_id is not None else current_reply_session_id("")
            ),
        )

    def _is_stale(self, reply: TaggedReply) -> bool:
        return (time.time() - reply.timestamp) > self.STALE_AFTER_SECONDS

    def _prune_deferred(self) -> int:
        kept: deque[TaggedReply] = deque()
        pruned = 0
        while self._deferred:
            reply = self._deferred.popleft()
            if self._is_stale(reply):
                pruned += 1
                continue
            kept.append(reply)
        self._deferred = kept
        return pruned

    def _pop_deferred_match(
        self,
        origin: str | None = None,
        session_id: str = "",
    ) -> TaggedReply | None:
        if not self._deferred:
            return None

        found: TaggedReply | None = None
        kept: deque[TaggedReply] = deque()
        while self._deferred:
            reply = self._deferred.popleft()
            if self._is_stale(reply):
                continue
            if found is None and (origin is None or reply.is_for(origin, session_id)):
                found = reply
                continue
            kept.append(reply)
        self._deferred = kept
        return found

    def _pop_waiter(self, reply: TaggedReply) -> asyncio.Future[TaggedReply] | None:
        keys = [(reply.origin, reply.session_id)] if reply.session_id else []
        keys.append((reply.origin, ""))
        for key in keys:
            waiters = self._waiters.get(key)
            if waiters is None:
                continue
            while waiters:
                waiter = waiters.popleft()
                if not waiter.done():
                    if not waiters:
                        self._waiters.pop(key, None)
                    return waiter
            self._waiters.pop(key, None)
        return None

    @staticmethod
    def _deliver_waiter(
        waiter: asyncio.Future[TaggedReply],
        reply: TaggedReply,
    ) -> None:
        def deliver() -> None:
            if not waiter.done():
                waiter.set_result(reply)

        waiter.get_loop().call_soon_threadsafe(deliver)

    def _dispatch_waiter(self, reply: TaggedReply) -> bool:
        with self._lock:
            waiter = self._pop_waiter(reply)
        if waiter is None:
            return False
        self._deliver_waiter(waiter, reply)
        return True

    def _drain_queued_match(self, origin: str, session_id: str) -> TaggedReply | None:
        deferred = self._pop_deferred_match(origin, session_id)
        if deferred is not None:
            return deferred
        found: TaggedReply | None = None
        while True:
            try:
                reply = self._coerce_reply(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            if self._is_stale(reply):
                continue
            if found is None and reply.is_for(origin, session_id):
                found = reply
            else:
                self._deferred.append(reply)
        self._prune_deferred()
        return found

    def _remove_waiter(
        self,
        key: tuple[str, str],
        waiter: asyncio.Future[TaggedReply],
    ) -> None:
        with self._lock:
            waiters = self._waiters.get(key)
            if waiters is None:
                return
            try:
                waiters.remove(waiter)
            except ValueError:
                return
            if not waiters:
                self._waiters.pop(key, None)

    async def put(
        self,
        content: Any,
        origin: str | None = None,
        session_id: str | None = None,
    ):
        reply = self._coerce_reply(content, origin=origin, session_id=session_id)
        if self._dispatch_waiter(reply):
            return
        try:
            await self._queue.put(reply)
        except asyncio.QueueFull:
            try:
                dropped = self._coerce_reply(self._queue.get_nowait())
                logger.warning(
                    "Reply queue full; dropped oldest reply (origin=%s, session=%s)",
                    dropped.origin,
                    dropped.session_id or "-",
                )
                self._queue.put_nowait(reply)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('tagged_reply_queue', exc)
                logger.warning("Could not enqueue tagged reply: %s", exc)

    def put_nowait(
        self,
        content: Any,
        origin: str | None = None,
        session_id: str | None = None,
    ):
        reply = self._coerce_reply(content, origin=origin, session_id=session_id)
        if self._dispatch_waiter(reply):
            return
        try:
            self._queue.put_nowait(reply)
        except asyncio.QueueFull:
            try:
                dropped = self._coerce_reply(self._queue.get_nowait())
                logger.warning(
                    "Reply queue full (nowait); dropped oldest reply (origin=%s, session=%s)",
                    dropped.origin,
                    dropped.session_id or "-",
                )
                self._queue.put_nowait(reply)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('tagged_reply_queue', exc)
                logger.warning("Could not enqueue tagged reply without waiting: %s", exc)

    async def get(self) -> Any:
        with self._lock:
            deferred = self._pop_deferred_match()
        if deferred is not None:
            return deferred.content
        item = await self._queue.get()
        return self._coerce_reply(item).content

    def get_nowait(self) -> Any:
        with self._lock:
            deferred = self._pop_deferred_match()
        if deferred is not None:
            return deferred.content
        item = self._queue.get_nowait()
        return self._coerce_reply(item).content

    async def get_for_origin(  # noqa: ASYNC109 - public compatibility timeout
        self,
        origin: str,
        session_id: str = "",
        timeout: float = 120.0,  # noqa: ASYNC109 - public compatibility name
    ) -> Any | None:
        if timeout <= 0:
            return None
        key = (str(origin), str(session_id or ""))
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[TaggedReply] = loop.create_future()
        with self._lock:
            self._prune_deferred()
            queued = self._drain_queued_match(*key)
            if queued is None:
                self._waiters[key].append(waiter)
        if queued is not None:
            return queued.content
        try:
            reply = await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout)
            return reply.content
        except TimeoutError:
            logger.warning(
                "Timed out waiting for reply origin=%s session=%s after %.0fs",
                origin,
                session_id or "-",
                timeout,
            )
            return None
        finally:
            self._remove_waiter(key, waiter)

    async def flush_origin(self, origin: str) -> None:
        flushed = 0
        kept: list[TaggedReply] = []

        with self._lock:
            while self._deferred:
                reply = self._deferred.popleft()
                if reply.origin == origin:
                    flushed += 1
                else:
                    kept.append(reply)
            self._deferred.extend(kept)

        retained: list[TaggedReply] = []
        while not self._queue.empty():
            try:
                reply = self._coerce_reply(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            if reply.origin == origin:
                flushed += 1
            else:
                retained.append(reply)

        for reply in retained:
            try:
                self._queue.put_nowait(reply)
            except asyncio.QueueFull:
                break

        if flushed:
            logger.debug("Flushed %d replies for origin=%s", flushed, origin)

    async def flush_all(self) -> None:
        count = 0
        with self._lock:
            count += len(self._deferred)
            self._deferred.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        if count:
            logger.debug("Flushed %d replies", count)

    def task_done(self):
        try:
            self._queue.task_done()
        except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
            record_degradation('tagged_reply_queue', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    def qsize(self) -> int:
        self._prune_deferred()
        return len(self._deferred) + self._queue.qsize()

    def size(self) -> int:
        return self.qsize()

    def empty(self) -> bool:
        self._prune_deferred()
        return not self._deferred and self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()
