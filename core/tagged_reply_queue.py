"""
Request-scoped reply delivery for Aura.

This replaces the old "shared reply queue + global flush" pattern with a queue
that preserves reply ownership across overlapping user, voice, and autonomous
flows. It remains compatible with the plain queue interface most of Aura uses.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import contextvars
import logging
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

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
        if session_id and self.session_id and self.session_id != session_id:
            return False
        return True


class TaggedReplyQueue:
    """Compatibility queue with tagged delivery semantics."""

    STALE_AFTER_SECONDS = 30.0

    def __init__(self, maxsize: int = 50):
        self.maxsize = maxsize
        self._queue = LoopAgnosticQueue(maxsize=maxsize)
        self._deferred: Deque[TaggedReply] = deque()
        self._lock = asyncio.Lock()
        logger.debug("TaggedReplyQueue initialized (maxsize=%d)", maxsize)

    def _coerce_reply(
        self,
        content: Any,
        origin: Optional[str] = None,
        session_id: Optional[str] = None,
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
        kept: Deque[TaggedReply] = deque()
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
        origin: Optional[str] = None,
        session_id: str = "",
    ) -> Optional[TaggedReply]:
        if not self._deferred:
            return None

        found: Optional[TaggedReply] = None
        kept: Deque[TaggedReply] = deque()
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

    async def put(
        self,
        content: Any,
        origin: Optional[str] = None,
        session_id: str = "",
    ):
        reply = self._coerce_reply(content, origin=origin, session_id=session_id)
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
            except Exception as exc:
                record_degradation('tagged_reply_queue', exc)
                logger.warning("Could not enqueue tagged reply: %s", exc)

    def put_nowait(
        self,
        content: Any,
        origin: Optional[str] = None,
        session_id: str = "",
    ):
        reply = self._coerce_reply(content, origin=origin, session_id=session_id)
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
            except Exception as exc:
                record_degradation('tagged_reply_queue', exc)
                logger.warning("Could not enqueue tagged reply without waiting: %s", exc)

    async def get(self) -> Any:
        async with self._lock:
            deferred = self._pop_deferred_match()
        if deferred is not None:
            return deferred.content
        item = await self._queue.get()
        return self._coerce_reply(item).content

    def get_nowait(self) -> Any:
        deferred = self._pop_deferred_match()
        if deferred is not None:
            return deferred.content
        item = self._queue.get_nowait()
        return self._coerce_reply(item).content

    async def get_for_origin(
        self,
        origin: str,
        session_id: str = "",
        timeout: float = 120.0,
    ) -> Optional[Any]:
        deadline = time.time() + timeout
        deferred_count = 0

        async with self._lock:
            self._prune_deferred()
            deferred = self._pop_deferred_match(origin, session_id)
        if deferred is not None:
            return deferred.content

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=min(remaining, 2.0),
                )
            except asyncio.TimeoutError:
                continue

            reply = self._coerce_reply(item)
            if self._is_stale(reply):
                continue

            if reply.is_for(origin, session_id):
                if deferred_count:
                    logger.debug(
                        "Found reply for %s/%s after deferring %d others",
                        origin,
                        session_id or "-",
                        deferred_count,
                    )
                return reply.content

            async with self._lock:
                self._deferred.append(reply)
                self._prune_deferred()
            deferred_count += 1

        logger.warning(
            "Timed out waiting for reply origin=%s session=%s after %.0fs",
            origin,
            session_id or "-",
            timeout,
        )
        return None

    async def flush_origin(self, origin: str) -> None:
        flushed = 0
        kept: list[TaggedReply] = []

        async with self._lock:
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
        async with self._lock:
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
        except Exception as _exc:
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
