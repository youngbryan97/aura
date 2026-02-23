"""
core/tagged_reply_queue.py
───────────────────────────
Replaces the bare asyncio.Queue used for replies in the orchestrator.

The problem it solves:
  The original orchestrator uses a single reply_queue for both user responses
  and autonomous thought responses. The race condition:

  1. User sends message
  2. Orchestrator cancels autonomous thought
  3. Autonomous thought had already queued its response
  4. Orchestrator clears the queue before processing
  5. Autonomous thought's finally block queues AGAIN after the clear
  6. User's response arrives but is consumed by code waiting for the wrong type
  7. User waits 240 seconds until timeout

This queue tags every entry with an origin and session_id.
get_for_origin() only returns responses matching the requested origin,
discarding anything else. No more cross-contamination.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("Aura.TaggedReplyQueue")


@dataclass
class TaggedReply:
    content: str
    origin: str                    # "user", "voice", "autonomous", "admin"
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
    """
    A reply queue where entries are tagged by origin.
    Consumers can wait for replies from a specific origin,
    and stale replies from other origins are automatically discarded.
    """

    MAX_SIZE = 50
    STALE_AFTER_SECONDS = 30.0

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_SIZE)
        self._lock = asyncio.Lock()
        logger.debug("TaggedReplyQueue initialized")

    async def put(self, content: str, origin: str, session_id: str = ""):
        """Put a tagged reply into the queue."""
        reply = TaggedReply(content=content, origin=origin, session_id=session_id)
        try:
            self._queue.put_nowait(reply)
            logger.debug("Queued reply (origin=%s, id=%s)", origin, reply.request_id)
        except asyncio.QueueFull:
            # Drop the oldest entry and retry
            try:
                dropped = self._queue.get_nowait()
                logger.debug("Queue full — dropped reply (origin=%s)", dropped.origin)
                self._queue.put_nowait(reply)
            except Exception as exc:
                logger.warning("Could not queue reply: %s", exc)

    def put_nowait(self, content: str, origin: str, session_id: str = ""):
        """Synchronous put — for use in non-async contexts."""
        reply = TaggedReply(content=content, origin=origin, session_id=session_id)
        try:
            self._queue.put_nowait(reply)
        except asyncio.QueueFull:
            pass

    async def get_for_origin(
        self,
        origin: str,
        session_id: str = "",
        timeout: float = 120.0,
    ) -> Optional[str]:
        """
        Wait for and return the next reply for the given origin.
        Discards replies for other origins (putting them back is not
        possible with asyncio.Queue, so we discard and wait for the
        right one to arrive).

        Returns None on timeout.
        """
        deadline = time.time() + timeout
        discarded_count = 0

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                reply = await asyncio.wait_for(
                    self._queue.get(), timeout=min(remaining, 2.0)
                )
            except asyncio.TimeoutError:
                continue

            # Check staleness
            age = time.time() - reply.timestamp
            if age > self.STALE_AFTER_SECONDS:
                logger.debug(
                    "Discarding stale reply (age=%.1fs, origin=%s)",
                    age, reply.origin
                )
                discarded_count += 1
                continue

            # Check if this reply is for us
            if reply.is_for(origin, session_id):
                if discarded_count:
                    logger.debug(
                        "Found reply for %s after discarding %d others",
                        origin, discarded_count
                    )
                return reply.content
            else:
                # Not for us — discard it
                # NOTE: In a more complex system you'd re-queue it.
                # But since autonomous thoughts getting discarded is
                # EXACTLY WHAT WE WANT when the user is waiting, this is correct.
                logger.debug(
                    "Discarding reply for wrong origin (got=%s, want=%s)",
                    reply.origin, origin
                )
                discarded_count += 1

        logger.warning(
            "get_for_origin(%s) timed out after %.0fs (discarded=%d)",
            origin, timeout, discarded_count
        )
        return None

    async def flush_origin(self, origin: str):
        """Discard all queued replies for a specific origin."""
        flushed = 0
        remaining = []

        # Drain everything
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item.origin != origin:
                    remaining.append(item)
                else:
                    flushed += 1
            except asyncio.QueueEmpty:
                break

        # Re-queue non-matching items
        for item in remaining:
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                break

        if flushed:
            logger.debug("Flushed %d replies for origin=%s", flushed, origin)

    async def flush_all(self):
        """Discard everything in the queue."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        if count:
            logger.debug("Flushed %d replies from queue", count)

    def size(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
