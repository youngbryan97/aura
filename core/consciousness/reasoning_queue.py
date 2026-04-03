"""core/consciousness/reasoning_queue.py
Zenith Architecture — Async Reasoning Queue.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("Consciousness.ReasoningQueue")

class ReasoningQueue:
    _instance: Optional['ReasoningQueue'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReasoningQueue, cls).__new__(cls)
            cls._instance._queue = None
            logger.info("ReasoningQueue singleton initialized (lazy)")
        return cls._instance

    @property
    def queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    async def put(self, item: Any):
        await self.queue.put(item)

    async def get(self) -> Any:
        return await self.queue.get()

    def qsize(self) -> int:
        return self.queue.qsize()

def get_reasoning_queue() -> ReasoningQueue:
    return ReasoningQueue()
