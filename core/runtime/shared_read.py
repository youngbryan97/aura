"""Share bounded read work without making a caller's timeout cancel it."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
import logging
import time
from typing import Any
from weakref import WeakKeyDictionary
from core.runtime.task_ownership import create_owned_asyncio_task

logger = logging.getLogger(__name__)


@dataclass
class _Read:
    task: asyncio.Task | None = None
    value: Any = None
    completed_at: float = 0.0


class SharedRead:
    """One loader per scope; completed values survive the next polling interval.

    The caller owns authorization and includes its full scope in the key.
    Only completed entries can be evicted. A timed-out or cancelled waiter
    cannot create replacement work while the original loader is running.
    """

    def __init__(self, *, capacity: int = 16, retention_s: float = 30.0):
        self._capacity = max(1, capacity)
        self._retention_s = max(0.0, retention_s)
        self._loops: WeakKeyDictionary = WeakKeyDictionary()

    async def read(
        self,
        key: Hashable,
        loader: Callable[[], Awaitable[Any]],
        *,
        wait_s: float,
    ) -> Any:
        loop = asyncio.get_running_loop()
        entries: OrderedDict = self._loops.setdefault(loop, OrderedDict())
        now = time.monotonic()
        for old_key, entry in list(entries.items()):
            if entry.task is None and now - entry.completed_at >= self._retention_s:
                entries.pop(old_key)
        entry = entries.get(key)
        if entry is None:
            if len(entries) >= self._capacity:
                completed = [k for k, v in entries.items() if v.task is None]
                if not completed:
                    raise RuntimeError("shared_read_capacity_busy")
                entries.pop(completed[0])
            entry = _Read()
            entries[key] = entry

            async def collect():
                try:
                    value = await loader()
                except BaseException as exc:
                    entries.pop(key, None)
                    if isinstance(exc, Exception):
                        logger.warning("Shared read failed before result delivery", exc_info=True)
                    raise
                entry.value = value
                entry.completed_at = time.monotonic()
                entry.task = None
                return value

            task = create_owned_asyncio_task(collect(), name="shared_read")
            entry.task = task
            # A waiter may leave before an exception arrives. Retrieve it
            # here as well; awaiting the task still raises the same failure.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            if task.done():
                entry.task = None
                return task.result()
        else:
            entries.move_to_end(key)
        if entry.task is None:
            return entry.value
        return await asyncio.wait_for(asyncio.shield(entry.task), timeout=wait_s)
