# core/utils/supervisor.py
"""
Managed Task Supervisor.
- create_managed_task(coro, name=...) schedules tasks and attaches a safe done-callback
- tracks tasks in a registry
- provides graceful cancellation
- exposes memory-based protective hooks to cancel/evict optional work when memory high
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Set
import psutil

logger = logging.getLogger("aura.supervisor")

# Tune these envs/values in your config
DEFAULT_MEMORY_HIGH_PERCENT = 80.0
DEFAULT_MEMORY_CRITICAL_PERCENT = 92.0
MEMORY_CHECK_INTERVAL = 5.0  # seconds


class ManagedTask:
    def __init__(self, task: asyncio.Task, created_at: float, name: Optional[str], meta: Optional[dict]):
        self.task = task
        self.created_at = created_at
        self.name = name or f"task-{int(created_at)}"
        self.meta = meta or {}
        self.cancel_reason: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.task.done()

    def cancel(self, reason: Optional[str] = None) -> None:
        self.cancel_reason = reason
        if not self.task.done():
            try:
                self.task.cancel()
            except Exception:
                logger.exception("Exception while cancelling task %s", self.name)


class Supervisor:
    def __init__(self,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 memory_high_percent: float = DEFAULT_MEMORY_HIGH_PERCENT,
                 memory_critical_percent: float = DEFAULT_MEMORY_CRITICAL_PERCENT):
        try:
            self.loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self.loop = loop  # Will be set when the event loop starts
        self._tasks: Dict[str, ManagedTask] = {}
        self._lock = asyncio.Lock()
        self.memory_high_percent = memory_high_percent
        self.memory_critical_percent = memory_critical_percent
        self._memory_watcher_task: Optional[asyncio.Task] = None
        self._optional_task_tags: Set[str] = set()  # tasks allowed to be auto-evicted

    def start_memory_watcher(self) -> None:
        if self._memory_watcher_task is None:
            self._memory_watcher_task = self.create_managed_task(self._memory_watcher(), name="memory_watcher", meta={"system": True})

    def stop_memory_watcher(self) -> None:
        if self._memory_watcher_task is not None:
            self._memory_watcher_task.cancel("supervisor-stopping")
            self._memory_watcher_task = None

    def create_managed_task(self, coro, name: Optional[str] = None, meta: Optional[dict] = None) -> ManagedTask:
        """
        Schedule a coroutine and return a ManagedTask wrapper.
        Use this instead of asyncio.create_task across the codebase.
        """
        if asyncio.iscoroutine(coro):
            task = self.loop.create_task(coro)
        elif callable(coro):
            task = self.loop.create_task(coro())
        else:
            raise TypeError("create_managed_task expects a coroutine or callable returning one")

        created_at = time.time()
        managed = ManagedTask(task, created_at, name, meta)
        # register
        key = managed.name
        self._tasks[key] = managed

        def _on_done(t: asyncio.Task):
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                logger.exception("ManagedTask '%s' finished with exception: %s", key, exc)
            else:
                logger.debug("ManagedTask '%s' finished successfully", key)
            # cleanup
            self._tasks.pop(key, None)

        task.add_done_callback(_on_done)
        logger.debug("Created ManagedTask %s (meta=%s)", key, meta)
        return managed

    async def _memory_watcher(self):
        """Background loop to monitor memory and evict optional tasks proactively."""
        # Fix: psutil.virtual_memory is a function, not a property of a proc
        while True:
            try:
                mem = psutil.virtual_memory()
                percent = mem.percent
                if percent >= self.memory_critical_percent:
                    logger.warning("Memory CRITICAL (%.1f%%) — cancelling optional tasks", percent)
                    # cancel optional tasks immediately
                    await self.cancel_optional_tasks(reason="memory_critical")
                elif percent >= self.memory_high_percent:
                    logger.warning("Memory HIGH (%.1f%%) — gently evicting optional tasks", percent)
                    await self.cancel_optional_tasks(reason="memory_high")
                await asyncio.sleep(MEMORY_CHECK_INTERVAL)
            except asyncio.CancelledError:
                logger.info("Memory watcher shutting down.")
                break
            except Exception as e:
                logger.error("Memory watcher error: %s", e)
                await asyncio.sleep(MEMORY_CHECK_INTERVAL)

    async def cancel_optional_tasks(self, reason: str):
        """Cancel tasks that are marked optional (by meta tag or registered name)."""
        async with self._lock:
            to_cancel = [m for m in self._tasks.values() if m.meta.get("optional") or m.name in self._optional_task_tags]
            for m in to_cancel:
                logger.info("Evicting optional task %s (reason=%s)", m.name, reason)
                m.cancel(reason)

    async def cancel_all(self, reason: Optional[str] = "shutdown"):
        async with self._lock:
            for m in list(self._tasks.values()):
                logger.info("Cancelling task %s (reason=%s)", m.name, reason)
                m.cancel(reason)

    def register_optional_tag(self, name: str):
        self._optional_task_tags.add(name)

    def get_task_names(self):
        return list(self._tasks.keys())

    def get_task(self, name: str) -> Optional[ManagedTask]:
        return self._tasks.get(name)
