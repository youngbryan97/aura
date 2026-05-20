from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.runtime.errors import Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ReasoningQueue")

_QUEUE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_reasoning_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "reasoning_queue",
        error,
        severity=severity,
        action=action,
        extra=extra,
    )


class ReasoningPriority(Enum):
    CRITICAL = 0    # Moral reasoning about an action about to be taken
    HIGH = 1        # Theory of mind update, belief conflict
    NORMAL = 2      # Temporal reflection, future prediction
    LOW = 3         # Self-modification diagnosis, background learning

@dataclass(order=True)
class ReasoningTask:
    priority: int
    coro_fn: Any = field(compare=False)
    task_id: str = field(compare=False, default_factory=lambda: str(__import__('uuid').uuid4())[:8])
    created_at: float = field(default_factory=time.time, compare=False)
    callback: Any = field(compare=False, default=None)
    description: str = field(compare=False, default="")

class BackgroundReasoningQueue:
    """Async queue for deep reasoning tasks.
    Conversation is never blocked — deep reasoning
    runs between turns or concurrently on a separate task.
    """
    
    def __init__(self, max_concurrent: int = 1):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._max_concurrent = max(1, int(max_concurrent))
        self._active_tasks: set[asyncio.Task] = set()
        self._results: dict[str, Any] = {}
        self._worker_tasks: set[asyncio.Task] = set()
        self._worker_task: asyncio.Task | None = None
        self._MAX_CACHED_RESULTS = 50
    
    async def submit(
        self,
        coro_fn: Callable,
        priority: ReasoningPriority = ReasoningPriority.NORMAL,
        callback: Callable | None = None,
        description: str = "",
    ) -> str:
        """Submit a reasoning task. Returns task_id."""
        if not callable(coro_fn):
            raise TypeError("reasoning task requires a callable")
        task_id = str(uuid.uuid4())[:8]
        
        task = ReasoningTask(
            priority=priority.value,
            task_id=task_id,
            coro_fn=coro_fn,
            callback=callback,
            description=description
        )
        
        await self._queue.put(task)
        logger.debug("Queued [%s]: %s (%s)", priority.name, description, task_id)

        self._schedule_registry_size_update(reason="submit")

        return task_id
    
    async def start(self):
        """Start the worker loop."""
        if self._running:
            return
        self._running = True
        tracker = get_task_tracker()
        for worker_id in range(self._max_concurrent):
            worker = tracker.create_task(
                self._run(worker_id=worker_id),
                name=f"reasoning_queue_worker_{worker_id}",
            )
            self._worker_tasks.add(worker)
            worker.add_done_callback(self._worker_tasks.discard)
            if self._worker_task is None:
                self._worker_task = worker
        logger.info(
            "Background Reasoning Queue started with %d worker(s).",
            self._max_concurrent,
        )

    async def _run(self, worker_id: int = 0):
        """Main worker loop."""
        while self._running:
            try:
                # Wait for a task
                task: ReasoningTask = await self._queue.get()

                logger.info(
                    "🧠 Processing Reasoning Task [%s] (%s) on worker %d...",
                    task.description,
                    task.task_id,
                    worker_id,
                )
                self._active_tasks.add(asyncio.current_task())
                try:
                    await self._execute_task(task, worker_id=worker_id)
                finally:
                    self._active_tasks.discard(asyncio.current_task())
                    self._queue.task_done()
                    self._schedule_registry_size_update(reason="task_done")

            except asyncio.CancelledError:
                break
            except _QUEUE_RECOVERABLE_ERRORS as e:
                _record_reasoning_degradation(
                    e,
                    action=(
                        "kept the reasoning worker alive, preserved queued work, "
                        "and applied a one-second backoff"
                    ),
                    extra={"worker_id": worker_id, "queue_size": self._queue.qsize()},
                )
                logger.error("Queue worker %d encountered error: %s", worker_id, e)
                await asyncio.sleep(1) # Prevent tight loop on persistent errors

    async def _execute_task(self, task: ReasoningTask, *, worker_id: int) -> None:
        start_time = time.time()
        try:
            candidate = task.coro_fn()
            result = await candidate if inspect.isawaitable(candidate) else candidate
            elapsed = time.time() - start_time
            logger.info("✓ [%s] completed in %.1fs", task.description, elapsed)

            self._remember_result(task.task_id, result)

            if task.callback:
                await self._invoke_callback(task, result, worker_id=worker_id)
        except _QUEUE_RECOVERABLE_ERRORS as e:
            elapsed = time.time() - start_time
            self._remember_result(
                task.task_id,
                {
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "description": task.description,
                    "elapsed_s": round(elapsed, 3),
                },
            )
            _record_reasoning_degradation(
                e,
                action=(
                    "stored a failure envelope for the task and allowed the "
                    "worker to continue draining the reasoning queue"
                ),
                extra={
                    "task_id": task.task_id,
                    "description": task.description,
                    "worker_id": worker_id,
                },
            )
            logger.error("✗ [%s] failed: %s", task.description, e)

    async def _invoke_callback(
        self,
        task: ReasoningTask,
        result: Any,
        *,
        worker_id: int,
    ) -> None:
        try:
            callback_result = task.callback(result)
            if inspect.isawaitable(callback_result):
                await callback_result
        except _QUEUE_RECOVERABLE_ERRORS as e:
            _record_reasoning_degradation(
                e,
                severity="warning",
                action=(
                    "preserved the completed reasoning result and isolated the "
                    "callback failure"
                ),
                extra={
                    "task_id": task.task_id,
                    "description": task.description,
                    "worker_id": worker_id,
                },
            )

    def _remember_result(self, task_id: str, result: Any) -> None:
        self._results[task_id] = result
        if len(self._results) > self._MAX_CACHED_RESULTS:
            oldest = next(iter(self._results))
            self._results.pop(oldest)

    def get_result(self, task_id: str) -> Any:
        return self._results.get(task_id)

    def results_snapshot(self) -> dict[str, Any]:
        return dict(self._results)

    def _schedule_registry_size_update(self, *, reason: str) -> None:
        try:
            from core.state_registry import get_registry
            get_task_tracker().create_task(
                get_registry().update(reasoning_queue_size=self._queue.qsize()),
                name=f"reasoning_queue_registry_update_{reason}",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _record_reasoning_degradation(
                e,
                severity="warning",
                action=(
                    "continued queue operation with in-memory queue size intact "
                    "after StateRegistry telemetry update failed"
                ),
                extra={"reason": reason, "queue_size": self._queue.qsize()},
            )
    
    async def prune_low_priority(self, threshold_priority: int = ReasoningPriority.NORMAL.value):
        """Drops all tasks with priority > threshold_priority (numerically higher values are lower priority)."""
        new_queue = asyncio.PriorityQueue()
        dropped_count = 0
        
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if task.priority <= threshold_priority:
                    await new_queue.put(task)
                else:
                    dropped_count += 1
                    logger.info("🗑️ Pruning low-priority task [%s] due to cognitive overwhelm.", task.description)
            except asyncio.QueueEmpty:
                break
        
        self._queue = new_queue
        self._schedule_registry_size_update(reason="prune_low_priority")

        return dropped_count

    def stop(self):
        """Stop the worker loop."""
        self._running = False
        for worker in list(self._worker_tasks):
            worker.cancel()
        self._worker_tasks.clear()
        self._worker_task = None

# Global instance
_reasoning_queue: BackgroundReasoningQueue | None = None

def get_reasoning_queue() -> BackgroundReasoningQueue:
    global _reasoning_queue
    if _reasoning_queue is None:
        _reasoning_queue = BackgroundReasoningQueue()
    return _reasoning_queue
