"""Background reasoning queue.

Conversation is never blocked — deep reasoning runs between turns or
concurrently on separate worker tasks.

That promise only holds if the queue cannot grow without bound, a worker
cannot die silently, and a task that is dropped or cancelled tells its
submitter so. None of those held before CP126: the queue was unbounded, an
unlisted exception killed a worker while ``_running`` stayed True (so
``start()`` refused to replace it), pruning swapped in a *new* queue object
while a worker was parked on the old one, and pruned or cancelled work
vanished without a result.

CP126 b3925b0c / 7c56145d / ddab4f5a / 99c0be71 / db963c45 / 9056ce92.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.runtime.errors import Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ReasoningQueue")

#: Default capacity. A reasoning backlog past this point is a symptom, not a
#: workload — admitting more of it only converts latency into memory.
DEFAULT_MAX_QUEUE = 256

#: Default per-task execution budget and time-to-live in the queue.
DEFAULT_TASK_TIMEOUT_S = 120.0
DEFAULT_TASK_TTL_S = 900.0
DEFAULT_CALLBACK_TIMEOUT_S = 15.0

#: How long submit() waits for space before applying the admission policy.
DEFAULT_ADMISSION_WAIT_S = 0.0

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


class QueueFull(RuntimeError):
    """Raised when admission policy refuses a submission."""


@dataclass(order=True)
class ReasoningTask:
    priority: int
    coro_fn: Any = field(compare=False)
    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time, compare=False)
    callback: Any = field(compare=False, default=None)
    description: str = field(compare=False, default="")
    #: Execution budget and queue time-to-live (CP126 b3925b0c).
    timeout_s: float = field(compare=False, default=DEFAULT_TASK_TIMEOUT_S)
    ttl_s: float = field(compare=False, default=DEFAULT_TASK_TTL_S)
    #: Tiebreaker so equal priorities never compare the payload.
    sequence: int = field(compare=True, default=0)

    def expired(self, *, now: float | None = None) -> bool:
        if self.ttl_s <= 0:
            return False
        return (now or time.time()) - self.created_at > self.ttl_s


class BackgroundReasoningQueue:
    """Async queue for deep reasoning tasks."""

    def __init__(
        self,
        max_concurrent: int = 1,
        *,
        maxsize: int = DEFAULT_MAX_QUEUE,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
        callback_timeout_s: float = DEFAULT_CALLBACK_TIMEOUT_S,
        task_ttl_s: float = DEFAULT_TASK_TTL_S,
    ):
        # CP126 b3925b0c: an unbounded PriorityQueue plus an always-awaiting
        # put() is a memory-exhaustion path with no backpressure signal.
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max(1, int(maxsize)))
        self._running = False
        self._max_concurrent = max(1, int(max_concurrent))
        self._active_tasks: set[asyncio.Task] = set()
        self._results: OrderedDict[str, Any] = OrderedDict()
        self._worker_tasks: set[asyncio.Task] = set()
        self._worker_task: asyncio.Task | None = None
        self._MAX_CACHED_RESULTS = 50
        self._pending_ids: set[str] = set()
        self._sequence = 0
        self._prune_lock = asyncio.Lock()
        self.task_timeout_s = float(task_timeout_s)
        self.callback_timeout_s = float(callback_timeout_s)
        self.task_ttl_s = float(task_ttl_s)
        self._shed_count = 0

    # -- admission (CP126 b3925b0c) ---------------------------------------
    async def submit(
        self,
        coro_fn: Callable,
        priority: ReasoningPriority = ReasoningPriority.NORMAL,
        callback: Callable | None = None,
        description: str = "",
        *,
        timeout_s: float | None = None,
        ttl_s: float | None = None,
        admission_wait_s: float = DEFAULT_ADMISSION_WAIT_S,
    ) -> str:
        """Submit a reasoning task. Returns task_id.

        When the queue is full, a task is admitted only if it outranks
        something already queued — that lower-priority task is shed with a
        receipt. Otherwise the submission is refused with ``QueueFull`` rather
        than blocking a caller (or the event loop) indefinitely.
        """
        if not callable(coro_fn):
            raise TypeError("reasoning task requires a callable")
        task_id = str(uuid.uuid4())[:8]
        self._sequence += 1

        task = ReasoningTask(
            priority=priority.value,
            task_id=task_id,
            coro_fn=coro_fn,
            callback=callback,
            description=description,
            timeout_s=self.task_timeout_s if timeout_s is None else float(timeout_s),
            ttl_s=self.task_ttl_s if ttl_s is None else float(ttl_s),
            sequence=self._sequence,
        )

        if not self._admit(task, admission_wait_s):
            if admission_wait_s > 0:
                try:
                    await asyncio.wait_for(self._queue.put(task), timeout=admission_wait_s)
                    self._pending_ids.add(task_id)
                    self._schedule_registry_size_update(reason="submit")
                    return task_id
                except TimeoutError:
                    pass
            reason = f"reasoning queue is full ({self._queue.qsize()} queued)"
            self._remember_result(
                task_id,
                {
                    "status": "rejected",
                    "reason": reason,
                    "description": description,
                    "priority": priority.name,
                },
            )
            _record_reasoning_degradation(
                QueueFull(reason),
                severity="warning",
                action="refused a reasoning submission instead of growing the queue without bound",
                extra={"task_id": task_id, "description": description},
            )
            raise QueueFull(reason)

        self._pending_ids.add(task_id)
        logger.debug("Queued [%s]: %s (%s)", priority.name, description, task_id)
        self._schedule_registry_size_update(reason="submit")
        return task_id

    def _admit(self, task: ReasoningTask, admission_wait_s: float) -> bool:
        """Try to place the task now, shedding a worse one if necessary."""
        try:
            self._queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            pass
        if admission_wait_s > 0:
            return False

        victim = self._worst_queued()
        if victim is None or victim.priority <= task.priority:
            return False
        # Shed the lower-priority task and take its place, with a receipt.
        drained = self._drain_all()
        kept = [item for item in drained if item.task_id != victim.task_id]
        self._refill(kept)
        self._finalize_dropped(victim, "shed", f"displaced by {task.description or task.task_id}")
        self._shed_count += 1
        try:
            self._queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            return False

    def _worst_queued(self) -> ReasoningTask | None:
        items = list(getattr(self._queue, "_queue", ()) or ())
        if not items:
            return None
        return max(items, key=lambda item: (item.priority, item.sequence))

    # -- workers (CP126 7c56145d) -----------------------------------------
    async def start(self):
        """Start (or top up) the worker pool.

        The old ``start`` returned immediately when ``_running`` was True, so a
        worker killed by an unlisted exception was never replaced and the queue
        stalled permanently while still reporting itself running.
        """
        self._running = True
        tracker = get_task_tracker()
        self._worker_tasks = {task for task in self._worker_tasks if not task.done()}
        missing = self._max_concurrent - len(self._worker_tasks)
        for worker_id in range(missing):
            worker = tracker.create_task(
                self._run(worker_id=len(self._worker_tasks)),
                name=f"reasoning_queue_worker_{worker_id}",
            )
            self._worker_tasks.add(worker)
            worker.add_done_callback(self._on_worker_done)
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = worker
        if missing > 0:
            logger.info(
                "Background Reasoning Queue running with %d worker(s).",
                len(self._worker_tasks),
            )

    def _on_worker_done(self, worker: asyncio.Task) -> None:
        self._worker_tasks.discard(worker)
        if not self._running or worker.cancelled():
            return
        exc = worker.exception() if worker.done() else None
        if exc is not None:
            _record_reasoning_degradation(
                exc,
                severity="error",
                action="respawning the reasoning worker that died on an unhandled exception",
                extra={"worker": worker.get_name()},
            )
            logger.error("Reasoning worker %s died: %s", worker.get_name(), exc)
        # Replace it: a queue that reports itself running must have workers.
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().call_soon(
                lambda: get_task_tracker().create_task(
                    self.start(),
                    name="ReasoningQueue.recover_workers",
                )
            )

    async def _run(self, worker_id: int = 0):
        """Main worker loop."""
        while self._running:
            task: ReasoningTask | None = None
            try:
                task = await self._queue.get()
                if task.expired():
                    self._finalize_dropped(
                        task, "expired", f"exceeded its {task.ttl_s:.0f}s time-to-live"
                    )
                    self._queue.task_done()
                    task = None
                    self._schedule_registry_size_update(reason="expired")
                    continue

                logger.info(
                    "🧠 Processing Reasoning Task [%s] (%s) on worker %d...",
                    task.description,
                    task.task_id,
                    worker_id,
                )
                current = asyncio.current_task()
                self._active_tasks.add(current)
                try:
                    await self._execute_task(task, worker_id=worker_id)
                finally:
                    self._active_tasks.discard(current)
                    self._pending_ids.discard(task.task_id)
                    self._queue.task_done()
                    task = None
                    self._schedule_registry_size_update(reason="task_done")

            except asyncio.CancelledError:
                if task is not None:
                    # CP126 99c0be71: the task was already consumed from the
                    # queue; without this its submitter waits on a result that
                    # can never arrive.
                    self._finalize_dropped(
                        task, "cancelled", "worker cancelled while the task was in flight"
                    )
                raise
            except Exception as exc:  # noqa: BLE001 - a worker must outlive its work
                _record_reasoning_degradation(
                    exc,
                    action=(
                        "kept the reasoning worker alive, preserved queued work, "
                        "and applied a one-second backoff"
                    ),
                    extra={"worker_id": worker_id, "queue_size": self._queue.qsize()},
                )
                logger.error("Queue worker %d encountered error: %s", worker_id, exc)
                await asyncio.sleep(1)  # Prevent tight loop on persistent errors

    async def _execute_task(self, task: ReasoningTask, *, worker_id: int) -> None:
        start_time = time.time()
        try:
            candidate = task.coro_fn()
            if inspect.isawaitable(candidate):
                # CP126 b3925b0c: one hung task per worker used to stop all
                # draining indefinitely.
                result = (
                    await asyncio.wait_for(candidate, timeout=task.timeout_s)
                    if task.timeout_s > 0
                    else await candidate
                )
            else:
                result = candidate
            elapsed = time.time() - start_time
            logger.info("✓ [%s] completed in %.1fs", task.description, elapsed)

            self._remember_result(task.task_id, result)

            if task.callback:
                await self._invoke_callback(task, result, worker_id=worker_id)
        except asyncio.CancelledError:
            self._remember_result(
                task.task_id,
                {
                    "status": "cancelled",
                    "description": task.description,
                    "elapsed_s": round(time.time() - start_time, 3),
                },
            )
            raise
        except TimeoutError as exc:
            self._store_failure(task, exc, start_time, worker_id, status="timeout")
        except Exception as exc:  # noqa: BLE001 - unlisted failures killed workers
            self._store_failure(task, exc, start_time, worker_id)

    def _store_failure(
        self,
        task: ReasoningTask,
        exc: BaseException,
        start_time: float,
        worker_id: int,
        *,
        status: str = "failed",
    ) -> None:
        elapsed = time.time() - start_time
        self._remember_result(
            task.task_id,
            {
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "description": task.description,
                "elapsed_s": round(elapsed, 3),
            },
        )
        _record_reasoning_degradation(
            exc,
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
        logger.error("✗ [%s] %s: %s", task.description, status, exc)

    async def _invoke_callback(
        self,
        task: ReasoningTask,
        result: Any,
        *,
        worker_id: int,
    ) -> None:
        """Run the submitter's callback without letting it own the worker.

        CP126 db963c45: an awaitable callback had no budget, so one hung
        callback consumed worker capacity indefinitely even though the
        reasoning result was already stored.
        """
        try:
            callback_result = task.callback(result)
            if inspect.isawaitable(callback_result):
                if self.callback_timeout_s > 0:
                    await asyncio.wait_for(callback_result, timeout=self.callback_timeout_s)
                else:
                    await callback_result
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - callbacks are submitter code
            _record_reasoning_degradation(
                exc,
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

    # -- results (CP126 9056ce92) -----------------------------------------
    def _remember_result(self, task_id: str, result: Any) -> None:
        self._results[task_id] = result
        self._results.move_to_end(task_id)
        while len(self._results) > self._MAX_CACHED_RESULTS:
            self._results.popitem(last=False)

    def get_result(self, task_id: str) -> Any:
        return self._results.get(task_id)

    def task_status(self, task_id: str) -> str:
        """pending | done | evicted_or_unknown — a submitter can tell these apart."""
        if task_id in self._pending_ids:
            return "pending"
        if task_id in self._results:
            result = self._results[task_id]
            if isinstance(result, dict) and result.get("status"):
                return str(result["status"])
            return "done"
        return "evicted_or_unknown"

    def results_snapshot(self) -> dict[str, Any]:
        return dict(self._results)

    def stats(self) -> dict[str, Any]:
        return {
            "queued": self._queue.qsize(),
            "maxsize": self._queue.maxsize,
            "pending_ids": len(self._pending_ids),
            "workers": len([task for task in self._worker_tasks if not task.done()]),
            "running": self._running,
            "shed": self._shed_count,
            "cached_results": len(self._results),
        }

    def _finalize_dropped(self, task: ReasoningTask, status: str, reason: str) -> None:
        """Give a dropped task a terminal envelope and fire its callback.

        CP126 9056ce92: pruning logged and counted, so a submitter holding a
        task id could not distinguish a pruned task from pending work, a cache
        eviction, an unknown id, or a real result of None.
        """
        envelope = {
            "status": status,
            "reason": reason,
            "description": task.description,
            "priority": task.priority,
            "task_id": task.task_id,
            "queued_for_s": round(time.time() - task.created_at, 3),
        }
        self._remember_result(task.task_id, envelope)
        self._pending_ids.discard(task.task_id)
        if task.callback:
            with contextlib.suppress(RuntimeError):
                get_task_tracker().create_task(
                    self._notify_dropped(task, envelope),
                    name=f"reasoning_queue_dropped_{task.task_id}",
                )

    async def _notify_dropped(self, task: ReasoningTask, envelope: dict[str, Any]) -> None:
        await self._invoke_callback(task, envelope, worker_id=-1)

    def _schedule_registry_size_update(self, *, reason: str) -> None:
        try:
            from core.state.state_registry import get_registry
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

    # -- pruning (CP126 ddab4f5a) -----------------------------------------
    def _drain_all(self) -> list[ReasoningTask]:
        drained: list[ReasoningTask] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        # Balance the unfinished-task accounting for everything we removed;
        # anything re-queued increments it again on put.
        for _ in drained:
            with contextlib.suppress(ValueError):
                self._queue.task_done()
        return drained

    def _refill(self, tasks: list[ReasoningTask]) -> None:
        for task in tasks:
            try:
                self._queue.put_nowait(task)
            except asyncio.QueueFull:
                self._finalize_dropped(
                    task, "shed", "queue was full while restoring after a prune"
                )

    async def prune_low_priority(
        self, threshold_priority: int = ReasoningPriority.NORMAL.value
    ) -> int:
        """Drop tasks whose priority is worse than the threshold.

        The queue OBJECT is preserved. Replacing ``self._queue`` stranded any
        worker already parked in ``get()`` on the old instance: new submissions
        went to the new object and never woke it.
        """
        async with self._prune_lock:
            retained: list[ReasoningTask] = []
            dropped: list[ReasoningTask] = []
            for task in self._drain_all():
                if task.priority <= threshold_priority and not task.expired():
                    retained.append(task)
                else:
                    dropped.append(task)
            self._refill(retained)

            for task in dropped:
                logger.info(
                    "🗑️ Pruning low-priority task [%s] due to cognitive overwhelm.",
                    task.description,
                )
                self._finalize_dropped(
                    task,
                    "expired" if task.expired() else "pruned",
                    f"priority {task.priority} exceeded threshold {threshold_priority}",
                )
            self.last_pruned_ids = [task.task_id for task in dropped]
            self._schedule_registry_size_update(reason="prune_low_priority")
            return len(dropped)

    # -- shutdown (CP126 99c0be71) ----------------------------------------
    def stop(self):
        """Stop the worker loop and give every queued task a terminal result."""
        self._running = False
        for worker in list(self._worker_tasks):
            worker.cancel()
        self._worker_tasks.clear()
        self._worker_task = None
        for task in self._drain_all():
            self._finalize_dropped(task, "cancelled", "queue stopped before execution")
        self._schedule_registry_size_update(reason="stop")

    async def aclose(self, timeout: float = 5.0) -> bool:
        """Stop and await the workers within a deadline."""
        workers = [task for task in self._worker_tasks if not task.done()]
        self.stop()
        if not workers:
            return True
        try:
            await asyncio.wait_for(
                asyncio.gather(*workers, return_exceptions=True), timeout=timeout
            )
        except TimeoutError:
            logger.warning("Reasoning workers did not stop within %.1fs", timeout)
            return False
        return True


# Global instance
_reasoning_queue: BackgroundReasoningQueue | None = None


def get_reasoning_queue() -> BackgroundReasoningQueue:
    global _reasoning_queue
    if _reasoning_queue is None:
        _reasoning_queue = BackgroundReasoningQueue()
    return _reasoning_queue
