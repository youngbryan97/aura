from core.runtime.errors import record_degradation
import asyncio
import contextvars
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_SKIP_FACTORY_TRACK = contextvars.ContextVar("aura_skip_factory_track", default=False)


@dataclass
class TaskRecord:
    task_id: int
    name: str
    tracker: str
    supervision: str
    source: str
    created_at: float
    coroutine: str = "unknown"
    done: bool = False
    cancelled: bool = False
    failed: bool = False
    finished_at: Optional[float] = None
    exception: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.monotonic)

    def age_s(self, now: Optional[float] = None) -> float:
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.created_at)

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        duration = None
        if self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.created_at)
        return {
            "task_id": self.task_id,
            "name": self.name,
            "tracker": self.tracker,
            "supervision": self.supervision,
            "source": self.source,
            "coroutine": self.coroutine,
            "age_s": self.age_s(now),
            "done": self.done,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "finished_at": self.finished_at,
            "duration_s": duration,
            "exception": self.exception,
            "last_heartbeat": self.last_heartbeat,
        }


class TaskTracker:
    """Track and manage background asyncio tasks to ensure graceful shutdown.

    Prevents "Task was destroyed but it is pending!" errors and provides
    lifecycle telemetry for tasks created both through the tracker and through
    raw asyncio task creation APIs.
    """

    def __init__(self, name: str = "Global", max_concurrent: int = 20):
        self.name = name
        self.tasks: Set[asyncio.Task] = set()
        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None  # Lazy init
        self._high_water = 0
        self._total_tracked = 0
        self._total_observed = 0
        self._completed_total = 0
        self._cancelled_total = 0
        self._failed_total = 0
        self._records: Dict[int, TaskRecord] = {}
        self._recently_completed: Deque[Dict[str, Any]] = deque(maxlen=128)
        self._installed_loop_factories: Dict[int, Any] = {}

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore (must be in event loop context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    def track(self, coro_or_task, name: Optional[str] = None) -> asyncio.Task:
        """Track a new task or coroutine (no concurrency limit)."""
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        else:
            child_context = contextvars.copy_context()
            child_context.run(_SKIP_FACTORY_TRACK.set, False)
            token = _SKIP_FACTORY_TRACK.set(True)
            try:
                try:
                    task = asyncio.create_task(coro_or_task, name=name, context=child_context)
                except TypeError as exc:
                    if "context" not in str(exc):
                        raise
                    task = asyncio.create_task(coro_or_task, name=name)
            finally:
                _SKIP_FACTORY_TRACK.reset(token)
        self._total_tracked += 1
        self._attach(task, name=name, supervision="explicit", source="track")
        return task

    # Alias for compatibility with components calling track_task or create_task
    track_task = track
    create_task = track

    def observe(self, task: asyncio.Task, name: Optional[str] = None, source: str = "loop_factory") -> asyncio.Task:
        """Observe a task created outside the tracker so it still gets cleaned up and audited."""
        self._attach(task, name=name, supervision="implicit", source=source)
        return task

    def bounded_track(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Track a task WITH concurrency limiting via semaphore.

        Use this for short-lived tasks (maintenance, learning, reflection).
        Long-running loops should use track() directly.
        """

        async def _bounded():
            sem = self._get_semaphore()
            async with sem:
                if asyncio.iscoroutine(coro):
                    return await coro
                if asyncio.iscoroutinefunction(coro):
                    return await coro()
                return await coro

        child_context = contextvars.copy_context()
        child_context.run(_SKIP_FACTORY_TRACK.set, False)
        token = _SKIP_FACTORY_TRACK.set(True)
        try:
            try:
                task = asyncio.create_task(_bounded(), name=name, context=child_context)
            except TypeError as exc:
                if "context" not in str(exc):
                    raise
                task = asyncio.create_task(_bounded(), name=name)
        finally:
            _SKIP_FACTORY_TRACK.reset(token)
        self._total_tracked += 1
        self._attach(task, name=name, supervision="explicit", source="bounded_track")
        return task

    def install_loop_hygiene(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Install a task factory so raw asyncio.create_task/loop.create_task calls are still observed."""
        target_loop = loop or asyncio.get_running_loop()
        loop_id = id(target_loop)
        if loop_id in self._installed_loop_factories:
            return

        previous_factory = target_loop.get_task_factory()
        tracker = self

        def _factory(factory_loop, coro, **kwargs):
            if previous_factory is not None:
                try:
                    task = previous_factory(factory_loop, coro, **kwargs)
                except TypeError:
                    kwargs.pop("context", None)
                    try:
                        task = previous_factory(factory_loop, coro, **kwargs)
                    except TypeError:
                        kwargs.pop("name", None)
                        task = previous_factory(factory_loop, coro, **kwargs)
            else:
                task = asyncio.Task(coro, loop=factory_loop, **kwargs)
            if not _SKIP_FACTORY_TRACK.get():
                try:
                    tracker.observe(task, source="loop_factory")
                except Exception as exc:
                    record_degradation('task_tracker', exc)
                    logger.debug("TaskTracker[%s]: failed to observe loop task: %s", tracker.name, exc)
            return task

        target_loop.set_task_factory(_factory)
        self._installed_loop_factories[loop_id] = (target_loop, previous_factory)

    def restore_loop_hygiene(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Restore a loop's original task factory."""
        if loop is not None:
            info = self._installed_loop_factories.pop(id(loop), None)
            if info is not None:
                target_loop, previous_factory = info
                try:
                    target_loop.set_task_factory(previous_factory)
                except Exception as exc:
                    record_degradation('task_tracker', exc)
                    logger.debug("TaskTracker[%s]: failed to restore loop factory: %s", self.name, exc)
            return

        for loop_id, info in list(self._installed_loop_factories.items()):
            target_loop, previous_factory = info
            try:
                target_loop.set_task_factory(previous_factory)
            except Exception as exc:
                record_degradation('task_tracker', exc)
                logger.debug("TaskTracker[%s]: failed to restore loop factory: %s", self.name, exc)
            finally:
                self._installed_loop_factories.pop(loop_id, None)

    def get_stale_tasks(self, min_age_s: float = 900.0, *, include_supervised: bool = False) -> List[Dict[str, Any]]:
        """Return a sample of long-lived tasks that may need inspection."""
        now = time.monotonic()
        stale: List[Dict[str, Any]] = []
        for task in list(self.tasks):
            if task.done():
                continue
            record = self._records.get(id(task))
            if record is None:
                continue
            if record.age_s(now) < min_age_s:
                continue
            if not include_supervised and record.supervision == "explicit":
                continue
            stale.append(record.to_dict(now))
        stale.sort(key=lambda item: item["age_s"], reverse=True)
        return stale

    def heartbeat(self, task: Optional[asyncio.Task] = None) -> None:
        """Register a heartbeat for the given task, or the current task if None."""
        target_task = task or asyncio.current_task()
        if not target_task:
            return
            
        record = self._records.get(id(target_task))
        if record:
            record.last_heartbeat = time.monotonic()

    def _mark_supervised(self, task: asyncio.Task) -> None:
        try:
            setattr(task, "_aura_supervised", True)
            setattr(task, "_aura_task_tracker", self.name)
            setattr(task, "_aura_task_supervision", "explicit")
        except Exception as e:
            record_degradation('task_tracker', e)
            logger.debug("TaskTracker[%s]: failed to mark task supervised: %s", self.name, e)

    def _attach(
        self,
        task: asyncio.Task,
        *,
        name: Optional[str],
        supervision: str,
        source: str,
    ) -> None:
        task_name = name or task.get_name()
        task_id = id(task)
        record = self._records.get(task_id)
        if record is None:
            record = TaskRecord(
                task_id=task_id,
                name=task_name,
                tracker=self.name,
                supervision=supervision,
                source=source,
                created_at=time.monotonic(),
                coroutine=self._describe_task(task),
            )
            self._records[task_id] = record
            self.tasks.add(task)
            task.add_done_callback(self._on_task_done)
            self._total_observed += 1
        else:
            if name:
                record.name = task_name
            if record.source == "loop_factory" and source != "loop_factory":
                record.source = source
            if supervision == "explicit":
                record.supervision = "explicit"

        try:
            setattr(task, "_aura_task_tracker", self.name)
            setattr(task, "_aura_task_supervision", record.supervision)
            setattr(task, "_aura_task_source", record.source)
            setattr(task, "_aura_task_created_at", record.created_at)
            if record.supervision == "explicit":
                self._mark_supervised(task)
            elif not hasattr(task, "_aura_supervised"):
                setattr(task, "_aura_supervised", False)
        except Exception as exc:
            record_degradation('task_tracker', exc)
            logger.debug("TaskTracker[%s]: failed to annotate task: %s", self.name, exc)

        if task.done():
            self._on_task_done(task)
        else:
            self._high_water = max(self._high_water, len(self.tasks))

    def _describe_task(self, task: asyncio.Task) -> str:
        try:
            coro = task.get_coro()
        except Exception:
            return "unknown"
        qualname = getattr(coro, "__qualname__", None)
        if qualname:
            return qualname
        return repr(coro)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        record = self._records.get(id(task))
        if record is None or record.done:
            return

        record.done = True
        record.finished_at = time.monotonic()
        self._completed_total += 1

        if task.cancelled():
            record.cancelled = True
            self._cancelled_total += 1
        else:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                record.cancelled = True
                self._cancelled_total += 1
            except Exception as exc:
                record_degradation('task_tracker', exc)
                record.failed = True
                record.exception = f"{type(exc).__name__}: {exc}"
                self._failed_total += 1
            else:
                if exc is not None:
                    record.failed = True
                    record.exception = f"{type(exc).__name__}: {exc}"
                    self._failed_total += 1
                    logger.warning(
                        "TaskTracker[%s]: task %s failed: %s",
                        self.name,
                        record.name,
                        record.exception,
                    )

        self._recently_completed.append(record.to_dict())

    @property
    def active_count(self) -> int:
        """Number of currently active (not done) tasks."""
        return len(self.tasks)

    async def shutdown(self, timeout: float = 5.0):
        """Cancel and wait for all tracked tasks."""
        pending = {
            task
            for task in self.tasks
            if not task.done() and task is not asyncio.current_task()
        }
        if not pending:
            return

        logger.info("Shutting down TaskTracker[%s]: %s tasks pending.", self.name, len(pending))

        for task in pending:
            task.cancel()

        try:
            await asyncio.wait(pending, timeout=timeout)
        except Exception as e:
            record_degradation('task_tracker', e)
            logger.error("Error during TaskTracker shutdown: %s", e)

        remaining = [task for task in pending if not task.done()]
        if remaining:
            logger.warning("%d tasks still pending after timeout. Forcing abandonment.", len(remaining))
        for task in remaining:
            self.tasks.discard(task)

    def get_stats(self) -> dict:
        explicit_active = 0
        implicit_active = 0
        for task in list(self.tasks):
            record = self._records.get(id(task))
            if record is None:
                continue
            if record.supervision == "explicit":
                explicit_active += 1
            else:
                implicit_active += 1
        stale_tasks = self.get_stale_tasks(min_age_s=300.0)
        return {
            "active": self.active_count,
            "high_water": self._high_water,
            "total_tracked": self._total_tracked,
            "total_observed": self._total_observed,
            "explicit_active": explicit_active,
            "implicit_active": implicit_active,
            "completed_total": self._completed_total,
            "cancelled_total": self._cancelled_total,
            "failed_total": self._failed_total,
            "max_concurrent": self._max_concurrent,
            "stale_tasks": stale_tasks[:5],
            "recently_completed": list(self._recently_completed)[-5:],
        }


_task_tracker = None


def get_task_tracker() -> TaskTracker:
    global _task_tracker
    if _task_tracker is None:
        _task_tracker = TaskTracker(name="Global")
    return _task_tracker


# For backward compatibility if any module imports task_tracker directly
# Note: it's better to use get_task_tracker() in async contexts.
task_tracker = TaskTracker()


def fire_and_track(coro, name: Optional[str] = None) -> asyncio.Task:
    """Convenience function to create and track a task in one go."""
    tracker = get_task_tracker()
    return tracker.track(coro, name=name)
