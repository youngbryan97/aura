import asyncio
import contextvars
import inspect
import itertools
import logging
import threading
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.task_ownership import close_awaitable, create_owned_asyncio_task

logger = logging.getLogger(__name__)

_SKIP_FACTORY_TRACK = contextvars.ContextVar("aura_skip_factory_track", default=False)
_ALLOW_SHUTDOWN_TASK_CREATION = contextvars.ContextVar(
    "aura_allow_shutdown_task_creation",
    default=False,
)
_ALLOW_SHUTDOWN_RESOURCE_CREATION = contextvars.ContextVar(
    "aura_allow_shutdown_resource_creation",
    default=False,
)


def shutdown_task_creation_allowed(context: contextvars.Context | None = None) -> bool:
    if context is not None:
        return bool(context.get(_ALLOW_SHUTDOWN_TASK_CREATION, False))
    return bool(_ALLOW_SHUTDOWN_TASK_CREATION.get())


_CANONICAL_SHUTDOWN_ENTRYPOINTS = frozenset(
    {
        ("core.ops.graceful_shutdown", "GracefulShutdown.trigger_shutdown"),
        ("core.runtime.shutdown_coordinator", "ShutdownCoordinator.shutdown"),
    }
)


def canonical_shutdown_awaitable(awaitable: Any) -> bool:
    """Recognize exact root shutdown coroutines at the raw task boundary."""
    if not inspect.iscoroutine(awaitable):
        return False
    code = getattr(awaitable, "cr_code", None)
    frame = getattr(awaitable, "cr_frame", None)
    if code is None or frame is None:
        return False
    module_name = str(frame.f_globals.get("__name__") or "")
    qualname = str(getattr(code, "co_qualname", "") or "")
    return (module_name, qualname) in _CANONICAL_SHUTDOWN_ENTRYPOINTS


def begin_shutdown_task_creation_scope() -> contextvars.Token[bool]:
    return _ALLOW_SHUTDOWN_TASK_CREATION.set(True)


def end_shutdown_task_creation_scope(token: contextvars.Token[bool]) -> None:
    _ALLOW_SHUTDOWN_TASK_CREATION.reset(token)


def shutdown_resource_creation_allowed() -> bool:
    """Whether this exact call site may create a teardown worker/resource."""

    return bool(_ALLOW_SHUTDOWN_RESOURCE_CREATION.get())


def begin_shutdown_resource_creation_scope() -> contextvars.Token[bool]:
    return _ALLOW_SHUTDOWN_RESOURCE_CREATION.set(True)


def end_shutdown_resource_creation_scope(token: contextvars.Token[bool]) -> None:
    _ALLOW_SHUTDOWN_RESOURCE_CREATION.reset(token)


def _runtime_shutdown_requested() -> bool:
    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        return bool(is_shutdown_requested())
    except (ImportError, AttributeError, RuntimeError):
        return False


def _record_shutdown_task_event(
    *,
    name: str | None,
    source: str,
    outcome: str,
) -> None:
    try:
        from core.runtime.shutdown_coordinator import record_shutdown_admission_event

        record_shutdown_admission_event(
            name or "unnamed_task",
            resource_kind="asyncio_task",
            outcome=outcome,
            detail=source,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return


def mark_task_protected(task: asyncio.Task[Any], *, owner: str = "task_tracker") -> asyncio.Task[Any]:
    """Mark a task as shutdown-critical without exempting it from cancellation."""
    try:
        task._aura_protected = True
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(owner, exc)
        logger.debug("Task protection annotation failed for %s: %s", owner, exc)
    return task


@dataclass
class TaskRecord:
    lifecycle_id: str
    task_id: int
    name: str
    owner: str
    tracker: str
    supervision: str
    source: str
    created_at: float
    coroutine: str = "unknown"
    done: bool = False
    cancelled: bool = False
    failed: bool = False
    finished_at: float | None = None
    exception: str | None = None
    outcome: str = "running"
    last_heartbeat: float = field(default_factory=time.monotonic)

    def age_s(self, now: float | None = None) -> float:
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.created_at)

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        duration = None
        if self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.created_at)
        return {
            "lifecycle_id": self.lifecycle_id,
            "task_id": self.task_id,
            "name": self.name,
            "owner": self.owner,
            "tracker": self.tracker,
            "supervision": self.supervision,
            "source": self.source,
            "coroutine": self.coroutine,
            "age_s": self.age_s(now),
            "done": self.done,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "outcome": self.outcome,
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
        self._state_lock = threading.RLock()
        self.tasks: set[asyncio.Task] = set()
        self._max_concurrent = max_concurrent
        self._semaphores: dict[
            int,
            tuple[
                weakref.ReferenceType[asyncio.AbstractEventLoop],
                asyncio.Semaphore,
            ],
        ] = {}
        self._high_water = 0
        self._total_tracked = 0
        self._total_observed = 0
        self._completed_total = 0
        self._cancelled_total = 0
        self._failed_total = 0
        self._shutdown_suppressed_total = 0
        self._tracker_instance_id = uuid.uuid4().hex[:12]
        self._lifecycle_sequence = itertools.count(1)
        self._records: dict[int, TaskRecord] = {}
        self._recently_completed: deque[dict[str, Any]] = deque(maxlen=128)
        self._installed_loop_factories: dict[int, Any] = {}
        self._max_records_in_memory = 256  # Bounded history of completed tasks

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return the bounded-work semaphore owned by the current event loop."""

        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        with self._state_lock:
            existing = self._semaphores.get(loop_id)
            if existing is not None and existing[0]() is loop:
                return existing[1]

            semaphore = asyncio.Semaphore(self._max_concurrent)
            self._semaphores[loop_id] = (weakref.ref(loop), semaphore)
            if len(self._semaphores) > 64:
                stale_loop_ids = [
                    candidate_id
                    for candidate_id, (loop_ref, _semaphore) in self._semaphores.items()
                    if loop_ref() is None
                    or bool(getattr(loop_ref(), "is_closed", lambda: True)())
                ]
                for candidate_id in stale_loop_ids:
                    self._semaphores.pop(candidate_id, None)
            return semaphore

    def track(
        self,
        coro_or_task,
        name: str | None = None,
        *,
        owner: str | None = None,
        allow_during_shutdown: bool = False,
    ) -> asyncio.Task:
        """Track a new task or coroutine (no concurrency limit)."""
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        else:
            if _runtime_shutdown_requested() and not allow_during_shutdown:
                return self._suppress_shutdown_start(coro_or_task, name=name, source="track")
            token = (
                begin_shutdown_task_creation_scope()
                if allow_during_shutdown
                else None
            )
            try:
                try:
                    task = create_owned_asyncio_task(coro_or_task, name=name)
                except RuntimeError:
                    close_awaitable(coro_or_task)
                    raise
            finally:
                if token is not None:
                    end_shutdown_task_creation_scope(token)
        if allow_during_shutdown:
            self._mark_shutdown_critical(task)
            if _runtime_shutdown_requested():
                _record_shutdown_task_event(
                    name=name,
                    source="track:explicit_shutdown_owner",
                    outcome="allowed_teardown",
                )
        with self._state_lock:
            self._total_tracked += 1
        self._attach(
            task,
            name=name,
            owner=owner,
            supervision="explicit",
            source="track",
        )
        return task

    # Compatibility methods for components calling track_task or create_task.
    # Keep these as forwarding methods instead of class-level aliases so future
    # keyword-only lifecycle controls cannot stale-bind to an older track
    # implementation during import-heavy tests or warm reloads.
    def track_task(self, coro_or_task, name: str | None = None, **kwargs: Any) -> asyncio.Task:
        return self.track(coro_or_task, name=name, **kwargs)

    def create_task(self, coro_or_task, name: str | None = None, **kwargs: Any) -> asyncio.Task:
        return self.track(coro_or_task, name=name, **kwargs)

    def observe(
        self,
        task: asyncio.Task,
        name: str | None = None,
        source: str = "loop_factory",
        *,
        owner: str | None = None,
    ) -> asyncio.Task:
        """Observe a task created outside the tracker so it still gets cleaned up and audited."""
        self._attach(
            task,
            name=name,
            owner=owner,
            supervision="implicit",
            source=source,
        )
        return task

    def bounded_track(
        self,
        coro,
        name: str | None = None,
        *,
        owner: str | None = None,
        allow_during_shutdown: bool = False,
    ) -> asyncio.Task:
        """Track a task WITH concurrency limiting via semaphore.

        Use this for short-lived tasks (maintenance, learning, reflection).
        Long-running loops should use track() directly.
        """
        if _runtime_shutdown_requested() and not allow_during_shutdown:
            return self._suppress_shutdown_start(coro, name=name, source="bounded_track")

        async def _bounded():
            sem = self._get_semaphore()
            async with sem:
                if asyncio.iscoroutine(coro):
                    return await coro
                if inspect.iscoroutinefunction(coro):
                    return await coro()
                return await coro

        bounded_coro = _bounded()
        token = (
            begin_shutdown_task_creation_scope()
            if allow_during_shutdown
            else None
        )
        try:
            try:
                task = create_owned_asyncio_task(bounded_coro, name=name)
            except RuntimeError:
                close_awaitable(coro)
                close_awaitable(bounded_coro)
                raise
        finally:
            if token is not None:
                end_shutdown_task_creation_scope(token)
        if allow_during_shutdown:
            self._mark_shutdown_critical(task)
        with self._state_lock:
            self._total_tracked += 1
        self._attach(
            task,
            name=name,
            owner=owner,
            supervision="explicit",
            source="bounded_track",
        )
        return task

    def _suppress_shutdown_start(self, awaitable: Any, *, name: str | None, source: str) -> asyncio.Task:
        """Close late runtime work after shutdown starts and return a completed owned task.

        Shutdown is not a valid time for ordinary subsystems to spawn new
        inference, recovery, telemetry, or repair work. Returning a tiny
        completed task preserves call-site compatibility while preventing the
        original coroutine from running after executors and event loops begin
        teardown.
        """
        close_awaitable(awaitable)
        with self._state_lock:
            self._shutdown_suppressed_total += 1
        _record_shutdown_task_event(name=name, source=source, outcome="suppressed")

        async def _shutdown_suppressed() -> None:
            return None

        suppressed_coro = _shutdown_suppressed()
        try:
            task = create_owned_asyncio_task(
                suppressed_coro,
                name=name or f"{self.name}.shutdown_suppressed",
            )
        except RuntimeError:
            close_awaitable(suppressed_coro)
            raise
        try:
            task._aura_shutdown_suppressed = True
            task._aura_shutdown_suppressed_source = source
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("task_tracker", exc)
            logger.debug("TaskTracker[%s]: failed to annotate suppressed task: %s", self.name, exc)
        with self._state_lock:
            self._total_tracked += 1
        self._attach(
            task,
            name=name,
            owner="runtime_shutdown",
            supervision="explicit",
            source=f"{source}:shutdown_suppressed",
        )
        logger.debug(
            "TaskTracker[%s]: suppressed late task start during runtime shutdown (name=%s source=%s).",
            self.name,
            name or "",
            source,
        )
        return task

    def _mark_shutdown_critical(self, task: asyncio.Task[Any]) -> None:
        try:
            task._aura_shutdown_critical = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("task_tracker", exc)
            logger.debug(
                "TaskTracker[%s]: failed to mark shutdown-critical task: %s",
                self.name,
                exc,
            )

    def install_loop_hygiene(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install a task factory so raw asyncio.create_task/loop.create_task calls are still observed."""
        target_loop = loop or asyncio.get_running_loop()
        loop_id = id(target_loop)
        if loop_id in self._installed_loop_factories:
            return

        previous_factory = target_loop.get_task_factory()
        tracker = self

        def _factory(factory_loop, coro, **kwargs):
            skip_factory_track = _SKIP_FACTORY_TRACK.get()
            shutdown_suppressed = False
            canonical_shutdown = canonical_shutdown_awaitable(coro)
            teardown_allowed = shutdown_task_creation_allowed() or canonical_shutdown
            if (
                not skip_factory_track
                and _runtime_shutdown_requested()
                and not teardown_allowed
            ):
                close_awaitable(coro)
                with tracker._state_lock:
                    tracker._shutdown_suppressed_total += 1
                _record_shutdown_task_event(
                    name=str(kwargs.get("name") or "raw_asyncio_task"),
                    source="loop_factory",
                    outcome="suppressed",
                )

                async def _shutdown_suppressed() -> None:
                    return None

                coro = _shutdown_suppressed()
                shutdown_suppressed = True
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
            if shutdown_suppressed:
                try:
                    task._aura_shutdown_suppressed = True
                    task._aura_shutdown_suppressed_source = "loop_factory"
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("task_tracker", exc)
            elif (
                not skip_factory_track
                and _runtime_shutdown_requested()
                and teardown_allowed
            ):
                tracker._mark_shutdown_critical(task)
                _record_shutdown_task_event(
                    name=str(kwargs.get("name") or "shutdown_teardown_task"),
                    source=(
                        "loop_factory:canonical_shutdown_entrypoint"
                        if canonical_shutdown
                        else "loop_factory:explicit_shutdown_scope"
                    ),
                    outcome="allowed_teardown",
                )
            if not skip_factory_track:
                try:
                    tracker.observe(task, source="loop_factory")
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('task_tracker', exc)
                    logger.debug("TaskTracker[%s]: failed to observe loop task: %s", tracker.name, exc)
            return task

        target_loop.set_task_factory(_factory)
        self._installed_loop_factories[loop_id] = (target_loop, previous_factory)

    def restore_loop_hygiene(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Restore a loop's original task factory."""
        if loop is not None:
            info = self._installed_loop_factories.pop(id(loop), None)
            if info is not None:
                target_loop, previous_factory = info
                try:
                    target_loop.set_task_factory(previous_factory)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('task_tracker', exc)
                    logger.debug("TaskTracker[%s]: failed to restore loop factory: %s", self.name, exc)
            return

        for loop_id, info in list(self._installed_loop_factories.items()):
            target_loop, previous_factory = info
            try:
                target_loop.set_task_factory(previous_factory)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('task_tracker', exc)
                logger.debug("TaskTracker[%s]: failed to restore loop factory: %s", self.name, exc)
            finally:
                self._installed_loop_factories.pop(loop_id, None)

    def get_stale_tasks(self, min_age_s: float = 900.0, *, include_supervised: bool = False) -> list[dict[str, Any]]:
        """Return a sample of long-lived tasks that may need inspection."""
        now = time.monotonic()
        stale: list[dict[str, Any]] = []
        with self._state_lock:
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

    def heartbeat(self, task: asyncio.Task | None = None) -> None:
        """Register a heartbeat for the given task, or the current task if None."""
        target_task = task or asyncio.current_task()
        if not target_task:
            return
            
        with self._state_lock:
            record = self._records.get(id(target_task))
            if record:
                record.last_heartbeat = time.monotonic()

    def _mark_supervised(self, task: asyncio.Task) -> None:
        try:
            task._aura_supervised = True
            task._aura_task_tracker = self.name
            task._aura_task_supervision = "explicit"
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('task_tracker', e)
            logger.debug("TaskTracker[%s]: failed to mark task supervised: %s", self.name, e)

    def _attach(
        self,
        task: asyncio.Task,
        *,
        name: str | None,
        owner: str | None,
        supervision: str,
        source: str,
    ) -> None:
        # The tracked set is typed asyncio.Task and shutdown relies on
        # .cancel()/awaitability. A non-Task slipping in (observed: a test
        # double returned by a monkeypatched asyncio.create_task, with
        # done() pinned False) poisons the global tracker permanently --
        # every later shutdown raised AttributeError and erred ten
        # unrelated teardowns in the chunked suite. Refuse loudly.
        if not isinstance(task, asyncio.Task):
            record_degradation(
                "task_tracker",
                TypeError(f"refused non-Task attach: {type(task).__name__}"),
                severity="warning",
                action="ignored non-Task object handed to tracker",
                extra={"source": source, "name": str(name or "")},
            )
            return
        coroutine = self._describe_task(task)
        runtime_name = str(task.get_name() or "").strip()
        task_name = str(name or "").strip()
        if not task_name:
            task_name = coroutine if runtime_name.startswith("Task-") else runtime_name
        task_name = task_name or "unknown_task"
        task_owner = str(owner or "").strip() or coroutine or "unknown_owner"
        task_id = id(task)
        with self._state_lock:
            record = self._records.get(task_id)
            if record is not None and (
                getattr(task, "_aura_task_lifecycle_id", None) != record.lifecycle_id
            ):
                # Completed task records intentionally outlive their Task objects for
                # bounded diagnostics. CPython may reuse the object id while that
                # record is retained; treating the new task as the old one skips the
                # done callback and leaves its exception unobserved.
                record = None
            if record is None:
                lifecycle_sequence = next(self._lifecycle_sequence)
                record = TaskRecord(
                    lifecycle_id=(
                        f"{self.name}:{self._tracker_instance_id}:"
                        f"{lifecycle_sequence:012d}"
                    ),
                    task_id=task_id,
                    name=task_name,
                    owner=task_owner,
                    tracker=self.name,
                    supervision=supervision,
                    source=source,
                    created_at=time.monotonic(),
                    coroutine=coroutine,
                )
                self._records[task_id] = record
                self.tasks.add(task)
                task.add_done_callback(self._on_task_done)
                self._total_observed += 1
            else:
                if name:
                    record.name = task_name
                if owner:
                    record.owner = task_owner
                if record.source == "loop_factory" and source != "loop_factory":
                    record.source = source
                if supervision == "explicit":
                    record.supervision = "explicit"

            task_done = task.done()
            if not task_done:
                self._high_water = max(self._high_water, len(self.tasks))

        try:
            task._aura_task_tracker = self.name
            task._aura_task_supervision = record.supervision
            task._aura_task_source = record.source
            task._aura_task_created_at = record.created_at
            task._aura_task_lifecycle_id = record.lifecycle_id
            task._aura_task_owner = record.owner
            if record.supervision == "explicit":
                self._mark_supervised(task)
            elif not hasattr(task, "_aura_supervised"):
                task._aura_supervised = False
        except (RuntimeError, AttributeError, TypeError) as exc:
            record_degradation('task_tracker', exc)
            logger.debug("TaskTracker[%s]: failed to annotate task: %s", self.name, exc)

        if task_done:
            self._on_task_done(task)

    def _describe_task(self, task: asyncio.Task) -> str:
        try:
            coro = task.get_coro()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return "unknown"
        qualname = getattr(coro, "__qualname__", None)
        if qualname:
            return qualname
        return repr(coro)

    def _on_task_done(self, task: asyncio.Task) -> None:
        cancelled = bool(task.cancelled())
        terminal_exception: BaseException | None = None
        observation_error: BaseException | None = None
        if not cancelled:
            try:
                terminal_exception = task.exception()
            except asyncio.CancelledError:
                cancelled = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                observation_error = exc

        with self._state_lock:
            self.tasks.discard(task)
            record = self._records.get(id(task))
            if record is None or record.done:
                # Reading task.exception() above is deliberate even at this early
                # return. It prevents a missing/stale record from becoming
                # "Task exception was never retrieved" at garbage collection.
                return

            record.done = True
            record.finished_at = time.monotonic()
            self._completed_total += 1

            if cancelled:
                record.cancelled = True
                record.outcome = "cancelled"
                self._cancelled_total += 1
            elif observation_error is not None:
                record.failed = True
                record.outcome = "observation_error"
                record.exception = (
                    f"{type(observation_error).__name__}: {observation_error}"
                )
                self._failed_total += 1
            elif terminal_exception is not None:
                record.failed = True
                record.outcome = "failed"
                record.exception = (
                    f"{type(terminal_exception).__name__}: {terminal_exception}"
                )
                self._failed_total += 1
            else:
                record.outcome = "succeeded"

            terminal_receipt = record.to_dict()
            self._recently_completed.append(terminal_receipt)

            if len(self._records) > self._max_records_in_memory:
                completed_records = [
                    (task_id, candidate)
                    for task_id, candidate in self._records.items()
                    if candidate.done and candidate.finished_at is not None
                ]
                if completed_records:
                    completed_records.sort(
                        key=lambda item: item[1].finished_at or 0
                    )
                    remove_count = max(1, len(completed_records) // 4)
                    for task_id, _candidate in completed_records[:remove_count]:
                        del self._records[task_id]

        if observation_error is not None:
            try:
                record_degradation("task_tracker", observation_error)
            except RuntimeError as degradation_error:
                logger.error(
                    "TaskTracker[%s]: terminal observation degradation failed: %s",
                    self.name,
                    degradation_error,
                )
        elif terminal_exception is not None:
            logger.warning(
                "TaskTracker[%s]: task %s failed: %s",
                self.name,
                record.name,
                record.exception,
                extra={"aura_task_terminal": terminal_receipt},
            )
        logger.debug(
            "TaskTracker[%s]: terminal task receipt %s outcome=%s owner=%s name=%s",
            self.name,
            record.lifecycle_id,
            record.outcome,
            record.owner,
            record.name,
            extra={"aura_task_terminal": terminal_receipt},
        )

    @property
    def active_count(self) -> int:
        """Number of currently active (not done) tasks."""
        with self._state_lock:
            return len(self.tasks)

    async def shutdown(self, timeout: float = 5.0) -> dict[str, Any]:  # noqa: ASYNC109
        """Cancel and wait for all tracked tasks.

        Tasks marked ``_aura_protected`` are cancelled after ordinary tracked
        work so shutdown can drain short-lived background jobs first. Protection
        never means "leave this task alive"; a clean runtime shutdown must not
        strand scheduler, substrate, or watchdog loops behind the caller.
        Coordinator/finalizer tasks marked ``_aura_shutdown_critical`` are
        excluded because cancelling the teardown owner would make cleanup lie
        about completion.
        """
        started = time.monotonic()
        current_task = asyncio.current_task()
        with self._state_lock:
            all_pending = {
                task
                for task in self.tasks
                if not task.done() and task is not current_task
            }
        shutdown_critical = {
            task for task in all_pending if getattr(task, "_aura_shutdown_critical", False)
        }
        pending = all_pending - shutdown_critical
        if not pending:
            return {
                "clean": True,
                "requested": 0,
                "cancelled": 0,
                "remaining": 0,
                "remaining_tasks": [],
                "shutdown_critical_active": len(shutdown_critical),
                "duration_seconds": round(time.monotonic() - started, 6),
            }

        ordinary = {task for task in pending if not getattr(task, "_aura_protected", False)}
        protected = pending - ordinary
        deadline = time.monotonic() + max(0.0, float(timeout))

        async def _cancel_group(
            group: set[asyncio.Task[Any]],
            label: str,
            *,
            budget_fraction: float,
        ) -> None:
            if not group:
                return
            logger.info(
                "TaskTracker[%s]: cancelling %s %s task(s) during shutdown.",
                self.name,
                len(group),
                label,
            )

            current_loop = asyncio.get_running_loop()
            for task in group:
                try:
                    owner_loop = task.get_loop()
                    if owner_loop is current_loop or not owner_loop.is_running():
                        task.cancel()
                    else:
                        owner_loop.call_soon_threadsafe(task.cancel)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("task_tracker", exc)
                    logger.debug(
                        "TaskTracker[%s]: failed to cancel %s task %s: %s",
                        self.name,
                        label,
                        task.get_name(),
                        exc,
                    )

            remaining_budget = max(0.0, deadline - time.monotonic())
            group_budget = remaining_budget * budget_fraction
            local_tasks = {task for task in group if task.get_loop() is current_loop}
            try:
                if local_tasks and group_budget > 0:
                    await asyncio.wait(local_tasks, timeout=group_budget)
                foreign_deadline = min(deadline, time.monotonic() + group_budget)
                foreign_tasks = [task for task in group if task.get_loop() is not current_loop]
                if foreign_tasks and foreign_deadline > time.monotonic():
                    foreign_completion = asyncio.Event()

                    def _notify_foreign_completion(_task: asyncio.Task[Any]) -> None:
                        current_loop.call_soon_threadsafe(foreign_completion.set)

                    for task in foreign_tasks:
                        task.add_done_callback(_notify_foreign_completion)
                    try:
                        while any(not task.done() for task in foreign_tasks):
                            remaining = foreign_deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            foreign_completion.clear()
                            if any(not task.done() for task in foreign_tasks):
                                async with asyncio.timeout(remaining):
                                    await foreign_completion.wait()
                    finally:
                        for task in foreign_tasks:
                            task.remove_done_callback(_notify_foreign_completion)
            except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as e:
                record_degradation('task_tracker', e)
                logger.error("Error during TaskTracker shutdown: %s", e)

        await _cancel_group(
            ordinary,
            "ordinary",
            budget_fraction=0.6 if protected else 1.0,
        )
        await _cancel_group(protected, "protected", budget_fraction=1.0)

        remaining = [task for task in pending if not task.done()]
        if remaining:
            logger.warning("%d tasks still pending after bounded cancellation.", len(remaining))
        remaining_tasks = []
        try:
            from core.runtime.how_a_task_should_end import the_policy_for
        except ImportError:  # a tracker in a process without the runtime
            the_policy_for = None  # type: ignore[assignment]
        for task in remaining[:20]:
            with self._state_lock:
                record = self._records.get(id(task))
            owner = record.source if record is not None else "unknown"
            _record_shutdown_task_event(
                name=record.name if record is not None else task.get_name(),
                source=owner,
                outcome="survived",
            )
            # Whether this survival is a defect is the owner's declaration, not
            # this loop's guess. A curiosity task outliving shutdown costs
            # nothing; a write that outlives it is how a state file is
            # truncated, and both used to be one line saying "survived".
            told = the_policy_for(owner) if the_policy_for is not None else None
            remaining_tasks.append(
                {
                    "name": record.name if record is not None else task.get_name(),
                    "source": owner,
                    "supervision": record.supervision if record is not None else "unknown",
                    "loop_running": task.get_loop().is_running(),
                    "an_orphan_is_a_defect": (
                        bool(told.an_orphan_is_a_defect) if told is not None else None
                    ),
                    "why_it_matters": told.why if told is not None else "",
                }
            )
        return {
            "clean": not remaining,
            "requested": len(pending),
            "cancelled": sum(1 for task in pending if task.cancelled()),
            "remaining": len(remaining),
            "remaining_tasks": remaining_tasks,
            "shutdown_critical_active": len(shutdown_critical),
            "duration_seconds": round(time.monotonic() - started, 6),
        }

    def cleanup_old_records(self, max_age_s: float = 300.0) -> int:
        """Explicitly clean up task records older than max_age_s.
        
        Called periodically to prevent unbounded memory growth from completed tasks.
        """
        now = time.monotonic()
        removed = 0
        with self._state_lock:
            for task_id in list(self._records.keys()):
                record = self._records[task_id]
                if record.done and record.finished_at is not None:
                    age = now - record.finished_at
                    if age > max_age_s:
                        del self._records[task_id]
                        removed += 1
        if removed > 0:
            logger.debug("TaskTracker[%s]: cleaned up %d old records", self.name, removed)
        return removed

    def get_stats(self) -> dict:
        explicit_active = 0
        implicit_active = 0
        shutdown_critical_active = 0
        with self._state_lock:
            active_tasks = list(self.tasks)
            for task in active_tasks:
                record = self._records.get(id(task))
                if record is None:
                    continue
                if record.supervision == "explicit":
                    explicit_active += 1
                else:
                    implicit_active += 1
                if getattr(task, "_aura_shutdown_critical", False):
                    shutdown_critical_active += 1
            counters = {
                "active": len(active_tasks),
                "high_water": self._high_water,
                "total_tracked": self._total_tracked,
                "total_observed": self._total_observed,
                "completed_total": self._completed_total,
                "cancelled_total": self._cancelled_total,
                "failed_total": self._failed_total,
                "shutdown_suppressed_total": self._shutdown_suppressed_total,
                "loop_semaphore_count": len(self._semaphores),
                "recently_completed": list(self._recently_completed)[-5:],
            }
        stale_tasks = self.get_stale_tasks(min_age_s=300.0)
        return {
            **counters,
            "explicit_active": explicit_active,
            "implicit_active": implicit_active,
            "shutdown_critical_active": shutdown_critical_active,
            "max_concurrent": self._max_concurrent,
            "stale_tasks": stale_tasks[:5],
        }

    def get_active_task_snapshot(
        self,
        *,
        exclude_current: bool = False,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return bounded evidence for tasks still alive at a final boundary."""

        current: asyncio.Task[Any] | None = None
        if exclude_current:
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
        with self._state_lock:
            active = [
                task
                for task in list(self.tasks)
                if not task.done() and task is not current
            ]
            samples: list[dict[str, Any]] = []
            for task in active[: max(0, int(limit))]:
                record = self._records.get(id(task))
                samples.append(
                    {
                        "name": record.name if record is not None else task.get_name(),
                        "owner": record.owner if record is not None else "unknown",
                        "lifecycle_id": (
                            record.lifecycle_id if record is not None else "unknown"
                        ),
                        "source": record.source if record is not None else "unknown",
                        "supervision": (
                            record.supervision if record is not None else "unknown"
                        ),
                        "shutdown_critical": bool(
                            getattr(task, "_aura_shutdown_critical", False)
                        ),
                        "protected": bool(getattr(task, "_aura_protected", False)),
                        "loop_running": task.get_loop().is_running(),
                    }
                )
        return {"count": len(active), "tasks": samples}


_task_tracker = TaskTracker(name="Global")


def get_task_tracker() -> TaskTracker:
    """Return the canonical process-wide task tracker.

    Both compatibility paths in this module must resolve to the same object.
    Otherwise tasks can be supervised by one tracker while shutdown, health
    checks, or imports of ``task_tracker`` query a different tracker.
    """
    return _task_tracker


# Backward compatibility for modules that import ``task_tracker`` directly.
task_tracker = _task_tracker


def fire_and_track(coro, name: str | None = None) -> asyncio.Task:
    """Convenience function to create and track a task in one go."""
    tracker = get_task_tracker()
    return tracker.track(coro, name=name)
