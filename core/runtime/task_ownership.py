"""Canonical task ownership helpers for Aura.

Use this instead of raw asyncio.create_task in production code.
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Runtime.TaskOwnership")


@dataclass(frozen=True, slots=True)
class OwnedTaskDrain[T]:
    """A tracked task drained to completion despite caller cancellation."""

    task: asyncio.Task[T]
    cancellation: asyncio.CancelledError | None = None


def _raw_asyncio_create_task(awaitable: Awaitable[Any], *, name: str | None, context: Any = None) -> asyncio.Task:
    create_task = getattr(asyncio.create_task, "__wrapped__", asyncio.create_task)
    try:
        if context is not None:
            return create_task(awaitable, name=name, context=context)
    except TypeError:
        # not a failure: an event loop whose create_task has no `context`
        # keyword takes the two-argument call below instead.
        pass
    if name is not None:
        return create_task(awaitable, name=name)
    return create_task(awaitable)


def _get_tracker() -> Any:
    try:
        from core.utils.task_tracker import get_task_tracker
        return get_task_tracker()
    except (ImportError, AttributeError, RuntimeError):
        # not a failure: no tracker to register with, and every caller
        # checks for None before registering.
        return None


def close_awaitable(awaitable: Any) -> None:
    if inspect.iscoroutine(awaitable):
        awaitable.close()
        return
    cancel = getattr(awaitable, "cancel", None)
    if callable(cancel):
        with suppress(Exception):
            cancel()


def _create_owned_asyncio_task(awaitable: Awaitable[Any], *, name: str | None) -> asyncio.Task:
    """Create a fallback task while preserving strict task-owner semantics."""

    try:
        from core.utils.task_tracker import _SKIP_FACTORY_TRACK
    except (ImportError, AttributeError, RuntimeError):
        return _raw_asyncio_create_task(awaitable, name=name)

    child_context = contextvars.copy_context()
    child_context.run(_SKIP_FACTORY_TRACK.set, False)
    token = _SKIP_FACTORY_TRACK.set(True)
    try:
        try:
            return _raw_asyncio_create_task(awaitable, name=name, context=child_context)
        except TypeError:
            return _raw_asyncio_create_task(awaitable, name=name)
    finally:
        _SKIP_FACTORY_TRACK.reset(token)


def create_owned_asyncio_task(awaitable: Awaitable[Any], *, name: str | None = None) -> asyncio.Task:
    """Create a raw asyncio task inside the canonical task-ownership sink."""
    return _create_owned_asyncio_task(awaitable, name=name)


def create_tracked_task(
    awaitable: Awaitable[Any],
    *,
    name: str | None = None,
    owner: str | None = None,
    bounded: bool = False,
    on_done: Callable[[asyncio.Task], Any] | None = None,
    cancel_on_fail: bool = True,
    allow_during_shutdown: bool = False,
) -> asyncio.Task:
    tracker = _get_tracker()
    task: asyncio.Task | None = None
    task_kwargs: dict[str, Any] = {"name": name}
    if owner is not None:
        task_kwargs["owner"] = owner
    if allow_during_shutdown:
        task_kwargs["allow_during_shutdown"] = True
    try:
        if tracker is not None:
            if bounded and hasattr(tracker, "bounded_track"):
                task = tracker.bounded_track(awaitable, **task_kwargs)
            elif hasattr(tracker, "create_task"):
                task = tracker.create_task(awaitable, **task_kwargs)
            elif hasattr(tracker, "track_task"):
                task = tracker.track_task(awaitable, **task_kwargs)
            elif hasattr(tracker, "track"):
                task = tracker.track(awaitable, **task_kwargs)

        if task is None:
            task = _create_owned_asyncio_task(awaitable, name=name)
            if allow_during_shutdown:
                try:
                    task._aura_shutdown_critical = True
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation("task_ownership", exc)
                    logger.debug(
                        "Failed to annotate fallback shutdown-critical task %s: %s",
                        name or task,
                        exc,
                    )
            if tracker is not None:
                try:
                    if hasattr(tracker, "observe"):
                        observe_kwargs: dict[str, Any] = {
                            "name": name,
                            "source": "task_ownership_fallback",
                        }
                        if owner is not None:
                            observe_kwargs["owner"] = owner
                        tracker.observe(task, **observe_kwargs)
                    elif hasattr(tracker, "track_task"):
                        tracker.track_task(task, name=name)
                except (RuntimeError, AttributeError, TypeError) as exc:
                    record_degradation('task_ownership', exc)
                    logger.debug("Failed to observe fallback task %s: %s", name or task, exc)

        if on_done is not None:
            task.add_done_callback(on_done)
        return task
    except (RuntimeError, AttributeError, TypeError):
        if cancel_on_fail:
            close_awaitable(awaitable)
        raise


async def drain_owned_awaitable[T](
    awaitable: Awaitable[T],
    *,
    name: str,
    owner: str,
    allow_during_shutdown: bool = False,
) -> OwnedTaskDrain[T]:
    """Observe non-cancellable work through completion under cancellation.

    The returned task is terminal but deliberately unresolved so the caller can
    apply domain-specific failure precedence before propagating the captured
    caller cancellation. `allow_during_shutdown` is only for work admitted
    before the shutdown latch that must finish to preserve durability or close
    authority; callers must reject new work before using it.
    """

    task = create_tracked_task(
        awaitable,
        name=name,
        owner=owner,
        allow_during_shutdown=allow_during_shutdown,
    )
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    return OwnedTaskDrain(task=task, cancellation=cancellation)


def runtime_shutdown_requested() -> bool:
    """Return the monotonic runtime shutdown latch without cold construction."""

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        return bool(is_shutdown_requested())
    except (ImportError, AttributeError, RuntimeError):
        # not a failure: the docstring says without cold construction, and
        # no coordinator means no shutdown has been requested.
        return False


def runtime_shutdown_blocks_new_work(
    operation: str,
    *,
    resource_kind: str,
) -> bool:
    """Refuse and receipt new work after the monotonic shutdown latch."""

    if not runtime_shutdown_requested():
        return False
    try:
        from core.runtime.shutdown_coordinator import record_shutdown_admission_event

        record_shutdown_admission_event(
            operation,
            resource_kind=resource_kind,
            outcome="suppressed",
            detail="new work refused after runtime shutdown request",
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # The work is refused below either way. What is lost is the
        # admission event saying so, which is the only record that this
        # shutdown turned work away.
        logger.warning("A shutdown refusal was not recorded: %s", exc)
    return True


def fire_and_forget(
    awaitable: Awaitable[Any],
    *,
    name: str | None = None,
    owner: str | None = None,
    bounded: bool = False,
    log_exceptions: bool = True,
) -> asyncio.Task | None:
    def _log_done(task: asyncio.Task) -> None:
        if not log_exceptions:
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            # not a failure: a cancelled task has no exception to log, and
            # cancellation is how most owned tasks end.
            return
        except (RuntimeError, AttributeError, TypeError, ValueError):
            # not a failure: a task still running has none to read either.
            return
        if exc is not None:
            logger.warning("Background task %s failed: %s", name or task.get_name(), exc, exc_info=exc)

    try:
        return create_tracked_task(
            awaitable,
            name=name,
            owner=owner,
            bounded=bounded,
            on_done=_log_done,
        )
    except RuntimeError:
        # not a failure: no running loop to schedule on. The coroutine is
        # closed rather than left unawaited, which is the whole reason this
        # wrapper exists.
        close_awaitable(awaitable)
        return None
    except (AttributeError, TypeError, ValueError) as exc:
        record_degradation('task_ownership', exc)
        close_awaitable(awaitable)
        logger.debug("fire_and_forget scheduling failed for %s: %s", name, exc)
        return None
