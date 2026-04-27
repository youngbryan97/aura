"""Canonical task ownership helpers for Aura.

Use this instead of raw asyncio.create_task in production code.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import suppress
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("Aura.Runtime.TaskOwnership")


def _get_tracker() -> Any:
    try:
        from core.utils.task_tracker import get_task_tracker
        return get_task_tracker()
    except Exception:
        return None


def close_awaitable(awaitable: Any) -> None:
    if inspect.iscoroutine(awaitable):
        awaitable.close()
        return
    cancel = getattr(awaitable, "cancel", None)
    if callable(cancel):
        with suppress(Exception):
            cancel()


def create_tracked_task(
    awaitable: Awaitable[Any],
    *,
    name: Optional[str] = None,
    bounded: bool = False,
    on_done: Optional[Callable[[asyncio.Task], Any]] = None,
    cancel_on_fail: bool = True,
) -> asyncio.Task:
    tracker = _get_tracker()
    task: Optional[asyncio.Task] = None
    try:
        if tracker is not None:
            if bounded and hasattr(tracker, "bounded_track"):
                task = tracker.bounded_track(awaitable, name=name)
            elif hasattr(tracker, "create_task"):
                task = tracker.create_task(awaitable, name=name)
            elif hasattr(tracker, "track_task"):
                task = tracker.track_task(awaitable, name=name)
            elif hasattr(tracker, "track"):
                task = tracker.track(awaitable, name=name)

        if task is None:
            task = asyncio.create_task(awaitable, name=name)
            if tracker is not None:
                try:
                    if hasattr(tracker, "observe"):
                        tracker.observe(task, name=name, source="task_ownership_fallback")
                    elif hasattr(tracker, "track_task"):
                        tracker.track_task(task, name=name)
                except Exception as exc:
                    logger.debug("Failed to observe fallback task %s: %s", name or task, exc)

        if on_done is not None:
            task.add_done_callback(on_done)
        return task
    except Exception:
        if cancel_on_fail:
            close_awaitable(awaitable)
        raise


def fire_and_forget(
    awaitable: Awaitable[Any],
    *,
    name: Optional[str] = None,
    bounded: bool = False,
    log_exceptions: bool = True,
) -> Optional[asyncio.Task]:
    def _log_done(task: asyncio.Task) -> None:
        if not log_exceptions:
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception:
            return
        if exc is not None:
            logger.warning("Background task %s failed: %s", name or task.get_name(), exc, exc_info=exc)

    try:
        return create_tracked_task(awaitable, name=name, bounded=bounded, on_done=_log_done)
    except RuntimeError:
        close_awaitable(awaitable)
        return None
    except Exception as exc:
        close_awaitable(awaitable)
        logger.debug("fire_and_forget scheduling failed for %s: %s", name, exc)
        return None
