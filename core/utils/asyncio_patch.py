"""Global asyncio task-supervision patch for Aura.

Aura has a strong runtime-hygiene/TaskTracker story, but a large historical
codebase inevitably contains direct ``get_task_tracker().create_task(...)`` calls. This
module makes those calls safer by routing them through ``TaskTracker`` when it
is available. The patch is idempotent, re-entrancy guarded, and fail-open during
earliest boot so it cannot prevent Aura from starting.

Install as early as possible in ``aura_main.py``::

    try:
        import core.utils.asyncio_patch  # noqa: F401
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.AsyncioPatch")

_original_create_task = getattr(asyncio, "create_task", None)
_ORIGINAL_CREATE_TASK: Callable[..., asyncio.Task[Any]] | None = (
    cast(Callable[..., asyncio.Task[Any]], _original_create_task)
    if callable(_original_create_task)
    else None
)
_REENTRY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aura_asyncio_patch_reentry",
    default=False,
)


def _call_original(
    coro: Awaitable[Any],
    *,
    name: str | None = None,
    context: contextvars.Context | None = None,
) -> asyncio.Task[Any]:
    if _ORIGINAL_CREATE_TASK is None:
        raise RuntimeError("asyncio.create_task is unavailable")
    try:
        if context is not None:
            return _ORIGINAL_CREATE_TASK(coro, name=name, context=context)
    # not a failure: an event loop whose create_task predates the context keyword
    # takes the call below instead.
    except TypeError:
        pass  # no-op: intentional
    if name is not None:
        return _ORIGINAL_CREATE_TASK(coro, name=name)
    return _ORIGINAL_CREATE_TASK(coro)


def install_asyncio_task_patch() -> bool:
    """Install the task patch once. Returns True when installed/active."""
    if _ORIGINAL_CREATE_TASK is None:
        return False
    if getattr(asyncio.create_task, "__aura_task_patch__", False):
        return True

    def _patched_create_task(
        coro: Awaitable[Any],
        *,
        name: str | None = None,
        context: contextvars.Context | None = None,
    ) -> asyncio.Task[Any]:
        if _REENTRY.get():
            return _call_original(coro, name=name, context=context)

        try:
            from core.utils.task_tracker import (
                canonical_shutdown_awaitable,
                get_task_tracker,
                shutdown_task_creation_allowed,
            )
            tracker = get_task_tracker()
        # not a failure: no task tracker yet, and the plain create_task below is used.
        except (ImportError, AttributeError, RuntimeError):
            tracker = None

        if tracker is None:
            return _call_original(coro, name=name, context=context)

        token = _REENTRY.set(True)
        try:
            allow_during_shutdown = bool(
                shutdown_task_creation_allowed(context)
                or canonical_shutdown_awaitable(coro)
            )
            return cast(
                asyncio.Task[Any],
                tracker.create_task(
                    coro,
                    name=name,
                    allow_during_shutdown=allow_during_shutdown,
                ),
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('asyncio_patch', exc)
            logger.debug("TaskTracker create_task fallback: %s", exc)
            return _call_original(coro, name=name, context=context)
        finally:
            _REENTRY.reset(token)

    _patched_create_task.__aura_task_patch__ = True  # type: ignore[attr-defined]
    _patched_create_task.__wrapped__ = _ORIGINAL_CREATE_TASK  # type: ignore[attr-defined]
    asyncio.create_task = _patched_create_task
    logger.debug("Installed Aura asyncio task supervision patch.")
    return True


install_asyncio_task_patch()
