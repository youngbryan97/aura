"""Global asyncio task-supervision patch for Aura.

Aura has a strong runtime-hygiene/TaskTracker story, but a large historical
codebase inevitably contains direct ``get_task_tracker().create_task(...)`` calls. This
module makes those calls safer by routing them through ``TaskTracker`` when it
is available. The patch is idempotent, re-entrancy guarded, and fail-open during
earliest boot so it cannot prevent Aura from starting.

Install as early as possible in ``aura_main.py``::

    try:
        import core.utils.asyncio_patch  # noqa: F401
    except Exception:
        pass  # no-op: intentional
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import contextvars
import logging
from typing import Any, Optional

logger = logging.getLogger("Aura.AsyncioPatch")

_ORIGINAL_CREATE_TASK = getattr(asyncio, "create_task", None)
_REENTRY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "aura_asyncio_patch_reentry",
    default=False,
)


def _call_original(coro: Any, *, name: Optional[str] = None, context: Any = None) -> asyncio.Task:
    if _ORIGINAL_CREATE_TASK is None:
        raise RuntimeError("asyncio.create_task is unavailable")
    try:
        if context is not None:
            return _ORIGINAL_CREATE_TASK(coro, name=name, context=context)  # type: ignore[misc]
    except TypeError:
        pass  # no-op: intentional
    if name is not None:
        return _ORIGINAL_CREATE_TASK(coro, name=name)  # type: ignore[misc]
    return _ORIGINAL_CREATE_TASK(coro)  # type: ignore[misc]


def install_asyncio_task_patch() -> bool:
    """Install the task patch once. Returns True when installed/active."""
    if _ORIGINAL_CREATE_TASK is None:
        return False
    if getattr(asyncio.create_task, "__aura_task_patch__", False):
        return True

    def _patched_create_task(coro: Any, *, name: Optional[str] = None, context: Any = None) -> asyncio.Task:
        if _REENTRY.get():
            return _call_original(coro, name=name, context=context)

        try:
            from core.utils.task_tracker import get_task_tracker
            tracker = get_task_tracker()
        except Exception:
            tracker = None

        if tracker is None:
            return _call_original(coro, name=name, context=context)

        token = _REENTRY.set(True)
        try:
            return tracker.create_task(coro, name=name)
        except Exception as exc:
            record_degradation('asyncio_patch', exc)
            logger.debug("TaskTracker create_task fallback: %s", exc)
            return _call_original(coro, name=name, context=context)
        finally:
            _REENTRY.reset(token)

    _patched_create_task.__aura_task_patch__ = True  # type: ignore[attr-defined]
    _patched_create_task.__wrapped__ = _ORIGINAL_CREATE_TASK  # type: ignore[attr-defined]
    asyncio.create_task = _patched_create_task  # type: ignore[assignment]
    logger.debug("Installed Aura asyncio task supervision patch.")
    return True


install_asyncio_task_patch()
