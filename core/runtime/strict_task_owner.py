"""Strict-runtime task ownership enforcement.

Audit constraint: in AURA_STRICT_RUNTIME=1 there must be no unowned
``asyncio.create_task`` / ``loop.create_task`` calls in core code paths.

This module installs a task factory on the event loop that, whenever a
task is created outside the canonical TaskTracker, records a degraded
event and (in strict mode) raises so tests can prove the contract.

The factory cooperates with TaskTracker: tasks created via the tracker
set a context variable so the factory knows to allow them.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("Aura.StrictTaskOwner")


_STRICT_VIOLATIONS: list = []
# Re-export the flag the TaskTracker uses to mark tracker-managed creations.
try:
    from core.utils.task_tracker import _SKIP_FACTORY_TRACK  # type: ignore
except Exception:  # pragma: no cover - defensive
    _SKIP_FACTORY_TRACK = contextvars.ContextVar("aura_skip_factory_track", default=False)


def install_strict_task_owner(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Install a strict task factory on ``loop``."""
    loop = loop or asyncio.get_event_loop()
    previous = loop.get_task_factory()

    def _factory(factory_loop, coro, context=None):
        def _create(l, c, ctx):
            if previous is not None:
                try:
                    return previous(l, c, context=ctx)
                except TypeError:
                    return previous(l, c)
            return asyncio.Task(c, loop=l, context=ctx)

        if _SKIP_FACTORY_TRACK.get():
            # Tracker-managed task — allowed.
            return _create(factory_loop, coro, context)
        # Unowned creation: record a violation.
        violation = {
            "coro": getattr(coro, "__qualname__", repr(coro)),
            "stack_summary": _capture_short_stack(),
        }
        _STRICT_VIOLATIONS.append(violation)
        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event("strict_runtime.unowned_task", violation)
        except Exception:
            pass
        if os.environ.get("AURA_STRICT_RUNTIME") == "1":
            # Cancel the coroutine to avoid silent leakage; raise so the
            # caller sees a clean failure.
            if hasattr(coro, "close"):
                try:
                    coro.close()
                except Exception:
                    pass
            raise RuntimeError(
                f"AURA_STRICT_RUNTIME: unowned asyncio.create_task: {violation['coro']}"
            )
        # Non-strict: still create the task so callers do not crash.
        return _create(factory_loop, coro, context)

    loop.set_task_factory(_factory)
    loop._aura_previous_factory = previous  # type: ignore[attr-defined]


def restore_strict_task_owner(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    loop = loop or asyncio.get_event_loop()
    previous = getattr(loop, "_aura_previous_factory", None)
    loop.set_task_factory(previous)


def violations() -> list:
    return list(_STRICT_VIOLATIONS)


def reset_violations() -> None:
    _STRICT_VIOLATIONS.clear()


def _capture_short_stack(limit: int = 6) -> list:
    import traceback

    return traceback.format_stack(limit=limit)
