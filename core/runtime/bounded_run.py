"""asyncio.run with a bound on the wait for cancelled tasks, and their names.

``asyncio.run`` finishes by cancelling every task still alive and waiting
for all of them, forever. A task that does not end on cancellation — one
that awaits a thread that never returns, or catches the cancellation and
keeps looping — holds the process open with the main thread in
``_cancel_all_tasks``. LIVE, 2026-09-16: a runtime whose root coroutine
had returned sat like that for fifteen minutes, port closed, voice thread
still logging, no line saying why, until it was dumped by hand.

The wait here is bounded by the process shutdown budget the coordinator
already declares, and a task that outlives it is named with its coroutine
and its current frame before the loop is closed underneath it.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger("Aura.BoundedRun")

T = TypeVar("T")

#: What the wait falls back to when the coordinator has no budget (a run
#: that never requested shutdown): the coordinator's own first-request
#: default, read at call time so an operator override applies.
_FALLBACK_BUDGET_S = 180.0


def _cancellation_budget_s() -> float:
    try:
        from core.runtime.shutdown_coordinator import (
            _shutdown_budget_seconds,
            shutdown_remaining_budget_seconds,
        )

        remaining = shutdown_remaining_budget_seconds()
        if remaining is not None:
            # Something is always allowed, so the stragglers get named.
            return max(5.0, float(remaining))
        return float(_shutdown_budget_seconds(first_request=True))
    except (ImportError, AttributeError, TypeError, ValueError):
        return _FALLBACK_BUDGET_S


def _describe(task: asyncio.Task[Any]) -> str:
    coro = task.get_coro()
    name = task.get_name()
    where = "unknown"
    try:
        frames = task.get_stack(limit=1)
        if frames:
            frame = frames[-1]
            where = f"{frame.f_code.co_filename}:{frame.f_lineno} {frame.f_code.co_name}"
    # not a failure: a task that has not started, or has already finished, has
    # no frame to name, and "unknown" is the honest description of it.
    except (RuntimeError, AttributeError):
        pass
    return f"{name} <{getattr(coro, '__qualname__', repr(coro))}> at {where}"


async def _cancel_and_wait(loop: asyncio.AbstractEventLoop, budget_s: float) -> list[str]:
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop) and not t.done()]
    if not tasks:
        return []
    for task in tasks:
        task.cancel()
    done, pending = await asyncio.wait(tasks, timeout=budget_s)
    for task in done:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during shutdown",
                    "exception": exc,
                    "task": task,
                }
            )
    return [_describe(task) for task in pending]


def run(main: Coroutine[Any, Any, T], *, debug: bool | None = None) -> T:
    """Run ``main`` to completion the way ``asyncio.run`` does, with a bounded close."""
    if asyncio._get_running_loop() is not None:  # type: ignore[attr-defined]
        raise RuntimeError("bounded_run.run() cannot be called from a running event loop")
    loop = asyncio.new_event_loop()
    stragglers: list[str] = []
    try:
        asyncio.set_event_loop(loop)
        if debug is not None:
            loop.set_debug(debug)
        return loop.run_until_complete(main)
    finally:
        try:
            budget = _cancellation_budget_s()
            stragglers = loop.run_until_complete(_cancel_and_wait(loop, budget))
            if stragglers:
                logger.warning(
                    "Exit: %d task(s) did not end within %.0fs of cancellation; closing the "
                    "loop underneath them:\n%s",
                    len(stragglers),
                    budget,
                    "\n".join(f"  - {line}" for line in stragglers)[:12000],
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor(timeout=min(budget, 30.0)))
        except Exception as exc:  # noqa: BLE001 — the close must not replace the run's outcome
            logger.error("Exit: loop close failed: %s\n%s", exc, traceback.format_exc()[-2000:])
        finally:
            asyncio.set_event_loop(None)
            loop.close()


__all__ = ["run"]
