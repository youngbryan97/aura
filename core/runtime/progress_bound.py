"""A wait that ends on a wedge, not on a slow host.

A boot step that runs on a thread was awaited with a fixed wall budget. On
an idle host the whole step took two seconds; on a host three times
oversubscribed by other processes it took seventeen, the budget was fifteen,
and the boot died at the last organ with the work all but done (2026-09-16,
load 34 on 18 cores, boot offset +286s).

A stopwatch cannot tell a step that is working slowly from one that is
stuck. Progress can: a step that is still completing its parts is working,
however slowly the host lets it, and a step whose parts have stopped
completing for a budget's worth of wall is wedged. The budget here is the
longest a single part may take to complete, and that is what the old wall
budget was really measuring on the idle host.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import Awaitable, Callable

from core.runtime.task_ownership import create_owned_asyncio_task
from core.runtime.thread_cpu import thread_cpu_seconds


async def _as_coroutine[T](awaitable: Awaitable[T]) -> T:
    """A coroutine around any awaitable.

    `ensure_future` took a future or a coroutine alike; the owned
    creator takes a coroutine, and a caller here may hand either.
    """
    return await awaitable

__all__ = [
    "await_while_it_progresses",
    "await_while_the_task_moves",
    "innermost_await",
    "run_on_a_thread_while_it_works",
]


async def await_while_it_progresses[T](
    awaitable: Awaitable[T],
    *,
    progress: Callable[[], object],
    stall_s: float,
    name: str,
) -> T:
    """Await ``awaitable``; give up only when ``progress()`` stops changing.

    ``progress`` is read from the waiting task and must be cheap. The wait
    ends with the awaitable's result, or raises ``TimeoutError`` naming the
    step and the last progress seen once no change has been observed for
    ``stall_s`` seconds. The awaitable is cancelled on that path.
    """
    # Owned, not raw: an unowned task whose failure nobody observes is
    # the thing ASYNC-TASK-001 exists to stop, and the production
    # surface audit blocks `ensure_future` for that reason.
    task = create_owned_asyncio_task(
        _as_coroutine(awaitable), name="progress_bound.await_while_it_progresses"
    )
    stall_s = max(0.0, float(stall_s))
    last = progress()
    last_at = time.monotonic()
    # Read progress at a resolution that bounds the overshoot past the stall
    # budget without turning the wait into a busy loop.
    period = max(0.05, min(1.0, stall_s / 10.0)) if stall_s else 1.0
    while True:
        done, _pending = await asyncio.wait({task}, timeout=period)
        if done:
            return task.result()
        seen = progress()
        now = time.monotonic()
        if seen != last:
            last, last_at = seen, now
            continue
        if now - last_at >= stall_s:
            task.cancel()
            raise TimeoutError(
                f"{name} made no progress for {now - last_at:.1f}s (last progress: {last!r})"
            )


def innermost_await(task: asyncio.Task[object]) -> tuple[object, ...]:
    """Where a task is right now: the innermost coroutine it is inside and its line.

    Walks the ``cr_await`` chain from the task's coroutine. A coroutine that
    is running, or that has moved on to a different line or a different
    awaitable, gives a different key. Cheap enough to read every period.
    """
    return _position(task)[0]


def _position(task: asyncio.Task[object]) -> tuple[tuple[object, ...], object]:
    """The key ``innermost_await`` builds, and the innermost object itself.

    The key alone cannot tell two awaits apart when the second is at the
    same line and its object took the freed address of the first — a loop
    of ``await asyncio.sleep(0.05)`` read as one await that never moved. The
    caller holds the object weakly: a different live object is a move.
    """
    coro: object = task.get_coro()
    key: list[object] = []
    depth = 0
    while depth < 64:
        depth += 1
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None:
            key.append(id(coro))
            break
        key.append((id(coro), frame.f_lineno))
        nxt = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
        if nxt is None:
            break
        coro = nxt
    return tuple(key), coro


def _hold(obj: object) -> object:
    try:
        return weakref.ref(obj)
    except TypeError:
        return obj


def _same_object(held: object, obj: object) -> bool:
    if isinstance(held, weakref.ref):
        return held() is obj
    return held is obj


async def await_while_the_task_moves[T](
    awaitable: Awaitable[T],
    *,
    stall_s: float,
    name: str,
) -> T:
    """Await a stage; give up only when it has sat on one await for ``stall_s``.

    The stage's own position is its progress: the innermost coroutine and
    line it is at. A stage that is running through synchronous work on the
    loop thread moves between the waiter's looks; one that is awaiting
    something which never resolves does not.

    Only time the waiter could observe counts toward the stall. When the
    stage blocks the loop synchronously for a long call, the waiter cannot
    run, and a single late wake is one look, not a verdict about everything
    that happened while it could not look.
    """
    task = create_owned_asyncio_task(
        _as_coroutine(awaitable), name="progress_bound.await_while_the_task_moves"
    )
    stall_s = max(0.0, float(stall_s))
    period = max(0.05, min(1.0, stall_s / 10.0)) if stall_s else 1.0
    last, innermost = _position(task)
    held = _hold(innermost)
    still_for = 0.0
    last_wake = time.monotonic()
    while True:
        done, _pending = await asyncio.wait({task}, timeout=period)
        if done:
            return task.result()
        now = time.monotonic()
        observed = min(now - last_wake, 2.0 * period)
        last_wake = now
        seen, innermost = _position(task)
        if seen != last or not _same_object(held, innermost):
            last, held, still_for = seen, _hold(innermost), 0.0
            continue
        still_for += observed
        if still_for >= stall_s:
            task.cancel()
            raise TimeoutError(
                f"{name} sat on one await for {still_for:.1f}s of observed time"
            )


async def run_on_a_thread_while_it_works[T](
    fn: Callable[..., T],
    /,
    *args: object,
    stall_s: float,
    name: str,
) -> T:
    """Run ``fn`` on a worker thread; give up only when that thread stops working.

    A health probe or a status snapshot on a thread was bounded by a fixed
    wall budget. On a host loaded 30 to 120 on 18 cores the probe was still
    running its lines, slowly, when the budget ran out (2026-09-16: the
    control-plane reconcile fell over seven times in forty minutes and the
    Skynet probes thirty-five). The thread's own CPU time is its progress: a
    thread that is being scheduled is working, and one that has sat on a
    lock or a socket for ``stall_s`` of observed wall is not.

    This is the bound for work that is done on the CPU. A wait on a network
    peer legitimately uses none, and needs a bound of its own kind.
    """
    ident: list[int | None] = [None]

    def _run() -> T:
        ident[0] = threading.get_ident()
        return fn(*args)

    def _progress() -> object:
        if ident[0] is None:
            return None
        return thread_cpu_seconds(ident[0])

    return await await_while_it_progresses(
        asyncio.to_thread(_run), progress=_progress, stall_s=stall_s, name=name
    )
