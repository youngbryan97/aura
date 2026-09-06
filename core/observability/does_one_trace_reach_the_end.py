"""Whether one trace id survives every boundary a turn actually crosses.

:mod:`core.runtime.causal_trace` holds the trace in a context variable, which
is the right place for it and is also why the question is worth asking. A
context variable follows the call stack. It does not follow a thread, it does
not follow a task spawned without a copied context, and it does not follow a
message onto a bus. Every one of those boundaries exists between the start of
a turn and the learning that comes out the far end.

So the interesting thing is not that Aura has trace ids. It is which of the
hops keep them. This module asks each boundary directly, one at a time, and
names the ones that drop it.

The six hops OpenHands names — turn, deliberation, action, tool, result,
learning — do not each cross the same kind of boundary. What matters is the
mechanism underneath: a plain call, a spawned task, a thread, an executor, a
queue, a bus message. :func:`where_the_trace_is_lost` runs all six mechanisms
and reports which carry the id through.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import functools
import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from core.runtime.causal_trace import (
    current_span,
    current_trace_id,
    extract_trace_carrier,
    new_trace,
    trace_scope,
)

logger = logging.getLogger("Aura.DoesOneTraceReachTheEnd")

__all__ = [
    "ABoundary",
    "where_the_trace_is_lost",
    "how_far_a_trace_reaches",
    "carrying_the_trace",
    "in_a_thread_that_carries_it",
    "on_an_executor_that_carries_it",
    "a_carrier_for_a_queue",
    "under_a_carrier",
]


@dataclass(frozen=True, slots=True)
class ABoundary:
    """One way control leaves the current stack, and whether the id follows."""

    name: str
    #: The turn stage that crosses it, so a loss has a consequence and not
    #: just a mechanism.
    crossed_by: str
    kept: bool
    saw: str
    expected: str

    @property
    def lost(self) -> bool:
        return not self.kept


def _plain_call() -> str:
    return current_trace_id()


async def _awaited() -> str:
    await asyncio.sleep(0)
    return current_trace_id()


async def _spawned_task() -> str:
    """asyncio.create_task copies the current context by default."""
    return await asyncio.create_task(_awaited())


async def _spawned_task_with_a_fresh_context() -> str:
    """A task given an empty context, which is how a trace is really lost."""
    return await asyncio.create_task(_awaited(), context=contextvars.Context())


def _a_bare_thread() -> str:
    """A thread started without copying the context. The classic loss."""
    out: list[str] = []
    thread = threading.Thread(target=lambda: out.append(current_trace_id()))
    thread.start()
    thread.join()
    return out[0] if out else ""


def _a_thread_that_carries_it() -> str:
    ctx = contextvars.copy_context()
    out: list[str] = []
    thread = threading.Thread(target=lambda: out.append(ctx.run(current_trace_id)))
    thread.start()
    thread.join()
    return out[0] if out else ""


async def _an_executor() -> str:
    """run_in_executor does not copy the context; the loop hands over a bare fn."""
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return await loop.run_in_executor(pool, _plain_call)


def _a_queue_hop() -> str:
    """A message put on a queue and read by another thread carries no context."""
    box: queue.Queue[str] = queue.Queue()
    carrier = current_trace_id()

    def consumer() -> None:
        _ = box.get()
        box.put(current_trace_id())

    worker = threading.Thread(target=consumer)
    worker.start()
    box.put(carrier)
    worker.join()
    return box.get()


async def _run_the_boundaries(trace: str) -> list[ABoundary]:
    checks: list[tuple[str, str, Any]] = [
        ("a plain call", "turn -> deliberation", _plain_call()),
        ("an awaited coroutine", "deliberation -> action", await _awaited()),
        ("a spawned task", "action -> tool", await _spawned_task()),
        (
            "a task given a fresh context",
            "tool run detached from its caller",
            await _spawned_task_with_a_fresh_context(),
        ),
        ("a bare thread", "tool -> result on a worker", _a_bare_thread()),
        (
            "a thread carrying a copied context",
            "tool -> result done properly",
            _a_thread_that_carries_it(),
        ),
        ("an executor", "result -> learning off the loop", await _an_executor()),
        ("a queue between threads", "learning enqueued for later", _a_queue_hop()),
    ]
    return [
        ABoundary(
            name=name,
            crossed_by=stage,
            kept=(saw == trace),
            saw=saw or "(nothing)",
            expected=trace,
        )
        for name, stage, saw in checks
    ]


def where_the_trace_is_lost() -> tuple[ABoundary, ...]:
    """Run every boundary and report which ones keep the id.

    Safe to call from a running loop or from a plain thread; it makes its own
    loop where there is none rather than touching the caller's.
    """
    span = new_trace("does_one_trace_reach_the_end", origin="observability")
    with trace_scope(span):
        trace = span.trace_id
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return tuple(asyncio.run(_run_the_boundaries(trace)))
        out: list[tuple[ABoundary, ...]] = []
        ctx = contextvars.copy_context()

        def _in_its_own_loop() -> None:
            out.append(tuple(ctx.run(lambda: asyncio.run(_run_the_boundaries(trace)))))

        worker = threading.Thread(target=_in_its_own_loop)
        worker.start()
        worker.join()
        return out[0] if out else ()


def how_far_a_trace_reaches() -> dict[str, Any]:
    """For the health report: which boundaries drop the id, and what crosses them."""
    boundaries = where_the_trace_is_lost()
    lost = [b for b in boundaries if b.lost]
    return {
        "boundaries": len(boundaries),
        "kept": len(boundaries) - len(lost),
        "lost": len(lost),
        "lost_at": [
            {"boundary": b.name, "crossed_by": b.crossed_by, "saw": b.saw}
            for b in lost
        ],
        "kept_at": [b.name for b in boundaries if b.kept],
        "a_context_variable_does_not_follow": sorted(b.name for b in lost),
    }


# --- Carrying it across the boundaries that drop it ---------------------------
#
# Three of the four losses above are the same loss: something ran a callable
# without the context. The remedy is small enough that the reason it was not
# already everywhere is that nobody had measured which hops needed it.


def carrying_the_trace(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Bind the caller's context to ``fn`` so a worker keeps the trace id.

    Capture happens now, at the call to this function, not when the returned
    callable runs. That is the whole point: the capture has to happen on the
    side of the boundary that still has the id.
    """
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def _carried(*args: Any, **kwargs: Any) -> Any:
        return ctx.run(fn, *args, **kwargs)

    return _carried


def in_a_thread_that_carries_it(
    target: Callable[..., Any], *args: Any, **kwargs: Any
) -> threading.Thread:
    """A Thread whose target runs under the caller's context. Not started."""
    return threading.Thread(target=carrying_the_trace(target), args=args, kwargs=kwargs)


async def on_an_executor_that_carries_it(
    pool: concurrent.futures.Executor | None,
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """run_in_executor with the trace attached. ``pool`` None means the default."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, carrying_the_trace(fn), *args)


def a_carrier_for_a_queue() -> dict[str, Any]:
    """The trace as plain data, for a hop no context variable can cross.

    A queue carries values, not stacks. Put this beside the message and open
    it with :func:`under_a_carrier` on the far side.
    """
    span = current_span()
    return span.to_carrier() if span is not None else {}


@contextmanager
def under_a_carrier(carrier: dict[str, Any] | None) -> Iterator[str]:
    """Re-enter a trace from the data a queue or a socket carried.

    Yields the trace id, empty where the carrier held nothing, so a caller can
    tell a resumed trace from a missing one instead of inventing a new id and
    reporting continuity it does not have.
    """
    span = extract_trace_carrier(carrier or {})
    if span is None:
        yield ""
        return
    with trace_scope(span):
        yield span.trace_id
