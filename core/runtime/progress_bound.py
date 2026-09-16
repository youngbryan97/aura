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
import time
from collections.abc import Awaitable, Callable

__all__ = ["await_while_it_progresses"]


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
    task = asyncio.ensure_future(awaitable)
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
