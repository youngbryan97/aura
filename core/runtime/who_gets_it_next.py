"""Who gets the scarce thing next, and how long they waited for it.

Aura already has locks — 230 modules use ``checked_lock`` — and a careful
transactional claim on the model lane. What it did not have is one place that
answers "who holds the screen", "how long did training wait", "did anyone
give up waiting", for every scarce thing rather than for one of them.

A lock answers none of those. It is a boolean with a queue you cannot see,
and its queue is whatever the OS or the event loop decides, which is not
fairness — a caller that asks often can starve one that asks once.

So: a claim is a named holder of a named resource, taken in the order asked.
Waiting has a deadline that comes from the caller's own context rather than a
number written here, and a claim given up because the work was stopped is a
different outcome from one that timed out, because they need different fixes.

Reentrancy is a declared policy, not an accident: the same holder asking
again gets the claim it already has, counted, and releases it once. That is
the one case where a lock silently deadlocks instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from core.runtime.what_stops_it import AnExecutionContext, Stopped, current

logger = logging.getLogger("Aura.WhoGetsItNext")

__all__ = [
    "AClaim",
    "GaveUp",
    "THE_RESOURCES",
    "claim",
    "observe_held",
    "observe_released",
    "how_it_has_gone",
    "who_holds_what",
    "who_is_waiting",
]

#: The scarce things: what makes each one scarce, and who decides who gets it.
#: Named here so a claim on something nobody declared is a typo rather than a
#: resource nobody noticed. Adding one is an edit to this table and a sentence
#: about why it is scarce.
#:
#: "here" means this module grants it, in the order asked. "elsewhere" means
#: something older and richer grants it and only reports in — the model lane
#: has a reservation with eviction and compensation, and replacing that with a
#: queue would lose the part that matters. There is deliberately no "files"
#: row: the write gateway writes to a temporary file and renames, so two
#: writers to one path do not interleave, and declaring a resource nothing
#: claims would be decoration.
THE_RESOURCES: dict[str, dict[str, str]] = {
    "model_lane": {
        "scarce": "one resident model at a time; a second load evicts the first",
        "granted": "elsewhere",
    },
    "screen": {
        "scarce": "one actor moving the pointer; two make neither's moves mean "
        "anything",
        "granted": "here",
    },
    "training": {
        "scarce": "a fine-tune holds the wired memory the resident model needs",
        "granted": "here",
    },
}


class GaveUp(Stopped):
    """Waiting ended without the claim. Carries which of the two reasons."""

    def __init__(self, resource: str, why: str, waited_s: float) -> None:
        super().__init__(f"{resource}: {why} after waiting {waited_s:.3f}s")
        self.resource = resource
        self.reason = why
        self.waited_s = waited_s


@dataclass
class AClaim:
    """One holder of one resource."""

    resource: str
    by: str
    since: float = field(default_factory=time.monotonic)
    #: How deep the same holder has re-entered. One means held once.
    depth: int = 1
    #: How long this holder waited before getting it.
    waited_s: float = 0.0
    trace: str = ""

    @property
    def held_for_s(self) -> float:
        return time.monotonic() - self.since


@dataclass
class _AWaiter:
    by: str
    asked_at: float
    ready: asyncio.Future
    trace: str = ""


@dataclass
class _HowItHasGone:
    """What has happened to one resource, since the process started."""

    granted: int = 0
    reentered: int = 0
    timed_out: int = 0
    stopped: int = 0
    waited_s_total: float = 0.0
    waited_s_worst: float = 0.0
    held_s_total: float = 0.0


_HELD: dict[str, AClaim] = {}
_WAITING: dict[str, deque[_AWaiter]] = {}
_RECORD: dict[str, _HowItHasGone] = {}

#: One lock over the bookkeeping only, per event loop. Held for the length of
#: a dict update, never across an await on the resource itself — a claim
#: manager that holds a lock while its callers work is the contention it was
#: built to remove.
#:
#: Per loop, not per process. An ``asyncio.Lock`` built at import binds its
#: waiters to whichever loop first had to wait on it, and a second loop then
#: waits forever on futures nothing will ever complete. That is not only a
#: test artifact: ``AuraKernel.hot_reboot`` builds a new loop in a live
#: process, and every claim after one would have hung.
_LOCKS: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _books() -> asyncio.Lock:
    """The bookkeeping lock for the loop that is running now."""
    loop = asyncio.get_running_loop()
    found = _LOCKS.get(loop)
    if found is None:
        found = asyncio.Lock()
        _LOCKS[loop] = found
    return found


def _record(resource: str) -> _HowItHasGone:
    return _RECORD.setdefault(resource, _HowItHasGone())


def who_holds_what() -> dict[str, dict[str, Any]]:
    """Every resource currently held, by whom, and for how long."""
    return {
        resource: {
            "by": held.by,
            "held_for_s": round(held.held_for_s, 3),
            "depth": held.depth,
            "waited_s": round(held.waited_s, 3),
            "trace": held.trace,
        }
        for resource, held in sorted(_HELD.items())
    }


def who_is_waiting() -> dict[str, list[dict[str, Any]]]:
    """Every queue, in the order it will be served."""
    now = time.monotonic()
    return {
        resource: [
            {"by": one.by, "waiting_s": round(now - one.asked_at, 3)}
            for one in queue
        ]
        for resource, queue in sorted(_WAITING.items())
        if queue
    }


def how_it_has_gone() -> dict[str, dict[str, Any]]:
    """The record per resource: grants, re-entries, and both ways of failing.

    Timed out and stopped are separate because they need different fixes. A
    timeout says the holder is too slow or the deadline too tight; a stop says
    the caller's own work was cancelled while it queued, which is not the
    resource's fault at all.
    """
    return {
        resource: {
            "granted": r.granted,
            "reentered": r.reentered,
            "timed_out": r.timed_out,
            "stopped": r.stopped,
            "waited_s_mean": round(r.waited_s_total / r.granted, 4) if r.granted else 0.0,
            "waited_s_worst": round(r.waited_s_worst, 4),
            "held_s_total": round(r.held_s_total, 3),
        }
        for resource, r in sorted(_RECORD.items())
    }


def observe_held(resource: str, by: str, *, trace: str = "") -> None:
    """Record that something granted elsewhere is now held.

    For a resource whose own mechanism decides who gets it. Nothing queues on
    this and nothing is granted by it — it exists so ``who_holds_what`` can
    answer for every scarce thing rather than for the two this module runs.
    """
    _check(resource, granted="elsewhere")
    _HELD[resource] = AClaim(resource=resource, by=by, trace=trace)
    _record(resource).granted += 1


def observe_released(resource: str, by: str) -> None:
    """Record that a resource granted elsewhere is no longer held."""
    _check(resource, granted="elsewhere")
    held = _HELD.get(resource)
    if held is not None and held.by == by:
        _record(resource).held_s_total += held.held_for_s
        _HELD.pop(resource, None)


def _check(resource: str, *, granted: str) -> None:
    row = THE_RESOURCES.get(resource)
    if row is None:
        raise KeyError(
            f"no such resource: {resource!r}; declare it in THE_RESOURCES first "
            f"({', '.join(sorted(THE_RESOURCES))})"
        )
    if row["granted"] != granted:
        raise ValueError(
            f"{resource} is granted {row['granted']}, not {granted}"
        )


def _next_in_line(resource: str) -> None:
    """Give the resource to whoever asked first and is still there.

    The transfer IS the invariant. This used to pop the holder, wake the next
    waiter's future and let the waiter install itself when it resumed — which
    is not mutual exclusion, because a third caller arriving between the wake
    and the resume sees an empty ``_HELD`` and takes it, and then the waiter
    resumes and overwrites the record without checking. Two holders inside
    what is supposed to be exclusive ownership, and the register showing one.

    So the claim is written here, under ``_books()``, before the future is
    resolved. A caller arriving in that window sees the resource held by the
    waiter and queues behind it. Callers must hold ``_books()``.
    """
    queue = _WAITING.get(resource)
    while queue:
        waiter = queue.popleft()
        if waiter.ready.done():
            # Cancelled or timed out while queued. Its claim is nobody's.
            continue
        waited = time.monotonic() - waiter.asked_at
        _HELD[resource] = AClaim(
            resource=resource,
            by=waiter.by,
            trace=waiter.trace,
            waited_s=waited,
        )
        record = _record(resource)
        record.granted += 1
        record.waited_s_total += waited
        record.waited_s_worst = max(record.waited_s_worst, waited)
        try:
            waiter.ready.set_result(True)
        except asyncio.InvalidStateError:
            # It finished between the check and here. Undo the grant rather
            # than leave a claim held by somebody who will never release it.
            _HELD.pop(resource, None)
            record.granted -= 1
            record.waited_s_total -= waited
            continue
        return


@asynccontextmanager
async def claim(
    resource: str,
    by: str,
    *,
    context: AnExecutionContext | None = None,
    seconds: float = 0.0,
):
    """Hold ``resource`` as ``by``, waiting in line for it.

    The deadline is the caller's own: ``context.seconds_left`` unless
    ``seconds`` narrows it. Nothing here writes a timeout down, because a
    number written here is a number that is wrong for some caller.
    """
    _check(resource, granted="here")
    here = context or current(whose=f"claim.{resource}")
    waiting = here.under(f"waiting for {resource}", seconds=seconds)
    asked_at = time.monotonic()
    record = _record(resource)

    # One decision, under the books lock, so two callers arriving together
    # cannot both read "free". Split across two lock holds this had a window
    # where each saw the resource unheld and each took it.
    waiter: _AWaiter | None = None
    async with _books():
        held = _HELD.get(resource)
        if held is not None and held.by == by:
            held.depth += 1
            record.reentered += 1
        elif held is None:
            _HELD[resource] = AClaim(
                resource=resource, by=by, trace=here.trace, waited_s=0.0
            )
            record.granted += 1
        else:
            waiter = _AWaiter(
                by=by,
                asked_at=asked_at,
                ready=asyncio.get_running_loop().create_future(),
                trace=here.trace,
            )
            _WAITING.setdefault(resource, deque()).append(waiter)

    if waiter is not None:
        try:
            left = waiting.seconds_left
            await asyncio.wait_for(
                waiter.ready, None if left == float("inf") else left
            )
        except TimeoutError:
            async with _books():
                _drop(resource, waiter)
                # It may have been given the claim in the instant before the
                # deadline. Hand it straight on rather than leave it held by
                # somebody who has already stopped waiting.
                _release(resource, by, record)
            record.timed_out += 1
            raise GaveUp(
                resource, "ran out of time", time.monotonic() - asked_at
            ) from None
        except asyncio.CancelledError:
            async with _books():
                _drop(resource, waiter)
                _release(resource, by, record)
            record.stopped += 1
            raise
        # The claim is already ours: _next_in_line wrote it under the books
        # lock before waking this future. Nothing is installed here, because
        # installing it here is what let a third caller slip in between.
        try:
            waiting.check()
        except Stopped:
            async with _books():
                _release(resource, by, record)
            record.stopped += 1
            raise

    try:
        yield _HELD.get(resource)
    finally:
        async with _books():
            _release(resource, by, record)


def _release(resource: str, by: str, record: "_HowItHasGone") -> None:
    """Give up one level of a claim, handing it on if that was the last.

    Callers must hold ``_books()``: the pop and the hand-on are one step, and
    the window between them is where two holders came from.
    """
    held = _HELD.get(resource)
    if held is None or held.by != by:
        return
    held.depth -= 1
    if held.depth > 0:
        return
    record.held_s_total += held.held_for_s
    _HELD.pop(resource, None)
    _next_in_line(resource)


def _drop(resource: str, waiter: _AWaiter) -> None:
    queue = _WAITING.get(resource)
    if not queue:
        return
    try:
        queue.remove(waiter)
    except ValueError:
        pass


def forget_everything() -> None:
    """Drop all bookkeeping. For tests only; a live runtime never calls this."""
    _HELD.clear()
    _WAITING.clear()
    _RECORD.clear()
