"""Giving up on an answer, and the work carrying on regardless.

A caller that awaits a reply and times out abandons the wait. The handler on
the other side has no idea: it goes on holding the model lane, writing state
for a turn nobody is listening to, and eventually answers into a future nobody
is holding. The turn that gave up is charged for none of it and the next turn
waits behind all of it.

AutoGen ties a cancellation token to the future representing the call, so
cancelling the wait cancels the work. This is that, with the two things Aura
needs on top.

The work must be able to see the cancellation without being written to look
for it. So a handle carries a stop signal the handler already reads, and
setting it is what a cancel does — a handler that never checks is a handler
that runs to completion, and that is visible here rather than mysterious.

And giving up must be distinguishable from failing. A caller that stopped
waiting learned nothing about whether the work was possible; a caller whose
call raised learned something. Collapsing them is how a timeout turns into a
permanent belief that a capability is broken.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.CancellingTheCall")

__all__ = [
    "ACall",
    "GaveUpWaiting",
    "call",
    "how_the_calls_ended",
    "forget_everything",
]


class GaveUpWaiting(asyncio.CancelledError):
    """The caller stopped waiting. Not the same as the work failing.

    A CancelledError so that every `except asyncio.CancelledError: raise`
    already in the tree keeps behaving correctly, and a distinct type so a
    caller that cares can tell giving up from a failure.
    """


@dataclass
class ACall:
    """One call in flight: the future being awaited and the work behind it."""

    what: str
    by: str = ""
    #: Set when the caller gives up. The handler reads it and stops.
    stop: threading.Event = field(default_factory=threading.Event, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    started: float = field(default_factory=time.monotonic)
    gave_up: bool = False
    checked_the_stop: bool = False

    @property
    def cancelled(self) -> bool:
        return self.stop.is_set()

    def should_stop(self) -> bool:
        """For the handler. Reading it is what makes a cancel arrive."""
        self.checked_the_stop = True
        return self.stop.is_set()

    def give_up(self, why: str = "") -> None:
        """Ask the work to stop. Does not cancel it — :func:`call` does that.

        Setting the signal is the polite half and it has to come first: a
        handler that checks gets to finish what it is holding. Cancelling
        immediately raises inside whatever `await` the handler happens to be
        on, so it never reaches its own stop branch, and "stopped cleanly"
        becomes "stopped wherever it was".
        """
        self.gave_up = True
        self.stop.set()
        if why:
            logger.info("%s gave up on %s: %s", self.by or "a caller", self.what, why)

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "by": self.by,
            "gave_up": self.gave_up,
            "seconds": round(time.monotonic() - self.started, 4),
            "the_work_checked": self.checked_the_stop,
        }


_HISTORY: list[dict[str, Any]] = []
_LOCK = threading.Lock()
_KEEP = 200


async def call(
    what: str,
    work: Callable[[ACall], Awaitable[Any]],
    *,
    by: str = "",
    seconds: float = 0.0,
) -> Any:
    """Await ``work``, and stop it if this wait is abandoned.

    ``work`` is handed the call so it can ask :meth:`ACall.should_stop`. It is
    not required to; a handler that never asks runs to completion and the
    report says so, which is the honest version of "cancellation did nothing".

    Raises :class:`GaveUpWaiting` where the caller stopped waiting — including
    on its own timeout — and whatever ``work`` raised where the work failed.
    """
    one = ACall(what=str(what), by=str(by))
    running = asyncio.ensure_future(work(one))
    one.task = running
    try:
        if seconds > 0:
            return await asyncio.wait_for(asyncio.shield(running), timeout=seconds)
        return await asyncio.shield(running)
    except asyncio.TimeoutError:
        one.give_up(f"nothing back inside {seconds:.1f}s")
        await _let_it_stop(one, running)
        raise GaveUpWaiting(
            f"{what}: nothing came back inside {seconds:.1f}s, and the work was "
            "stopped rather than left running"
        ) from None
    except asyncio.CancelledError:
        # The caller's own task was cancelled. The work must go too, or it
        # answers into a future nobody is holding.
        one.give_up("the caller was cancelled")
        # Shielded, because this task is being cancelled and an unshielded
        # await here would be cancelled before the work heard anything.
        await asyncio.shield(_let_it_stop(one, running))
        raise
    finally:
        with _LOCK:
            _HISTORY.append(one.to_dict())
            del _HISTORY[:-_KEEP]


#: How long a handler gets to notice the signal before it is cancelled
#: outright. Long enough for a loop that checks between awaits, short enough
#: that a deaf handler does not hold the caller up.
A_MOMENT_TO_NOTICE = 0.25


async def _let_it_stop(one: ACall, running: asyncio.Future) -> None:
    """Give the work a moment to stop itself, then cancel it. Never raises.

    Two steps rather than one: a handler that checks the signal gets to
    finish what it is holding, and a handler that does not gets cancelled
    anyway. The report says which happened, so "cancellation did nothing" is
    a measurement rather than a suspicion.
    """
    if running.done():
        return
    try:
        await asyncio.wait({running}, timeout=A_MOMENT_TO_NOTICE)
        if not running.done():
            running.cancel()
            await asyncio.wait({running}, timeout=A_MOMENT_TO_NOTICE)
    except Exception as exc:  # noqa: BLE001 - stopping must not be the failure
        logger.debug("waiting for cancelled work to stop: %s", exc)


def how_the_calls_ended() -> dict[str, Any]:
    """For the health report: how often work outlived the caller waiting on it."""
    with _LOCK:
        rows = list(_HISTORY)
    abandoned = [row for row in rows if row["gave_up"]]
    deaf = [row for row in abandoned if not row["the_work_checked"]]
    return {
        "calls": len(rows),
        "given_up_on": len(abandoned),
        "work_that_never_checked": len(deaf),
        "never_checked_what": sorted({row["what"] for row in deaf}),
        "longest": max((row["seconds"] for row in rows), default=0.0),
        "recent": rows[-5:],
    }


def forget_everything() -> None:
    with _LOCK:
        _HISTORY.clear()
