"""core/runtime/what_stops_it.py — one thing that says stop, carried everywhere.

Two peer architectures make the same point. OpenHands threads a cancellation
token through agent and tool calls so cooperative cancellation composes;
AutoGen puts the token in the message API itself, so a handler cannot forget
it. Aura has asyncio cancellation, deadlines, and a narrow token inside the
voice duplex — three mechanisms that do not compose, and a fourth pattern where
a long-running step polls a module-level flag.

Polling a global is the specific thing this replaces. It cannot say WHY, it
cannot be scoped to one turn, a caller cannot hand a narrower one to a
subcall, and a test cannot cancel one operation without cancelling everything.

What is here is one token and one context to carry it.

The token composes. A child is stopped when its parent is, never the other way
round, so a turn can hand a subagent a token that dies with the turn and the
subagent can hand its tool a token it can stop on its own. That is what
"cooperative cancellation composes" comes to in code.

The context is ambient AND explicit. An API that takes it is honest about
being interruptible; ``current()`` exists so that a deep call which nobody has
threaded yet still sees the turn's token rather than a global. The ambient
read is the migration path, not the destination — a call that only ever reads
the ambient one has not been threaded, and ``what_is_not_threaded_yet`` says
which those are.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatStopsIt")

__all__ = [
    "AnExecutionContext",
    "Stopped",
    "Stopping",
    "current",
    "stopping_with",
    "under",
    "what_is_not_threaded_yet",
]


class Stopped(Exception):
    """Raised where a stop was asked for and the caller wanted an exception."""

    def __init__(self, why: str = "") -> None:
        super().__init__(why or "stopped")
        self.why = why


class Stopping:
    """One thing that says stop, and why, and takes its children with it.

    Named a token in both peer systems. What matters is not the name but the
    three properties none of Aura's existing mechanisms had together: it says
    why, it composes, and it can be handed to a callee that has no idea what
    a turn is.
    """

    __slots__ = ("_event", "_why", "_at", "_children", "_lock", "_when_stopped", "name")

    def __init__(self, name: str = "") -> None:
        self.name = str(name or "")
        self._event = threading.Event()
        self._why = ""
        self._at = 0.0
        self._children: list[Stopping] = []
        self._when_stopped: list[Callable[[str], None]] = []
        self._lock = threading.Lock()

    def stop(self, why: str = "") -> bool:
        """Ask for a stop. Returns False when one was already asked for.

        Idempotent on purpose: two subsystems noticing the same reason must
        not produce two reasons, and the first one is the true one.
        """

        with self._lock:
            if self._event.is_set():
                return False
            self._why = str(why or "")
            self._at = time.monotonic()
            children = list(self._children)
            callbacks = list(self._when_stopped)
            self._event.set()
        for child in children:
            child.stop(why)
        for called in callbacks:
            try:
                called(self._why)
            except Exception as exc:  # noqa: BLE001 — a listener must not block a stop
                logger.debug("a stop listener raised: %s", exc)
        return True

    @property
    def stopped(self) -> bool:
        return self._event.is_set()

    @property
    def why(self) -> str:
        return self._why

    @property
    def stopped_for_s(self) -> float:
        return max(0.0, time.monotonic() - self._at) if self._at else 0.0

    def raise_if_stopped(self) -> None:
        if self._event.is_set():
            raise Stopped(self._why)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until stopped, or the timeout. True when it stopped."""

        return self._event.wait(timeout)

    def when_stopped(self, called: Callable[[str], None]) -> None:
        """Run this when the stop arrives, or now if it already has."""

        with self._lock:
            if not self._event.is_set():
                self._when_stopped.append(called)
                return
            why = self._why
        try:
            called(why)
        except Exception as exc:  # noqa: BLE001
            logger.debug("a stop listener raised: %s", exc)

    def child(self, name: str = "") -> "Stopping":
        """A token that dies with this one and can also die alone.

        The direction is the whole design. A subagent stopping does not stop
        the turn that made it; a turn stopping stops everything under it.
        """

        made = Stopping(name or f"{self.name}/child")
        with self._lock:
            already = self._event.is_set()
            why = self._why
            if not already:
                self._children.append(made)
        if already:
            made.stop(why)
        return made

    def __repr__(self) -> str:
        state = f"stopped: {self._why!r}" if self._event.is_set() else "running"
        return f"<Stopping {self.name!r} {state}>"


@dataclass(frozen=True)
class AnExecutionContext:
    """What one piece of work is allowed, carried with it.

    AutoGen puts this in the message API so a handler receives it rather than
    reconstructing it. That is the property worth copying: a handler that
    infers its context from the payload has a different context depending on
    who called it.
    """

    stopping: Stopping = field(default_factory=Stopping)
    #: Monotonic time after which the work is late. Zero for no deadline.
    due_by: float = 0.0
    #: What this work is doing, for logs and receipts.
    doing: str = ""
    #: Who asked. An authority name, not a user id.
    asked_by: str = ""
    #: The turn or tick this belongs to, so records join up.
    trace: str = ""

    @property
    def out_of_time(self) -> bool:
        return bool(self.due_by) and time.monotonic() >= self.due_by

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.due_by - time.monotonic()) if self.due_by else float("inf")

    def check(self) -> None:
        """Raise where the work should not continue. The one call a loop needs."""

        self.stopping.raise_if_stopped()
        if self.out_of_time:
            raise Stopped(f"out of time after {self.doing or 'this work'}")

    def under(self, doing: str, *, seconds: float = 0.0) -> "AnExecutionContext":
        """A narrower context for a subcall: its own token, its own deadline.

        The deadline never widens. A subcall given ten seconds inside a turn
        with three left has three, which is the property a caller cannot get
        by passing a number.
        """

        due = self.due_by
        if seconds > 0:
            asked = time.monotonic() + seconds
            due = min(due, asked) if due else asked
        return AnExecutionContext(
            stopping=self.stopping.child(doing),
            due_by=due,
            doing=doing,
            asked_by=self.asked_by,
            trace=self.trace,
        )


_HERE: contextvars.ContextVar[AnExecutionContext | None] = contextvars.ContextVar(
    "aura_execution_context", default=None
)

#: Names of callers that read the ambient context instead of taking one. The
#: migration list: a call on it has not been threaded, and the number only
#: goes down.
_READ_THE_AMBIENT_ONE: dict[str, int] = {}


def current(*, whose: str = "") -> AnExecutionContext:
    """The context for this work, or a fresh one that stops nothing.

    ``whose`` records that a caller read the ambient context rather than being
    handed one. That is not an error and it is not the destination either, so
    it is counted rather than logged.
    """

    if whose:
        _READ_THE_AMBIENT_ONE[whose] = _READ_THE_AMBIENT_ONE.get(whose, 0) + 1
    found = _HERE.get()
    return found if found is not None else AnExecutionContext(doing="unbound")


def what_is_not_threaded_yet() -> dict[str, int]:
    """Who is still reading the ambient context, and how often."""

    return dict(_READ_THE_AMBIENT_ONE)


@contextmanager
def under(context: AnExecutionContext) -> Iterator[AnExecutionContext]:
    """Run this block with that context ambient."""

    token = _HERE.set(context)
    try:
        yield context
    finally:
        _HERE.reset(token)


@contextmanager
def stopping_with(
    doing: str, *, seconds: float = 0.0, asked_by: str = "", trace: str = ""
) -> Iterator[AnExecutionContext]:
    """Open a context for a piece of work, nested under whatever is ambient."""

    here = _HERE.get()
    made = (
        here.under(doing, seconds=seconds)
        if here is not None
        else AnExecutionContext(
            stopping=Stopping(doing),
            due_by=(time.monotonic() + seconds) if seconds > 0 else 0.0,
            doing=doing,
            asked_by=asked_by,
            trace=trace,
        )
    )
    with under(made):
        yield made


def from_asyncio(context: AnExecutionContext) -> Callable[[Any], None]:
    """Bridge a task's cancellation into the token.

    asyncio cancellation is the mechanism Aura already has and the one that
    does not compose: a cancelled task raises inside itself and says nothing
    to the thing it called. Attach this as a done-callback and the token
    carries the news to everything below.
    """

    def when_done(task: Any) -> None:
        try:
            if task.cancelled():
                context.stopping.stop("the task was cancelled")
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not read a task's cancellation: %s", exc)

    return when_done
