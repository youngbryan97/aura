"""A checkpoint must not become durable before the writes that produced it.

:class:`core.state.a_checkpoint_and_its_writes.TheChannels` already refuses a
checkpoint while writes are pending in memory. That is the easy half. The half
that costs a real recovery is durability ordering: the writes are committed in
memory, a task is off writing them to disk, and the checkpoint reaches the disk
first. A reader then restores a state whose values were never written, and
every check passes — the digest matches what the checkpoint recorded, and what
it recorded was a future that did not happen.

LangGraph's loop drains the futures from delta writes before making the next
checkpoint durable, with a comment saying why: otherwise a checkpoint can
become visible before the writes that produced it. This is that, in the shape
Aura already has.

A write registers itself while it is in flight. A checkpoint store awaits every
write for its scope before persisting, and the wait is bounded — a write that
will not land is a reason to refuse the checkpoint, not a reason to hang the
runtime holding it.

Nothing here writes anything. It only orders what other things write, which is
why it can sit under both stores without either knowing about the other.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.NothingLandsBeforeItsWrites")

__all__ = [
    "AWriteInFlight",
    "TookTooLong",
    "a_write_in_flight",
    "still_in_flight",
    "wait_for_the_writes",
    "wait_for_the_writes_async",
    "how_the_drains_have_gone",
    "forget_everything",
]

#: Longer than any single durable write should take. A checkpoint that cannot
#: be taken is a worse outcome than a slow one and a much better outcome than a
#: runtime wedged behind a write that will never finish.
LONG_ENOUGH = 30.0


class TookTooLong(TimeoutError):
    """Writes for this scope did not land inside the drain deadline."""


@dataclass
class AWriteInFlight:
    """One durable write that has started and not finished."""

    scope: str
    what: str
    started: float = field(default_factory=time.monotonic)
    #: Set when the write finishes, so a waiter can be woken rather than poll.
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "what": self.what,
            "seconds": round(self.seconds, 4),
            "finished": self.done.is_set(),
        }


_IN_FLIGHT: dict[int, AWriteInFlight] = {}
_HISTORY: list[dict[str, Any]] = []
_LOCK = threading.Lock()
_KEEP = 200


@contextlib.contextmanager
def a_write_in_flight(scope: str, what: str) -> Iterator[AWriteInFlight]:
    """Register a durable write for the length of the block.

    Registered before the write starts and cleared however it ends, including
    a raise: a write that failed is not in flight, and leaving it registered
    would block every checkpoint after it.
    """
    one = AWriteInFlight(scope=str(scope), what=str(what))
    with _LOCK:
        _IN_FLIGHT[id(one)] = one
    try:
        yield one
    finally:
        one.done.set()
        with _LOCK:
            _IN_FLIGHT.pop(id(one), None)


def still_in_flight(scope: str | None = None) -> tuple[AWriteInFlight, ...]:
    """Writes that have started and not finished, for this scope or all."""
    with _LOCK:
        rows = list(_IN_FLIGHT.values())
    if scope is None:
        return tuple(rows)
    return tuple(one for one in rows if one.scope == str(scope))


def _record(scope: str, waited: float, left: tuple[AWriteInFlight, ...]) -> None:
    with _LOCK:
        _HISTORY.append(
            {
                "scope": scope,
                "waited": round(waited, 4),
                "drained": not left,
                "still_in_flight": [one.to_dict() for one in left],
            }
        )
        del _HISTORY[:-_KEEP]


def wait_for_the_writes(scope: str, *, seconds: float = LONG_ENOUGH) -> None:
    """Block until this scope's writes have landed. Raises if they do not.

    For a synchronous caller. The deadline is the whole wait, not per write:
    ten writes that each take a second must not add up to ten deadlines.
    """
    started = time.monotonic()
    ends_at = started + max(0.0, float(seconds))
    while True:
        left = still_in_flight(scope)
        if not left:
            _record(scope, time.monotonic() - started, ())
            return
        left[0].done.wait(timeout=max(0.0, ends_at - time.monotonic()))
        if time.monotonic() >= ends_at:
            left = still_in_flight(scope)
            if not left:
                _record(scope, time.monotonic() - started, ())
                return
            _record(scope, time.monotonic() - started, left)
            raise TookTooLong(
                f"{len(left)} write(s) for {scope} did not land in "
                f"{seconds:.1f}s: "
                + ", ".join(sorted(one.what for one in left))
                + ". The checkpoint is refused rather than taken over writes "
                "that may not have happened."
            )


async def wait_for_the_writes_async(
    scope: str, *, seconds: float = LONG_ENOUGH
) -> None:
    """The same wait, off the event loop.

    The wait itself blocks a thread, so it goes to one rather than to the
    loop: a drain on the loop is the freeze this whole ordering exists to
    avoid causing somewhere else.
    """
    await asyncio.to_thread(wait_for_the_writes, scope, seconds=seconds)


def how_the_drains_have_gone() -> dict[str, Any]:
    """For the health report: whether a checkpoint ever outran its writes."""
    with _LOCK:
        rows = list(_HISTORY)
        flying = [one.to_dict() for one in _IN_FLIGHT.values()]
    refused = [row for row in rows if not row["drained"]]
    return {
        "drains": len(rows),
        "in_flight_now": flying,
        "refused_checkpoints": len(refused),
        "longest_wait": max((row["waited"] for row in rows), default=0.0),
        "scopes": sorted({row["scope"] for row in rows}),
        "recent_refusals": refused[-5:],
    }


def forget_everything() -> None:
    with _LOCK:
        _IN_FLIGHT.clear()
        _HISTORY.clear()
