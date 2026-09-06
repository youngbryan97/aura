"""What the event loop's thread may not do, and what it did anyway.

Home Assistant took the top maturity score partly for its root runtime:
explicit lifecycle states, thread-affinity enforcement, typed job
distinctions, task ownership, startup and shutdown stages, and carefully
controlled executor behaviour.

Thread affinity is the one Aura has already been bitten by. An on-loop fsync
once froze the live event loop for twenty minutes, and the fix was a rule
written in a guide: from async code, use the async writers. A rule in a guide
is not enforcement — nothing could tell you it had been broken until the loop
stopped answering.

So the loop's thread says who it is at boot, and a blocking call that names
itself is checked against it. Three kinds of work:

* ``on_the_loop`` — must run on the loop thread. Touching loop-owned state
  from a worker thread is the mirror-image defect.
* ``anywhere`` — no opinion.
* ``never_on_the_loop`` — a blocking call: a synchronous fsync, a subprocess
  wait, a large parse. On the loop it stalls every other turn.

Recorded rather than raised by default. A rule that crashes the runtime the
first time somebody breaks it gets turned off; a rule that counts gets fixed.
``strictly`` is there for tests, which want the failure loud.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterator

logger = logging.getLogger("Aura.WhichThreadMayDoThis")

__all__ = [
    "AKindOfWork",
    "OnTheWrongThread",
    "how_it_has_gone",
    "note_the_loop_thread",
    "the_loop_thread",
    "this_is",
    "we_are_on_the_loop",
]


class AKindOfWork(StrEnum):
    """What thread a piece of work belongs on."""

    ON_THE_LOOP = "on the loop"
    ANYWHERE = "anywhere"
    NEVER_ON_THE_LOOP = "never on the loop"


class OnTheWrongThread(RuntimeError):
    """Work ran on a thread it must not. Raised only under ``strictly``."""


@dataclass
class _Broken:
    """One rule, and how often it has been broken."""

    what: str
    kind: AKindOfWork
    times: int = 0
    first_at: float = 0.0
    last_at: float = 0.0
    seconds_spent: float = 0.0
    where: list[str] = field(default_factory=list)


_LOOP_THREAD: int | None = None
_BROKEN: dict[str, _Broken] = {}
_STRICT = threading.local()

#: How many distinct call sites are kept per broken rule. Enough to find it.
_HOW_MANY_SITES = 5


def note_the_loop_thread() -> int:
    """Say that this thread is the event loop's. Called once, at boot."""
    global _LOOP_THREAD
    _LOOP_THREAD = threading.get_ident()
    logger.debug("the event loop is thread %d", _LOOP_THREAD)
    return _LOOP_THREAD


def the_loop_thread() -> int | None:
    """Which thread the loop is on, or None before anything said."""
    return _LOOP_THREAD


def we_are_on_the_loop() -> bool:
    """Whether this is the loop's thread.

    False before the loop has said who it is, which is the safe answer: a
    check that assumed yes would let a blocking call through unmeasured.
    """
    return _LOOP_THREAD is not None and threading.get_ident() == _LOOP_THREAD


def _wrong(what: str, kind: AKindOfWork, spent: float) -> None:
    broken = _BROKEN.setdefault(what, _Broken(what=what, kind=kind))
    broken.times += 1
    broken.last_at = time.time()
    broken.first_at = broken.first_at or broken.last_at
    broken.seconds_spent += spent
    where = _where_from()
    if where and where not in broken.where:
        broken.where.append(where)
        del broken.where[:-_HOW_MANY_SITES]


#: Frames that are the plumbing rather than the caller. `contextlib` is how
#: this is entered, so without it every site reads as contextlib.py:144.
_NOT_THE_CALLER = ("contextlib.py", "which_thread_may_do_this.py")


def _where_from() -> str:
    """The first frame that is neither this module nor the machinery."""
    import traceback

    for frame in reversed(traceback.extract_stack()[:-2]):
        name = frame.filename.rsplit("/", 1)[-1]
        if name in _NOT_THE_CALLER:
            continue
        return f"{frame.filename.rsplit('/', 2)[-1]}:{frame.lineno}"
    return ""


@contextmanager
def this_is(kind: AKindOfWork, what: str, *, strictly: bool = False) -> Iterator[None]:
    """Declare what kind of work this is, and check the thread it is on.

    ``strictly`` raises instead of recording. Off by default because a rule
    that crashes the runtime the first time somebody breaks it gets turned
    off, and a rule that counts gets fixed.
    """
    on_the_loop = we_are_on_the_loop()
    wrong = (
        (kind is AKindOfWork.NEVER_ON_THE_LOOP and on_the_loop)
        or (kind is AKindOfWork.ON_THE_LOOP and _LOOP_THREAD is not None
            and not on_the_loop)
    )
    if wrong and (strictly or getattr(_STRICT, "on", False)):
        raise OnTheWrongThread(
            f"{what} is {kind} and ran on "
            + ("the loop thread" if on_the_loop else "a worker thread")
        )
    began = time.monotonic()
    try:
        yield
    finally:
        if wrong:
            spent = time.monotonic() - began
            _wrong(what, kind, spent)
            logger.warning(
                "%s is %s and ran on %s for %.3fs",
                what, kind,
                "the loop thread" if on_the_loop else "a worker thread",
                spent,
            )


@contextmanager
def strictly() -> Iterator[None]:
    """Make every check in this thread raise. For tests."""
    was = getattr(_STRICT, "on", False)
    _STRICT.on = True
    try:
        yield
    finally:
        _STRICT.on = was


def how_it_has_gone() -> dict[str, Any]:
    """Every rule that has been broken, how often, and where.

    ``seconds_on_the_loop`` is the number that matters: a blocking call that
    takes a millisecond on the loop is untidy, and one that takes twenty
    minutes is the runtime being down.
    """
    return {
        "loop_thread": _LOOP_THREAD,
        "rules_broken": len(_BROKEN),
        "seconds_on_the_loop": round(
            sum(
                one.seconds_spent
                for one in _BROKEN.values()
                if one.kind is AKindOfWork.NEVER_ON_THE_LOOP
            ),
            3,
        ),
        "broken": {
            what: {
                "kind": str(one.kind),
                "times": one.times,
                "seconds": round(one.seconds_spent, 3),
                "where": list(one.where),
            }
            for what, one in sorted(_BROKEN.items())
        },
    }


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    global _LOOP_THREAD
    _LOOP_THREAD = None
    _BROKEN.clear()
