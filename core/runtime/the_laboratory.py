"""core/runtime/the_laboratory.py — step it, freeze it, look at it, again.

Soar can be told to run for one phase, one decision or N cycles, and that one
capability is why its experiments reproduce. Aura's runtime is continuous and
event driven, which is right for a thing that lives on a desktop and wrong for
an experiment: two runs of the same question differ because the clock moved,
because a background tick fired between two phases, because a search was
bounded by seconds and the host was busy.

This session found that three separate times — a search bounded by a wall
clock gave a family on an idle machine and refused it on a loaded one, an
ablation reversed itself twice under adaptive depth, and a measurement of the
same eighty-seven tasks took four hundred and fourteen seconds once and would
not finish the next time. Every one of those is the same missing thing.

So: a mode where time does not move unless somebody moves it, the background
clocks are held, the seeds are fixed, and one step means one step.

Three properties matter and the third is the one that makes it honest.

* Time is a number this module owns. ``now()`` in laboratory mode returns the
  virtual clock, and ``advance(seconds)`` is the only thing that changes it.
  A subsystem that reads ``time.monotonic()`` directly is not in the
  laboratory, and ``what_still_reads_the_wall_clock`` says which those are.
* Nothing starts on its own. Background work registers itself as due at a
  virtual time and runs when the clock reaches it, in a deterministic order.
* The mode is visible. A receipt made in the laboratory says so, because a
  measurement taken with the clock held is a different measurement and
  reporting it as an ordinary one would be the dishonest part.
"""

from __future__ import annotations

import logging
import random
import threading
import time as _time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.TheLaboratory")

__all__ = [
    "ALaboratory",
    "in_the_laboratory",
    "under_the_laboratory",
    "the_laboratory",
    "now",
    "seeded",
    "what_still_reads_the_wall_clock",
]

_LOCK = threading.RLock()
_ACTIVE: "ALaboratory | None" = None
_WALL_CLOCK_READERS: dict[str, int] = {}


@dataclass
class _Due:
    at: float
    order: int
    what: Callable[[], Any]
    name: str


@dataclass
class ALaboratory:
    """A held clock, a fixed seed, and work that only runs when told."""

    #: Where the virtual clock starts. Any number; what matters is that it
    #: moves only when advance() is called.
    started_at: float = 1_000_000.0
    seed: int = 0
    _at: float = field(default=0.0, init=False)
    _due: list[_Due] = field(default_factory=list, init=False)
    _issued: int = field(default=0, init=False)
    _steps: int = field(default=0, init=False)
    _ran: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._at = float(self.started_at)

    @property
    def at(self) -> float:
        return self._at

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def what_ran(self) -> tuple[str, ...]:
        return tuple(self._ran)

    def due_in(self, seconds: float, what: Callable[[], Any], *, name: str = "") -> None:
        """Register work for a virtual time. It will not run until then."""

        with _LOCK:
            self._issued += 1
            self._due.append(
                _Due(
                    at=self._at + max(0.0, float(seconds)),
                    order=self._issued,
                    what=what,
                    name=name or f"work-{self._issued}",
                )
            )

    def advance(self, seconds: float) -> tuple[str, ...]:
        """Move the clock, running whatever comes due, in a fixed order.

        Ties break on registration order rather than on anything about the
        work, so the same registrations always produce the same sequence.
        """

        ran: list[str] = []
        target = self._at + max(0.0, float(seconds))
        while True:
            with _LOCK:
                ready = sorted(
                    (one for one in self._due if one.at <= target),
                    key=lambda one: (one.at, one.order),
                )
                if not ready:
                    self._at = target
                    break
                first = ready[0]
                self._due.remove(first)
                self._at = first.at
            ran.append(first.name)
            self._ran.append(first.name)
            try:
                first.what()
            except Exception as exc:  # noqa: BLE001 — one piece of work failing is a result
                logger.info("laboratory work %s raised: %s", first.name, exc)
        return tuple(ran)

    def step(self, what: Callable[[], Any], *, name: str = "") -> Any:
        """Run one thing, count it, and do not move the clock.

        The unit Soar calls a step. Time not moving is the point: two phases
        run back to back see the same clock, so anything that differs between
        them differs for a reason other than when it happened.
        """

        self._steps += 1
        self._ran.append(name or f"step-{self._steps}")
        return what()

    def report(self) -> dict[str, Any]:
        with _LOCK:
            waiting = sorted((one.at, one.name) for one in self._due)
        return {
            "in_the_laboratory": True,
            "at": round(self._at, 6),
            "since_it_started": round(self._at - self.started_at, 6),
            "steps": self._steps,
            "ran": list(self._ran)[-32:],
            "waiting": [{"at": round(at, 6), "name": name} for at, name in waiting],
            "seed": self.seed,
        }


def the_laboratory() -> ALaboratory | None:
    """The laboratory this thread is in, or None for the ordinary world."""

    return _ACTIVE


def in_the_laboratory() -> bool:
    return _ACTIVE is not None


def now(*, whose: str = "") -> float:
    """The time, virtual in the laboratory and monotonic outside it.

    ``whose`` records that a caller asked through here. A caller that reads
    ``time.monotonic()`` directly is outside the laboratory whatever mode is
    set, and the difference between the two lists is the migration.
    """

    lab = _ACTIVE
    if lab is not None:
        return lab.at
    if whose:
        _WALL_CLOCK_READERS[whose] = _WALL_CLOCK_READERS.get(whose, 0) + 1
    return _time.monotonic()


def what_still_reads_the_wall_clock() -> dict[str, int]:
    """Who asked for the time outside the laboratory, and how often."""

    return dict(_WALL_CLOCK_READERS)


def seeded(salt: str = "") -> random.Random:
    """A generator fixed by the laboratory's seed, or an ordinary one outside.

    Salted per caller so two subsystems drawing from "the seed" do not draw
    the same numbers — which is a bug that looks like a coincidence.
    """

    lab = _ACTIVE
    if lab is None:
        return random.Random()
    return random.Random(f"{lab.seed}:{salt}")


@contextmanager
def under_the_laboratory(
    *, seed: int = 0, started_at: float = 1_000_000.0
) -> Iterator[ALaboratory]:
    """Hold the clocks for this block. Nested entry is an error, not a nesting.

    Two laboratories would each think they own the time, and the inner one
    leaving would hand the outer one a clock that had moved.
    """

    global _ACTIVE
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("already in the laboratory; nesting would share a clock")
        made = ALaboratory(started_at=started_at, seed=seed)
        _ACTIVE = made
    try:
        yield made
    finally:
        with _LOCK:
            _ACTIVE = None
