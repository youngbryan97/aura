"""What every background task promises about its own ending.

The tracker in :mod:`core.utils.task_tracker` already knows which tasks are
alive, who owns them, and it reports what survived a shutdown. What it does
not know is what any of them *should* do when the runtime is going down. It
has one bit — shutdown-critical or not — and one global timeout for all of
them. So a five-second drain is applied to a task that needed twenty and to a
task that could have been dropped instantly, and neither owner ever said which
it was.

The three things an owner has to declare:

* **when it may be cancelled.** Straight away, after its current unit of work,
  or not until it finishes.
* **how long its drain may take.** A number, per owner, not one number for the
  whole runtime.
* **what an orphan means.** A task of this owner surviving shutdown is either
  expected noise or a defect, and only the owner knows which.

The point of writing it down is the question it makes answerable:
:func:`owners_that_have_not_said` names every owner that spawns tasks and has
never declared any of this. A count of live tasks cannot tell you that.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.HowATaskShouldEnd")

__all__ = [
    "WhenToCancel",
    "AnEndingPolicy",
    "declare",
    "the_policy_for",
    "the_drain_deadline_for",
    "owners_that_have_not_said",
    "how_the_endings_are_declared",
    "note_an_owner_spawned",
    "forget_everything",
]


class WhenToCancel(StrEnum):
    """How soon this owner's tasks may be interrupted."""

    #: Cancel immediately. Losing the work costs nothing.
    AT_ONCE = "at once"
    #: Cancel at the next checkpoint the task itself declares.
    AT_A_CHECKPOINT = "at a checkpoint"
    #: Do not cancel. Wait for the drain deadline, then record an orphan.
    ONLY_WHEN_DONE = "only when done"


@dataclass(frozen=True, slots=True)
class AnEndingPolicy:
    """One owner's answer to all three questions."""

    owner: str
    when: WhenToCancel
    drain_seconds: float
    #: True where a surviving task of this owner is a defect worth a
    #: degradation record, rather than expected noise from a slow drain.
    an_orphan_is_a_defect: bool
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "when": str(self.when),
            "drain_seconds": self.drain_seconds,
            "an_orphan_is_a_defect": self.an_orphan_is_a_defect,
            "why": self.why,
        }


#: What ships declared. Owners here are the ones whose task lifetimes were
#: read rather than guessed; everything else shows up in
#: :func:`owners_that_have_not_said` until somebody reads it too.
_DECLARED: dict[str, AnEndingPolicy] = {}
_SPAWNED: dict[str, int] = {}
_LOCK = threading.Lock()

#: The tracker's own default, used where an owner has said nothing. Keeping it
#: explicit is the difference between a default and an accident.
THE_DEFAULT = AnEndingPolicy(
    owner="(undeclared)",
    when=WhenToCancel.AT_ONCE,
    drain_seconds=5.0,
    an_orphan_is_a_defect=False,
    why="nobody said, so the runtime's old behaviour applies",
)


def declare(
    owner: str,
    *,
    when: WhenToCancel,
    drain_seconds: float,
    an_orphan_is_a_defect: bool,
    why: str,
) -> AnEndingPolicy:
    """Say how this owner's background tasks should end."""
    if drain_seconds < 0:
        raise ValueError(f"{owner}: a drain deadline cannot be negative")
    if when is WhenToCancel.AT_ONCE and drain_seconds > 0:
        logger.debug(
            "%s says cancel at once but asks for a %.1fs drain; the drain wins "
            "only for work already in flight",
            owner,
            drain_seconds,
        )
    policy = AnEndingPolicy(
        owner=owner,
        when=when,
        drain_seconds=float(drain_seconds),
        an_orphan_is_a_defect=an_orphan_is_a_defect,
        why=why,
    )
    with _LOCK:
        _DECLARED[owner] = policy
    return policy


def the_policy_for(owner: str | None) -> AnEndingPolicy:
    """This owner's declaration, or the runtime's stated default."""
    if not owner:
        return THE_DEFAULT
    with _LOCK:
        return _DECLARED.get(owner, THE_DEFAULT)


def the_drain_deadline_for(owner: str | None, *, ceiling: float) -> float:
    """How long to wait for this owner, never past the shutdown's own ceiling."""
    return min(float(ceiling), the_policy_for(owner).drain_seconds)


def note_an_owner_spawned(owner: str | None) -> None:
    """Record that this owner started a background task.

    Cheap on purpose: it is called on a spawn path, so it counts and returns.
    """
    if not owner:
        owner = "(unnamed)"
    with _LOCK:
        _SPAWNED[owner] = _SPAWNED.get(owner, 0) + 1


def owners_that_have_not_said() -> tuple[str, ...]:
    """Owners that spawned background work and never declared how it ends."""
    with _LOCK:
        return tuple(sorted(set(_SPAWNED) - set(_DECLARED)))


def how_the_endings_are_declared() -> dict[str, Any]:
    """For the health report: who declared, who did not, and what they said."""
    with _LOCK:
        declared = {name: p.to_dict() for name, p in sorted(_DECLARED.items())}
        spawned = dict(sorted(_SPAWNED.items()))
    silent = tuple(sorted(set(spawned) - set(declared)))
    return {
        "declared": len(declared),
        "owners_that_spawned": len(spawned),
        "owners_that_have_not_said": list(silent),
        "orphan_is_a_defect_for": sorted(
            name for name, p in declared.items() if p["an_orphan_is_a_defect"]
        ),
        "longest_drain": max(
            (p["drain_seconds"] for p in declared.values()), default=0.0
        ),
        "policies": declared,
    }


def forget_everything() -> None:
    with _LOCK:
        _DECLARED.clear()
        _SPAWNED.clear()
    _declare_what_ships()


def _declare_what_ships() -> None:
    """The owners whose lifetimes were read out of the code that spawns them."""
    declare(
        "cognitive_engine",
        when=WhenToCancel.AT_A_CHECKPOINT,
        drain_seconds=8.0,
        an_orphan_is_a_defect=True,
        why="a half-finished turn leaves state written by some phases and not others",
    )
    declare(
        "file_write_gateway",
        when=WhenToCancel.ONLY_WHEN_DONE,
        drain_seconds=20.0,
        an_orphan_is_a_defect=True,
        why="cancelling mid-write is how a state file is truncated",
    )
    declare(
        "event_spine",
        when=WhenToCancel.ONLY_WHEN_DONE,
        drain_seconds=10.0,
        an_orphan_is_a_defect=True,
        why="a dropped append breaks the hash chain and the next boot cannot verify it",
    )
    declare(
        "curiosity",
        when=WhenToCancel.AT_ONCE,
        drain_seconds=0.0,
        an_orphan_is_a_defect=False,
        why="background wondering; nothing downstream waits on it",
    )
    declare(
        "telemetry",
        when=WhenToCancel.AT_ONCE,
        drain_seconds=1.0,
        an_orphan_is_a_defect=False,
        why="a lost sample is a gap in a chart, not a corrupted state",
    )
    declare(
        "memory_consolidation",
        when=WhenToCancel.AT_A_CHECKPOINT,
        drain_seconds=15.0,
        an_orphan_is_a_defect=True,
        why="consolidation rewrites what recall reads; stopping halfway is worse than not starting",
    )
    declare(
        "model_lane",
        when=WhenToCancel.AT_A_CHECKPOINT,
        drain_seconds=12.0,
        an_orphan_is_a_defect=False,
        why="a generation can be abandoned, but the lane must be released or the next turn waits forever",
    )
    declare(
        "shutdown_coordinator",
        when=WhenToCancel.ONLY_WHEN_DONE,
        drain_seconds=25.0,
        an_orphan_is_a_defect=True,
        why="the thing running the shutdown cannot be cancelled by the shutdown",
    )


_declare_what_ships()
