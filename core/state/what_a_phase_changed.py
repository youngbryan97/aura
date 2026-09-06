"""Working-memory changes buffered, committed at the phase boundary, refcounted.

Soar scored above Aura on engineering maturity even after its reputation was
stripped away, and the reason given was authority: one agent state, one
decision cycle over it, and working-memory changes buffered and committed at
defined phase boundaries with explicit addition, removal and refcount
semantics. Aura has far more varieties of internal process; Soar more often
makes "which mechanism owns this transition?" obvious.

Two things follow from buffering rather than mutating in place.

A phase sees the state as it was at the boundary, not as the phase before it
left it half-changed. Which means a phase's result does not depend on how far
through the previous phase the reader happened to look.

And a removal is refcounted. Two phases can both add the same item for their
own reasons, and one of them removing it must not take away the other's. This
is the semantics a list of dicts silently does not have: ``remove()`` on a
list takes the first match away from whoever put it there.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhatAPhaseChanged")

__all__ = [
    "AChange",
    "ABoundary",
    "at_the_boundary",
    "how_the_boundaries_have_gone",
    "the_boundary_for",
]


def _a_key(item: Any) -> str:
    """A stable name for an item, so two equal items refcount together.

    Dicts are unhashable and working memory is a list of them, so the key is
    the item's own content in a canonical order. Falls back to repr for
    anything that will not serialise, which is worse but never wrong in a way
    that merges two different items.
    """
    try:
        return json.dumps(item, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(item)


@dataclass(frozen=True)
class AChange:
    """One addition or removal, waiting for the boundary."""

    what: Any
    added: bool
    by: str


@dataclass
class ABoundary:
    """What one phase changed, and what it will take to undo it."""

    phase: str
    changes: list[AChange] = field(default_factory=list)

    def add(self, item: Any) -> None:
        self.changes.append(AChange(what=item, added=True, by=self.phase))

    def remove(self, item: Any) -> None:
        self.changes.append(AChange(what=item, added=False, by=self.phase))

    @property
    def additions(self) -> int:
        return sum(1 for one in self.changes if one.added)

    @property
    def removals(self) -> int:
        return sum(1 for one in self.changes if not one.added)


#: How many times each item has been added, and by whom. The refcount is per
#: item; the names are what makes a stuck count answerable rather than a
#: mystery.
_HELD: Counter[str] = Counter()
_WHO: dict[str, set[str]] = {}
_LOCK = checked_lock("core.state.what_a_phase_changed")

_HOW_IT_WENT: dict[str, dict[str, int]] = {}


def the_boundary_for(phase: str) -> ABoundary:
    """A buffer for one phase. Nothing it records happens until it commits."""
    return ABoundary(phase=str(phase))


def _commit(state: Any, boundary: ABoundary) -> dict[str, int]:
    """Apply a phase's buffered changes to the working memory."""
    from core.state.one_working_memory import the_working_memory

    memory = the_working_memory(state)
    added = removed = held_by_another = 0
    with _LOCK:
        for change in boundary.changes:
            key = _a_key(change.what)
            if change.added:
                _HELD[key] += 1
                _WHO.setdefault(key, set()).add(change.by)
                if _HELD[key] == 1:
                    memory.append(change.what)
                    added += 1
                continue
            if _HELD.get(key, 0) <= 0:
                # Removing something nobody added. Counted rather than raised:
                # a phase that cleans up defensively is not a defect, and a
                # boundary that raised here would make cleanup dangerous.
                continue
            _HELD[key] -= 1
            _WHO.get(key, set()).discard(change.by)
            if _HELD[key] > 0:
                held_by_another += 1
                continue
            del _HELD[key]
            _WHO.pop(key, None)
            for index, held in enumerate(memory):
                if _a_key(held) == key:
                    del memory[index]
                    removed += 1
                    break
    went = {
        "additions": added,
        "removals": removed,
        "still_held_by_another": held_by_another,
    }
    _HOW_IT_WENT[boundary.phase] = {
        name: _HOW_IT_WENT.get(boundary.phase, {}).get(name, 0) + count
        for name, count in went.items()
    }
    return went


@contextmanager
def at_the_boundary(state: Any, phase: str) -> Iterator[ABoundary]:
    """Buffer this phase's working-memory changes; commit them on the way out.

    Commits on the way out even when the phase raises. A phase that fails
    halfway has still done the part it did, and dropping those changes would
    make the state depend on where the exception happened rather than on what
    the phase decided.
    """
    boundary = the_boundary_for(phase)
    try:
        yield boundary
    finally:
        went = _commit(state, boundary)
        if went["additions"] or went["removals"]:
            logger.debug(
                "%s committed +%d -%d at the boundary",
                phase, went["additions"], went["removals"],
            )


def how_the_boundaries_have_gone() -> dict[str, Any]:
    """What each phase has committed, and what is still held by more than one."""
    with _LOCK:
        shared = sorted(
            (count, sorted(_WHO.get(key, ())), key[:80])
            for key, count in _HELD.items()
            if count > 1
        )
    return {
        "by_phase": {name: dict(went) for name, went in sorted(_HOW_IT_WENT.items())},
        "items_held": len(_HELD),
        "held_by_more_than_one": [
            {"count": count, "by": who, "item": what} for count, who, what in shared
        ],
    }


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    with _LOCK:
        _HELD.clear()
        _WHO.clear()
        _HOW_IT_WENT.clear()
