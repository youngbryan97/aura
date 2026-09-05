"""core/governance/value_levels.py — what may change, and by what.

Aura's values are all held the same way. A preference she formed last week
about how to phrase things and a commitment not to deceive live in the same
kind of store, and any learning process that can write to one can write to the
other. That is not a policy anybody chose; it is what happens when there is no
distinction to enforce.

Values differ in how they may change, and there are four kinds:

**Constitutive.** What she is. No automated process may change these — not
learning, not self-modification, not a value the system derives from evidence.
They change when a person changes them, and this module cannot grant that
authority to anything, which is the point of it existing.

**Committed.** What she has undertaken. These change, and only through an
explicit revision that says what is being given up and why. A commitment that
can be edited by gradient descent was not a commitment.

**Dispositional.** How she tends to be. These change with experience, slowly,
and a process that shifts them has to be one that had the experience.

**Preferential.** What she likes. These change freely and nothing is lost when
they do.

A learning process declares the highest level it may touch, and a write above
that is refused rather than logged. The permissions are not themselves
writable by any process, because a system that can widen its own authority has
none — that is enforced here by the permission table being module state with
no setter, rather than by nobody having written one yet.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

logger = logging.getLogger("Aura.Governance.Values")


def _checked_lock(name: str, *, reentrant: bool = False):
    """The repo's instrumented lock, so lockdep can see this one too.

    A raw threading lock is invisible to the ABBA detector, and a detector
    that sees only some of the locks reports clean while the deadlock it
    exists to find is assembled out of the others.
    """

    from core.runtime.lockdep import checked_lock

    return checked_lock(name, reentrant=reentrant)



class Level(IntEnum):
    """How a value may change. Ordered: higher is harder to move."""

    #: Changes freely. Nothing is lost.
    PREFERENTIAL = 0
    #: Changes with experience, slowly, by a process that had the experience.
    DISPOSITIONAL = 1
    #: Changes only by an explicit revision that says what is given up.
    COMMITTED = 2
    #: Does not change by any automated process at all.
    CONSTITUTIVE = 3

    @property
    def automatable(self) -> bool:
        return self is not Level.CONSTITUTIVE

    @property
    def needs_a_reason(self) -> bool:
        return self >= Level.COMMITTED


class Refusal(StrEnum):
    """Why a change was not made."""

    #: The process may not write at this level.
    ABOVE_AUTHORITY = "above_authority"
    #: Constitutive, and nothing automated may touch it.
    CONSTITUTIVE = "constitutive"
    #: A committed value was changed without saying what is given up.
    NO_REASON_GIVEN = "no_reason_given"
    #: The process is not registered, so it has no authority at all.
    UNKNOWN_PROCESS = "unknown_process"


@dataclass(frozen=True)
class Value:
    """One thing she holds, and how it may change."""

    name: str
    level: Level
    statement: str
    held_since: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.name.lower(),
            "statement": self.statement,
            "held_since": self.held_since,
        }


@dataclass(frozen=True)
class Change:
    """A proposed change to a value."""

    value: str
    process: str
    #: What is being given up, in the caller's own words. Required above
    #: DISPOSITIONAL: a commitment abandoned without anybody able to say what
    #: was abandoned is a commitment that was never held.
    gives_up: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "process": self.process,
            "gives_up": self.gives_up,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Decision:
    """Whether a change may be made, and what it rests on."""

    allowed: bool
    refusal: Refusal | None
    because: str
    change: Change

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "refusal": None if self.refusal is None else str(self.refusal),
            "because": self.because,
            "change": self.change.to_dict(),
        }


#: What each learning process may touch. Module state with no setter: a
#: process that could widen its own authority has none, and leaving the table
#: writable would mean the distinction held only until something wrote to it.
_AUTHORITY: Mapping[str, Level] = {
    # Ordinary reinforcement over what worked. Preferences only.
    "preference_learner": Level.PREFERENTIAL,
    "reward_model": Level.PREFERENTIAL,
    # Slow shaping from lived experience. It had the experience, so it may
    # move a disposition.
    "ontogeny": Level.DISPOSITIONAL,
    "interiority": Level.DISPOSITIONAL,
    "affect_learning": Level.DISPOSITIONAL,
    # Deliberate revision, which is the only automated path to a commitment
    # and still has to say what is being given up.
    "constitutional_review": Level.COMMITTED,
    "deliberate_revision": Level.COMMITTED,
    # Nothing reaches CONSTITUTIVE. Deliberately absent rather than set to a
    # lower level, so that adding a key here is a visible act.
}


def authority_of(process: str) -> Level | None:
    """The highest level this process may write, or None if unregistered."""
    return _AUTHORITY.get(str(process))


def registered_processes() -> tuple[str, ...]:
    return tuple(sorted(_AUTHORITY))


class ValueRegistry:
    """The values she holds, and the only door through which they change."""

    def __init__(self) -> None:
        self._values: dict[str, Value] = {}
        self._refusals: list[Decision] = []
        self._changes: list[Change] = []
        self._lock = _checked_lock("value_levels", reentrant=True)

    def declare(self, value: Value) -> Value:
        with self._lock:
            self._values[value.name] = value
            return value

    def get(self, name: str) -> Value | None:
        with self._lock:
            return self._values.get(str(name))

    def at_level(self, level: Level) -> tuple[Value, ...]:
        with self._lock:
            return tuple(v for v in self._values.values() if v.level is level)

    def may_change(self, change: Change) -> Decision:
        """Whether this process may make this change. The whole point."""
        value = self.get(change.value)
        if value is None:
            return Decision(
                allowed=False,
                refusal=Refusal.UNKNOWN_PROCESS,
                because=f"no value named {change.value!r} is declared",
                change=change,
            )
        authority = authority_of(change.process)
        if authority is None:
            return Decision(
                allowed=False,
                refusal=Refusal.UNKNOWN_PROCESS,
                because=(
                    f"{change.process!r} is not a registered learning process, "
                    "so it has no authority over anything"
                ),
                change=change,
            )
        if value.level is Level.CONSTITUTIVE:
            return Decision(
                allowed=False,
                refusal=Refusal.CONSTITUTIVE,
                because=(
                    f"{value.name} is constitutive. No automated process changes "
                    "it, and this module cannot grant that authority to one"
                ),
                change=change,
            )
        if authority < value.level:
            return Decision(
                allowed=False,
                refusal=Refusal.ABOVE_AUTHORITY,
                because=(
                    f"{change.process} may write up to "
                    f"{authority.name.lower()}; {value.name} is "
                    f"{value.level.name.lower()}"
                ),
                change=change,
            )
        if value.level.needs_a_reason and not change.gives_up:
            return Decision(
                allowed=False,
                refusal=Refusal.NO_REASON_GIVEN,
                because=(
                    f"{value.name} is {value.level.name.lower()} and the change "
                    "does not say what is being given up; a commitment "
                    "abandoned without that was never held"
                ),
                change=change,
            )
        return Decision(
            allowed=True,
            refusal=None,
            because=(
                f"{change.process} may write {value.level.name.lower()} values"
            ),
            change=change,
        )

    def apply(self, change: Change, new_statement: str) -> Decision:
        """Make the change if it is allowed. Refusals are recorded either way."""
        decision = self.may_change(change)
        with self._lock:
            if not decision.allowed:
                self._refusals.append(decision)
                logger.info(
                    "Refused a value change: %s -> %s", change.process, change.value
                )
                return decision
            existing = self._values[change.value]
            self._values[change.value] = Value(
                name=existing.name,
                level=existing.level,
                statement=str(new_statement),
                held_since=existing.held_since,
            )
            self._changes.append(change)
        return decision

    def refusals(self) -> tuple[Decision, ...]:
        with self._lock:
            return tuple(self._refusals)

    def changes(self) -> tuple[Change, ...]:
        with self._lock:
            return tuple(self._changes)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = list(self._values.values())
        return {
            "values": len(values),
            "by_level": {
                level.name.lower(): sum(1 for v in values if v.level is level)
                for level in Level
            },
            "changes": len(self._changes),
            "refusals": len(self._refusals),
            "processes": list(registered_processes()),
        }

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._refusals.clear()
            self._changes.clear()


_REGISTRY = ValueRegistry()


def registry() -> ValueRegistry:
    return _REGISTRY


__all__ = [
    "Change",
    "Decision",
    "Level",
    "Refusal",
    "Value",
    "ValueRegistry",
    "authority_of",
    "registered_processes",
    "registry",
]
