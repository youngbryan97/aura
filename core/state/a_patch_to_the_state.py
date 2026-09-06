"""core/state/a_patch_to_the_state.py — what one organ hands another.

LangGraph's nodes return ``Partial[State]`` and the framework applies it with a
declared reducer per key. That is a smaller idea than it looks: the value is
not the immutability, it is that a write becomes a VALUE — something you can
log, compare, refuse, replay and merge — instead of an assignment that has
already happened by the time anyone could object.

Aura derives a new state and phases mutate fields on the copy. Inside one
organ that is right and this does not touch it: an organ owning a field and
changing it is the organ doing its job. What has no representation is the
cross-organ write, and that is the one worth having as a value, because it is
the one another organ has to reason about.

So: a patch is a set of dotted paths and their new values, with who wrote it
and why. Applying it goes through the same write-mode declaration the compiled
plan uses, so "several phases write this field and the order settles it" is
one rule with one place to change it rather than two mechanisms that can
disagree.

What this does not do is pretend the conversion is finished. Every direct
mutation still works, ``how_much_still_bypasses_this`` counts what has not
moved, and the number is the migration.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.APatchToTheState")

__all__ = [
    "APatch",
    "apply_a_patch",
    "how_much_still_bypasses_this",
    "note_a_direct_write",
    "read_a_path",
]

_LOCK = threading.Lock()
_DIRECT_WRITES: dict[str, int] = {}


@dataclass(frozen=True)
class APatch:
    """A cross-organ write, as a value.

    ``changes`` are dotted paths into the state. A path rather than a nested
    dict because the reducer is declared per path, and a nested dict makes the
    question "which key is this" a parse rather than a lookup.
    """

    changes: Mapping[str, Any]
    #: The organ that wrote it. Not decoration: the write mode for a field
    #: that several organs write is settled by the plan's order, and the
    #: order is over organs.
    by: str = ""
    #: One line. What a reader of the log needs to know about this write.
    because: str = ""
    #: The plan seal this was written under, where the writer knows it.
    under: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "by": self.by,
            "because": self.because,
            "under": self.under,
            "changes": dict(self.changes),
        }

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.changes))


def read_a_path(state: Any, path: str) -> Any:
    """The value at a dotted path, or None where the path does not resolve."""

    found: Any = state
    for step in str(path or "").split("."):
        if not step:
            return None
        if isinstance(found, Mapping):
            found = found.get(step)
        else:
            found = getattr(found, step, None)
        if found is None:
            return None
    return found


def _combine(mode: str, was: Any, now: Any) -> Any:
    """What the field becomes, given the declared mode."""

    if mode in {"last in the order", "single writer"} or was is None:
        return now
    try:
        if mode == "highest":
            return max(was, now)
        if mode == "lowest":
            return min(was, now)
        if mode == "sum":
            return was + now
        if mode == "union":
            return sorted(set(was) | set(now)) if not isinstance(was, set) else was | now
    except (TypeError, ValueError) as exc:
        logger.info("could not combine %r and %r by %s: %s", was, now, mode, exc)
    return now


def apply_a_patch(state: Any, patch: APatch) -> dict[str, Any]:
    """Apply one patch. Returns what changed, per path, before and after.

    A path whose parent does not resolve is refused rather than created: an
    organ writing into a branch that does not exist is a typo, and creating it
    turns the typo into a field nothing reads.
    """

    changed: dict[str, Any] = {}
    refused: list[str] = []
    try:
        from core.runtime.the_shape_of_one_turn import write_mode_for
    except (ImportError, RuntimeError):  # pragma: no cover — the default is the order
        def write_mode_for(_path: str) -> str:
            return "last in the order"

    for path, value in sorted(patch.changes.items()):
        parts = str(path).split(".")
        if len(parts) < 2:
            owner, key = state, parts[0] if parts else ""
        else:
            owner = read_a_path(state, ".".join(parts[:-1]))
            key = parts[-1]
        if owner is None or not key:
            refused.append(path)
            continue
        was = owner.get(key) if isinstance(owner, Mapping) else getattr(owner, key, None)
        mode = write_mode_for(path) or "last in the order"
        now = _combine(mode, was, value)
        if isinstance(owner, dict):
            owner[key] = now
        else:
            try:
                setattr(owner, key, now)
            except (AttributeError, TypeError) as exc:
                logger.info("could not write %s: %s", path, exc)
                refused.append(path)
                continue
        changed[path] = {"was": was, "now": now, "combined_by": mode}
    return {
        "by": patch.by,
        "because": patch.because,
        "changed": changed,
        "refused": refused,
    }


def note_a_direct_write(who: str) -> None:
    """Record that an organ wrote across the boundary without a patch.

    Not an error. It is what every write does today, and counting it is how
    the conversion becomes a number rather than an intention.
    """

    with _LOCK:
        _DIRECT_WRITES[str(who)] = _DIRECT_WRITES.get(str(who), 0) + 1


def how_much_still_bypasses_this() -> dict[str, int]:
    with _LOCK:
        return dict(_DIRECT_WRITES)
