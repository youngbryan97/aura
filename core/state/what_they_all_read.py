"""A group of phases reads one state, not each other's half-finished work.

LangGraph runs parallel nodes as a superstep: every node in the group reads
the state as it was when the group started, writes into its own buffer, and
the buffers are merged at a barrier. Nothing a node writes is visible to its
siblings until the barrier. The closure asked for the same shape here.

Aura's parallel phase groups read live state. Two phases in a group can
therefore see different values for the same field depending on which finished
first, and the turn is not reproducible from its inputs. Worse, the failure is
invisible: both orderings produce an answer, and neither says which state it
read.

Three parts, and the third is the one that pays:

* **one snapshot** — the group reads what was there when it began.
* **a buffer each** — a phase's write is its own until the barrier.
* **a named conflict** — where two phases wrote the same field to different
  values, the merge says so instead of picking. Last-write-wins is how a race
  becomes a result nobody can explain.

Composes with :mod:`core.state.what_a_phase_changed`, which handles the other
half: the additions and removals a single phase makes at its own boundary.
This module is about what a *group* of them may see while they work.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatTheyAllRead")

__all__ = [
    "AConflictingWrite",
    "ASuperstep",
    "TheyDisagreed",
    "a_superstep_over",
    "how_the_supersteps_have_gone",
    "forget_everything",
]


class TheyDisagreed(RuntimeError):
    """Two phases in one group wrote the same field to different values."""


@dataclass(frozen=True)
class AConflictingWrite:
    """One field, two phases, two values."""

    field: str
    by: tuple[str, ...]
    values: tuple[Any, ...]

    def __str__(self) -> str:
        pairs = ", ".join(f"{who}={val!r}" for who, val in zip(self.by, self.values))
        return f"{self.field}: {pairs}"


@dataclass
class ASuperstep:
    """One parallel group: a frozen read, a buffer per phase, one barrier."""

    group: str
    _read: dict[str, Any]
    _wrote: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def read(self, key: str, default: Any = None) -> Any:
        """What the field held when the group began. Never what a sibling wrote."""
        return self._read.get(key, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._read))

    def write(self, phase: str, key: str, value: Any) -> None:
        """Buffer a write. Invisible to siblings until the barrier."""
        with self._lock:
            self._wrote.setdefault(phase, {})[key] = value

    def what_a_phase_wrote(self, phase: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._wrote.get(phase, {}))

    def conflicts(self) -> tuple[AConflictingWrite, ...]:
        """Fields two or more phases wrote to values that are not equal."""
        with self._lock:
            wrote = {phase: dict(kv) for phase, kv in self._wrote.items()}
        by_field: dict[str, list[tuple[str, Any]]] = {}
        for phase in sorted(wrote):
            for key, value in wrote[phase].items():
                by_field.setdefault(key, []).append((phase, value))
        found: list[AConflictingWrite] = []
        for key, writes in sorted(by_field.items()):
            if len(writes) < 2:
                continue
            values = [value for _, value in writes]
            if all(_same(values[0], other) for other in values[1:]):
                continue
            found.append(
                AConflictingWrite(
                    field=key,
                    by=tuple(who for who, _ in writes),
                    values=tuple(values),
                )
            )
        return tuple(found)

    def merged(self) -> dict[str, Any]:
        """The group's writes as one dict. Raises where they disagree."""
        clashes = self.conflicts()
        if clashes:
            raise TheyDisagreed(
                f"{self.group}: {len(clashes)} field(s) written twice — "
                + "; ".join(str(c) for c in clashes)
            )
        out: dict[str, Any] = {}
        with self._lock:
            for phase in sorted(self._wrote):
                out.update(self._wrote[phase])
        return out


def _same(a: Any, b: Any) -> bool:
    """Equality that survives values which refuse to compare."""
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 — identity is the answer when equality refuses
        # Not a loss and not a guess. A value whose __eq__ raises (a numpy
        # array, a lazily-loading proxy, a half-built object) has no equality
        # to report, and "the same object" is true of exactly the cases where
        # nothing changed. Nothing needs counting because nothing was dropped.
        return a is b


_HISTORY: list[dict[str, Any]] = []
_HISTORY_LOCK = threading.Lock()
_KEEP = 200


@contextmanager
def a_superstep_over(
    state: Any,
    group: str,
    *,
    fields: tuple[str, ...] | None = None,
    strict: bool = True,
) -> Iterator[ASuperstep]:
    """Run a parallel phase group against one snapshot, commit at the barrier.

    ``state`` may be a mapping or an object; ``fields`` names what to snapshot,
    defaulting to every public attribute that is not callable. ``strict``
    raises on a conflicting write; False records it and keeps the first
    phase's value in sorted-name order, which is at least deterministic.
    """
    snapshot = _snapshot(state, fields)
    step = ASuperstep(group=group, _read=snapshot)
    committed = 0
    clashes: tuple[AConflictingWrite, ...] = ()
    try:
        yield step
        clashes = step.conflicts()
        if clashes and strict:
            raise TheyDisagreed(
                f"{group}: {len(clashes)} field(s) written twice — "
                + "; ".join(str(c) for c in clashes)
            )
        with step._lock:
            wrote = {phase: dict(kv) for phase, kv in step._wrote.items()}
        settled: dict[str, Any] = {}
        for phase in sorted(wrote):
            for key, value in wrote[phase].items():
                settled.setdefault(key, value)
        for key, value in settled.items():
            _put(state, key, value)
            committed += 1
    finally:
        with _HISTORY_LOCK:
            _HISTORY.append(
                {
                    "group": group,
                    "read": len(snapshot),
                    "phases": len(step._wrote),
                    "committed": committed,
                    "conflicts": [str(c) for c in clashes],
                }
            )
            del _HISTORY[:-_KEEP]


def _snapshot(state: Any, fields: tuple[str, ...] | None) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state) if fields is None else {k: state.get(k) for k in fields}
    if fields is not None:
        return {k: getattr(state, k, None) for k in fields}
    out: dict[str, Any] = {}
    unreadable: list[str] = []
    for name in dir(state):
        if name.startswith("_"):
            continue
        try:
            value = getattr(state, name)
        except Exception as exc:  # noqa: BLE001 — a property that raises is data
            # Named rather than dropped. A state object whose properties raise
            # produces a smaller dict, and a smaller dict is indistinguishable
            # from a state that genuinely holds less — which is how a reader
            # comes to believe a field simply is not there.
            unreadable.append(f"{name} ({type(exc).__name__})")
            continue
        if callable(value):
            continue
        out[name] = value
    if unreadable:
        out["_unreadable"] = unreadable
    return out


def _put(state: Any, key: str, value: Any) -> None:
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def how_the_supersteps_have_gone() -> dict[str, Any]:
    """For the health report: how many groups ran, and where they disagreed."""
    with _HISTORY_LOCK:
        rows = list(_HISTORY)
    disagreed = [r for r in rows if r["conflicts"]]
    return {
        "supersteps": len(rows),
        "groups": sorted({r["group"] for r in rows}),
        "fields_committed": sum(r["committed"] for r in rows),
        "with_conflicts": len(disagreed),
        "conflicts": [c for r in disagreed for c in r["conflicts"]][:20],
        "recent": rows[-5:],
    }


def forget_everything() -> None:
    with _HISTORY_LOCK:
        _HISTORY.clear()
