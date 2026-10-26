"""Readings of `time.monotonic` that a snapshot holds, and their age after a restore.

A restore puts her state back and cannot put the machine's clock back. Much of
what she keeps is a `time.monotonic()` reading that something later turns into
an elapsed time: when volition last saw activity, when the body was last
sampled, when a goal last went on cooldown. Restored an hour after it was
taken, each of those came back an hour older, and every idle timer and cooldown
read an hour she never lived. Seed 71's interventional test rolled untouched
anchors an hour into its process and three of the eight came out different
(26 October), and moving `time.monotonic` an hour forward in a fresh process
did the same to the same three.

The wall clock has its own answer (the experiment clock, and `_reanchor` in
snapshot.py). The monotonic clock cannot be rewound, because asyncio's timeouts
read it, so the readings move instead. A snapshot tags every float it holds
that is a reading of this process's monotonic clock, and a restore hands each
one back at the same distance before now that it stood before the snapshot.

What counts as a reading is decided by its value. Every reading this process
can have made lies between the moment the process started and now, plus a day
for deadlines set ahead; a float in that window is taken as one. The window
sits at the machine's uptime, hundreds of thousands of seconds, and is a few
hours wide in a run. A count, a rate or a duration held as a float does not
land there by accident, and a census of what was tagged on a real snapshot is
in docs/ISC_V6_PREREGISTRATION.md (26 October addendum).
"""

from __future__ import annotations

import contextlib
import logging
import time
import types
from collections import deque
from collections.abc import Iterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["HORIZON_S", "PROCESS_START", "MonotonicInstant", "rebased", "tag_instants", "tag_snapshot"]

#: How far past the snapshot a reading can lie and still be one: a deadline
#: set for later, a cooldown that ends tomorrow.
HORIZON_S: float = 86400.0

#: How deep into a captured value the tagging walks. A captured organ is a
#: dict of fields, a field can be an object holding a list of records, and a
#: record holds the stamp: four levels is common, and cycles are cut by
#: identity, so the bound only stops a pathological chain.
_DEPTH: int = 24


def _process_start() -> float:
    """Where the monotonic clock stood when this process began.

    Without it the window starts at this import, and a reading made before
    then is carried unchanged, as every reading was before this module.
    """
    now = time.monotonic()
    try:
        import psutil
    except ImportError as exc:
        logger.info("process start unknown (%s); instants are counted from import", exc)
        return now
    from core.subject.clock import real_time

    try:
        started = psutil.Process().create_time()
    except (psutil.Error, OSError) as exc:
        logger.info("process start unknown (%s); instants are counted from import", exc)
        return now
    return now - max(0.0, real_time() - started) - 1.0


#: Where the monotonic clock stood when this process began.
PROCESS_START: float = _process_start()

#: How far a restore in progress moves each tagged instant; None outside one.
_SHIFT: float | None = None


class MonotonicInstant(float):
    """A float a snapshot holds that was a reading of `time.monotonic`.

    It reads as the number it was. Copied out of the snapshot during a
    restore, it comes back moved by however long the snapshot sat, so the
    elapsed time anything computes from it is the elapsed time it had.
    """

    __slots__ = ()

    def placed(self) -> float:
        return float(self) + (_SHIFT or 0.0)

    def __copy__(self) -> Any:
        return self if _SHIFT is None else self.placed()

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        return self if _SHIFT is None else self.placed()

    def __reduce__(self) -> tuple[Any, tuple[float]]:
        return (MonotonicInstant, (float(self),))


@contextlib.contextmanager
def rebased(taken_at: float | None) -> Iterator[None]:
    """Hand every tagged instant copied out in this block back at the age it had at `taken_at`."""
    global _SHIFT
    if taken_at is None:
        yield
        return
    previous = _SHIFT
    _SHIFT = time.monotonic() - float(taken_at)
    try:
        yield
    finally:
        _SHIFT = previous


def tag_instants(value: Any, low: float, high: float, seen: set[int], depth: int = 0) -> Any:
    """`value` with each float in (low, high] tagged; containers are changed in place.

    Returns the value to keep in the parent: a tagged float, a rebuilt tuple,
    or `value` itself.
    """
    kind = type(value)
    if kind is float:
        return MonotonicInstant(value) if low < value <= high else value
    if kind in (str, bytes, int, bool, type(None), MonotonicInstant) or isinstance(value, np.ndarray):
        return value
    if depth > _DEPTH or id(value) in seen:
        return value
    seen.add(id(value))
    deeper = depth + 1
    if isinstance(value, dict):
        for key, item in list(value.items()):
            new = tag_instants(item, low, high, seen, deeper)
            if new is not item:
                value[key] = new
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            new = tag_instants(item, low, high, seen, deeper)
            if new is not item:
                value[index] = new
        return value
    if isinstance(value, deque):
        items = [tag_instants(item, low, high, seen, deeper) for item in value]
        if any(new is not old for new, old in zip(items, value, strict=True)):
            value.clear()
            value.extend(items)
        return value
    if isinstance(value, tuple):
        items = [tag_instants(item, low, high, seen, deeper) for item in value]
        if all(new is old for new, old in zip(items, value, strict=True)):
            return value
        if kind is tuple:
            return tuple(items)
        make = getattr(kind, "_make", None)
        return make(items) if callable(make) else value
    if isinstance(value, (types.ModuleType, types.FunctionType, types.MethodType, type)):
        return value
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        for name, item in list(fields.items()):
            new = tag_instants(item, low, high, seen, deeper)
            if new is not item:
                fields[name] = new
    return value


def tag_snapshot(snapshot: Any, taken_at: float) -> None:
    """Tag the instants in everything a snapshot copied, and nothing it only points at.

    A module holding is kept as `(object, copy)`: the object is the live one,
    so only the copy is tagged. The effort ledger and the chaos engines are
    restored by their own methods, which take what they are given rather than
    copying it, and their numbers are not instants.
    """
    low, high = PROCESS_START, float(taken_at) + HORIZON_S
    seen: set[int] = set()
    for part in ("organs", "services", "phases", "singletons", "layers"):
        held = getattr(snapshot, part, None)
        if isinstance(held, dict):
            tag_instants(held, low, high, seen)
    state = getattr(snapshot, "state", None)
    if state is not None:
        tag_instants(state, low, high, seen)
    for key, pair in list((getattr(snapshot, "module_state", None) or {}).items()):
        if isinstance(pair, tuple) and len(pair) == 2:
            copied = tag_instants(pair[1], low, high, seen)
            if copied is not pair[1]:
                snapshot.module_state[key] = (pair[0], copied)
    intentions = getattr(snapshot, "intentions", None)
    if isinstance(intentions, dict):
        for part in ("active", "completed"):
            if intentions.get(part) is not None:
                intentions[part] = tag_instants(intentions[part], low, high, seen)
