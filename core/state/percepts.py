"""One reading of a percept, for everything that consumes the percept stream.

`AuraState.world.recent_percepts` is a list of plain dicts written by nine
different places, and until this module existed every consumer invented its own
reading of them. Affect keyed on `type` and dropped anything it did not
recognise. The workspace priced its perception bid from `salience`, which no
producer in the codebase has ever written. Half the producers omit `timestamp`,
so the recency of what she just saw read as zero.

The result was a sensory stream that arrived and went nowhere: a vision frame
was appended, counted, and could not move affect, could not compete for
broadcast, and aged instantly. A percept nothing can read is not perception.

So the reading lives in one place, next to the state it reads. Producers stay
as they are — they write what is natural at the site — and every consumer gets
the same four values out: what kind of event it was, how strong, how much it
deserves attention, and when it happened.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_INTENSITY",
    "Percept",
    "drop_consumed",
    "emit_percept",
    "fresh_for",
    "mark_consumed",
    "read_percept",
]

#: Where a percept records which consumers have already taken it. A consumer
#: that deletes what it has processed is not preventing double-processing, it
#: is deleting the event for everyone downstream of it — which is what the
#: affect phase did, so the workspace, the world model, the phi estimate and
#: the state's own reading of perception all saw an empty stream on every turn
#: after affect had run.
CONSUMED_KEY: str = "consumed_by"

#: What a percept with no stated strength is worth. Half, because a producer
#: that does not say has no opinion, and the alternative — zero — silently
#: deletes the event on the way to every consumer that multiplies by it.
DEFAULT_INTENSITY: float = 0.5


def _number(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if out != out:  # NaN
        return fallback
    return out


@dataclass(frozen=True, slots=True)
class Percept:
    """A percept as every consumer should see it."""

    kind: str
    content: str
    intensity: float
    salience: float
    timestamp: float
    raw: Mapping[str, Any]

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.timestamp) if self.timestamp > 0 else 0.0


def read_percept(item: Any, *, now: float | None = None) -> Percept:
    """The four numbers, however the producer chose to write them.

    `salience` falls back to `intensity` rather than to zero. A producer that
    states how strong an event was has said how much attention it deserves; the
    two are separate only when a producer sets them apart on purpose.
    """
    if not isinstance(item, Mapping):
        return Percept(
            kind="unknown",
            content=str(item)[:240],
            intensity=DEFAULT_INTENSITY,
            salience=DEFAULT_INTENSITY,
            timestamp=0.0,
            raw={},
        )
    kind = str(item.get("type") or item.get("role") or item.get("source") or "unknown")
    content = item.get("content")
    if content is None:
        payload = item.get("payload")
        content = str(payload) if payload is not None else ""
    intensity = max(0.0, min(1.0, _number(item.get("intensity"), DEFAULT_INTENSITY)))
    salience = max(0.0, min(1.0, _number(item.get("salience"), intensity)))
    stamp = _number(item.get("timestamp"), 0.0)
    if stamp <= 0.0:
        # A producer that did not stamp it is telling us it happened now; the
        # alternative reading, epoch zero, makes it infinitely old.
        stamp = float(now if now is not None else time.time())
    return Percept(
        kind=kind,
        content=str(content)[:240],
        intensity=intensity,
        salience=salience,
        timestamp=stamp,
        raw=item,
    )


def mark_consumed(item: Any, by: str) -> None:
    """Record that one consumer has taken this percept. It stays in the stream."""
    if not isinstance(item, MutableMapping):
        return
    seen = item.get(CONSUMED_KEY)
    if isinstance(seen, list):
        if by not in seen:
            seen.append(by)
    else:
        item[CONSUMED_KEY] = [by]


def fresh_for(percepts: Any, by: str) -> list[Any]:
    """The percepts this consumer has not taken yet, oldest first."""
    if not isinstance(percepts, list):
        return []
    out = []
    for item in percepts:
        seen = item.get(CONSUMED_KEY) if isinstance(item, Mapping) else None
        if not isinstance(seen, list) or by not in seen:
            out.append(item)
    return out


def drop_consumed(world: Any, by: str) -> int:
    """Forget the percepts this consumer already took on an earlier pass.

    Called by the consumer at the start of its own pass, this gives a percept a
    lifetime of exactly one turn: long enough for every later stage of the turn
    to see what arrived, short enough that the stream never grows and the count
    of it still means something.
    """
    percepts = getattr(world, "recent_percepts", None)
    if not isinstance(percepts, list):
        return 0
    keep = fresh_for(percepts, by)
    dropped = len(percepts) - len(keep)
    if dropped:
        percepts[:] = keep
    return dropped


def emit_percept(
    world: Any,
    kind: str,
    *,
    content: str = "",
    intensity: float = DEFAULT_INTENSITY,
    salience: float | None = None,
    **extra: Any,
) -> dict[str, Any] | None:
    """Append a percept carrying everything a consumer needs. Trims the stream."""
    percepts = getattr(world, "recent_percepts", None)
    if not isinstance(percepts, list):
        return None
    record: dict[str, Any] = {
        "type": str(kind),
        "content": str(content)[:512],
        "intensity": max(0.0, min(1.0, float(intensity))),
        "salience": max(0.0, min(1.0, float(intensity if salience is None else salience))),
        "timestamp": time.time(),
    }
    record.update(extra)
    percepts.append(record)
    trim = getattr(world, "trim_percepts", None)
    if callable(trim):
        trim()
    return record
