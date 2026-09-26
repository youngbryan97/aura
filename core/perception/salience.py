"""How much an arrival stands out, which is hers to decide.

A producer states how strong an event was. How much of her attention it
deserves is a different question, and until now nothing answered it: `salience`
falls back to `intensity` whenever no producer sets the two apart, and no
producer ever does. So the perception domain carried the world's own number
twice and nothing of her, and the workspace's perception bid, the temporal
binding weight and the world's own "worth keeping" filter were all reading the
environment directly.

That shows up in the battery as the cheapest cut. On the seed-7 run of 24
September the cheapest bipartition was perception against the other nine
domains, a third of the way to the bar: a domain that is a function of the
world alone is severable from a mind by construction, because nothing she is
changes it.

What makes an arrival stand out is how it compares with the arrivals she has
been having, and how sharply she is attending when it lands. Both are readings
she already has. Neither is a threshold:

    attention = where her arousal sits in her recent arousal
    weighted  = intensity * attention
    salience  = where that sits in her recent weighted arrivals

A rank cannot saturate, and it keeps the scale every consumer already reads
salience on: half her arrivals sit above a half, which is what the filters that
test against a half were written against. The same knock is nothing on a busy
afternoon and everything at three in the morning, and that is the whole of what
this says.
"""

from __future__ import annotations

from collections import deque
from typing import Any

__all__ = [
    "ENOUGH",
    "WINDOW",
    "RankLedger",
    "attention_of",
    "get_arrival_ledger",
    "get_attention_ledger",
    "reset_for_test",
    "salience_of",
]

#: How much of her own recent life a rank is taken over. A rank within it
#: cannot saturate at any length, so this sets how quickly the comparison
#: itself follows her rather than what the reading can reach.
WINDOW: int = 256

#: Readings before a rank is a rank rather than a report of the first one.
#: Below this the answer is the middle, which is neither end.
ENOUGH: int = 8


class RankLedger:
    """Where a reading sits among the recent ones, and nothing else."""

    def __init__(self, window: int = WINDOW) -> None:
        self._seen: deque[float] = deque(maxlen=max(ENOUGH, int(window)))

    def note(self, reading: Any) -> float:
        """Record one reading and return the share of the recent ones it is above."""
        try:
            value = float(reading)
        except (TypeError, ValueError):
            return 0.5
        if value != value:
            return 0.5
        past = list(self._seen)
        self._seen.append(value)
        if len(past) < ENOUGH:
            return 0.5
        below = sum(1 for one in past if one < value)
        ties = sum(1 for one in past if one == value)
        # Ties take the middle of the ground they share, so a run of identical
        # readings sits at the centre instead of drifting to an end.
        return (below + ties / 2.0) / len(past)

    def seen(self) -> int:
        return len(self._seen)


_ATTENTION: RankLedger | None = None
_ARRIVALS: RankLedger | None = None


def get_attention_ledger() -> RankLedger:
    global _ATTENTION
    if _ATTENTION is None:
        _ATTENTION = RankLedger()
    return _ATTENTION


def get_arrival_ledger() -> RankLedger:
    global _ARRIVALS
    if _ARRIVALS is None:
        _ARRIVALS = RankLedger()
    return _ARRIVALS


def reset_for_test() -> None:
    global _ATTENTION, _ARRIVALS
    _ATTENTION = None
    _ARRIVALS = None


def attention_of(arousal: Any, *, ledger: RankLedger | None = None) -> float:
    """How sharply she is attending, against how sharply she usually is."""
    book = ledger if ledger is not None else get_attention_ledger()
    return book.note(arousal)


def salience_of(
    intensity: Any,
    arousal: Any,
    *,
    attention: RankLedger | None = None,
    arrivals: RankLedger | None = None,
) -> float:
    """What this arrival is worth to her, given the ones before it.

    Both readings are her own. An arrival of the same strength is worth more
    when she is more awake than usual and less when the last hundred were
    stronger, which is why this cannot be computed from the event alone.
    """
    try:
        strength = float(intensity)
    except (TypeError, ValueError):
        strength = 0.0
    if strength != strength:
        strength = 0.0
    strength = max(0.0, min(1.0, strength))
    sharpness = attention_of(arousal, ledger=attention)
    book = arrivals if arrivals is not None else get_arrival_ledger()
    return book.note(strength * sharpness)
