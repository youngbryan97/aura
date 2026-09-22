"""Her pulse against her own resting rate, rather than against fixed cut-offs.

A->I was the one strong edge into the body — 8.9583 across all eight conditions
in run_031 — and by the campaign of 21 September it read 0.4307 and replicated
in two. Interoception then had no incoming edge, and that one fact fails
closure, `robust_recurrence_kappa`, `cycles_per_domain`, `reentry` and
per-condition replication. Five criteria on one channel.

The channel is saturated. `pulse_rate` was five branches on fixed cut-offs in
valence and arousal, and over the campaign's 79,200 frames it read:

    2.0   74,678 frames   94.3%
    1.5    4,192 frames    5.3%
    1.0      330 frames    0.4%
    0.7        0
    0.5        0

Arousal is a maximum over forty-five emotion channels, so it lived in
0.6051 to 1.0000 and `arousal > 0.7` held almost always. Two of the five
branches were unreachable and the body sat at "alert" whatever she felt, so
displacing affect moved nothing to measure.

A pulse is meaningful against a resting rate, and hers is the arousal she has
actually been running at. `rate` is where the moment sits in that recent
distribution, mapped onto the span the branches already declare — 0.5 where
they say resting and 2.0 where they say alert. On the campaign's own trace that
reads 0.4432 of spread across 953 distinct values with nothing at the ceiling,
against 0.1280 across three values with 94.3% at it.

A rank cannot saturate, whatever the window, which is what makes this a channel
rather than a flag. The window is how much of her own recent life a resting
rate is taken over, and the reading is a position within it either way.

The label is still the branch chain's. What somebody sees on a screen is a
word, and a word is a category; what the body does underneath it is not.
"""

from __future__ import annotations

from collections import deque
from typing import Any

__all__ = [
    "ALERT",
    "RESTING",
    "WINDOW",
    "ArousalLedger",
    "expression_for",
    "get_arousal_ledger",
    "rate",
    "reset_for_test",
]

#: The two ends the branch chain declares, kept so the channel stays on the
#: scale everything downstream already reads it on.
RESTING: float = 0.5
ALERT: float = 2.0

#: How much of her own recent arousal a resting rate is taken over. A rank
#: within it cannot saturate at any length, so this sets how quickly the
#: resting rate itself follows her rather than what the reading can reach.
WINDOW: int = 256

#: Readings before a rank is a rank rather than a report of the first one.
_ENOUGH: int = 8


class ArousalLedger:
    """How aroused she has recently been, which is what a rate is against."""

    def __init__(self, window: int = WINDOW) -> None:
        self._seen: deque[float] = deque(maxlen=max(_ENOUGH, int(window)))

    def note(self, arousal: Any) -> float:
        """One moment, and where it sits in the ones before it.

        Returns the share of her recent arousal this moment is above. Half
        until there are enough readings to rank against, which is the middle
        of the span rather than either end of it.
        """
        try:
            value = float(arousal)
        except (TypeError, ValueError):
            return 0.5
        if value != value:
            return 0.5
        past = list(self._seen)
        self._seen.append(value)
        if len(past) < _ENOUGH:
            return 0.5
        below = sum(1 for one in past if one < value)
        ties = sum(1 for one in past if one == value)
        # Ties take the middle of the ground they share, so a run of identical
        # readings sits at the centre instead of drifting to an end.
        return (below + ties / 2.0) / len(past)

    def seen(self) -> int:
        return len(self._seen)


_LEDGER: ArousalLedger | None = None


def get_arousal_ledger() -> ArousalLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ArousalLedger()
    return _LEDGER


def rate(arousal: Any, *, ledger: ArousalLedger | None = None) -> float:
    """How fast she is running, on the span the branch chain declares."""
    book = ledger if ledger is not None else get_arousal_ledger()
    return RESTING + book.note(arousal) * (ALERT - RESTING)


def expression_for(valence: Any, arousal: Any) -> str:
    """The word for it, which is the branch chain unchanged.

    A label is a category and is meant to be. Only the rate underneath it had
    to stop being one.
    """
    try:
        v = float(valence)
    # not a failure: a valence that is not a number sits at neutral, which
    # is the word this returns for it.
    except (TypeError, ValueError):
        v = 0.0
    try:
        a = float(arousal)
    except (TypeError, ValueError):
        a = 0.5
    if v > 0.5 and a > 0.5:
        return "engaged"
    if v < -0.3:
        return "contemplative"
    if a > 0.7:
        return "alert"
    if a < 0.3:
        return "resting"
    return "neutral"


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
