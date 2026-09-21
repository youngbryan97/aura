"""What the convenient route costs in distinctiveness, measured on her own output.

Oddisee's "People Watching" puts it as a trade nobody notices making: "I can't
help but notice what's distinct becoming scarce — we gladly trade our freedom
for convenience, cause it's comfort over quality, a burden to think logically."
The record has the highest ornament rate of any measured here, 7.01 direction
changes a second, which is the opposite of what it describes.

The trade is real in her too. The response phase asks for three drafts when
there is time and two when there is not, and the cheap route is the one that
takes the first thing that works. Nothing in the system noticed what that cost.

This keeps her own replies and measures how close each new one is to them:

    sameness      the median closeness of her recent replies to the ones
                  before, in shared vocabulary
    scarce        true when the recent half is closer to her history than the
                  earlier half was — distinctiveness going out of the set
    distinct(t)   how far one candidate sits from what she has been saying

`distinct` is a feature of a candidate, so the selector prefers the one that
is not another copy — and the reply that wins is the next entry in the history
the following comparison is made against. Nothing here writes a word; it only
scores words she already wrote.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Convenience",
    "ConvenienceLedger",
    "distinct",
    "get_convenience_ledger",
    "reset_for_test",
]

#: Replies held. Long enough that a run of short answers does not read as her
#: whole style, short enough that a month-old habit is not held against a
#: reply today.
WINDOW: int = 60

#: Replies before the two halves are worth comparing.
MIN_REPLIES: int = 8

#: Words this common carry no style and are dropped before comparing.
_STOP = frozenset(
    "the a an and or but if then than that this those these of to in on for with "
    "is are was were be been being it its as at by from you your i me my we our "
    "they them their he she his her do does did not no yes so just can could would "
    "should will have has had here there what which who how when where why".split()
)

_WORD = re.compile(r"[a-z']+")


def _shape(text: str) -> frozenset[str]:
    """The words of an utterance that carry its style."""
    return frozenset(w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP and len(w) > 2)


def _closeness(left: frozenset[str], right: frozenset[str]) -> float:
    """Shared vocabulary over the vocabulary of both, in [0, 1]."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass
class Convenience:
    """How much of her recent output is a copy of the rest of it."""

    sameness: float = 0.0
    earlier: float = 0.0
    #: True when the recent half repeats her more than the earlier half did.
    scarce: bool = False
    replies: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sameness": round(self.sameness, 4),
            "earlier": round(self.earlier, 4),
            "scarce": self.scarce,
            "replies": self.replies,
            "measured": self.measured,
        }


class ConvenienceLedger:
    """Her recent replies, and how close each was to the ones before it."""

    def __init__(self, window: int = WINDOW) -> None:
        self._shapes: deque[frozenset[str]] = deque(maxlen=window)
        self._closeness: deque[float] = deque(maxlen=window)

    def note_reply(self, text: str) -> float:
        """One reply she actually sent. Returns how close it was to the rest."""
        shape = _shape(text)
        if not shape:
            return 0.0
        closeness = self.closeness_of(text)
        self._shapes.append(shape)
        self._closeness.append(closeness)
        return closeness

    def closeness_of(self, text: str) -> float:
        """How close a candidate sits to her recent replies, in [0, 1]."""
        shape = _shape(text)
        if not shape or not self._shapes:
            return 0.0
        scores = sorted((_closeness(shape, held) for held in self._shapes), reverse=True)
        # The nearest three rather than the mean: a reply is a repeat of
        # something, not of everything.
        top = scores[: min(3, len(scores))]
        return sum(top) / len(top)

    def read(self) -> Convenience:
        held = list(self._closeness)
        if not held:
            return Convenience()
        if len(held) < MIN_REPLIES:
            return Convenience(sameness=_median(held), replies=len(held))
        half = len(held) // 2
        earlier, recent = _median(held[:half]), _median(held[half:])
        return Convenience(
            sameness=recent,
            earlier=earlier,
            scarce=bool(recent > earlier + 1e-9),
            replies=len(held),
            measured=True,
        )


def distinct(text: str) -> float:
    """How far a candidate is from what she has been saying, in [0, 1].

    One when she has said nothing like it. This is the feature the selector
    reads; it is a fact about her own history and about this candidate, and
    nothing about the person she is answering.
    """
    return max(0.0, 1.0 - get_convenience_ledger().closeness_of(text))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


_LEDGER: ConvenienceLedger | None = None


def get_convenience_ledger() -> ConvenienceLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ConvenienceLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
