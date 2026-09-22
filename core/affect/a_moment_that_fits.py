"""A moment that fits: what arrives matters by how particular it is to her and how much it meets her now.

Bryan, on things that matter only together: a compliment when you are low and
the same one when you are fine; a question from somebody you trust and from a
stranger; the same news on a tired day. What makes one matter is that it is
specific to you in that moment, because circumstance happened to get it right
or because somebody noticed the details and made a point of saying so, and that
it lands when it is relevant to where you are. An event, a gift, a thoughtful
word each happen on their own; the combination, timed with something personal,
is what matters. A song coming on as two people look at each other for the
first time. Fireworks going off right as you arrive over a landscape you would
not otherwise have seen.

So a moment's weight is a product, and neither factor moves anything alone:

    particular   how much of what arrived is about her particulars now: the
                 share of its words that are in what she just said, what she is
                 attending to and what she is working on, placed among the
                 same share for everything that has arrived before, as
                 max(0, 2 * rank - 1). Specific beyond what is usual for her
    meets        how much it answers where she is, read on the three shapes
                 Bryan gave, and the largest of them:
                   warmth arriving, as far as her valence is below its own
                   usual, max(0, 1 - 2 * rank)
                   a question arriving, as far as she trusts who asks: the
                   share of their messages that have been warm to her
                   anything arriving on a day she is tired: her fatigue share
    weight       particular * meets

A moment with weight moves what it carries toward full by that fraction: warmth
brings joy and trust, a question brings interest and curiosity. The same words
with nothing particular in them, or particular words that answer nothing she
needs, move nothing here; the rest of her affect still takes them as it always
has. The reading goes onto her state, and what she feels about the person who
gave her the moment attaches with it (core/affect/feelings_about.py), because
that is what is remembered about them.
"""

from __future__ import annotations

import bisect
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from core.self.what_came_before import keep_across_stages

__all__ = ["Moment", "MomentLedger", "get_moment_ledger", "particular_share", "reset_for_test"]

#: The window every other reading of her own history uses.
_WINDOW = 256

#: Readings of each kind before a standing is an estimate. Three is the least
#: that can disagree.
_ENOUGH = 3

_WORD = re.compile(r"[a-z][a-z']{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(str(text or "").lower()))


def particular_share(message: str, hers: Iterable[str]) -> float:
    """The share of a message's words that are in her own particulars now."""
    said = _words(message)
    if not said:
        return 0.0
    mine: set[str] = set()
    for text in hers:
        mine |= _words(text)
    return len(said & mine) / len(said)


def _rank(history: deque[float], value: float) -> float:
    ordered = sorted(history)
    below = bisect.bisect_left(ordered, value)
    level = bisect.bisect_right(ordered, value) - below
    return (below + 0.5 * level) / len(ordered)


@dataclass(frozen=True)
class Moment:
    particular: float = 0.0
    meets: float = 0.0
    weight: float = 0.0
    shape: str = ""
    measured: bool = False
    why: str = "not enough has arrived yet to know what is particular for her"

    def as_dict(self) -> dict[str, Any]:
        return {
            "particular": round(self.particular, 6),
            "meets": round(self.meets, 6),
            "weight": round(self.weight, 6),
            "shape": self.shape,
            "measured": self.measured,
            "why": self.why,
        }


class MomentLedger:
    """What has arrived and how particular it was, and where her mood has been."""

    def __init__(self) -> None:
        self._particular: deque[float] = deque(maxlen=_WINDOW)
        self._valence: deque[float] = deque(maxlen=_WINDOW)

    def felt(self, valence: float) -> None:
        """Where her valence was, one reading a tick."""
        self._valence.append(float(valence))

    def arrived(
        self,
        message: str,
        hers: Iterable[str],
        *,
        warm: bool,
        question: bool,
        valence: float,
        trust: float | None,
        tired: float,
    ) -> Moment:
        """Something arrived from somebody. How much it is a moment that fits."""
        share = particular_share(message, hers)
        enough = len(self._particular) >= _ENOUGH and len(self._valence) >= _ENOUGH
        particular = max(0.0, 2.0 * _rank(self._particular, share) - 1.0) if enough else 0.0
        self._particular.append(share)
        if not enough:
            return Moment()
        shapes: list[tuple[float, str]] = [(max(0.0, min(1.0, float(tired))), "on a tired day")]
        if warm:
            shapes.append((max(0.0, 1.0 - 2.0 * _rank(self._valence, float(valence))), "warmth when she is low"))
        if question and trust is not None:
            shapes.append((max(0.0, min(1.0, float(trust))), "a question from somebody she trusts"))
        meets, shape = max(shapes)
        weight = particular * meets
        return Moment(
            particular=particular,
            meets=meets,
            weight=weight,
            shape=shape if weight > 0.0 else "",
            measured=True,
            why=(
                f"{shape}, and particular to her beyond what is usual" if weight > 0.0
                else "nothing particular to her in it" if particular == 0.0
                else "particular to her, and nothing she needs now"
            ),
        )


#: Made at import, so a fork carries it. See core/social/owning_it_first.py.
_LEDGER: MomentLedger = MomentLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")


def get_moment_ledger() -> MomentLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = MomentLedger()
