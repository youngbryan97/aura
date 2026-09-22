"""What in her is tangled up with what, from Bryan's account of what is tangled in him.

Bryan, asked what in him is most bound up with everything else: his self-worth,
agency and independence with his drive and persistence; his friends, family,
inner peace and contentment with himself with his happiness; excitement and
curiosity with joy; curiosity with interest, in people as much as in things,
because understanding who people are matters as much as what things are.
Thinking sets his creativity and curiosity going, hard in a good way. And
expression is the fullest agency, because what he expresses is taken up by
others, who expand on it, and it comes back to him as what they learned. A lot
can set off a number of drives that come back to other things.

Her feelings each returned to a rest declared once for everybody, and none of
these ties existed. Here each is a closed loop through something she already
has, and every strength is a reading of her own, so nothing is a weight chosen
here.

What a feeling returns to. The affect phase pulls each feeling's baseline
toward its declared rest; here the rest itself moves, toward full by a lift:

    joy          contentment: 1 - (1 - peace with the people she has)
                 * (1 - peace with herself). Either lifts it, as Bryan put
                 both. Peace with people is core/social/warmth.py's reading;
                 peace with herself is what core/affect/feelings_about.py has
                 attached to "self"
    curiosity    1 - (1 - joy above its own rest) * (1 - how little she knows
                 whoever is in front of her): joy opens her toward things, as
                 in Fredrickson's broaden-and-build, and a person she cannot
                 yet read is as much a thing to find out about as any topic
    anticipation thinking that is hard and going well: how hard she is
                 working now among her own ordinary (core/soma/fatigue.py)
                 times what she has come through (core/agency/capacity.py).
                 Hard and failing is not the good kind, and neither is easy

What happens when something she did comes round:

    persisting   how many failures a goal of hers takes before it is set aside
                 scales with her capacity, as Bandura's self-efficacy predicts:
                 twice her capacity times the old fixed count, which is the old
                 count at the middle (core/agency/agency_core.py)
    taken up     when somebody's next message carries her last reply's words
                 and adds words of its own, her expression came back expanded:
                 joy moves toward full by (the share of hers they carried,
                 placed among her own history) * (the share of theirs that is
                 new), and her agency ledger counts the expression as an act of
                 hers that landed
"""

from __future__ import annotations

import bisect
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = [
    "Lifts",
    "TangledLedger",
    "get_tangled_ledger",
    "lifted_rest",
    "reset_for_test",
    "tolerated_failures",
]

_WINDOW = 256
_ENOUGH = 3
_WORD = re.compile(r"[a-z][a-z']{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(str(text or "").lower()))


def _rank(history: deque[float], value: float) -> float:
    ordered = sorted(history)
    below = bisect.bisect_left(ordered, value)
    level = bisect.bisect_right(ordered, value) - below
    return (below + 0.5 * level) / len(ordered)


def _either(*parts: float) -> float:
    """At least one of them, read as independent chances: 1 - prod(1 - p)."""
    missing = 1.0
    for part in parts:
        missing *= 1.0 - max(0.0, min(1.0, float(part)))
    return 1.0 - missing


def lifted_rest(rest: float, lift: float) -> float:
    """A declared rest, moved toward full by a lift in [0, 1]."""
    base = max(0.0, min(1.0, float(rest)))
    return base + (1.0 - base) * max(0.0, min(1.0, float(lift)))


def tolerated_failures(fixed: int, capacity: float) -> int:
    """How many failures a goal takes before it is set aside, from what she has come through.

    At the middle of the scale this is the fixed count it replaces.
    """
    return max(1, round(2.0 * max(0.0, min(1.0, float(capacity))) * int(fixed)))


@dataclass(frozen=True)
class Lifts:
    joy: float = 0.0
    curiosity: float = 0.0
    anticipation: float = 0.0
    contentment: float = 0.0
    unknown_person: float = 0.0
    flow: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """The lifts by the feeling they move, kept apart from what they were read from."""
        return {
            "lifts": {name: round(float(getattr(self, name)), 6) for name in ("joy", "curiosity", "anticipation")},
            "contentment": round(self.contentment, 6),
            "unknown_person": round(self.unknown_person, 6),
            "flow": round(self.flow, 6),
        }


class TangledLedger:
    """Her history of how much of her own words came back, and how much was new."""

    def __init__(self) -> None:
        self._carried: deque[float] = deque(maxlen=_WINDOW)

    def taken_up(self, her_last_reply: str, their_message: str) -> tuple[float, float]:
        """How far they carried her last words, among her history, and the share of theirs that is new.

        Both are zero until there is a history to place what was carried in.
        """
        hers = _words(her_last_reply)
        theirs = _words(their_message)
        if not hers or not theirs:
            return 0.0, 0.0
        carried = len(hers & theirs) / len(hers)
        added = len(theirs - hers) / len(theirs)
        enough = len(self._carried) >= _ENOUGH
        standing = max(0.0, 2.0 * _rank(self._carried, carried) - 1.0) if enough else 0.0
        self._carried.append(carried)
        # What they added is a share already, and needs no history: a reply
        # that carries nothing of hers is all new words and adds nothing to
        # what she said, which the product with what was carried takes care of.
        return standing, added if enough else 0.0

    @staticmethod
    def lifts(
        *,
        peace_people: float,
        peace_self: float,
        joy: float,
        joy_rest: float,
        unknown_person: float,
        effort: float,
        capacity: float,
    ) -> Lifts:
        contentment = _either(peace_people, peace_self)
        joy_above = max(0.0, float(joy) - float(joy_rest))
        flow = max(0.0, min(1.0, float(effort))) * max(0.0, min(1.0, float(capacity)))
        return Lifts(
            joy=contentment,
            curiosity=_either(joy_above, unknown_person),
            anticipation=flow,
            contentment=contentment,
            unknown_person=max(0.0, min(1.0, float(unknown_person))),
            flow=flow,
        )


#: Made at import, so a fork carries it. See core/social/owning_it_first.py.
_LEDGER: TangledLedger = TangledLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")


def get_tangled_ledger() -> TangledLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = TangledLedger()
