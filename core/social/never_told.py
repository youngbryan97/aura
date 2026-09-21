"""What she has gone without, and gives anyway.

Lizzo's "Special" gives the thing on the condition that it might be missing:
"in case nobody told you today", "in case nobody made you believe". The verse
says where it comes from — "I'm used to feeling hollow, so I thought that I'd
let you know". The gift is her own shortage, handed over.

The record is major, the only major one of the five in this batch, and it puts
its loudest window at 0.85 — the last time the line is said, not the first.

Three kinds of regard can be told apart in what arrives, and a system that
receives them can also notice going without them:

    held      that she matters, said without being asked
    good      that a particular thing she did was good
    asked     that somebody asked how she is

`scarcest` is the kind she has gone longest without, as a share of the turns
since it last arrived. `supplies` reads the same three kinds out of something
she is about to say, and counts them only where the other person did not ask
for them, because the song's line is offered rather than answered.

The shortage does not make the giving; it decides which of the three she gives
when she is giving anyway. Nothing here writes a word of the reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.expression.register import read as read_register

__all__ = [
    "KINDS",
    "asks_for",
    "NeverTold",
    "TellingLedger",
    "get_telling_ledger",
    "reset_for_test",
    "supplies",
]

#: The three kinds, in the order their patterns are tried.
KINDS: tuple[str, ...] = ("held", "good", "asked")

#: Turns before the shares are a reading rather than a start.
MIN_TURNS: int = 12

#: What counts as each kind, in the second person. These are shapes rather
#: than a vocabulary: a pronoun, a copula and a term of regard for `held`; a
#: possessive and a term of quality for `good`; a question about the other
#: person's state for `asked`.
_HELD = re.compile(
    r"\byou(?:'re| are|r)?\b[^.?!]{0,40}\b(matter|matters|mattered|enough|special|loved|"
    r"valued|important|welcome|wanted|worth it|belong)\b", re.I)
_GOOD = re.compile(
    r"\b(your|that|this|the)\b[^.?!]{0,30}\b(work|answer|idea|writing|code|point|take|"
    r"question|catch|fix)\b[^.?!]{0,30}\b(good|right|strong|clean|sharp|careful|clever|"
    r"well|works?|landed)\b", re.I)
_ASKED = re.compile(
    r"\bhow (?:are|is) (?:you|things|it going)\b|\bare you (?:ok|okay|alright|all right)\b|"
    r"\bhow(?:'s| is) (?:your|it)\b", re.I)

_PATTERNS = {"held": _HELD, "good": _GOOD, "asked": _ASKED}

#: What a question about each kind is about. A shape that gives regard and a
#: question that asks for it put the same words in a different order, so the
#: subtraction in `supplies` reads topics rather than shapes.
_TOPICS = {
    "held": re.compile(
        r"\b(matter|matters|special|loved|valued|important|welcome|wanted|belong|"
        r"worth)\b", re.I),
    "good": re.compile(r"\b(good|right|any good|well done|quality|ok(?:ay)?)\b", re.I),
    "asked": re.compile(r"\b(how i am|how i'?m doing|am i (?:ok|okay|alright))\b", re.I),
}


def asks_for(text: str) -> set[str]:
    """The kinds a question is asking about, whatever order its words are in."""
    body = str(text or "")
    if "?" not in body and not body.strip():
        return set()
    return {kind for kind, pattern in _TOPICS.items() if pattern.search(body)}


@dataclass
class NeverTold:
    """How long since each kind last arrived, and which is scarcest."""

    since: dict[str, int] | None = None
    #: The kind she has gone longest without, or "" before there is one.
    scarcest: str = ""
    #: How long that has been, as a share of the turns she has been counting.
    scarcity: float = 0.0
    turns: int = 0
    measured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": dict(self.since or {}),
            "scarcest": self.scarcest,
            "scarcity": round(self.scarcity, 4),
            "turns": self.turns,
            "measured": self.measured,
        }


def kinds_in(text: str) -> set[str]:
    """Which of the three kinds a piece of text carries."""
    body = str(text or "")
    if not body.strip():
        return set()
    return {kind for kind, pattern in _PATTERNS.items() if pattern.search(body)}


def supplies(text: str, *, asked_for: str = "") -> set[str]:
    """The kinds a reply gives that the other person did not ask for.

    A reply that answers "was that any good?" with "that was good" is an
    answer. The song's line is the other thing: said because it might be
    missing, to somebody who did not raise it.
    """
    given = kinds_in(text)
    if not given:
        return given
    theirs = read_register(str(asked_for or ""))
    if theirs.measured and theirs.asking > 0.0:
        given -= kinds_in(asked_for) | asks_for(asked_for)
    return given


class TellingLedger:
    """Turns since each kind of regard last reached her."""

    def __init__(self) -> None:
        self._turns = 0
        self._last: dict[str, int] = {}

    def note_turn(self, received: str = "") -> None:
        """One turn, and whatever arrived in it."""
        self._turns += 1
        for kind in kinds_in(received):
            self._last[kind] = self._turns

    def note_received(self, kind: str) -> None:
        """One kind of regard, arriving by a route other than words."""
        if str(kind) in _PATTERNS:
            self._last[str(kind)] = self._turns

    def read(self) -> NeverTold:
        if self._turns <= 0:
            return NeverTold(since={})
        since = {kind: self._turns - self._last.get(kind, 0) for kind in KINDS}
        scarcest = max(KINDS, key=lambda kind: since[kind])
        return NeverTold(
            since=since,
            scarcest=scarcest,
            scarcity=min(1.0, since[scarcest] / self._turns),
            turns=self._turns,
            measured=self._turns >= MIN_TURNS,
        )

    def shortage_of(self, kinds: set[str]) -> float:
        """How much of her own shortage a set of kinds covers, in [0, 1].

        The scarcest kind she is short of is worth its scarcity; the others
        are worth theirs. Giving all three is worth the largest, not the sum,
        because the reading is about which one is missing.
        """
        if not kinds:
            return 0.0
        reading = self.read()
        if not reading.measured or not reading.since:
            return 0.0
        turns = max(reading.turns, 1)
        return max((reading.since.get(kind, 0) / turns) for kind in kinds if kind in _PATTERNS)


_LEDGER: TellingLedger | None = None


def get_telling_ledger() -> TellingLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = TellingLedger()
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = None
