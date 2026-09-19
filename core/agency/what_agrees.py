"""Whether what she was told is borne out by what she read.

She has more than one place to find out about something: her own record of
doing it, the encyclopedia she carries, the web where there is one, and her
own voice, which knows a great deal and is sometimes wrong about all of it.
Each on its own is a single witness. What makes a thing worth leaning on is
two witnesses that did not copy each other saying the same thing, and what
makes a thing worth doubting is one saying it and nothing else she can reach
agreeing.

So this counts witnesses. Two statements say the same thing when they share
what is distinctive in them — the words that are not the furniture of every
sentence — and when every number one of them states, the other states too. A
number is where two accounts of the same thing most often disagree, and it is
the part that decides what she does.

Nothing here knows what any statement is about. It compares words she was
given with words she read, and says which of them have a second witness.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Borne", "borne_out", "witnesses"]

#: Words every account of anything uses, which two statements sharing proves
#: nothing about.
_FURNITURE = frozenset(
    """a an the and or but if then than that this these those it its it's is are was
    were be been being to of in on at by for with from as into onto over under
    you your yours i me my we our they them their he she his her not no do does
    did can could should would will shall may might must so such very more most
    less least much many some any all each every just only also too up down left
    right one two way ways thing things keep make made get got have has had""".split()
)

#: How much of the smaller statement's distinctive words the other has to
#: share. Half: a paraphrase keeps the nouns and changes the rest.
SHARED_ENOUGH = 0.5


def _distinctive(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z'\-]{2,}", str(text or "").lower())
    return {word.rstrip("s") for word in words if word not in _FURNITURE}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d[\d,.]*\b", str(text or "")))


def say_the_same(one: str, other: str) -> bool:
    """Whether two statements say the same thing, as far as words can show it."""
    mine, theirs = _distinctive(one), _distinctive(other)
    if not mine or not theirs:
        return False
    shared = len(mine & theirs) / min(len(mine), len(theirs))
    if shared < SHARED_ENOUGH:
        return False
    stated, stated_back = _numbers(one), _numbers(other)
    return not stated or not stated_back or stated <= stated_back or stated_back <= stated


@dataclass(frozen=True)
class Borne:
    """A statement and the sources that bear it out."""

    says: str
    source: str
    by: tuple[str, ...] = ()

    @property
    def second_witness(self) -> bool:
        return bool(self.by)

    def says_so(self) -> str:
        if self.by:
            return f"{', '.join(self.by)} agree"
        return f"only {self.source or 'one source'} says so"


def witnesses(findings: Sequence[Any]) -> list[Borne]:
    """Each finding, with the other sources that say the same thing.

    A source never corroborates itself: two sentences from one page are one
    witness saying something twice.
    """
    borne: list[Borne] = []
    for finding in findings:
        said = str(getattr(finding, "says", "") or "")
        source = str(getattr(finding, "source", "") or "")
        by = tuple(
            dict.fromkeys(
                str(getattr(other, "source", "") or "")
                for other in findings
                if other is not finding
                and str(getattr(other, "source", "") or "") != source
                and say_the_same(said, str(getattr(other, "says", "") or ""))
            )
        )
        borne.append(Borne(says=said, source=source, by=tuple(name for name in by if name)))
    return borne


def borne_out(claim: str, findings: Sequence[Any]) -> Borne:
    """Whether something her voice said is borne out by anything she read."""
    by = tuple(
        dict.fromkeys(
            str(getattr(finding, "source", "") or "what I read")
            for finding in findings
            if say_the_same(claim, str(getattr(finding, "says", "") or ""))
        )
    )
    return Borne(says=str(claim or ""), source="my own voice", by=by)
