"""Things that go together: what each place showed when she turned it over, and pairing two that showed alike.

A game of turning over cards and finding the pairs, a game of giving each person
what they ask for, a page of matching each word to its picture: each asks her to
remember what was where and bring together what is alike. A person turns one
over, sees what is on it, remembers, and when a second shows the same picture,
turns both.

So after each click on a place, what that place looks like at the next look is
what it showed (core/perception/how_a_place_looks.py). Two places that showed
alike looks are a pair, offered as one move ('match "A" with "B"': a click on
the one, then on the other) ahead of any move made to find out, since it is
known to answer. A pair once made is not offered again.

Nothing here knows what a card, a customer or a word is.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["ThingsThatGoTogether", "a_match_of", "what_is_matched"]

#: The most pairs offered at once.
PAIRS_OFFERED = 6

_MATCH = re.compile(r'^match "(.+)" with "(.+)"$')


def a_match_of(one: str, other: str) -> str:
    """The move that pairs two places, named by what each is called."""
    return f'match "{" ".join(str(one).split())}" with "{" ".join(str(other).split())}"'


def what_is_matched(move: str) -> tuple[str, str] | None:
    """The two places a pairing move pairs, or None when the move is not a pairing."""
    found = _MATCH.match(str(move or "").strip())
    return (found.group(1), found.group(2)) if found else None


@dataclass
class ThingsThatGoTogether:
    """What each place showed after she clicked it, and the pairs she has made, for WhatWorksHere."""

    #: What a place (by what she calls it) looked like before she clicked it and at the look after, where it changed.
    showed: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(default_factory=dict)
    to_see: str = ""
    made: set[frozenset[str]] = field(default_factory=set)
    #: The looks at the last look, by what she calls each place: what a place looked like when she clicked it.
    looked: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def turned_over(self, label: str) -> None:
        """A click on a place: what it shows is read at the next look."""
        self.to_see = label

    def saw_looks(self, looks: Mapping[str, tuple[float, ...]]) -> None:
        """The looks of the places on the screen now, by what she calls them or by the move that clicks them."""
        from core.agency.what_i_can_do_here import what_is_clicked
        from core.perception.how_a_place_looks import alike

        by_label = {what_is_clicked(name) or name: look for name, look in looks.items()}
        before, after = self.looked.get(self.to_see), by_label.get(self.to_see)
        # Turned over: the place itself looks otherwise than it did when she clicked it. A click that led to another
        # screen shows nothing about the place, and is not remembered as having shown anything.
        if self.to_see and before and after and not alike(before, after):
            self.showed[self.to_see] = (before, after)
        self.to_see = ""
        self.looked = by_label

    def pairs(self, on_screen: Sequence[str]) -> tuple[str, ...]:
        """Two places on the screen that showed alike looks, as pairing moves, not made before."""
        from core.agency.what_i_can_do_here import what_is_clicked
        from core.perception.how_a_place_looks import alike

        here = [label for label in (what_is_clicked(move) for move in on_screen) if label in self.showed]
        offered: list[str] = []
        for at, one in enumerate(here):
            for other in here[at + 1:]:
                (back, face), (other_back, other_face) = self.showed[one], self.showed[other]
                # Alike before (two of the same kind of place) and alike when turned (the same thing on each).
                if frozenset((one, other)) not in self.made and alike(back, other_back) and alike(face, other_face):
                    offered.append(a_match_of(one, other))
        return tuple(offered[:PAIRS_OFFERED])

    def paired(self, move: str) -> None:
        """A pairing move made: not offered again."""
        found = what_is_matched(move)
        if found:
            self.made.add(frozenset(found))
