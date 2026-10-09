"""Putting a thing in place: a press on it, carried with the button held to another place, and let go there.

A part dragged onto a machine, a piece dragged to its square, a file dragged to
a folder, a word dragged into the gap in a sentence: one act, and a click on
either place does not do it. LIVE 2026-10-09 a game's own rules said "drag and
drop the piece to the highlighted area", and she had clicks only.

Offered where the place's words, or what she found out about it before she
began, speak of carrying a thing somewhere. Every thing on the screen worth
clicking may be carried, and every other place there is somewhere to carry it
('drag "pipe" to "attach here!"'); a place marked as the one (a label that says
"here", the one shape that stands out from its alike ones) is offered first.
Which carry to make is a choice among those on offer, as any other move is. A
carry that changed nothing is not offered again.

Nothing here knows what a part, a piece or a file is.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["PuttingInPlace", "a_carry_of", "speaks_of_carrying", "what_is_carried"]

#: The most carries offered at once.
CARRIES_OFFERED = 12

_CARRY = re.compile(r'^drag "(.+)" to "(.+)"$')

#: Words that speak of carrying a thing somewhere: "drag the piece", "drag and drop", "drop it on the", "move it to
#: the", "put the ... in", "place it on". A button called Drop is not one ("click the green Drop button").
_CARRYING = re.compile(
    r"\bdrag(?:s|ged|ging)?\b|\bdrop\s+(?:it|them|the|each|a|an|your)\b"
    r"|\b(?:move|put|place|carry)\s+(?:it|them|the|each|a|an|your)\b[^.!?]{0,40}?\b(?:to|into|onto|on|in|over)\b",
    re.IGNORECASE,
)

#: Words that mark a place as where things go ("attach here!", "drop here").
_MARKS_A_PLACE = re.compile(r"\bhere\b", re.IGNORECASE)


def a_carry_of(thing: str, place: str) -> str:
    """The move that carries a thing to a place, each named by what she calls it."""
    return f'drag "{" ".join(str(thing).split())}" to "{" ".join(str(place).split())}"'


def what_is_carried(move: str) -> tuple[str, str] | None:
    """What a carry carries and where to, or None when the move is not a carry."""
    found = _CARRY.match(str(move or "").strip())
    return (found.group(1), found.group(2)) if found else None


def speaks_of_carrying(words: str) -> bool:
    """Whether words speak of carrying a thing to a place."""
    return bool(_CARRYING.search(" ".join(str(words or "").split())))


@dataclass
class PuttingInPlace:
    """Whether this place is one where things are carried, and the carries made, for WhatWorksHere."""

    #: Whether the place's words, or what she found out about it, have spoken of carrying a thing somewhere.
    carrying_said: bool = False
    #: Each carry made, and whether the screen answered it.
    carried_to: dict[str, bool] = field(default_factory=dict)

    def told_of_carrying(self, words: str) -> None:
        """Words read at this place: once they speak of carrying, it is a place where things are carried."""
        self.carrying_said = self.carrying_said or speaks_of_carrying(words)

    def carries(self, on_screen: Sequence[str]) -> tuple[str, ...]:
        """Each thing on the screen carried to each other place there, the places marked as the one first, untried
        before tried, and none that changed nothing."""
        if not self.carrying_said:
            return ()
        from core.agency.what_i_can_do_here import _goes_on, _not_read_only, what_is_clicked
        from core.perception.shapes_that_look_pressable import STANDS_OUT

        worth = [move for move in on_screen if what_is_clicked(move) and _not_read_only(move)]
        things = [what_is_clicked(move) or "" for move in worth if not _goes_on(move)]
        places = [what_is_clicked(move) or "" for move in worth]
        marked = [p for p in places if p.startswith(STANDS_OUT) or _MARKS_A_PLACE.search(p)]
        places = marked + [p for p in places if p not in marked]
        offered = [a_carry_of(thing, place) for place in places for thing in things
                   if thing != place and self.carried_to.get(a_carry_of(thing, place)) is not False]
        offered.sort(key=lambda move: move in self.carried_to)
        return tuple(offered[:CARRIES_OFFERED])

    def carried(self, move: str, changed: bool) -> None:
        """A carry made, and whether the screen answered it."""
        if what_is_carried(move):
            self.carried_to[move] = changed
