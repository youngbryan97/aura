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
from typing import Any

__all__ = ["PuttingInPlace", "a_carry_of", "in_chain_order", "served_first", "speaks_of_carrying", "what_is_carried"]

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

#: A label that tells the place to do something ("SKIP INSTRUCTIONS", "TEST TRAP", "DELETE X", "Undo"): pressed, never
#: picked up. LIVE 2026-10-10 she carried "SKIP INSTRUCTIONS" to "DELETE X" in a game of devices dragged into a room.
_A_COMMAND = re.compile(r"^\W*(?:skip|delete|remove|test|run|go|start|clear|reset|undo|redo|back|menu|help|quit|exit|"
                        r"options|settings|sound|music|mute|pause|save|load|submit|done|finish|launch|play|next|continue|"
                        r"restart|retry|ok|cancel|close|show|hide|rotate|flip|zoom)\b", re.IGNORECASE)
#: A command whose place takes what is carried to it away: a bin is a place things are carried to.
_A_BIN = re.compile(r"\b(?:delete|remove|trash|bin|discard|recycle)\b", re.IGNORECASE)

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
    #: Where the last carry that answered put its thing: where a chain built so far ends.
    chain_ends_at: tuple[float, float] | None = None
    #: The place the words say things are carried to ("the end of another device's arrow"), found by her eyes.
    place_named: str = ""

    def told_of_carrying(self, words: str) -> None:
        """Words read at this place: once they speak of carrying, it is a place where things are carried."""
        self.carrying_said = self.carrying_said or speaks_of_carrying(words)
        from core.perception.where_the_words_point import the_place_named

        self.place_named = the_place_named(words) or self.place_named

    def carries(self, on_screen: Sequence[str]) -> tuple[str, ...]:
        """Each thing on the screen carried to each other place there, the places marked as the one first, untried
        before tried, and none that changed nothing."""
        if not self.carrying_said:
            return ()
        from core.agency.what_i_can_do_here import _goes_on, _not_read_only, what_is_clicked
        from core.perception.shapes_that_look_pressable import STANDS_OUT

        worth = [move for move in on_screen if what_is_clicked(move) and _not_read_only(move)]
        things = [what_is_clicked(move) or "" for move in worth
                  if not _goes_on(move) and not _A_COMMAND.search(what_is_clicked(move) or "")]
        # A bin is somewhere a thing is carried to be got rid of, not to be used: it is offered last.
        places = [what_is_clicked(move) or "" for move in worth if not _A_COMMAND.search(what_is_clicked(move) or "")]
        places += [what_is_clicked(move) or "" for move in worth if _A_BIN.search(what_is_clicked(move) or "")]
        # The place the words name is where things go, first; it is no thing to carry.
        things = [t for t in things if t != self.place_named]
        marked = [p for p in places if p == self.place_named] + [
            p for p in places if p != self.place_named and (p.startswith(STANDS_OUT) or _MARKS_A_PLACE.search(p))]
        places = marked + [p for p in places if p not in marked]
        offered = [a_carry_of(thing, place) for place in places for thing in things
                   if thing != place and self.carried_to.get(a_carry_of(thing, place)) is not False]
        offered.sort(key=lambda move: move in self.carried_to)
        said = getattr(self, "mechanics_said", set())
        where = getattr(self, "where", {})
        if "chains" in said:
            offered = in_chain_order(offered, where, self.chain_ends_at, falls="pull" in said)
        if "serving" in said:
            offered = served_first(offered, getattr(self, "looked", {}), what_is_carried)
        return tuple(offered[:CARRIES_OFFERED])

    def carried(self, move: str, changed: bool) -> None:
        """A carry made, and whether the screen answered it; one that answered extends the chain built so far."""
        found = what_is_carried(move)
        if found:
            self.carried_to[move] = changed
            if changed:
                from core.perception.where_the_words_point import PLACES

                PLACES.moved()
            at = getattr(self, "where", {}).get(found[1])
            if changed and at is not None:
                self.chain_ends_at = at


#: How near, as a share of the screen, a place is to where the last part went to be that place, taken.
TAKEN_WITHIN = 0.03


def in_chain_order(offered: Sequence[str], where: dict[str, tuple[float, float]], ends_at: tuple[float, float] | None,
                   *, falls: bool = False) -> list[str]:
    """Carries in the order a chain is built: each part put on from where the last one went, the nearest place first,
    leaning the way things fall where they fall. A chain fails at its weakest link: a part put far from the last leaves
    a gap no thing crosses, whether the parts are tubes, devices or ropes."""
    if ends_at is None:
        return list(offered)

    def far(move: str) -> float:
        found = what_is_carried(move)
        at = where.get(found[1]) if found else None
        if at is None:
            return 9.0
        across, down = at[0] - ends_at[0], at[1] - ends_at[1]
        gap = (across ** 2 + down ** 2) ** 0.5
        if gap < TAKEN_WITHIN:
            return 8.0  # where the last part went: taken
        return gap - (0.15 * down if falls and down > 0 else 0.0)

    return sorted(offered, key=far)


def served_first(offered: Sequence[str], looked: dict[str, tuple[float, ...]], parts: Any) -> list[str]:
    """Moves that give a thing to another, the thing given to what looks most like it first: where each asks for
    something, what it asks for is drawn by it, and giving each what it asks for is giving it what it looks like."""
    from core.perception.how_a_place_looks import apart

    def unlike(move: str) -> float:
        found = parts(move)
        return apart(looked.get(found[0], ()), looked.get(found[1], ())) if found else 1.0

    return sorted(offered, key=unlike)
