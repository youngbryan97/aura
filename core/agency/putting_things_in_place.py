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
#: Words that say what is carried is writing (words to a gap, letters to a slot, names to a box): only there is a
#: label carried. Everywhere else writing is a control to click (a category, a tab, a heading), and what is carried is
#: a picture. LIVE 2026-10-10 she carried a device library's tabs ("HANGERS", "ROLLERS") to a spot in the room, thirty
#: times, and the screen never changed.
_WRITING_IS_CARRIED = re.compile(r"\b(?:drag|drop|move|put|place|carry)\s+(?:the\s+|each\s+|a\s+|your\s+)?"
                                 r"(?:words?|letters?|labels?|names?|answers?|captions?|tiles?|cards?)\b", re.IGNORECASE)
#: Carries in a row that changed nothing before carrying rests until the screen changes: as many as it takes to judge an
#: input dead (core/agency/what_i_can_do_here.py ENOUGH_TO_JUDGE).
RESTS_AFTER = 4


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
    #: Whether the place's words say writing is what is carried (words to gaps, letters to slots).
    writing_carried: bool = False
    #: Carries made in a row that changed nothing, since the screen last changed; and what was on it then.
    carries_unanswered: int = 0
    screen_carried_on: tuple[str, ...] = ()
    #: What the screen reading found drawn on the screen now, by what she calls each: pictures, not writing.
    pictures: set[str] = field(default_factory=set)
    #: Turns given the part put down last (core/agency/aiming_what_was_placed.py).
    turned_since_placed: int = 0
    #: Delivered transfers and their independently observed current effects.
    placement_receipts: dict[str, Any] = field(default_factory=dict)
    placement_states: dict[str, Any] = field(default_factory=dict)

    def told_of_carrying(self, words: str) -> None:
        """Words read at this place: once they speak of carrying, it is a place where things are carried."""
        self.carrying_said = self.carrying_said or speaks_of_carrying(words)
        self.writing_carried = self.writing_carried or bool(_WRITING_IS_CARRIED.search(str(words or "")))
        from core.perception.where_the_words_point import the_place_named

        self.place_named = the_place_named(words) or self.place_named

    def carries(self, on_screen: Sequence[str]) -> tuple[str, ...]:
        """Each thing on the screen carried to each other place there, the places marked as the one first, untried
        before tried, and none that changed nothing."""
        if not self.carrying_said:
            return ()
        from core.agency.what_i_can_do_here import _goes_on, _not_read_only, what_is_clicked
        from core.perception.shapes_that_look_pressable import STANDS_OUT

        # Carrying rests once several carries in a row changed nothing on this screen: what does something here now is
        # a click (a category opened, a part picked), and the screen it brings is carried on afresh.
        screen = tuple(sorted(on_screen))
        if screen != self.screen_carried_on:
            self.screen_carried_on, self.carries_unanswered = screen, 0
        if self.carries_unanswered >= RESTS_AFTER:
            return ()
        worth = [move for move in on_screen if what_is_clicked(move) and _not_read_only(move)]
        things = [what_is_clicked(move) or "" for move in worth
                  if not _goes_on(move) and not _A_COMMAND.search(what_is_clicked(move) or "")
                  and (self.writing_carried or _a_picture(what_is_clicked(move) or "", STANDS_OUT, self.pictures))]
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

    def carried(self, move: str, changed: bool, *, receipt: Any = None) -> None:
        """Response teaches responsiveness; only a bound observed effect extends the build."""
        found = what_is_carried(move)
        if found:
            from core.runtime.skill_contract import PredicateState

            verified = (receipt is not None and receipt.move == move
                        and receipt.state is PredicateState.SATISFIED)
            if receipt is not None and receipt.move == move:
                if receipt.attempt_id in self.placement_receipts:
                    return  # a receipt cannot earn another part on a later pass
                if len(self.placement_receipts) >= 64:
                    oldest = next(iter(self.placement_receipts))
                    self.placement_receipts.pop(oldest)
                    self.placement_states.pop(oldest, None)
                self.placement_receipts[receipt.attempt_id] = receipt
                self.placement_states[receipt.attempt_id] = receipt.state
            # An unknown effect stays eligible for a bounded new observation or
            # repair. A measured unchanged input is remembered as unresponsive.
            if verified:
                self.carried_to[move] = True
            elif not changed and self.carried_to.get(move) is not True:
                self.carried_to[move] = False
            self.carries_unanswered = 0 if verified else self.carries_unanswered + 1
            if verified:
                from core.perception.where_the_words_point import PLACES

                PLACES.moved()
                self.turned_since_placed = 0
                self.chain_ends_at = receipt.destination_at

    def observe_placements(self, observation: dict[str, Any]) -> None:
        """Revalidate retained occupants; an edit or reset cannot leave stale build credit."""
        from core.perception.observed_transfer import revalidate_placement

        epoch = str(observation.get("_capture_epoch") or "")
        for attempt_id, receipt in self.placement_receipts.items():
            if epoch and epoch == receipt.after_epoch:
                self.placement_states[attempt_id] = receipt.state
            else:
                self.placement_states[attempt_id] = revalidate_placement(receipt, observation)
        from core.runtime.skill_contract import PredicateState

        self.chain_ends_at = next((receipt.destination_at for attempt_id, receipt
                                  in reversed(tuple(self.placement_receipts.items()))
                                  if self.placement_states.get(attempt_id) is PredicateState.SATISFIED), None)

    def placed_count(self) -> int:
        """Count measured current occupants, not move strings or replayed receipts."""
        from core.runtime.skill_contract import PredicateState

        boxes: list[tuple[float, ...]] = []
        for attempt_id, receipt in self.placement_receipts.items():
            box = receipt.effect_bbox
            if self.placement_states.get(attempt_id) is not PredicateState.SATISFIED or box is None:
                continue
            if any(abs(box[0] - other[0]) + abs(box[1] - other[1]) < 0.025 for other in boxes):
                continue
            boxes.append(box)
        return len(boxes)


def _a_picture(name: str, stands_out: str, pictures: set[str]) -> bool:
    """Whether a thing on the screen is a picture rather than writing: what the screen reading found drawn, by the name
    it was given (where it is, or what her eyes saw it as)."""
    return name in pictures or name.startswith(("the shape at", stands_out))


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
