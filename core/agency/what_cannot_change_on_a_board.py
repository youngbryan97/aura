"""Whether anything can still change on a board of pieces, under the rules she holds of it.

From what can never move again and where every other piece could ever go (core/cognition/what_can_still_change.py),
whether a piece of either side could ever be taken, a royal ever attacked, or a piece ever reach the row where it
becomes another kind or wins. Where none of these can ever happen, nothing that decides a game can: it can only be
drawn, however far one side stands ahead. A search cannot see that, for the change it waits for lies past every look;
this sees it without looking ahead (suggestions of 23 November 2026, item 18: a fortress, or a position locked shut).

What it proves holds for every future under the rules given, taking every piece as able to move wherever its steps
could carry it with the pieces that stay as walls, and the pieces that move as gone. That is more than any of them
will do, so what it says cannot happen cannot. It knows nothing of a game beyond the rules it is given: kinds, their
steps, which kinds are royal, become others or win arriving, and what a side with no move comes to.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from core.agency.pieces_by_their_kind import FIRST, _forward, _inside
from core.cognition.what_can_still_change import could_reach, staying

__all__ = ["WhatCanChange", "what_can_change"]

#: The ways of a step that move a piece without a take.
MOVING = ("either", "moves", "swaps")
#: The ways of a step that take.
TAKING = ("either", "takes", "takes over")


@dataclass(frozen=True)
class WhatCanChange:
    """Whether anything that decides a game can still happen, and if so the first such thing found."""

    can: bool
    why: str = ""
    #: The squares of the pieces that can never move again.
    staying: frozenset = frozenset()


def _lines(family: Any, square: tuple, side: int, step: Any) -> Iterator[list[tuple]]:
    """Each line of squares a step goes along from ``square``, nearest first, as far as it may go on an empty
    board."""
    ahead = _forward(side)
    for forward, right in step.offsets:
        dr, dc = forward * ahead, right * ahead
        line, (r, c), times = [], square, 0
        while True:
            r, c = r + dr, c + dc
            times += 1
            if not _inside(family, (r, c)):
                break
            line.append((r, c))
            if step.rides and times >= step.rides:
                break
        yield line


def what_can_change(family: Any, position: Any) -> WhatCanChange:
    """Whether a take, an attack on a royal, or a piece becoming another or arriving can ever happen from
    ``position`` under ``family``'s rules."""
    if family.no_move_unattacked != "draw":
        # Where a side left no move loses, a game locked shut is still decided: by who runs out of moves first.
        return WhatCanChange(True, "a side may run out of moves")
    cells = position.cells
    crossing = {kind.crosses_with for kind in family.kinds.values() if kind.royal and kind.crosses_with}

    def kind_of(square: tuple) -> Any:
        return family.kind(cells[square][1])

    def has_somewhere_to_step(square: tuple, held: frozenset) -> bool:
        side, letter = cells[square]
        kind = family.kind(letter)
        if letter in crossing and square in position.unmoved:
            return True
        for step in kind.steps:
            if step.may not in MOVING or (step.only_unmoved and square not in position.unmoved):
                continue
            if any(line and line[0] not in held for line in _lines(family, square, side, step)):
                return True
        return False

    stay = staying(cells, has_somewhere_to_step)

    def attacked_for_good(by: int) -> set:
        """Squares a piece of ``by`` that stays attacks by a step nothing can stand in the way of."""
        found = set()
        for square in stay:
            side, letter = cells[square]
            if side != by:
                continue
            for step in family.kind(letter).steps:
                if step.may in TAKING and step.may != "takes over" and step.rides == 1:
                    for line in _lines(family, square, side, step):
                        found.update(line)
        return found

    kept_off = {side: attacked_for_good(1 - side) for side in (FIRST, 1 - FIRST)}

    def region(square: tuple) -> frozenset:
        side, letter = cells[square]
        kind = family.kind(letter)

        def onward(at: tuple) -> Iterator[tuple]:
            for step in kind.steps:
                if step.may not in MOVING or (step.only_unmoved and (at != square or square not in position.unmoved)):
                    continue
                for line in _lines(family, at, side, step):
                    for there in line:
                        if there in stay:
                            break
                        if kind.royal and there in kept_off[side]:
                            continue
                        yield there

        return could_reach(square, onward)

    regions = {square: region(square) for square in cells if square not in stay}
    standing = {side: set() for side in (FIRST, 1 - FIRST)}
    for square, (side, letter) in cells.items():
        standing[side].update(regions.get(square, {square}))
        # A square passed over may be taken on as if the piece stood there.
        if square not in stay and square in position.unmoved:
            for step in family.kind(letter).steps:
                if step.leaves_passable:
                    for line in _lines(family, square, side, step):
                        standing[side].update(line[:-1])
    for square, (side, letter) in cells.items():
        kind = family.kind(letter)
        far = family.rows - 1 if side == FIRST else 0
        if (kind.becomes or kind.wins_arriving) and any(r == far for r, _c in regions.get(square, ())):
            return WhatCanChange(True, "a piece may reach its far row", stay)
    # Riders first: theirs is the widest reach, so a position that can still change is told soonest.
    order = sorted(cells, key=lambda square: -sum(step.rides == 0 for step in kind_of(square).steps))
    for square in order:
        side, letter = cells[square]
        theirs = standing[1 - side]
        for at in regions.get(square, {square}):
            for step in family.kind(letter).steps:
                if step.may not in TAKING:
                    continue
                for line in _lines(family, at, side, step):
                    for there in line:
                        if there in theirs:
                            return WhatCanChange(True, "a piece may be taken", stay)
                        if there in stay:
                            break
    return WhatCanChange(False, "nothing either side can do changes it", stay)
