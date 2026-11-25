"""What each area of a board comes to: a board read as fights apart from each other, under the rules she holds of it
(core/cognition/what_each_area_comes_to.py).

An area is a set of pieces tied by what can take what: two pieces are in one area where either could take on the
other's square on its next move, whichever side each is on (one guarding the other is a tie as much as one threatening
it). A piece tied to nothing is an area of its own. Each area is read alone, the rest of the board standing as it is:
her moves of the area's pieces and theirs, a few moves deep, either side free to leave the area and move elsewhere after
the first move there, by what each side stands at in what she holds the kinds worth. Put back together, the areas say
which fight is most urgent, what each move of hers comes to in the whole, what each move of theirs would cost her, and
which part of the board they keep working in.

The rules, the worth of each kind and her side are given; nothing here knows a game.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from core.agency.pieces_by_their_kind import FIRST, Family, Move, Position, _reach, after, moves
from core.cognition.what_each_area_comes_to import (
    HERS,
    Choice,
    Possibilities,
    Reading,
    Threat,
    Whole,
    put_together,
    read_part,
    threats,
    what_can_happen,
    what_to_do,
)

__all__ = ["BoardReading", "area_of", "areas_of", "possible_on_a_board", "read_board"]

Square = tuple[int, int]


def _takes_from(family: Family, position: Position, square: Square) -> set[Square]:
    """The occupied squares the piece on ``square`` could take on, or guard, on its next move."""
    side, letter = position.cells[square]
    kind = family.kinds.get(letter)
    if kind is None:
        return set()
    return {end for end, _step, _passed in _reach(family, position, square, side, kind, only_takes=True)
            if end in position.cells}


def areas_of(family: Family, position: Position) -> list[frozenset[Square]]:
    """The board's pieces in areas: the sets tied together by what can take, or guard, what."""
    tied: dict[Square, Square] = {square: square for square in position.cells}

    def root(square: Square) -> Square:
        while tied[square] != square:
            tied[square] = tied[tied[square]]
            square = tied[square]
        return square

    for square in position.cells:
        for other in _takes_from(family, position, square):
            tied[root(square)] = root(other)
    areas: dict[Square, set[Square]] = {}
    for square in position.cells:
        areas.setdefault(root(square), set()).add(square)
    return sorted((frozenset(area) for area in areas.values()), key=lambda area: sorted(area))


def area_of(areas: Sequence[frozenset[Square]], square: Square) -> frozenset[Square] | None:
    return next((area for area in areas if square in area), None)


@dataclass(frozen=True)
class BoardReading:
    """A board read by its areas: each area's reading, the whole put together with her to move, her choice, and the
    other side's acts by what they would cost her."""

    readings: tuple[Reading, ...]
    whole: Whole
    choice: Choice
    threats: tuple[Threat, ...]


def _named(area: frozenset[Square]) -> Hashable:
    return tuple(sorted(area))


def read_board(family: Family, position: Position, worth: dict[str, float], *, hers: int = FIRST, plies: int = 3,
               seconds: float | None = None) -> BoardReading:
    """Read ``position`` area by area for ``hers``: ``worth`` what she holds each kind worth, ``plies`` how many moves
    deep each area is looked at, ``seconds`` the time it may take in all."""
    areas = areas_of(family, position)
    budget = None if seconds is None else seconds / max(1, 2 * len(areas))
    acts, moved, judge = _the_world_of(family, position, worth, hers)
    readings = tuple(read_part(_named(area), (position, area), acts=acts, after=moved, judge=judge, plies=plies,
                               seconds=budget) for area in areas)
    return BoardReading(readings=readings, whole=put_together(readings, first=HERS), choice=what_to_do(readings),
                        threats=tuple(threats(readings)))


def possible_on_a_board(family: Family, position: Position, worth: dict[str, float], *, hers: int = FIRST,
                        plies_at_most: int, seconds: float) -> tuple[dict[tuple, tuple[Move | None, float, int]],
                                                                       tuple[Possibilities, ...]]:
    """Everything each side can do in each area of ``position``, worked out as deep as ``seconds`` allows, shared out
    over the areas with any act in them (core/cognition/what_each_area_comes_to.py ``what_can_happen``): for each move
    the other side could make, her answer to it, what that comes to, and how many moves deep it was worked out; and
    each area's possibilities."""
    acts, moved, judge = _the_world_of(family, position, worth, hers)
    areas = [area for area in areas_of(family, position) if acts((position, area), HERS) or acts((position, area), 1)]
    each = seconds / max(1, len(areas))
    found = tuple(what_can_happen(_named(area), (position, area), acts=acts, after=moved, judge=judge,
                                  plies_at_most=plies_at_most, seconds=each,
                                  name_of=lambda state: (state[0].key(), state[1])) for area in areas)
    counters: dict[tuple, tuple[Move | None, float, int]] = {}
    for possible in found:
        for theirs, answer, comes_to in possible.counters:
            counters[(theirs.start, theirs.end, theirs.becomes)] = (answer, comes_to, possible.deep)
    return counters, found


def _the_world_of(family: Family, position: Position, worth: dict[str, float], hers: int) -> tuple[Any, Any, Any]:
    """A board as a world of areas: each side's moves of an area's pieces (the rest of the board standing), the board
    and area a move leaves, and what an area stands at for her beside where the board stands now."""
    base = _standing(position, worth, hers)

    def judge(state: tuple[Position, frozenset[Square]]) -> float:
        return _standing(state[0], worth, hers) - base

    def acts(state: tuple[Position, frozenset[Square]], side: int) -> list[Move]:
        board, squares = state
        mover = hers if side == HERS else 1 - hers
        return [move for move in moves(family, replace(board, turn=mover)) if move.start in squares]

    def moved(state: tuple[Position, frozenset[Square]], move: Move) -> tuple[Position, frozenset[Square]]:
        board, squares = state
        mover = board.cells[move.start][0]
        made = after(family, replace(board, turn=mover), move)
        gone = {move.start, move.taken_at} | set(move.also_taken or ())
        return made, frozenset((squares - gone) | {move.end} | ({move.partner[1]} if move.partner else set()))

    return acts, moved, judge


def _standing(position: Position, worth: dict[str, float], hers: int) -> float:
    total = 0.0
    for side, letter in position.cells.values():
        value = worth.get(letter, 0.0)
        total += value if side == hers else -value
    return total
