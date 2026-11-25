"""Positioning in time on a board of pieces (core/cognition/where_forces_can_meet.py): each piece a mover, how its kind
goes (her rules of the board) its reach, and the places that matter to a side the squares of what it must keep safe and
the squares around them.

Nothing here knows a game: a board's kinds, which of them the game keeps safe, and how each goes are what she learned
of it (core/agency/how_the_pieces_move.py).
"""
from __future__ import annotations

from typing import Any

from core.agency.pieces_by_their_kind import Family, Position, _reach, attacked
from core.cognition.where_forces_can_meet import Mover, intended, time_to_reach_safely

__all__ = ["intended_on_a_board", "movers_of", "places_that_matter", "safe_ways_on_a_board"]


def movers_of(position: Position) -> list[Mover]:
    return [Mover(name=square, side=side, at=square) for square, (side, _kind) in position.cells.items()]


def places_that_matter(family: Family, position: Position) -> dict[int, list[tuple[int, int]]]:
    """For each side, the squares of what it must keep safe and every square next to one; and, among what the other
    side must hold, the squares where a kind of this side becomes another or wins by arriving, as she learned them: a
    place where one side grows stronger is a gate the other holds. Bryan, 24 November 2026, of a board game against
    the hardest computer opponent he plays: he never moved the pieces standing where theirs would grow stronger, and
    nothing of theirs ever did."""
    matters: dict[int, list[tuple[int, int]]] = {}
    for (r, c), (side, letter) in position.cells.items():
        kind = family.kinds.get(letter)
        if kind is None or not kind.royal:
            continue
        around = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if 0 <= r + dr < family.rows and 0 <= c + dc < family.columns]
        matters.setdefault(side, []).extend(around)
    for side in (0, 1):
        if any(position.cells.get(sq, (None,))[0] == side and (family.kinds[letter].becomes
                                                               or family.kinds[letter].wins_arriving)
               for sq, (_s, letter) in position.cells.items() if letter in family.kinds):
            far = family.rows - 1 if side == 0 else 0
            # Where this side grows stronger is what the other side holds.
            matters.setdefault(1 - side, []).extend((far, c) for c in range(family.columns))
    return matters


def _reach_on(family: Family, position: Position) -> Any:
    held = dict(position.cells)
    cache: dict[tuple[Any, Any, frozenset], list[tuple[int, int]]] = {}

    def reach(mover: Mover, place: tuple[int, int], standing: frozenset) -> list[tuple[int, int]]:
        key = (mover.name, place, standing)
        if key in cache:
            return cache[key]
        side, letter = held[mover.name]
        kind = family.kinds.get(letter)
        if kind is None:
            cache[key] = []
            return []
        cells = {sq: held[sq] for sq in standing if sq in held}
        cells[place] = (side, letter)
        there = Position(cells=cells, turn=side, unmoved=position.unmoved if place == mover.name else frozenset())
        found = [end for end, _step, _passed in _reach(family, there, place, side, kind)
                 if cells.get(end, (None,))[0] != side]
        cache[key] = found
        return found

    return reach


def intended_on_a_board(family: Family, position: Position, side: int, horizon: int = 3) -> dict[tuple, float]:
    """``side``'s moves that serve its position in time, each with a share, keyed by (from square, to square)."""
    matters = places_that_matter(family, position)
    if 1 - side not in matters:
        return {}
    return intended(movers_of(position), _reach_on(family, position), side, matters, horizon)


def _safe_on(family: Family, position: Position) -> Any:
    held = dict(position.cells)

    def safe(mover: Mover, place: tuple[int, int], standing: frozenset) -> bool:
        side, letter = held[mover.name]
        cells = {sq: held[sq] for sq in standing if sq in held}
        cells[place] = (side, letter)
        return not attacked(family, Position(cells=cells, turn=1 - side), place, 1 - side)

    return safe


def safe_ways_on_a_board(family: Family, position: Position, square: tuple[int, int], horizon: int) -> dict:
    """Where the piece on ``square`` can get, and how soon, going only by squares where nothing of the other side can
    take it on arriving, by the board's rules of taking as she learned them."""
    side, _letter = position.cells[square]
    mover = Mover(name=square, side=side, at=square)
    return time_to_reach_safely(mover, _reach_on(family, position), position.cells, horizon,
                                _safe_on(family, position))
