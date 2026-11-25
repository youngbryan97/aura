"""What each area of a board comes to: a board read as fights apart from each other, under the rules she holds of it
(core/cognition/what_each_area_comes_to.py).

An area is a set of pieces tied by what can take what: two pieces are in one area where either could take on the
other's square on its next move, whichever side each is on (one guarding the other is a tie as much as one threatening
it). A piece tied to nothing is an area of its own. Each area is read alone, the rest of the board standing as it is:
her moves of the area's pieces and theirs, a few moves deep, either side free to leave the area and move elsewhere after
the first move there, by what each side stands at in what she holds the kinds worth. Where her weights are given, her
whole judgement of the board (what she holds, and the ways each piece has, what stands guarded and what does not, and
whatever else her games have taught it to count) says which of her moves that keep everything held leave her worse all
the same: none of those is a move that only passes the turn. Put back together, the areas say which fight is most
urgent, what each move of hers comes to in the whole, what each move of theirs would cost her, and which part of the
board they keep working in. The areas are also played as one game (``play_the_sum``), her to move and them to move, and
where that is done as deep as the areas were read, it is what she chooses by.

The rules, the worth of each kind or her weights, and her side are given; nothing here knows a game.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from core.agency.pieces_by_their_kind import (
    FIRST,
    Family,
    Move,
    Position,
    _possible,
    _reach,
    after,
    in_check,
)
from core.agency.where_pieces_can_meet import _reach_on, movers_of
from core.cognition.what_each_area_comes_to import (
    HERS,
    THEIRS,
    Choice,
    Possibilities,
    Reading,
    SumPlayed,
    Threat,
    Whole,
    play_the_sum,
    put_together,
    read_part,
    threats,
    what_can_happen,
    what_to_do,
)
from core.cognition.who_can_wait_longer import spare_moves

__all__ = ["BoardReading", "area_of", "areas_of", "possible_on_a_board", "read_board"]

logger = logging.getLogger("Aura.WhatEachAreaOfABoardComesTo")

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
    """A board read by its areas: each area's reading, the whole put together by the quick sum with her to move, her
    choice, the other side's acts by what they would cost her, and the areas played as one game with her to move and
    with them to move (None where there was no time to play them as deep as the areas were read)."""

    readings: tuple[Reading, ...]
    whole: Whole
    choice: Choice
    threats: tuple[Threat, ...]
    played: tuple[SumPlayed, SumPlayed] | None = None


def _named(area: frozenset[Square]) -> Hashable:
    return tuple(sorted(area))


def read_board(family: Family, position: Position, worth: dict[str, float], *, hers: int = FIRST, plies: int = 3,
               seconds: float | None = None, weights: Any = None) -> BoardReading:
    """Read ``position`` area by area for ``hers``: ``worth`` what she holds each kind worth, ``plies`` how many moves
    deep each area is looked at, ``seconds`` the time it may take in all, half for the areas alone and half for playing
    them as one game. ``weights``, where given, are her whole judgement (core/agency/searching_by_kind.py ``Weights``):
    a move of hers that keeps everything held but that it judges leaves her worse is not one that changes nothing. An
    area that stands as it stood when it was last read (its pieces, and everything on their lines and on the royal
    pieces' lines) keeps that reading: what is already worked out is not worked out again."""
    areas = areas_of(family, position)
    budget = None if seconds is None else seconds / max(1, 2 * len(areas))
    acts, moved, judge = _the_world_of(family, position, worth, hers)
    rules = _the_rules(family)
    fuller = _her_judgement(family, weights, hers) if weights is not None else None
    readings = []
    for area in areas:
        key = ("read", plies, hers, tuple(sorted(worth.items())), rules, _as_it_stands(family, position, area))
        held = _WORKED_OUT.get(key)
        if held is None:
            held = read_part(_named(area), (position, area), acts=acts, after=moved, judge=judge, plies=plies,
                             seconds=budget)
            _keep(key, held)
        if fuller is not None:
            held = replace(held, weakens=_weakening(fuller, position, area, held, moved))
        readings.append(replace(held, part=_named(area)))
    readings = tuple(readings)
    movers = movers_of(position)
    reach = _reach_on(family, position)
    her_spare = spare_moves(movers, reach, hers)
    their_spare = spare_moves(movers, reach, 1 - hers)
    played = _played_as_one(family, position, areas, acts, moved, judge, plies,
                            math.inf if seconds is None else seconds / 4)
    return BoardReading(readings=readings, whole=put_together(readings, first=HERS),
                        choice=what_to_do(readings, her_spare=her_spare, their_spare=their_spare, played=played),
                        threats=tuple(threats(readings)), played=played)


def _held_together(states: tuple, i: int) -> list[tuple[Position, frozenset[Square]]]:
    """The areas after a move in area ``i``, as one board: every area stands on the board the move left, so that what
    the move did (a royal piece left open, a way closed or opened) is what every area's moves are worked out on; and an
    area the move reached (one of its squares now tied to the mover's area, or a piece of it taken or moved) is made one
    with the mover's area, so that no piece is counted, or moved, in two of them."""
    board, mine = states[i]
    mine = set(mine)
    others = [(j, area) for j, (_board, area) in enumerate(states) if j != i]
    merged = True
    while merged:
        merged = False
        for j, area in list(others):
            was = states[j][0]
            reached = bool(area & mine) or any(board.cells.get(sq) != was.cells.get(sq) for sq in area)
            if reached:
                mine |= {sq for sq in area if sq in board.cells}
                others.remove((j, area))
                merged = True
    kept = dict(others)
    return [(board, frozenset(mine)) if j == i else (board, kept[j]) for j in range(len(states)) if j == i or j in kept]


def _played_as_one(family: Family, position: Position, areas: Sequence[frozenset[Square]], acts: Any, moved: Any,
                   judge: Any, plies: int, seconds: float) -> tuple[SumPlayed, SumPlayed] | None:
    """The areas played as one game, as deep as each was read, her to move and them to move, each given ``seconds``:
    a move is a move in one area, and the sides take turns over all of them, on one board (``_held_together``). The
    whole stands where the board does. None where either was not played that deep in its time: the quick sum is then
    all there is."""
    parts = [(_named(area), (position, area)) for area in areas]
    played = tuple(play_the_sum(parts, acts=acts, after=moved, judge=judge, plies_at_most=plies, seconds=seconds,
                                name_of=lambda state: (state[0].key(), state[1]), first=side,
                                joined=_held_together, whole=lambda states: judge(states[0]))
                   for side in (HERS, THEIRS))
    if not all(one.complete and (one.deep >= plies or one.ended) for one in played):
        return None
    return played


def possible_on_a_board(family: Family, position: Position, worth: dict[str, float], *, hers: int = FIRST,
                        plies_at_most: int, seconds: float, stop: Any = None,
                        ) -> tuple[dict[tuple, tuple[Move | None, float, int]], tuple[Possibilities, ...]]:
    """Everything each side can do in each area of ``position``, worked out as deep as ``seconds`` allows, shared out
    over the areas with any act in them (core/cognition/what_each_area_comes_to.py ``what_can_happen``): for each move
    the other side could make, her answer to it, what that comes to, and how many moves deep it was worked out; and
    each area's possibilities. ``stop()``, where given, ends the working-out where it has got to: what each area had
    finished is kept, and an area not yet begun is left."""
    acts, moved, judge = _the_world_of(family, position, worth, hers)
    areas = [area for area in areas_of(family, position) if acts((position, area), HERS) or acts((position, area), 1)]
    each = seconds / max(1, len(areas))
    rules = _the_rules(family)
    found = []
    for area in areas:
        key = ("possible", hers, tuple(sorted(worth.items())), rules, _as_it_stands(family, position, area))
        held = _WORKED_OUT.get(key)
        if held is None and stop is not None and stop():
            continue
        # Worked out before as deep as it was asked, or to its end: kept; else worked out again, deeper.
        if held is None or (held.deep < plies_at_most and not held.complete):
            fresh = what_can_happen(_named(area), (position, area), acts=acts, after=moved, judge=judge,
                                    plies_at_most=plies_at_most, seconds=each,
                                    name_of=lambda state: (state[0].key(), state[1]), stop=stop)
            held = fresh if held is None or fresh.deep >= held.deep else held
            _keep(key, held)
        found.append(replace(held, part=_named(area)))
    found = tuple(found)
    counters: dict[tuple, tuple[Move | None, float, int]] = {}
    for possible in found:
        for theirs, answer, comes_to in possible.counters:
            counters[(theirs.start, theirs.end, theirs.becomes)] = (answer, comes_to, possible.deep)
    return counters, found


def _her_judgement(family: Family, weights: Any, hers: int) -> Callable[[Position], float] | None:
    """Her whole judgement of a board, from her side, as her search judges where it stops looking: the compiled
    search's, parsed once for the reading. None where the compiled search cannot judge for her (a build from before it
    could hold a judgement, or no build), and what she holds is then all an area is judged by."""
    from core.agency.searching_by_kind import _compiled, described, placed

    engine = _compiled()
    if engine is None or not hasattr(engine, "Judgement"):
        logger.info("areas judged by what is held alone: the compiled search holds no judgement here")
        return None
    judgement = engine.Judgement(described(family), weights.as_json())
    seen: dict = {}

    def of(board: Position) -> float:
        # Judged with her to move on every board, so that whose turn it is counts the same on both sides of a
        # difference.
        mine = replace(board, turn=hers)
        key = mine.key()
        if key not in seen:
            seen[key] = float(judgement.of_each([placed(mine)])[0])
        return seen[key]

    return of


def _weakening(fuller: Callable[[Position], float], position: Position, area: frozenset[Square], reading: Reading,
               moved: Any) -> tuple[Move, ...]:
    """Her moves in the area that leave it standing as it stood by what is held, after which her whole judgement of the
    board is lower than before: what the move itself does to the ways her pieces have, to what stands guarded and to
    whatever else that judgement counts. What their answers could then do is the reading's own part."""
    now = fuller(position)
    return tuple(act for act, comes_to in reading.acts
                 if comes_to == reading.as_it_stands and fuller(moved((position, area), act)[0]) < now)


def _the_world_of(family: Family, position: Position, worth: dict[str, float], hers: int) -> tuple[Any, Any, Any]:
    """A board as a world of areas: each side's moves of an area's pieces (the rest of the board standing), the board
    and area a move leaves, and what an area stands at for her beside where the board stands now."""
    base = _standing(position, worth, hers)

    def judge(state: tuple[Position, frozenset[Square]]) -> float:
        return _standing(state[0], worth, hers) - base

    def acts(state: tuple[Position, frozenset[Square]], side: int) -> list[Move]:
        """The moves of the area's pieces that do not leave the mover's royal pieces where they can be taken: the
        area's possible moves first, and only those checked."""
        board, squares = state
        mover = hers if side == HERS else 1 - hers
        moving = replace(board, turn=mover)
        return [move for move in _possible(family, moving) if move.start in squares
                and not in_check(family, after(family, moving, move), mover)]

    def moved(state: tuple[Position, frozenset[Square]], move: Move) -> tuple[Position, frozenset[Square]]:
        """The board and the area a move leaves. A move that ties the area to pieces outside it (what it can now take
        or guard, and what can now take on where it landed) brings them in for the rest of the line: two areas a move
        joins are read together from there (suggestions of 24 November 2026: what changes in one region changes
        another)."""
        board, squares = state
        mover = board.cells[move.start][0]
        made = after(family, replace(board, turn=mover), move)
        gone = {move.start, move.taken_at} | set(move.also_taken or ())
        area = (squares - gone) | {move.end} | ({move.partner[1]} if move.partner else set())
        tied = {sq for sq in _takes_from(family, made, move.end) if sq not in area}
        tied |= {sq for sq in made.cells if sq not in area and move.end in _takes_from(family, made, sq)}
        return made, frozenset(area | tied)

    return acts, moved, judge


#: What has been worked out, by what it was worked out from: kept the latest, up to as many as fill a few games.
_WORKED_OUT: OrderedDict = OrderedDict()
_KEPT_AT_MOST = 4096


def _keep(key: Hashable, held: Any) -> None:
    _WORKED_OUT[key] = held
    _WORKED_OUT.move_to_end(key)
    while len(_WORKED_OUT) > _KEPT_AT_MOST:
        _WORKED_OUT.popitem(last=False)


def _the_rules(family: Family) -> Hashable:
    """The rules a reading was worked out under. A reading under rules she has since corrected, or under another
    game's, is not the reading under these, however the pieces stand."""
    return (family.rows, family.columns, tuple(sorted(family.kinds.items())), family.count_to_draw,
            family.repeats_to_draw, family.no_move_unattacked, family.takes_first)


def _as_it_stands(family: Family, position: Position, area: frozenset[Square]) -> Hashable:
    """What an area's reading rests on: its pieces, and every piece on their lines and on the royal pieces' lines, with
    who is on the move and what may still be done once. The same where none of that has changed, whatever moved
    elsewhere."""
    royal = [sq for sq, (_side, letter) in position.cells.items()
             if letter in family.kinds and family.kinds[letter].royal]
    seen: set[Square] = set(area) | set(royal)
    for square in list(seen):
        side, letter = position.cells[square]
        kind = family.kinds.get(letter)
        if kind is None:
            continue
        seen.update(end for end, _step, _passed in _reach(family, position, square, side, kind) if end in position.cells)
    return (tuple(sorted((sq, position.cells[sq]) for sq in seen)), position.turn,
            tuple(sorted(sq for sq in position.unmoved if sq in seen)), position.passable)


def _standing(position: Position, worth: dict[str, float], hers: int) -> float:
    total = 0.0
    for side, letter in position.cells.values():
        value = worth.get(letter, 0.0)
        total += value if side == hers else -value
    return total
