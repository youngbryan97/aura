"""Searching a game of pieces that move by their kind: the best move she can find in the time a move is worth.

The search is general game-tree search over the description (core/agency/pieces_by_their_kind.py): alpha-beta deepened
one move at a time, a table of positions already judged, takes followed to the end of a line before a position is
judged, a pass tried to see whether a position is so good the other side cannot answer it, late moves looked at less
deeply. It runs compiled (rust_extensions/aura_board_search) where that is built, and in Python here where it is not;
the two read the same description and make the same moves.

What a position is worth is judged by weights, not by rules: what each kind is worth, how much each of its moves adds,
where it stands, how pressed a royal piece is. Before she has played a game of a family, a kind's worth is what it can
do: the squares it reaches, on average, from each square of an empty board, a far row it may become something better on
counted by what it may become. The weights she measures by playing replace these (core/agency/what_a_kind_is_worth.py).
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.agency.pieces_by_their_kind import Family, Move, Position, after, in_check, moves, outcome

logger = logging.getLogger("Aura.SearchingByKind")

__all__ = ["Found", "Weights", "described", "first_weights", "placed", "search"]

#: The scale a kind's worth is counted on: the worth of reaching one square.
PER_SQUARE = 25


@dataclass
class Weights:
    """What a position is worth, by its parts, in hundredths of what reaching a square is worth times 4."""

    worth: dict[str, float] = field(default_factory=dict)
    mobility: dict[str, float] = field(default_factory=dict)
    squares: dict[str, list[list[float]]] = field(default_factory=dict)
    tempo: float = 0.0
    royal_pressed: float = 0.0
    #: Per kind, against its side, for each piece of it the other side attacks and its own does not defend
    #: (core/cognition/what_she_cannot_afford_to_lose.py). Nothing until rehearsal measures it.
    exposed: dict[str, float] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(asdict(self))

    def under_other_names(self, names: dict[str, str], *, turned_round: bool = False) -> Weights:
        """These weights for a board that begins as theirs does with each kind called what ``names`` maps it to, seen
        from its other side where ``turned_round``. Each side's squares count rows from its own back row and columns
        from the first side's left, so seen from the other side a kind's squares keep their rows and swap their
        columns. A kind with no name there is left out."""
        def by_name(of: dict[str, Any]) -> dict[str, Any]:
            return {names[kind]: value for kind, value in of.items() if kind in names}

        return Weights(worth=by_name(self.worth), mobility=by_name(self.mobility),
                       squares={kind: [list(reversed(row)) if turned_round else list(row) for row in table]
                                for kind, table in by_name(self.squares).items()},
                       tempo=self.tempo, royal_pressed=self.royal_pressed, exposed=by_name(self.exposed))


@dataclass
class Found:
    move: Move | None
    score: int
    depth: int
    nodes: int
    line: list[Move]
    mate: bool = False
    ms: int = 0
    by: str = ""
    #: How far the best score moved between the last two depths the search completed: under this, two moves' scores
    #: do not tell them apart.
    unsteady: int = 0


def _reached_on_empty(family: Family, letter: str) -> float:
    """How many squares a kind reaches, on average over the squares of an empty board, as the first side."""
    from core.agency.pieces_by_their_kind import FIRST, _reach

    kind = family.kind(letter)
    empty = Position(cells={}, turn=FIRST, unmoved=frozenset())
    total, squares = 0, 0
    for row in range(family.rows):
        for column in range(family.columns):
            square = (row, column)
            lone = Position(cells={square: (FIRST, letter)}, turn=FIRST, unmoved=frozenset())
            reached = {end for end, step, _ in _reach(family, lone, square, FIRST, kind) if step.may != "takes"}
            reached |= {end for end, step, _ in _reach(family, lone, square, FIRST, kind) if step.may == "takes"}
            total += len(reached)
            squares += 1
    del empty
    return total / max(1, squares)


def first_weights(family: Family) -> Weights:
    """Weights from what each kind can do, before any game of the family is played: a kind is worth the squares it
    reaches on an empty board, and a kind that may become another on the far row is worth, besides, its share of the
    best it may become (a share of one over the rows it must cross). A royal piece is not counted: it is never taken."""
    reach = {letter: _reached_on_empty(family, letter) for letter in family.kinds}
    worth: dict[str, float] = {}
    for letter, kind in family.kinds.items():
        if kind.royal:
            worth[letter] = 0.0
            continue
        value = reach[letter]
        if kind.becomes:
            value += max(reach[b] for b in kind.becomes) / max(1, family.rows - 2)
        worth[letter] = round(value * PER_SQUARE, 1)
    # A kind not yet seen to move is worth what the kinds seen are, in the middle: not nothing, which would give it
    # away, and not the most, which would never trade it.
    seen = sorted(w for letter, w in worth.items() if w > 0)
    middle = seen[len(seen) // 2] if seen else PER_SQUARE * 4
    for letter, kind in family.kinds.items():
        if not kind.royal and worth[letter] == 0.0:
            worth[letter] = middle
    mobility = {letter: 0.0 for letter in family.kinds}
    return Weights(worth=worth, mobility=mobility, tempo=0.0)


def described(family: Family) -> str:
    """The family as the compiled search reads it."""
    kinds = []
    for letter, kind in family.kinds.items():
        kinds.append({
            "letter": letter,
            "steps": [{"offsets": [list(o) for o in s.offsets], "rides": s.rides, "may": s.may,
                       "only_unmoved": s.only_unmoved, "leaves_passable": s.leaves_passable,
                       "takes_in_passing": s.takes_in_passing} for s in kind.steps],
            "royal": kind.royal, "becomes": list(kind.becomes), "cannot_go_back": kind.cannot_go_back,
            "crosses_with": kind.crosses_with,
        })
    return json.dumps({"rows": family.rows, "columns": family.columns, "kinds": kinds,
                       "count_to_draw": family.count_to_draw, "repeats_to_draw": family.repeats_to_draw,
                       "no_move_unattacked": family.no_move_unattacked})


def _position_data(position: Position) -> dict[str, Any]:
    return {"cells": [[r, c, side, letter] for (r, c), (side, letter) in sorted(position.cells.items())],
            "turn": position.turn, "unmoved": [list(sq) for sq in sorted(position.unmoved)],
            "passable": [list(position.passable[0]), list(position.passable[1])] if position.passable else None,
            "count": position.count}


def placed(position: Position, earlier: list[Position] | None = None) -> str:
    """A position as the compiled search reads it, with the positions before it for counting repeats."""
    data = _position_data(position)
    data["history"] = [_position_data(p) for p in (earlier or [])]
    return json.dumps(data)


def _compiled() -> Any | None:
    try:
        import aura_board_search

        return aura_board_search
    except ImportError:
        return None


def _matching(family: Family, position: Position, found: tuple) -> Move | None:
    fr, fc, tr, tc, becomes = found
    for move in moves(family, position):
        if move.start == (fr, fc) and move.end == (tr, tc) and move.becomes == (becomes or ""):
            return move
    return None


def search(family: Family, position: Position, *, seconds: float, weights: Weights | None = None,
           earlier: list[Position] | None = None, nodes: int = 0, set_aside: list[Move] | None = None) -> Found:
    """The best move for the side to move, searched for ``seconds`` (or ``nodes``), leaving out ``set_aside``: moves
    the game has shown are not moves."""
    weights = weights or first_weights(family)
    aside = list(set_aside or [])
    engine = _compiled()
    if engine is not None:
        result = engine.search(described(family), placed(position, earlier), weights.as_json(), float(seconds),
                               nodes=int(nodes),
                               excluded=[(m.start[0], m.start[1], m.end[0], m.end[1], m.becomes) for m in aside])
        if result.get("move") is None:
            return Found(None, 0, 0, int(result.get("nodes", 0)), [], by="compiled")
        line: list[Move] = []
        walk = position
        for step in result.get("line") or []:
            move = _matching(family, walk, tuple(step))
            if move is None:
                break
            line.append(move)
            walk = after(family, walk, move)
        first = line[0] if line else _matching(family, position, tuple(result["move"]))
        return Found(first, int(result["score"]), int(result["depth"]), int(result["nodes"]), line,
                     mate=bool(result.get("mate")), ms=int(result.get("ms", 0)), by="compiled",
                     unsteady=int(result.get("unsteady", 0)))
    return _search_here(family, position, seconds=seconds, weights=weights, set_aside=aside)


_MATE = 1_000_000


def _worth_here(family: Family, position: Position, weights: Weights) -> float:
    score = [0.0, 0.0]
    for _square, (side, letter) in position.cells.items():
        score[side] += weights.worth.get(letter, 0.0)
    return score[position.turn] - score[1 - position.turn]


def _search_here(family: Family, position: Position, *, seconds: float, weights: Weights,
                 set_aside: list[Move] | None = None) -> Found:
    """The same search in Python, slower: alpha-beta deepened move by move, takes followed to quiet."""
    ends_at = time.monotonic() + seconds
    looked = [0]

    def quiet(pos: Position, alpha: float, beta: float, depth: int) -> float:
        looked[0] += 1
        stand = _worth_here(family, pos, weights)
        if stand >= beta or depth <= -6:
            return stand
        alpha = max(alpha, stand)
        for move in moves(family, pos):
            if move.taken_at is None and not move.becomes:
                continue
            value = -quiet(after(family, pos, move), -beta, -alpha, depth - 1)
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return alpha

    def negamax(pos: Position, depth: int, alpha: float, beta: float, ply: int) -> float:
        looked[0] += 1
        if looked[0] % 256 == 0 and time.monotonic() > ends_at:
            raise TimeoutError
        legal = moves(family, pos)
        ended = outcome(family, pos, legal)
        if ended.startswith("won"):
            return -(_MATE - ply)
        if ended:
            return 0.0
        if depth <= 0:
            return quiet(pos, alpha, beta, 0)
        best = -math.inf
        for move in sorted(legal, key=lambda m: m.taken_at is None):
            value = -negamax(after(family, pos, move), depth - 1, -beta, -alpha, ply + 1)
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best

    aside = {(m.start, m.end, m.becomes) for m in set_aside or []}
    legal = [m for m in moves(family, position) if (m.start, m.end, m.becomes) not in aside]
    if not legal:
        return Found(None, 0, 0, 0, [], by="here")
    chosen, chosen_value, depth_done = legal[0], 0.0, 0
    ordered = list(legal)
    for depth in range(1, 64):
        try:
            alpha, best, best_value = -math.inf, None, -math.inf
            for move in ordered:
                value = -negamax(after(family, position, move), depth - 1, -math.inf, -alpha, 1)
                if value > best_value:
                    best, best_value = move, value
                alpha = max(alpha, value)
        except TimeoutError:
            break
        chosen, chosen_value, depth_done = best, best_value, depth
        ordered = [best, *(m for m in ordered if m != best)]
        if abs(best_value) > _MATE - 1000:
            break
    return Found(chosen, int(chosen_value), depth_done, looked[0], [chosen], mate=abs(chosen_value) > _MATE - 1000,
                 by="here")


def gives_check(family: Family, position: Position, move: Move) -> bool:
    made = after(family, position, move)
    return in_check(family, made, made.turn)
