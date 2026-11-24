"""What a small ending comes to: a board with few pieces left, worked out to the end under the rules she holds.

With few pieces on a board, every placement of them can be listed (three on a board of 64 squares is half a million
situations), and the general working-out of a small world (core/cognition/what_a_small_world_comes_to.py) settles
each: won for her, lost, or neither, and in how many turns under best play on both sides. What each move leads to is
what her rules say; the ends are the ones her rules give (a side with no move, a piece that wins by arriving). A take,
or a piece that becomes another, leads into an ending of other pieces, which is listed and worked out with it.

An ending rests only on the board, on how a side with no move fares, and on the rules of the kinds in it, so a change
to her rules for any other kind leaves it worked out. Nothing here knows a game: an ending is a list of pieces by side
and kind, a situation a placement of them.

Measured 23 November 2026 under the rules of chess held as a test's reference: the royal and the strongest piece
against a lone royal took 48 s and 417 MB at most to work out, every one of its situations with the stronger side to
move came out won, and the longest took 19 turns, the longest such ending known (ten moves).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.WhatASmallEndingComesTo")

__all__ = ["Ending", "an_ending_worked_out", "endings_as_bytes", "endings_from_bytes", "endings_worth_working_out",
           "rules_of_the_ending", "the_ending_of", "worked_to_the_end"]

Piece = tuple[int, str]


def the_ending_of(position: Any) -> tuple[Piece, ...]:
    """The pieces a situation has, by side and kind, in a fixed order: which ending it belongs to."""
    return tuple(sorted(position.cells.values()))


def rules_of_the_ending(family: Any, pieces: tuple[Piece, ...]) -> str:
    """What an ending rests on, as a short name: the board, how a side with no move fares, and how its kinds go."""
    from core.agency.searching_by_kind import described

    held = json.loads(described(family))
    # Its kinds and every kind they may become: a piece that becomes another leads into an ending of that kind.
    kinds = {piece[1] for piece in pieces}
    becomes = {str(kind.get("letter")): [str(b) for b in kind.get("becomes") or []] for kind in held.get("kinds", [])}
    grown = True
    while grown:
        more = {b for kind in kinds for b in becomes.get(kind, [])} - kinds
        kinds |= more
        grown = bool(more)
    rested = {key: value for key, value in held.items() if key != "kinds"}
    rested["kinds"] = [kind for kind in held.get("kinds", []) if str(kind.get("letter")) in kinds]
    return hashlib.sha1(json.dumps(rested, sort_keys=True).encode()).hexdigest()[:16]


def _placed(cells: dict, pieces: tuple[Piece, ...], columns: int) -> tuple[int, ...] | None:
    """Where each of ``pieces`` stands, each square as one number, pieces alike in rising order: one name for one
    placement."""
    by_piece: dict[Piece, list[int]] = {}
    for (row, column), held in cells.items():
        by_piece.setdefault(held, []).append(row * columns + column)
    flat: list[int] = []
    for piece, alike in itertools.groupby(pieces):
        have = sorted(by_piece.get(piece, []))
        if len(have) != len(list(alike)):
            return None
        flat.extend(have)
    return tuple(flat)


def _index_of(flat: tuple[int, ...], turn: int, squares: int) -> int:
    index = 0
    for at in flat:
        index = index * squares + at
    return index * 2 + turn


@dataclass
class Ending:
    """One ending worked out, for every placement of its pieces and either side to move: above nought, won for her
    (the first side) in that many turns less one; below nought, lost in that many less one; nought, neither."""

    pieces: tuple[Piece, ...]
    rows: int
    columns: int
    rules: str
    values: np.ndarray = field(repr=False)
    #: Whether every situation that can follow was listed. A situation past what was listed is worked out as
    #: neither, which is no answer; an ending not wholly worked out is not used (suggestions of 23 November 2026).
    complete: bool = False

    def value_of(self, position: Any) -> int | None:
        """What a situation of this ending comes to for her; None where it is not of this ending."""
        if the_ending_of(position) != self.pieces:
            return None
        flat = _placed(position.cells, self.pieces, self.columns)
        return None if flat is None else int(self.values[_index_of(flat, position.turn, self.rows * self.columns)])

    def as_bytes(self) -> bytes:
        head = json.dumps({"pieces": self.pieces, "rows": self.rows, "columns": self.columns,
                           "rules": self.rules, "complete": self.complete}).encode()
        return len(head).to_bytes(4, "big") + head + self.values.astype("<i2").tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> Ending | None:
        try:
            size = int.from_bytes(data[:4], "big")
            head = json.loads(data[4:4 + size])
            values = np.frombuffer(data[4 + size:], dtype="<i2").astype(np.int16)
            return cls(pieces=tuple((int(s), str(k)) for s, k in head["pieces"]), rows=int(head["rows"]),
                       columns=int(head["columns"]), rules=str(head["rules"]), values=values,
                       complete=bool(head.get("complete", False)))
        # not a failure: nothing kept, or nothing readable, is an ending not worked out.
        except (ValueError, KeyError, TypeError):
            return None


def an_ending_worked_out(family: Any, pieces: tuple[Piece, ...], *, seconds: float | None = None) -> Ending:
    """Every placement of ``pieces`` (side, kind) under ``family``, with every ending a move from them leads into,
    worked out back from the ends."""
    from core.agency.pieces_by_their_kind import (
        FIRST,
        Position,
        after,
        in_check,
        moves,
        outcome,
        with_turn,
    )
    from core.cognition.what_a_small_world_comes_to import AGAINST, HERS, worked_out

    rows, columns = family.rows, family.columns
    squares = rows * columns
    pieces = tuple(sorted(pieces))
    every = [(r, c) for r in range(rows) for c in range(columns)]
    groups = [len(list(alike)) for _piece, alike in itertools.groupby(pieces)]
    legal_of: dict[int, list] = {}

    def starts():
        for placed in itertools.permutations(range(squares), len(pieces)):
            at, ordered = 0, True
            for size in groups:
                run = placed[at:at + size]
                ordered = ordered and all(a < b for a, b in zip(run, run[1:], strict=False))
                at += size
            if not ordered:
                continue
            cells = {every[at]: piece for at, piece in zip(placed, pieces, strict=True)}
            for turn in (0, 1):
                position = Position(cells=cells, turn=turn)
                # A side whose royal piece stands attacked on the other side's turn: no game reaches it.
                if not in_check(family, with_turn(position, 1 - turn)):
                    yield position

    # Each situation named by one number: which ending, then where in it. No ending a move leads into has more
    # pieces than this one, so each fits in the room this one takes.
    room = squares ** len(pieces) * 2
    endings: dict[tuple, int] = {pieces: 0}

    def name_of(position: Any) -> int:
        ending = the_ending_of(position)
        which = endings.setdefault(ending, len(endings))
        return which * room + _index_of(_placed(position.cells, ending, columns), position.turn, squares)

    def end_of(position: Any) -> float | None:
        legal = moves(family, position)
        ended = outcome(family, position, legal)
        if ended.startswith("won:"):
            return 1.0 if int(ended.split(":")[1]) == FIRST else -1.0
        if ended:
            return 0.0
        legal_of[id(position)] = legal
        return None

    def next_of(position: Any):
        legal = legal_of.pop(id(position), None) or moves(family, position)
        return [(move, after(family, position, move)) for move in legal]

    worked = worked_out(starts(), next_of, end_of, lambda p: HERS if p.turn == FIRST else AGAINST,
                        name_of=name_of, most=room * 4, seconds=seconds, keep=False)
    values = np.zeros(room, dtype=np.int16)
    for name, index in worked.names.items():
        if name < room:
            forced = int(worked.forced[index])
            values[name] = forced * (int(worked.turns[index]) + 1) if forced else 0
    solved = Ending(pieces=pieces, rows=rows, columns=columns, rules=rules_of_the_ending(family, pieces),
                    values=values, complete=worked.complete)
    logger.info("worked out the ending %s to the end: %d situations listed%s; %d won for her, %d lost", pieces,
                len(worked), "" if worked.complete else " (not all)", int((values > 0).sum()),
                int((values < 0).sum()))
    return solved


def endings_as_bytes(endings: dict[tuple, Ending]) -> bytes:
    """Every ending worked out, one after another, each led by its length."""
    out = b""
    for ending in endings.values():
        held = ending.as_bytes()
        out += len(held).to_bytes(8, "big") + held
    return out


def endings_from_bytes(data: bytes | None) -> dict[tuple, Ending]:
    endings: dict[tuple, Ending] = {}
    at = 0
    while data and at + 8 <= len(data):
        size = int.from_bytes(data[at:at + 8], "big")
        ending = Ending.from_bytes(data[at + 8:at + 8 + size])
        at += 8 + size
        if ending is not None:
            endings[ending.pieces] = ending
    return endings


def endings_worth_working_out(rules: Any) -> list[tuple[Piece, ...]]:
    """The smallest endings a game at this board can come to that are not already over: each side's royal piece with
    one piece of any other kind either side began with, beside it. Those of the kinds that reach furthest first, for
    they are the ones that can force an end."""
    from core.agency.pieces_by_their_kind import FIRST, SECOND
    from core.agency.searching_by_kind import _reached_on_empty

    began = set(rules.start.values())
    royal = {side: next((kind for owner, kind in sorted(began) if owner == side and rules.royal(kind)), None)
             for side in (FIRST, SECOND)}
    if None in royal.values():
        return []
    family = rules.family(unseen_go_every_way=True)
    others = sorted({(side, kind) for side, kind in began if kind != royal[side]},
                    key=lambda piece: -_reached_on_empty(family, piece[1]))
    return [tuple(sorted(((FIRST, royal[FIRST]), (SECOND, royal[SECOND]), piece))) for piece in others]


def _as_it_was_worked_out(family: Any, position: Any) -> bool:
    """Whether a situation has the history an ending was worked out with: no piece that may cross with a partner still
    standing unmoved with one, since an ending is worked out with every piece as having moved, and nothing passable."""
    if position.passable:
        return False
    for square in position.unmoved:
        held = position.cells.get(square)
        partner = family.kind(held[1]).crosses_with if held is not None else ""
        if partner and any(position.cells.get(other) == (held[0], partner) for other in position.unmoved):
            return False
    return True


def worked_to_the_end(endings: dict[tuple, Ending] | None, family: Any, position: Any, legal: list) -> Any:
    """Her move in a situation of an ending she has worked out under the rules she plays by, where it is won or lost:
    the quickest win, else the longest a loss can be put off. None where it is neither, or not worked out."""
    if not endings:
        return None
    ending = endings.get(the_ending_of(position))
    if ending is None or not ending.complete or ending.rules != rules_of_the_ending(family, ending.pieces):
        return None
    if not _as_it_was_worked_out(family, position):
        return None
    worth = ending.value_of(position)
    if not worth:
        return None
    # A win or loss that takes more turns than the count toward a draw has left is a draw: each situation before the
    # last of its ``abs(worth) - 1`` turns is one the count may draw, and the last is decided before the count is read.
    if family.count_to_draw and position.count + abs(worth) - 2 >= family.count_to_draw:
        return None
    from core.agency.pieces_by_their_kind import after

    def comes_to(move: Any) -> int:
        nxt = after(family, position, move)
        there = endings.get(the_ending_of(nxt))
        held = there.value_of(nxt) if there is not None and there.complete \
            and there.rules == rules_of_the_ending(family, there.pieces) else None
        return 0 if held is None else held

    # For her (the first side): a win soonest, else neither, else a loss as late as it can come.
    def preference(move: Any) -> tuple:
        value = comes_to(move)
        return (0, value) if value > 0 else (1, 0) if value == 0 else (2, value)

    return min(legal, key=preference) if legal else None
