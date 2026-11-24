"""What the game showed: every answer the world gave about what may be done, kept so that her rules can be checked
against all of it at once.

Her rules of a game are counts of what she saw: how often a kind was offered a square, seen to go somewhere, refused.
Counts can agree with most of what was seen and still be wrong in a way that matters: one false move in a model is not
noise to a search, which goes looking for it, and one missing defence turns a lost plan into a won one. A model is
qualified by being checked against what was shown, as a whole, not by how much it has seen.

So each answer is kept as it was given: the situation, the piece picked up and the squares the game offered it (its
withholding every other square is as much an answer); a move the game refused. Checked against her rules as they stand,
each answer says what her rules would have the game offer that it did not (spurious), and what it offered that her rules
would not (missing). Both should be none. Either, named, is where her rules are wrong, and the situations they are wrong
in are the counterexamples to learn from.

Nothing here knows a game: a situation is pieces on squares and whose turn it is, an answer a set of squares.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhatTheGameShowed")

__all__ = ["KEPT_AT_MOST", "Checked", "WhatTheGameShowed", "offered_where_it_began"]

#: Answers kept at most, the latest: each is a whole situation written out.
KEPT_AT_MOST = 800


def _written(position: Any) -> dict[str, Any]:
    return {"cells": [[r, c, side, kind] for (r, c), (side, kind) in sorted(position.cells.items())],
            "turn": position.turn, "unmoved": [list(sq) for sq in sorted(position.unmoved)],
            "passable": [list(position.passable[0]), list(position.passable[1])] if position.passable else None}


def _read(held: dict[str, Any]) -> Any:
    from core.agency.pieces_by_their_kind import Position

    passable = held.get("passable")
    return Position(cells={(int(r), int(c)): (int(side), str(kind)) for r, c, side, kind in held["cells"]},
                    turn=int(held["turn"]), unmoved=frozenset((int(r), int(c)) for r, c in held.get("unmoved", [])),
                    passable=((tuple(passable[0]), tuple(passable[1])) if passable else None))


@dataclass
class Checked:
    """Her rules against what the game showed: answers checked, and the squares her rules got wrong."""

    answers: int = 0
    missing: int = 0
    spurious: int = 0
    #: Kind -> squares wrong, either way.
    by_kind: dict[str, int] = field(default_factory=dict)
    #: A few of the situations, for whoever has to find out why.
    examples: list[dict[str, Any]] = field(default_factory=list)

    def says(self) -> str:
        if not self.answers:
            return "the game has shown her nothing to check her rules against yet"
        if not self.missing and not self.spurious:
            return f"her rules agree with all {self.answers} answers the game gave"
        return (f"her rules disagree with the game on {self.missing + self.spurious} square(s) over {self.answers} "
                f"answers: {self.missing} it offered that her rules would not, {self.spurious} her rules would offer "
                f"that it did not")

    def as_dict(self) -> dict[str, Any]:
        return {"answers": self.answers, "missing": self.missing, "spurious": self.spurious,
                "by_kind": dict(self.by_kind)}


@dataclass
class WhatTheGameShowed:
    """The answers the game gave: squares offered to a piece picked up, and moves refused."""

    offers: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)

    def offered(self, position: Any, square: tuple[int, int], squares: Any) -> None:
        self.offers.append({"at": _written(position), "from": list(square), "to": sorted(list(sq) for sq in squares)})
        del self.offers[:-KEPT_AT_MOST]

    def refused(self, position: Any, move: Any) -> None:
        self.refusals.append({"at": _written(position), "from": list(move.start), "to": list(move.end)})
        del self.refusals[:-KEPT_AT_MOST]

    def checked(self, family: Any) -> Checked:
        """Her rules, as ``family``, against every answer kept. An offer is read as the act it stands for before it is
        checked: a game may offer a crossing as its partner's square, and the learner reads that square as the crossing
        (how_the_pieces_move), so the checker does too. Read as squares, a crossing offered by its partner's square
        was two disagreements, one square missing and one spurious, about a move her rules and the game agree on
        (suggestions of 23 November 2026)."""
        from core.agency.pieces_by_their_kind import _possible, moves

        found = Checked()
        for answer in self.offers:
            try:
                position = _read(answer["at"])
            # not a failure: an answer kept by an older reader that cannot be read back is passed over.
            except (KeyError, TypeError, ValueError):
                continue
            start = tuple(answer["from"])
            # An offer of nothing may be a piece not picked up, or a board not yet taking moves, and the learner reads
            # it as saying nothing (how_the_pieces_move): so does the checker. LIVE 23 November 2026 six pieces picked
            # up on a game's setup board were counted as six disagreements.
            if start not in position.cells or position.cells[start][0] != position.turn or not answer["to"]:
                continue
            kind = position.cells[start][1]
            # Each partner's square her rules have the piece crossing with, read as the square the crossing ends on.
            crossing = {m.partner[0]: m.end for m in _possible(family, position)
                        if m.start == start and m.partner is not None and m.partner[0] != m.end}
            theirs = {crossing.get(tuple(sq), tuple(sq)) for sq in answer["to"]}
            mine = {m.end for m in moves(family, position) if m.start == start}
            missing, spurious = theirs - mine, mine - theirs
            found.answers += 1
            found.missing += len(missing)
            found.spurious += len(spurious)
            if missing or spurious:
                found.by_kind[kind] = found.by_kind.get(kind, 0) + len(missing) + len(spurious)
                if len(found.examples) < 5:
                    found.examples.append({"kind": kind, "from": list(start), "missing": sorted(missing),
                                           "spurious": sorted(spurious)})
        for answer in self.refusals:
            try:
                position = _read(answer["at"])
            # not a failure: as above.
            except (KeyError, TypeError, ValueError):
                continue
            start, end = tuple(answer["from"]), tuple(answer["to"])
            if start not in position.cells:
                continue
            found.answers += 1
            if any(m.start == start and m.end == end for m in moves(family, position)):
                found.spurious += 1
                kind = position.cells[start][1]
                found.by_kind[kind] = found.by_kind.get(kind, 0) + 1
        return found

    def as_bytes(self) -> bytes:
        return json.dumps({"offers": self.offers, "refusals": self.refusals}, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes | None) -> WhatTheGameShowed:
        try:
            held = json.loads(data or b"{}")
        # not a failure: nothing kept, or nothing readable, is nothing shown yet.
        except (TypeError, ValueError):
            return cls()
        if not isinstance(held, dict):
            return cls()
        return cls(offers=list(held.get("offers") or [])[-KEPT_AT_MOST:],
                   refusals=list(held.get("refusals") or [])[-KEPT_AT_MOST:])


def offered_where_it_began(shown: WhatTheGameShowed, start: dict, family: Any) -> list[tuple[tuple[int, int], set]]:
    """The acts the game offered at the state it began in that ``family`` does not allow, by the square each was
    offered from: what her rules can least afford to have wrong, since every game she plays there begins there."""
    from core.agency.pieces_by_their_kind import moves

    began = {square: tuple(held) for square, held in start.items()}
    out: list[tuple[tuple[int, int], set]] = []
    for answer in shown.offers:
        try:
            position = _read(answer["at"])
        # not a failure: an answer kept by an older reader that cannot be read back is passed over.
        except (KeyError, TypeError, ValueError):
            continue
        if not answer["to"] or {sq: tuple(held) for sq, held in position.cells.items()} != began:
            continue
        start_square = tuple(answer["from"])
        allowed = {m.end for m in moves(family, position) if m.start == start_square}
        missing = {tuple(sq) for sq in answer["to"]} - allowed
        if missing:
            out.append((start_square, missing))
    return out
