"""Playing a well: a game of falling pieces known for what it is by what it shows, and each piece put where it does most.

Nobody is told that a game is one of falling pieces. A person sees a tall well cut into cells, something coming down
it by itself, and things resting on its floor, and knows the whole family at once: the piece can be moved across, most
likely turned, and dropped; what lands stays; some arrangement of what has landed clears; and the well filling to the
top is the end. Which keys do which, and which arrangement clears, are that game's own, and a person finds them in the
first few pieces by pressing each key and watching the piece, and by watching what goes when a piece lands.

That is what is done here. The well is seen (core/perception/a_well_seen.py). Each key she has is pressed once on a
piece and known by what it did to it: took it a column across, changed it, or brought it down. Every landing the piece
could make is then tried in her head (core/agency/where_a_piece_goes.py) and the piece taken to the best, a press at a
time, each press checked against where the piece then is. What clears is held as rival rules, each kept by how often
it foretold what a landing left; beside the rules that have names (full rows, three alike in a line, four together)
is one learned outright, of kinds that clear when they touch, for games where a thing goes with another unlike it.
Kinds not yet seen touching are worth bringing together, since that is how what goes with what is found out.

Nothing here knows a game. The keys are whichever of hers move the piece; a kind is whatever a cell's look tells apart.
"""
from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.where_a_piece_goes import (
    RULES,
    Well,
    dropped,
    ready,
    rule_from_words,
    score,
    settled,
)
from core.verify import invariant

logger = logging.getLogger("Aura.PlayingAWell")

__all__ = ["NOTHING_MOVES_IT", "GoesWith", "HerKeys", "WhatClears", "a_well_with_a_piece_coming_down", "orientations", "play_a_well",
           "what_a_turn_did"]

Cell = tuple[int, int]
Parts = tuple[tuple[int, int, int], ...]

#: How a stretch ends where no key of hers takes the piece across.
NOTHING_MOVES_IT = "nothing she presses moves the piece across"
#: The name of the learned rule, beside the named ones.
GOES_WITH = "what goes with what"
#: How long a key is held for a press, and how long after it the piece is looked at.
HELD_S, SHOWS_S = 0.06, 0.16
#: How long a key that seemed to do nothing is held to see whether it brings the piece down.
HELD_TO_DROP_S = 0.35
#: How many rows more than it would have fallen by itself a piece must come down for a key to be the one that drops it.
ROWS_MORE = 2.0
#: How often a key is tried before it is taken to do nothing to a piece.
TRIES_OF_A_KEY = 2
#: How long after a piece lands what it left is believed: what clears takes a moment to go.
SETTLES_S = 1.2
#: How many looks at most are taken to get two running that agree.
STEADY_LOOKS = 5
#: How long what rests must be seen unchanged, after changing, before it is taken in as it then is.
COMES_TO_REST_S = 0.3
#: How long a thing read as a piece may hang unmoved before it is taken for part of what rests.
STUCK_S = 2.5
#: How long with no piece coming down before the game is taken to be over, and with no well in sight.
OVER_AFTER_S, HIDDEN_AT_MOST_S = 6.0, 4.0
#: How long ground that could be a well's is watched for something coming down in it, and how often it is looked at.
WATCHED_S, LOOKS_APART_S = 3.0, 0.12
#: How long she has, from first pressing at a piece, to find the two keys that take it across before it is no well.
ACROSS_FOUND_WITHIN_S = 12.0
#: How many looks may show something moving where the piece is not before the ground is no well: one may be a clear.
MOVES_ELSEWHERE = 2
#: How long a well cut into cells is watched: a piece that comes down a row a second, three rows tall, takes this
#: long to be seen coming.
WELL_WATCHED_S = 8.0
#: What bringing two kinds together for the first time is worth, in cells cleared: one, since one clear is what it may show.
FINDS_OUT = 1.0
#: What a landing that leaves the next piece no room to come in costs, in cells cleared: more than any landing clears.
NO_ROOM_TO_COME_IN = 1000.0
#: What each row short of four clear rows above a piece's height, where pieces come in, costs: as a hole does.
CROWDS_THE_WAY_IN = 4.0
#: The keys tried on a piece, in the order tried: across first, since nothing can be placed without them.
KEYS_TRIED = ("left", "right", "up", "space", "down", "z", "x")


def _now() -> float:
    return time.monotonic()


async def _wait(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _normal(parts: Sequence[tuple[int, int, int]]) -> Parts:
    """A piece's parts from its own corner, in one order: the same piece is the same tuple wherever it is."""
    least_dc, least_dr = min(p[0] for p in parts), min(p[1] for p in parts)
    return tuple(sorted((dc - least_dc, dr - least_dr, kind) for dc, dr, kind in parts))


def _reordered(parts: Parts, order: Sequence[int]) -> Parts:
    """The piece with its kinds put in another order over the same places: each place takes the kind of the place
    ``order`` gives for it."""
    places = sorted((dc, dr) for dc, dr, _kind in parts)
    kinds = [kind for _dc, _dr, kind in sorted(parts)]
    if len(order) != len(kinds):
        # An order learned on a piece of so many parts says nothing of one with more or fewer: it lies as it lies.
        return _normal(parts)
    return _normal([(dc, dr, kinds[order[i]]) for i, (dc, dr) in enumerate(places)])


def _turned(parts: Parts, quarters: int) -> Parts:
    for _ in range(quarters % 4):
        tallest = max(dr for _dc, dr, _kind in parts)
        parts = _normal([(tallest - dr, dc, kind) for dc, dr, kind in parts])
    return _normal(parts)


def what_a_turn_did(before: Parts, after: Parts) -> tuple[str, Any] | None:
    """What a key did to a piece it changed in place: put its kinds in another order over the same places (the order,
    place by place), or turned its shape by quarters. Where more than one order would give what is seen, the plainest
    is taken: a step round before a turn about before anything else."""
    before, after = _normal(before), _normal(after)
    if before == after or len(before) != len(after):
        return None
    count = len(before)
    if sorted((dc, dr) for dc, dr, _k in before) == sorted((dc, dr) for dc, dr, _k in after):
        round_about = [tuple((i + by) % count for i in range(count)) for by in range(1, count)]
        plain = [*round_about, tuple(reversed(range(count)))]
        others = itertools.permutations(range(count)) if count <= 5 else ()
        for order in (*plain, *others):
            if _reordered(before, order) == after:
                return ("order", tuple(order))
    for quarters in (1, 2, 3):
        if _turned(before, quarters) == after:
            return ("shape", quarters)
    return None


def orientations(parts: Parts, turn: tuple[str, Any] | None) -> list[Parts]:
    """Every way the piece can be made to lie by pressing the key that changes it, in the order the presses give."""
    found = [_normal(parts)]
    while turn is not None and len(found) < 8:
        nxt = _reordered(found[-1], turn[1]) if turn[0] == "order" else _turned(found[-1], turn[1])
        if nxt in found:
            break
        found.append(nxt)
    return found


# -- what clears ------------------------------------------------------------------


def _gone(placed: Well, after: Well) -> set[Cell]:
    """The cells of a well as a piece landed that are not in what was left: in each column, what was left is what
    stayed, in its order, fallen down."""
    gone: set[Cell] = set()
    for column in range(placed.columns):
        was = [(row, placed.cells[(column, row)]) for row in range(placed.rows - 1, -1, -1) if (column, row) in placed.cells]
        left = [after.cells[(column, row)] for row in range(after.rows - 1, -1, -1) if (column, row) in after.cells]
        at = 0
        for row, kind in was:
            if at < len(left) and left[at] == kind:
                at += 1
            else:
                gone.add((column, row))
    return gone


@dataclass
class GoesWith:
    """Kinds that clear when they touch, learned from what went and what stayed as pieces landed: for each two kinds
    seen touching, how often both went and how often they stayed."""

    went: dict[frozenset, int] = field(default_factory=dict)
    stayed: dict[frozenset, int] = field(default_factory=dict)

    def goes(self, one: int, other: int) -> bool:
        pair = frozenset((one, other))
        return self.went.get(pair, 0) > self.stayed.get(pair, 0)

    def met(self, one: int, other: int) -> bool:
        pair = frozenset((one, other))
        return pair in self.went or pair in self.stayed

    def clear(self, well: Well) -> set[Cell]:
        gone: set[Cell] = set()
        for (column, row), kind in well.cells.items():
            for near in ((column + 1, row), (column, row + 1)):
                other = well.cells.get(near)
                if other is not None and self.goes(kind, other):
                    gone |= {(column, row), near}
        return gone

    def learn(self, placed: Well, gone: set[Cell]) -> None:
        """A piece landed and these cells went. Kinds that touched and did not both go are seen to stay. Of what went,
        each lot of touching cells is read on its own: a lot of one kind shows that kind goes with itself, a lot of
        two kinds shows the two go together, and a lot of more shows nothing, since any of them may have gone with any
        other, or later, as what was above fell."""
        touching = [((column, row), near) for (column, row) in placed.cells for near in ((column + 1, row), (column, row + 1))
                    if near in placed.cells]
        for one, other in touching:
            if not (one in gone and other in gone):
                pair = frozenset((placed.cells[one], placed.cells[other]))
                self.stayed[pair] = self.stayed.get(pair, 0) + 1
        left = set(gone)
        while left:
            lot, frontier = set(), [left.pop()]
            while frontier:
                column, row = cell = frontier.pop()
                lot.add(cell)
                for near in ((column + 1, row), (column - 1, row), (column, row + 1), (column, row - 1)):
                    if near in left:
                        left.discard(near)
                        frontier.append(near)
            kinds = frozenset(placed.cells[cell] for cell in lot if cell in placed.cells)
            if len(lot) >= 2 and 1 <= len(kinds) <= 2:
                self.went[kinds] = self.went.get(kinds, 0) + 1

    def pairs(self, shown: int = 1) -> list[tuple[int, int]]:
        """The pairs taken to go together: those that went more often than they stayed, and at least ``shown`` times."""
        return [_two(pair) for pair in self.went
                if self.went[pair] >= shown and self.went[pair] > self.stayed.get(pair, 0)]


def _two(pair: frozenset) -> tuple[int, int]:
    kinds = sorted(pair)
    return (kinds[0], kinds[-1])


@dataclass
class WhatClears:
    """The rival rules of what clears, each with how often it foretold a clear and how often it was wrong."""

    named: str | None = None
    foretold: dict[str, int] = field(default_factory=dict)
    wrong: dict[str, int] = field(default_factory=dict)
    goes_with: GoesWith = field(default_factory=GoesWith)

    def told(self, words: str) -> None:
        self.named = rule_from_words(words) or self.named

    def _clears(self, name: str) -> Callable[[Well], set[Cell]]:
        return self.goes_with.clear if name == GOES_WITH else RULES[name]

    def saw(self, placed: Well, after: Well) -> str | None:
        """A piece landed: the well as it landed and as it was left. Each rule is held to what it would have left;
        the name of a rule that has now foretold a clear for the first time, for saying."""
        gone = _gone(placed, after)
        first: str | None = None
        for name in (*RULES, GOES_WITH):
            would, cleared, _chain = settled(placed, self._clears(name))
            if would.cells != after.cells:
                self.wrong[name] = self.wrong.get(name, 0) + 1
            elif cleared:
                first = first or (name if not self.foretold.get(name) else None)
                self.foretold[name] = self.foretold.get(name, 0) + 1
        # The learned rule is held to the landing before it learns from it, as the others are.
        self.goes_with.learn(placed, gone)
        if gone and first is None and settled(placed, self.goes_with.clear)[0].cells == after.cells \
                and not self.foretold.get(GOES_WITH):
            # What it has just learned accounts for what went: said now, and held to the next landing.
            first = GOES_WITH
        return first

    def name(self) -> str:
        """The rule to place by: the one that has foretold most over its mistakes; the one the words name until some
        rule has foretold anything; the learned one where nothing has."""
        standing = max((*RULES, GOES_WITH), key=lambda name: self.foretold.get(name, 0) - self.wrong.get(name, 0)
                       if self.foretold.get(name) else -10**6)
        if self.foretold.get(standing):
            return standing
        if self.goes_with.pairs():
            return GOES_WITH
        return self.named or GOES_WITH

    def rule(self) -> Callable[[Well], set[Cell]]:
        return self._clears(self.name())


# -- what her keys do to a piece ----------------------------------------------------


@dataclass
class HerKeys:
    """Which of her keys takes the piece left, which right, which changes it and how, and which brings it down."""

    left: str = ""
    right: str = ""
    turn: str = ""
    drop: str = ""
    turn_does: tuple[str, Any] | None = None
    #: How often each key was pressed on a piece and showed nothing, and how often held and showed nothing.
    nothing: dict[str, int] = field(default_factory=dict)
    held: dict[str, int] = field(default_factory=dict)

    def known(self) -> set[str]:
        return {key for key in (self.left, self.right, self.turn, self.drop) if key}

    def to_try(self, offered: Sequence[str]) -> str:
        """The next key worth pressing to see what it does: none once every use is found or every key tried. Until two
        keys take the piece across, each is pressed; after, a key pressed to no effect is still owed one holding,
        since a key that brings a piece down faster may show only when held."""
        if self.left and self.right and self.turn and self.drop:
            return ""
        mine = [key for key in KEYS_TRIED if key in offered] + [key for key in offered if key not in KEYS_TRIED]
        mine = [key for key in mine if key not in self.known()]
        if not (self.left and self.right):
            return next((key for key in mine if self.nothing.get(key, 0) < TRIES_OF_A_KEY), "")
        return next((key for key in mine if not self.nothing.get(key) or (not self.drop and not self.held.get(key))), "")

    def said(self) -> str:
        told = []
        if self.left and self.right:
            told.append(f"{self.left} and {self.right} move the piece across")
        if self.turn and self.turn_does:
            told.append(f"{self.turn} " + ("changes the order of its parts" if self.turn_does[0] == "order" else "turns it"))
        if self.drop:
            told.append(f"{self.drop} brings it down")
        return "; ".join(told)


# -- the play ---------------------------------------------------------------------


@dataclass
class _Game:
    well: Any
    hands: Any
    look: Any
    tell: Callable[[str], None]
    offered: tuple[str, ...]
    clears: WhatClears = field(default_factory=WhatClears)
    keys: HerKeys = field(default_factory=HerKeys)
    #: Current-piece horizontal responses; recalled key names remain hypotheses.
    current_controls: dict[str, int] = field(default_factory=dict)
    pieces: int = 0
    cleared: int = 0
    #: The cells that rest, as she holds them: changed only when a piece lands or what rests is seen to have changed.
    stack: dict[Cell, int] | None = None
    #: The piece as last seen, the cells of it that had something under them then, and since when it has not moved.
    last: Any = None
    stood: dict[Cell, int] = field(default_factory=dict)
    unmoved_since: float = 0.0
    #: For each cell holding something not counted as resting: what it holds, and since when it has held it.
    held_since: dict[Cell, tuple[int, float]] = field(default_factory=dict)
    #: Since when what rests has been seen to differ from what she holds, and the cells as last read.
    changing_since: float | None = None
    cells_before: dict[Cell, int] | None = None
    #: The well as the last piece landed, waiting to be held against what it left, and when it landed.
    landed: Well | None = None
    landed_at: float = 0.0
    #: Where the piece in the air is to go (how it lies, and its column), and the cells that plan was made over.
    plan: tuple[Parts, int] | None = None
    planned_over: dict[Cell, int] | None = None
    holding: str = ""
    #: The columns pieces come in by, and the tallest a piece has stood: where they come in must be kept clear.
    enters: set = field(default_factory=set)
    tallest: int = 1
    said: set = field(default_factory=set)
    #: When the piece's row last changed by itself, and the times between such changes.
    row_changed_at: float = 0.0
    fall_periods: list[float] = field(default_factory=list)

    def once(self, key: str, line: str) -> None:
        if key not in self.said:
            self.said.add(key)
            self.tell(line)

    def the_well(self, cells: dict[Cell, int]) -> Well:
        return Well(self.well.columns, self.well.rows, dict(cells))

    async def seen(self) -> tuple[dict[Cell, int], Any, bool] | None:
        """One look: every cell that holds something, the piece coming down, and whether what rests has changed from
        what she holds (then no piece is told apart until it has been taken in again)."""
        from core.perception.a_well_seen import the_piece_in, what_rests, what_stands_in

        got = await self.look()
        if got is None:
            return None
        cells = what_stands_in(got[0], self.well)
        if self.stack is None:
            self.stack = what_rests(cells, self.well)
        if any(cells.get(cell) != kind for cell, kind in self.stack.items()):
            return cells, None, True
        return cells, the_piece_in(got[0], self.well, self.stack), False

    async def steady(self) -> tuple[dict[Cell, int], Any, bool] | None:
        """A look to act on: looks, a moment apart, until two running agree on where the piece is and how it lies. A
        piece caught part-way across or part-way through changing is read as it is not: LIVE-like 2026-10-11 one read
        mid-slide with a part missing was taken for a new piece, and the key that slid it for the key that drops."""
        before: tuple[Any, ...] | None = None
        seen = None
        for _ in range(STEADY_LOOKS):
            seen = await self.seen()
            if seen is None or seen[2]:
                return seen
            piece = seen[1]
            mark = (None,) if piece is None else (piece.column, round(piece.row), _normal(piece.parts))
            if mark == before:
                return seen
            before = mark
            await _wait(LOOKS_APART_S)
        return seen

    async def let_go(self) -> None:
        if self.holding:
            await self.hands.up(self.holding)
            self.holding = ""

    async def press(self, key: str, held: float = HELD_S) -> None:
        await self.hands.down(key)
        await _wait(held)
        await self.hands.up(key)
        await _wait(SHOWS_S)


def _cells_of(piece: Any) -> dict[Cell, int]:
    """The cells a piece is in, to the nearest row."""
    return {(piece.column + dc, round(piece.row + dr)): kind for dc, dr, kind in piece.parts}


def _stands(game: _Game, piece: Any) -> dict[Cell, int]:
    """The cells of what was read as the piece that have something under them, down to the floor or to what rests:
    a piece that has landed, or the part of the reading that has, where the next piece is already in sight above it."""
    cells, stack = _cells_of(piece), game.stack or {}
    standing: dict[Cell, int] = {}
    for (column, row) in sorted(cells, key=lambda cell: -cell[1]):
        if row + 1 >= game.well.rows or (column, row + 1) in stack or (column, row + 1) in standing:
            standing[(column, row)] = cells[(column, row)]
    if not standing:
        return {}
    # A piece of blocks lands as one: what touches a part that stands, stands with it.
    grew = True
    while grew:
        grew = False
        for (column, row), kind in cells.items():
            if (column, row) not in standing and any(near in standing for near in (
                    (column + 1, row), (column - 1, row), (column, row + 1), (column, row - 1))):
                standing[(column, row)], grew = kind, True
    return standing


def _a_new_piece(piece: Any, last: Any) -> bool:
    """Whether a piece is the next one and not the one last seen: at the top, where the last was well below it."""
    return last is not None and piece is not None and piece.row < 1.0 and last.row > piece.row + 1.5


def _falls_in(game: _Game, seconds: float) -> float:
    """How many rows a piece comes down by itself in a while, from how often its row has been seen to change."""
    if not game.fall_periods:
        return 1.0
    usual = sorted(game.fall_periods)[len(game.fall_periods) // 2]
    return 1.0 + seconds / max(0.05, usual)


def _room_under(game: _Game, piece: Any) -> float:
    """How many rows the piece can still come down before it rests on something."""
    room = float(game.well.rows)
    for dc, dr, _kind in piece.parts:
        column, row = piece.column + dc, piece.row + dr
        floor = min((under for (col, under) in (game.stack or {}) if col == column and under > row), default=game.well.rows)
        room = min(room, floor - row - 1.0)
    return room


async def _find_out(game: _Game, key: str, before: Any) -> None:
    """Press a key on the piece and know it by what it did: took the piece across, changed it, or brought it down.
    What it did is judged from a steady look, and where the look cannot say (what rests is changing, the piece is out
    of sight, less of it is read than there was) nothing is concluded and the key is tried again."""
    keys = game.keys
    if _controls_found(game) and keys.nothing.get(key) and not keys.drop and before.whole:
        # Pressed before to no effect: all that is left to find is whether, held, it brings the piece down.
        await _held_to_see(game, key, before)
        return
    room = _room_under(game, before)
    await game.press(key)
    seen = await game.steady()
    after = seen[1] if seen is not None and not seen[2] else None
    logger.info("pressed %s on a piece at column %d, row %.1f, %s: then %s", key, before.column, before.row,
                [kind for _dc, _dr, kind in sorted(before.parts)],
                (after.column, round(after.row, 1), [kind for _dc, _dr, kind in sorted(after.parts)]) if after is not None else None)
    if after is None or (before.whole and len(after.parts) < len(before.parts) and not _a_new_piece(after, before)):
        return
    if _a_new_piece(after, before) or (_stands(game, after) and after.row - before.row >= room - 0.5):
        # Down the whole way, with room there was to fall: the key that drops a piece. With no room it was landing
        # anyway, and the press showed nothing either way.
        if room >= ROWS_MORE + 1.0:
            keys.drop = keys.drop or key
        return
    if after.column != before.column:
        if not before.whole or not after.whole or _normal(before.parts) != _normal(after.parts):
            return
        if after.column < before.column:
            if keys.right == key:
                keys.right = ''
            keys.left = keys.left or key
            game.current_controls[key] = -1
        else:
            if keys.left == key:
                keys.left = ''
            keys.right = keys.right or key
            game.current_controls[key] = 1
        return
    did = what_a_turn_did(before.parts, after.parts) if before.whole and after.whole else None
    if did is not None and not keys.turn:
        keys.turn, keys.turn_does = key, did
        return
    keys.nothing[key] = keys.nothing.get(key, 0) + 1
    if not keys.drop and before.whole:
        await _held_to_see(game, key, after)


async def _held_to_see(game: _Game, key: str, piece: Any) -> None:
    """Hold a key that did nothing when pressed, to see whether it brings the piece down faster than the piece comes
    by itself: tried only with room enough under the piece that coming down by itself could not be taken for it, and
    left for a later piece where there is not."""
    keys = game.keys
    if keys.held.get(key) or _room_under(game, piece) < _falls_in(game, HELD_TO_DROP_S + SHOWS_S) + ROWS_MORE + 0.5:
        return
    began = _now()
    await game.hands.down(key)
    await _wait(HELD_TO_DROP_S)
    await game.hands.up(key)
    seen = await game.steady()
    lower = seen[1] if seen is not None and not seen[2] else None
    logger.info("held %s on a piece at row %.1f: then %s", key, piece.row, round(lower.row, 1) if lower is not None else None)
    # Out of sight, or what rests already changing: with the room there was under it, it could only have got down that
    # fast by the key.
    fell = float(game.well.rows) if lower is None or _a_new_piece(lower, piece) else lower.row - piece.row
    if fell >= _falls_in(game, _now() - began) + ROWS_MORE:
        keys.drop = key
    else:
        keys.held[key] = keys.held.get(key, 0) + 1


#: The rules under which cells of one kind lying together are part-way to clearing.
_ALIKE_CLEARS = frozenset(name for name in RULES if name != "full rows")


def _top_of(cells: dict[Cell, int], column: int, rows: int) -> int:
    """The row of the highest thing resting in a column; the floor's row where the column is empty."""
    return min((row for (col, row) in cells if col == column), default=rows)


def _can_get_there(game: _Game, piece: Any, lying: Parts, column: int) -> bool:
    """Whether the piece can be taken across to a column from where it is: nothing that rests stands as high as the
    piece's foot in any column it would pass through."""
    stack, wide = game.stack or {}, 1 + max(dc for dc, _dr, _k in lying)
    foot = piece.row + piece.shape()[1]
    passes = range(min(piece.column, column), max(piece.column + piece.shape()[0], column + wide))
    return all(_top_of(stack, col, game.well.rows) >= foot for col in passes if 0 <= col < game.well.columns)


def _where_it_goes(game: _Game, piece: Any) -> tuple[Parts, int] | None:
    """The best landing for the piece over what rests: how it should lie and the column for its corner. Each landing
    is scored by what it would clear and leave. Like kinds left touching count toward the next clear only where like
    kinds are what clears. While what clears is still being found, bringing together two kinds that have not yet met
    counts for what it may show, less as the well fills. And where pieces come in is kept clear: a landing that would
    leave no room there for the next piece is the end of the game, and is taken only when every landing would."""
    name = game.clears.name()
    well, rule, goes = game.the_well(game.stack or {}), game.clears.rule(), game.clears.goes_with
    ways = orientations(piece.parts, game.keys.turn_does if game.keys.turn else None)
    reach = range(-3, well.columns + 3) if game.keys.left and game.keys.right else (piece.column,)
    best: tuple[float, Parts, int] | None = None
    for lying in ways:
        for column in reach:
            placed = dropped(well, lying, column)
            if placed is None or (column != piece.column and not _can_get_there(game, piece, lying, column)):
                continue
            after, cleared, chain = settled(placed, rule)
            value = score(after, cleared, chain)
            if name not in _ALIKE_CLEARS:
                value -= ready(after)
            if name == GOES_WITH:
                value += 10.0 * FINDS_OUT * max(0.0, 1.0 - 2.0 * after.height() / well.rows) * len(
                    {frozenset((kind, placed.cells[near])) for (col, row), kind in placed.cells.items()
                     for near in ((col + 1, row), (col, row + 1))
                     if near in placed.cells and ((col, row) not in well.cells or near not in well.cells)
                     and not goes.met(kind, placed.cells[near])})
            room = min((_top_of(after.cells, col, well.rows) for col in game.enters), default=well.rows)
            if room < game.tallest + 1:
                value -= NO_ROOM_TO_COME_IN
            else:
                value -= CROWDS_THE_WAY_IN * max(0, game.tallest + 4 - room)
            if best is None or value > best[0]:
                best = (value, lying, column)
    return (best[1], best[2]) if best is not None else None


def _kind_said(game: _Game, kind: int) -> str:
    from core.agency.naming_what_she_sees import colour_name

    try:
        look = game.well.looks[kind]
        return colour_name([int(v) for v in look.reshape(-1, 3).mean(axis=0)])
    except (IndexError, AttributeError, TypeError, ValueError):
        return "one"


#: How often two kinds must have gone together before she says they do: once is a sign, twice is shown.
SHOWN = 2


def _the_rule_said(name: str) -> str:
    return {"full rows": "A row filled the whole way across clears. I'll fill rows."}.get(
        name, f"What clears here is {name}. I'll line those up.")


def _pairs_said(game: _Game) -> None:
    """Each two kinds shown to go together, said once, in what she sees them as."""
    for one, other in game.clears.goes_with.pairs(SHOWN):
        first, second = _kind_said(game, one), _kind_said(game, other)
        game.once(f"pair:{one}:{other}", f"A {first} piece and a {second} one clear when they touch. I'll bring those together."
                  if first != second or one == other else "Two kinds I can tell apart by sight clear when they touch. "
                  "I'll bring those together.")


def _held_to_what_it_left(game: _Game, left: dict[Cell, int]) -> None:
    """The last landing held against what it left: every rule is tried against it, and what went is learned from."""
    if game.landed is None:
        return
    if not _controls_found(game):
        game.landed = None
        return
    after = game.the_well(left)
    gone = _gone(game.landed, after)
    game.cleared += len(gone)
    first = game.clears.saw(game.landed, after)
    game.landed = None
    if gone:
        logger.info("a piece landed and %d cell(s) went; the rule held to is %s (%s)", len(gone), game.clears.name(),
                    {name: (game.clears.foretold.get(name, 0), game.clears.wrong.get(name, 0)) for name in (*RULES, GOES_WITH)})
    del first
    standing = game.clears.name()
    if standing != GOES_WITH and game.clears.foretold.get(standing, 0) >= SHOWN:
        # Said once it has foretold what went twice over: once may be chance.
        game.once(f"rule:{standing}", _the_rule_said(standing))
    _pairs_said(game)


def _it_landed(game: _Game, standing: dict[Cell, int]) -> None:
    """A piece is down: it joins what rests, and the well as it landed is kept to be held against what it leaves."""
    _held_to_what_it_left(game, game.stack or {})
    game.stack = {**(game.stack or {}), **standing}
    game.landed, game.landed_at = (game.the_well(game.stack) if _controls_found(game) else None), _now()
    game.pieces += int(_controls_found(game))
    game.last, game.plan, game.stood = None, None, {}


def _taken_in_again(game: _Game, cells: dict[Cell, int]) -> None:
    """What rests has changed and come to rest again: it is taken in as it now is. Where a rule she holds foretells
    what the last landing would leave and that is what is there, that is what rests and anything more is the next
    piece; where none does, what rests is what stands on the floor with nothing missing under it."""
    if game.landed is not None:
        for name in sorted((*RULES, GOES_WITH), key=lambda n: game.clears.wrong.get(n, 0) - game.clears.foretold.get(n, 0)):
            would, cleared, _chain = settled(game.landed, game.clears._clears(name))
            if cleared and all(cells.get(cell) == kind for cell, kind in would.cells.items()):
                _held_to_what_it_left(game, would.cells)
                game.stack = dict(would.cells)
                return
    resting: dict[Cell, int] = {}
    for column in range(game.well.columns):
        row = game.well.rows - 1
        while (column, row) in cells:
            resting[(column, row)] = cells[(column, row)]
            row -= 1
    _held_to_what_it_left(game, resting)
    game.stack = resting


async def _tried(game: _Game, key: str, piece: Any) -> None:
    before = game.keys.said()
    await game.let_go()
    await _find_out(game, key, piece)
    game.plan = None
    if game.keys.said() != before:
        if _controls_found(game):
            # What it is is said once it has answered as such a thing does: a piece taken a column across by her keys.
            # LIVE 2026-10-11 it was said on sight, of a field with creatures walking down it.
            game.once("well", f"This is a well, {game.well.columns} across: pieces come down it and stack up, and I can "
                              "move each one across. I'll put each where it does the most.")
        game.once(f"keys:{game.keys.said()}", f"I've found that {game.keys.said()}.")


async def _taken_there(game: _Game, piece: Any) -> bool:
    """One press toward where the piece is to go: changed until it lies right, then across. Whether a press was made:
    not, once the piece is where it is to go or she has no key that would take it nearer."""
    if game.plan is None or game.planned_over != game.stack:
        game.plan, game.planned_over = _where_it_goes(game, piece), dict(game.stack or {})
    if game.plan is None:
        return False
    lying, column = game.plan
    keys = game.keys
    if _normal(piece.parts) != lying and keys.turn and lying in orientations(piece.parts, keys.turn_does):
        await game.let_go()
        await game.press(keys.turn)
        return True
    if piece.column != column and (keys.left if column < piece.column else keys.right):
        await game.let_go()
        await game.press(keys.left if column < piece.column else keys.right)
        return True
    return False


def _its_fall_noted(game: _Game, piece: Any, pressed: bool) -> None:
    now = _now()
    if game.last is not None and not pressed and piece.row > game.last.row + 0.2 and game.row_changed_at:
        game.fall_periods = [*game.fall_periods[-8:], (now - game.row_changed_at) / max(1.0, piece.row - game.last.row)]
    if game.last is None or piece.row != game.last.row or piece.column != game.last.column \
            or piece.parts != game.last.parts:
        game.row_changed_at = game.unmoved_since = now


def _what_stayed_put_is_taken_in(game: _Game, cells: dict[Cell, int]) -> bool:
    """Cells that hold something she does not count among what rests, and have held the same thing for longer than
    any part of a falling piece stays in one cell, rest: a piece whose landing she missed. They join what rests, so
    one landing missed is not carried through the rest of the game as part of every piece after it. Whether any did."""
    now, stack = _now(), game.stack or {}
    game.held_since = {cell: (kind, game.held_since[cell][1] if game.held_since.get(cell, (None, 0.0))[0] == kind else now)
                       for cell, kind in cells.items() if cell not in stack}
    usual = sorted(game.fall_periods)[len(game.fall_periods) // 2] if game.fall_periods else 0.0
    long_enough = max(STUCK_S, (game.tallest + 1.5) * usual)
    stayed = {cell: kind for cell, (kind, since) in game.held_since.items() if now - since >= long_enough}
    if not stayed:
        return False
    logger.info("%d cell(s) not counted as resting have not moved for %.1fs: taken in as resting", len(stayed), long_enough)
    _held_to_what_it_left(game, stack)
    game.stack = {**stack, **stayed}
    game.last, game.plan, game.stood, game.held_since = None, None, {}, {}
    return True


async def _while_it_changes(game: _Game, cells: dict[Cell, int]) -> None:
    """What rests is changing (something is clearing, or falling into the gap it left): nothing is pressed, and once
    two looks a moment apart agree it is taken in again."""
    await game.let_go()
    if game.changing_since is None and game.last is not None and game.last.whole:
        # What rests changed while a piece was in the air: the piece is down, and what it landed on has begun to
        # clear before she saw it standing. It landed where a piece lying as it lay, over that column, comes to rest.
        placed = dropped(game.the_well(game.stack or {}), _normal(game.last.parts), game.last.column)
        if placed is not None:
            _it_landed(game, {cell: kind for cell, kind in placed.cells.items() if cell not in (game.stack or {})})
    game.changing_since = game.changing_since or _now()
    if cells == game.cells_before and _now() - game.changing_since >= COMES_TO_REST_S:
        _taken_in_again(game, cells)
        game.changing_since, game.last, game.plan = None, None, None
    game.cells_before = cells
    await _wait(LOOKS_APART_S)


async def _one_piece_on(game: _Game, piece: Any) -> bool:
    """What is done with a whole piece in the air on one look. Whether there is anything she can do with it at all."""
    if not _controls_found(game):
        # Nothing can be placed until she can take a piece across: her keys are tried until two of them do.
        for direction, sign in (('left', -1), ('right', 1)):
            candidate = getattr(game.keys, direction)
            if candidate and game.current_controls.get(candidate) != sign and game.keys.nothing.get(candidate, 0) >= TRIES_OF_A_KEY:
                setattr(game.keys, direction, '')
        pending = [key for key, sign in ((game.keys.left, -1), (game.keys.right, 1))
                   if key and game.current_controls.get(key) != sign]
        key = pending[0] if pending else game.keys.to_try(game.offered)
        if not key:
            return False
        await _tried(game, key, piece)
    elif not await _taken_there(game, piece):
        # In place, where trying a key costs the piece nothing: her other keys are tried until what changes a piece
        # and what brings it down are both found, each held only while there is room under the piece to tell by.
        key = game.keys.to_try(game.offered)
        roomy = _room_under(game, piece) >= _falls_in(game, HELD_TO_DROP_S + SHOWS_S) + ROWS_MORE + 0.5
        if key and (roomy or not game.keys.nothing.get(key)):
            await _tried(game, key, piece)
        elif game.keys.drop and not game.holding:
            await game.hands.down(game.keys.drop)
            game.holding = game.keys.drop
        else:
            await _wait(LOOKS_APART_S)
    return True


async def play_a_well(look: Any, hands: Any, well: Any, words: str = "", *, seconds: float, say: Any = None,
                      offered: Sequence[str] = KEYS_TRIED, known: dict[str, Any] | None = None) -> dict[str, Any]:
    """Play the well in front of her for ``seconds``. ``look`` gives a picture and its time; ``hands`` has ``down`` and
    ``up`` for a key. ``known`` is what she kept of this game from before (her keys, what clears), and is kept up.
    What came of it, for whoever handed the game over."""
    from core.perception.a_well_seen import a_well_in

    began = _now()
    game = _Game(well, hands, look, say or (lambda _line: None), tuple(offered))
    _recalled(game, known)
    game.clears.told(words)
    nothing_since: float | None = None
    hidden_since: float | None = None
    ended = "out of time"
    pressed = False
    try:
        while _now() - began < seconds:
            # Every provisional path spends the same admission budget. A
            # changing picture cannot postpone it by repeatedly continuing.
            if not _controls_found(game) and _now()-began > ACROSS_FOUND_WITHIN_S:
                ended = NOTHING_MOVES_IT
                break
            got = await look()
            if got is None or a_well_in(got[0], game.well) is None:
                hidden_since = hidden_since or _now()
                if _now() - hidden_since > HIDDEN_AT_MOST_S:
                    ended = "the well is gone"
                    break
                await _wait(LOOKS_APART_S)
                continue
            hidden_since = None
            seen = await game.seen()
            if seen is None:
                continue
            cells, piece, changing = seen
            if changing:
                await _while_it_changes(game, cells)
                continue
            if _what_stayed_put_is_taken_in(game, cells):
                continue
            if piece is None:
                nothing_since = nothing_since or _now()
                if _now() - nothing_since > OVER_AFTER_S:
                    ended = "nothing more comes down" if _controls_found(game) else NOTHING_MOVES_IT
                    break
                await _wait(LOOKS_APART_S)
                continue
            nothing_since = None
            _its_fall_noted(game, piece, pressed)
            pressed = False
            standing = _stands(game, piece)
            if standing and (standing == game.stood or len(standing) < len(piece.parts)):
                # Down, and seen so twice running; or part of what was read stands and the rest is the next piece.
                await game.let_go()
                _it_landed(game, standing)
                continue
            if not standing and _now() - game.unmoved_since > max(STUCK_S, 3.0 * _falls_in(game, 0.0)):
                # Hanging there unmoved for longer than any piece hangs: it is part of what rests, misread as coming.
                counted = game.pieces
                _it_landed(game, _cells_of(piece))
                game.pieces = counted
                continue
            if game.last is None and piece.row < 1.0:
                game.enters |= {piece.column + dc for dc, _dr, _kind in piece.parts}
            if piece.whole:
                game.tallest = max(game.tallest, piece.shape()[1])
            game.stood, game.last = standing, piece
            if game.landed is not None and piece.whole and _now() - game.landed_at >= SETTLES_S:
                # The last landing has had its time and nothing went.
                _held_to_what_it_left(game, game.stack or {})
            if not piece.whole or standing:
                await _wait(LOOKS_APART_S)
                continue
            pressed = True
            if not await _one_piece_on(game, piece):
                ended = NOTHING_MOVES_IT
                break
            if not _controls_found(game) and _now() - began > ACROSS_FOUND_WITHIN_S:
                # A well answers at once: a piece goes a column across at the first press of the right key. Where
                # none has in this long, it is no well, whatever it looked like.
                ended = NOTHING_MOVES_IT
                break
    finally:
        await game.let_go()
    if ended == NOTHING_MOVES_IT:
        # What she held of her keys did not let her place a piece: it is let go, and found again from nothing next time.
        game.keys = HerKeys()
    if _controls_found(game):
        _kept(game, known)
    return {"ended": ended, "pieces": game.pieces, "cleared": game.cleared,
            "rule": game.clears.name() if _controls_found(game) else '',
            "keys": game.keys.said() if _controls_found(game) else '',
            "controls_confirmed": _controls_found(game), "seconds": round(_now() - began, 1)}


def _controls_found(game: _Game) -> bool:
    keys = game.keys
    return (bool(keys.left and keys.right) and keys.left != keys.right
            and game.current_controls.get(keys.left) == -1 and game.current_controls.get(keys.right) == 1)


@invariant('agency.provisional_well_motion_cannot_supply_a_played_result', scope='agency',
           owner='core/agency/playing_a_well.py', observational=False)
def _provisional_well_invariant() -> tuple:
    from types import SimpleNamespace

    game = _Game(SimpleNamespace(columns=8, rows=16), None, None, lambda line: None, ())
    game.keys.left, game.keys.right = 'a', 'd'
    _it_landed(game, {(0, 15): 1})
    _held_to_what_it_left(game, {})
    assert game.pieces == game.cleared == 0 and game.landed is None
    assert not _controls_found(game), 'recalled names supplied current control evidence'
    game.current_controls = {'a': -1, 'd': 1}
    assert _controls_found(game)
    return ()


def _recalled(game: _Game, known: dict[str, Any] | None) -> None:
    """Reuse prior hypotheses; current responses must admit the controls again."""
    kept = (known or {}).get("well_keys")
    if isinstance(kept, dict):
        keys = game.keys
        keys.left, keys.right = str(kept.get("left") or ""), str(kept.get("right") or "")
        keys.turn, keys.drop = str(kept.get("turn") or ""), str(kept.get("drop") or "")
        does = kept.get("turn_does")
        if isinstance(does, (list, tuple)) and len(does) == 2:
            keys.turn_does = (str(does[0]), tuple(does[1]) if isinstance(does[1], (list, tuple)) else int(does[1]))
        if keys.turn and keys.turn_does is None:
            keys.turn = ""
    held = (known or {}).get("well_clears")
    verified = (known or {}).get('well_control_verified') is True
    if isinstance(held, WhatClears) and verified:
        game.clears = copy.deepcopy(held)
    looks = (known or {}).get("well_looks")
    if isinstance(looks, list) and verified and not game.well.looks:
        game.well.looks = copy.deepcopy(looks)
    said = (known or {}).get("well_said")
    if isinstance(said, set) and verified:
        game.said = set(said)
    enters = (known or {}).get("well_enters")
    if isinstance(enters, (list, set)) and verified:
        game.enters = {int(column) for column in enters}


def _kept(game: _Game, known: dict[str, Any] | None) -> None:
    if known is None:
        return
    known['well_control_verified'] = True
    keys = game.keys
    does = keys.turn_does
    known["well_keys"] = {"left": keys.left, "right": keys.right, "turn": keys.turn, "drop": keys.drop,
                          "turn_does": [does[0], list(does[1]) if isinstance(does[1], tuple) else does[1]] if does else None}
    known["well_clears"] = game.clears
    known["well_looks"] = game.well.looks
    known["well_said"] = game.said
    known["well_enters"] = sorted(game.enters)


def _what_a_well_picture_shows(picture: Any, known: Any) -> tuple:
    """Read detached pixels together; temporal history stays with the capture owner."""
    from core.perception.a_well_seen import (
        a_ground_like_a_wells,
        a_well_in,
        the_piece_in,
        what_rests,
        what_stands_in,
    )

    well = a_well_in(picture, known)
    if well is None:
        return None, a_ground_like_a_wells(picture), {}, None
    cells = what_stands_in(picture, well)
    return well, True, cells, the_piece_in(picture, well, what_rests(cells, well))


async def a_well_with_a_piece_coming_down(look: Any, known: Any = None, seen: list[Any] | None = None) -> Any:
    """Watch for falling pieces while pure picture work leaves the control loop free."""
    from core.perception.a_picture_examined_without_holding_the_loop import PictureProbe

    probe = PictureProbe()
    try:
        return await _watched_for_a_well(look, known, seen, probe)
    finally:
        await probe.close()


async def _watched_for_a_well(look: Any, known: Any, seen: list[Any] | None, probe: Any) -> Any:
    """The well in front of her, where there is one with something coming down in it by itself: the two things that
    make a game one of falling pieces. Watched for a few seconds, with nothing pressed. A well seen with nothing yet
    coming down it is put in ``seen``, for whoever would rather wait for a piece than take it for something else."""
    from core.perception.a_picture_examined_without_holding_the_loop import UNMEASURED

    began = _now()
    well = None
    sightings: list[Any] = []
    filled: set[Cell] | None = None
    last: Any = None
    elsewhere = 0
    while _now() - began < (WELL_WATCHED_S if well is not None else WATCHED_S):
        got = await look()
        if got is None:
            return None
        measured = await probe.examine(_what_a_well_picture_shows, got[0], well or known)
        if measured is UNMEASURED:
            return UNMEASURED
        well, ground, cells, piece = measured
        if well is None:
            # No ground a well could be: not this kind of game. Ground with nothing in it yet is watched on.
            if not ground:
                return None
            await _wait(LOOKS_APART_S)
            continue
        if seen is not None and not seen:
            seen.append(well)
        # In a well the piece is the one thing that moves. Where cells fill and empty in columns the piece is not over,
        # other things are about: a field with creatures in it, not a well. LIVE 2026-10-11 one was taken for a well.
        over = {piece.column + dc for dc, _dr, _kind in piece.parts} if piece is not None else set()
        over |= {last.column + dc for dc, _dr, _kind in last.parts} if last is not None else set()
        if filled is not None and {column for column, _row in set(cells) ^ filled} - over:
            elsewhere += 1
            if elsewhere >= MOVES_ELSEWHERE:
                return None
        filled, last = set(cells), piece
        if piece is not None:
            # The same thing lower down than it was, as much of it in sight, over the same column; or more of it in
            # sight at the top than there was, which is a piece coming in.
            if any(piece.column == before.column and (
                    (len(piece.parts) == len(before.parts) and piece.row > before.row + 0.2)
                    or (len(piece.parts) > len(before.parts) and not before.whole and before.row < 1.0))
                   for before in sightings):
                return well
            sightings.append(piece)
        await _wait(LOOKS_APART_S)
    return None
