"""Playing a game of pieces on a board she sees, rules learned as she goes: following it, moving, saying why.

A person sitting down at a board of pieces they have never seen does a few things, whatever the game. They see two sides
facing each other and take the one nearest them. They find out who goes first by waiting a moment. They pick pieces up
to see where the game lets them go, and watch where the other side's pieces go; that is how they learn the rules
(core/agency/how_the_pieces_move.py). On their turn they think as far ahead as the rules they hold let them, for the
time a move is worth, and make the move by picking the piece up and putting it down. They say what they are doing and
why. When the game will not let them move any more, it is over, and they say how it went.

Her eyes are pictures (core/perception/pieces_by_their_look.py reads squares, sides and looks); her hands are clicks at
shares of the picture. What she follows is what she sees: the position is built from the board, the move that made it
from the squares that changed, so rules she has not learned yet never put her out of step with the game. The rules she
holds are for looking ahead, and the game corrects them. Nothing here knows a game or an application.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.agency.how_the_pieces_move import HowThePiecesMove, a_move_her_rules_allow, moves_seen
from core.agency.pieces_by_their_kind import (
    FIRST,
    SECOND,
    Family,
    Move,
    Position,
    after,
    in_check,
    moves,
    outcome,
)
from core.agency.thinking_on_their_time import Pondering, WhatTheyAnsweredWhere
from core.cognition.how_far_to_go_before_looking import HowFarToGo
from core.cognition.how_long_a_change_takes_to_settle import HowLongItSettles
from core.cognition.keeping_her_own_count import KeptCount
from core.cognition.the_furthest_she_has_got import WhatFailedWhere
from core.perception.pieces_by_their_look import Lattice, Looks, PiecesSeen, squares_of_one_ground

logger = logging.getLogger("Aura.PlayingPiecesByKind")

__all__ = ["Seat", "a_board_of_two_sides_at_its_start", "play_pieces_by_kind", "said_of_a_move"]

#: How long the board must look the same before it is believed, until she has watched how long its changes take to
#: settle (core/cognition/how_long_a_change_takes_to_settle.py): a piece being drawn across it is not a position.
STILL_S = 0.5
#: How long she waits at the start for the other side to move, before taking it that she goes first.
WAITS_FOR_THE_FIRST_MOVE_S = 6.0
#: How long a move she made is waited on to show, before it is made again.
MOVE_SHOWS_S = 3.0
#: How many times a move is made before the game is taken not to accept it.
TRIES_OF_A_MOVE = 2
#: How long a move is thought about, when the place says nothing of its clock.
THINKS_S = 4.0
#: The moment between picking a piece up and looking at what the game offers for it.
OFFERS_SHOW_S = 0.2
#: The share of a square's pixels that must change for the game to have marked it.
MARKED_SHARE = 0.02
#: Pieces she picks up to learn from, at most, on one turn: the rest of the turn is for thinking.
PROBES_A_TURN = 6
#: Offers seen of a kind after which it is not picked up just to learn from.
SEEN_ENOUGH = 8
#: Moves she picks up and finds not offered, at most, before she moves the best one the game does offer.
CHECKS_A_TURN = 8
#: The least she waits on an answer before taking it that none is coming, whatever answers have taken so far.
NO_ANSWER_AFTER_S = 90.0
#: Pictures in a row that no move explains before she says she has lost track.
MISREAD_AT_MOST = 20

Look = Callable[[], Awaitable[Any]]
Click = Callable[[float, float], Awaitable[bool]]


@dataclass
class Seat:
    """Where she sits at a board: its squares, the looks of its pieces, and her side, the one nearest her."""

    lattice: Lattice
    looks: Looks
    #: The side seen nearest the bottom of the picture, which is hers.
    her_seen_side: int
    rows: int
    columns: int
    #: The board as it stood when she sat down, in the rules' frame.
    start: dict[tuple[int, int], tuple[int, str]] = field(default_factory=dict)
    #: How long a change on this board takes to settle, as she has watched its changes settle, and what the board read
    #: when she last saw it stand still: a change between two looks at it is a change watched.
    settles: HowLongItSettles = field(default_factory=lambda: HowLongItSettles(STILL_S))
    last_still: str = ""

    def rules_square(self, row: int, column: int) -> tuple[int, int]:
        """Rows counted from her back row, columns from her left: her side is the rules' first side."""
        return self.rows - 1 - row, column

    def seen_square(self, square: tuple[int, int]) -> tuple[int, int]:
        return self.rows - 1 - square[0], square[1]

    def as_rules(self, seen: PiecesSeen) -> dict[tuple[int, int], tuple[int, str | None]]:
        """The board read, in the rules' frame: each piece's side and kind, the kind None where its look was unread."""
        cells: dict[tuple[int, int], tuple[int, str | None]] = {}
        for row, line in enumerate(seen.cells):
            for column, cell in enumerate(line):
                if cell is not None:
                    side, look = cell
                    cells[self.rules_square(row, column)] = (FIRST if side == self.her_seen_side else SECOND,
                                                             None if look is None else str(look))
        return cells

    def filled(self, seen: PiecesSeen, believed: dict) -> dict[tuple[int, int], tuple[int, str]]:
        """The board read with every unread kind given: the kind she holds on that square where her side there agrees,
        else the kind the look is most like."""
        cells = self.as_rules(seen)
        for square, (side, kind) in list(cells.items()):
            if kind is None:
                held = believed.get(square)
                cells[square] = (side, held[1] if held is not None and held[0] == side
                                 else str(seen.nearest.get(self.seen_square(square), 0)))
        return cells

    def centre_share(self, square: tuple[int, int], picture: Any) -> tuple[float, float]:
        row, column = self.seen_square(square)
        x, y = self.lattice.centre(row, column)
        return x / picture.shape[1], y / picture.shape[0]


def _ruled_by_its_period(picture: Any) -> Lattice | None:
    """The board found by how its picture repeats, where its squares are not flat ground to be found square by square
    (core/perception/a_board_seen.py)."""
    from core.perception.a_board_seen import the_lattice

    ruled = the_lattice(np.asarray(picture))
    if ruled is None:
        return None
    left, top, width, height, columns, rows = ruled
    return Lattice(left=left, top=top, width=width, height=height, columns=columns, rows=rows)


def a_board_of_two_sides_at_its_start(picture: Any, looks: Looks | None = None) -> Seat | None:
    """A board with two sides of pieces facing each other, each in the rows at its own end, nothing between: a game of
    pieces about to begin. Her seat is at the side nearest the bottom of the picture. ``looks`` are the looks she knows
    from an earlier sitting, so each kind keeps its name. None where the picture shows none."""
    lattice = squares_of_one_ground(picture) or _ruled_by_its_period(picture)
    if lattice is None or lattice.rows < 4:
        return None
    looks = looks if looks is not None else Looks()
    seen = looks.read(picture, lattice)
    if seen is None:
        return None
    if all(any(cell is not None for cell in line) for line in seen.cells):
        # Every square reads as something: the board's squares are drawn with more than flat ground. At a game's start
        # the rows between the two sides are empty, so what its middle rows show is what an empty square looks like
        # here. LIVE 11 October 2026 a web page's board drawn in two patterned pinks read as pieces on all 64 squares.
        looks.the_empty_squares_are(np.asarray(picture), lattice, (lattice.rows // 2 - 1, lattice.rows // 2))
        seen = looks.read(picture, lattice)
        if seen is None:
            return None
    rows_of: dict[int, list[int]] = {}
    for row, line in enumerate(seen.cells):
        for cell in line:
            if cell is not None:
                rows_of.setdefault(cell[0], []).append(row)
    if len(rows_of) != 2 or min(len(v) for v in rows_of.values()) < 2:
        return None
    (one, one_rows), (other, other_rows) = sorted(rows_of.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    # Facing each other from their ends: every piece of the top side above every piece of the bottom side, with at
    # least one empty row between.
    if max(one_rows) + 1 >= min(other_rows):
        return None
    if min(one_rows) > lattice.rows // 3 or max(other_rows) < lattice.rows - 1 - lattice.rows // 3:
        return None
    seat = Seat(lattice=lattice, looks=looks, her_seen_side=other, rows=lattice.rows, columns=lattice.columns)
    seat.start = seat.as_rules(seen)
    return seat


def _square_named(square: tuple[int, int], first_is_hers: bool, rows: int, columns: int) -> str:
    """A square as players of a board name it: its column lettered from the left of the side that went first, its row
    numbered from that side's back row."""
    row, column = square
    if not first_is_hers:
        row, column = rows - 1 - row, columns - 1 - column
    return f"{'abcdefghijklmnopqrstuvwxyz'[column]}{row + 1}"


def _owned(what: str, whose: str) -> str:
    """A piece's description as one side's: "the piece that leaps" becomes "my piece that leaps", and "a piece I
    haven't seen move" becomes "a piece of mine I haven't seen move", since "my a piece" is no English."""
    if what.startswith("the "):
        return f"{whose} {what.removeprefix('the ')}"
    if what.startswith("a "):
        noun, _, rest = what.removeprefix("a ").partition(" ")
        return f"a {noun} of {'mine' if whose == 'my' else 'theirs'} {rest}".rstrip()
    return f"{whose} {what}"


def said_of_a_move(family: Family, position: Position, move: Move, learned: HowThePiecesMove, first_is_hers: bool,
                   *, whose: str = "my") -> str:
    """A move in words: what moved, from where to where, what it took, what it became, whether it now attacks the
    royal piece. A kind is named by what she has seen it do."""
    def named(square: tuple[int, int]) -> str:
        return _square_named(square, first_is_hers, family.rows, family.columns)

    def kind_named(letter: str) -> str:
        return family.kinds[letter].name if letter in family.kinds and family.kinds[letter].name else \
            learned.what_it_does(letter)

    _side, letter = position.cells[move.start]
    words = f"{_owned(kind_named(letter), whose)} from {named(move.start)} to {named(move.end)}"
    if move.partner is not None:
        words += f", crossing with {kind_named(position.cells[move.partner[0]][1])}"
    if move.taken_at is not None and move.taken_at in position.cells:
        words += f", taking {kind_named(position.cells[move.taken_at][1])}"
    if move.becomes:
        words += f", which becomes {kind_named(move.becomes)}"
    if any(k.royal for k in family.kinds.values()):
        made = after(family, position, move)
        if in_check(family, made, made.turn):
            words += ", attacking the piece that must be kept safe"
    return words


def _how_it_stands(found: Any, weights: Any, start: dict | None = None) -> str:
    """How the position stands, in what her commonest piece is worth: the kind she has most of at the start. A kind
    rehearsal finds worth next to nothing made "of my least piece" a unit of nothing: LIVE 21 November 2026, "I judge
    myself behind by about 48.1 of my least piece"."""
    if getattr(found, "mate", False):
        return "I can force a win from here" if found.score > 0 else "They can force a win from here"
    worth = dict(weights.worth) if weights else {}
    mine = [kind for side, kind in (start or {}).values() if side == FIRST]
    commonest = max(set(mine), key=mine.count) if mine else ""
    unit = worth.get(commonest, 0.0)
    noun = "my commonest piece"
    if unit <= 0:
        positive = sorted(w for w in worth.values() if w > 0)
        unit, noun = (positive[len(positive) // 2], "a middling piece of mine") if positive else (100.0, "a piece")
    ahead = found.score / unit
    if abs(ahead) < 0.3:
        return "It looks about even"
    return f"I judge myself {'ahead' if ahead > 0 else 'behind'} by about {abs(ahead):.1f} of {noun}"


async def _seen_still(seat: Seat, look: Look, *, within_s: float = 10.0) -> tuple[Any, PiecesSeen] | None:
    """The board once it has stood still: two pictures that read the same, the second as long after the first as
    changes on this board have kept going. A change watched to its end is learned from. LIVE 21 November 2026, game
    16, seeing each of their moves stand still took 0.6 s of her clock, every turn."""
    last = ""
    began = time.monotonic()
    first_change: float | None = None
    last_change = 0.0
    while time.monotonic() - began < within_s:
        picture = await look()
        if picture is None:
            return None
        seen = seat.looks.read(picture, seat.lattice, settled=True)
        if seen is None:
            lattice = squares_of_one_ground(picture)
            if lattice is not None:
                seat.lattice = lattice
            await asyncio.sleep(0.3)
            continue
        said = seen.says()
        if said == last:
            if first_change is not None:
                seat.settles.watched(last_change - first_change)
            seat.last_still = said
            return picture, seen
        # A reading that differs from the one before it, or from the board as it last stood still, is a change seen.
        # LIVE 22 November 2026 her wait never learned: each of their moves came between her looks, and only a
        # change seen within one wait was counted.
        before = last or seat.last_still
        if before and said != before:
            now = time.monotonic()
            first_change = now if first_change is None else first_change
            last_change = now
        last = said
        await asyncio.sleep(seat.settles.wait())
    return None


def _marked(seat: Seat, before: Any, now: Any, picked: tuple[int, int] | None) -> set[tuple[int, int]]:
    """The squares the game marked between two pictures: where enough of a square's pixels changed. The square picked
    up is left out; it is marked as picked, not offered."""
    marked: set[tuple[int, int]] = set()
    lattice = seat.lattice
    for row in range(lattice.rows):
        for column in range(lattice.columns):
            square = seat.rules_square(row, column)
            if square == picked:
                continue
            a = lattice.square(before, row, column)
            b = lattice.square(now, row, column)
            if a.shape != b.shape or a.size == 0:
                continue
            changed = np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16)).max(axis=2) > 30
            if changed.mean() >= MARKED_SHARE:
                marked.add(square)
    return marked


async def _offers_for(seat: Seat, look: Look, click: Click, picture: Any, square: tuple[int, int]) -> set | None:
    """Pick the piece on ``square`` up, see which squares the game marks, and put it down again."""
    await click(*seat.centre_share(square, picture))
    await asyncio.sleep(OFFERS_SHOW_S)
    marked_picture = await look()
    if marked_picture is None or marked_picture.shape != picture.shape:
        await _put_down(seat, look, click, picture, square, set())
        return None
    marked = _marked(seat, picture, marked_picture, square)
    await _put_down(seat, look, click, picture, square, marked)
    return marked


async def _put_down(seat: Seat, look: Look, click: Click, picture: Any, square: tuple[int, int],
                    marked: set[tuple[int, int]]) -> bool:
    """Put a piece picked up down again, and see the board as it was before. A second click on it puts it down in some
    applications and leaves it picked up in others, where a click may begin a drag; there, a click on an empty square
    the game did not mark puts it down. Left picked up, its marks are read as the next piece's offers."""
    await click(*seat.centre_share(square, picture))
    await asyncio.sleep(OFFERS_SHOW_S)
    if await _as_it_was(seat, look, picture):
        return True
    seen = seat.looks.read(picture, seat.lattice, settled=True)
    occupied = set(seat.as_rules(seen)) if seen is not None else set()
    empty = [sq for sq in (seat.rules_square(r, c) for r in range(seat.rows) for c in range(seat.columns))
             if sq not in occupied and sq not in marked and sq != square]
    if not empty:
        return False
    # The empty square furthest from the piece: the least likely to be anything the piece could be meant to do.
    away = max(empty, key=lambda sq: abs(sq[0] - square[0]) + abs(sq[1] - square[1]))
    await click(*seat.centre_share(away, picture))
    await asyncio.sleep(OFFERS_SHOW_S)
    put = await _as_it_was(seat, look, picture)
    if not put:
        logger.info("a piece picked up on %s would not be put down again", square)
    return put


async def _as_it_was(seat: Seat, look: Look, picture: Any) -> bool:
    now = await look()
    return now is not None and now.shape == picture.shape and not _marked(seat, picture, now, None)


@dataclass
class _Game:
    learned: HowThePiecesMove
    believed: Position
    earlier: list[Position] = field(default_factory=list)
    #: The last move followed, whose piece may show on the square it came to what the move made of it.
    last_followed: Move | None = None
    first_is_hers: bool = True
    made: int = 0
    theirs: int = 0
    #: Since when she has waited on the other side, and the longest it has taken to answer.
    waiting_since: float = field(default_factory=time.monotonic)
    longest_answer_s: float = 0.0
    #: What her hands cost on her clock this game, measured: picking a piece up to see its offers, and making a move
    #: and seeing it shown. Each is the mean of the times taken, with how many there were.
    probe_s: float = 0.0
    probes_timed: int = 0
    finish_s: float = 0.0
    finishes_timed: int = 0
    #: Every move seen made, hers and theirs, as the board's players write one down: "g1-f3", "e4xd5".
    played: list[str] = field(default_factory=list)
    #: When she last saw a move of theirs made: from then her clock runs, though her turn has not begun.
    their_move_seen_at: float = 0.0
    #: The part of her clock the last turn she could read it took: what a turn whose clock she cannot read is given.
    last_share: float = 0.0
    #: Her clock as her own count keeps it, as it shows while it stands (core/cognition/keeping_her_own_count.py), and
    #: whether she has read it already while this wait for their answer lasts.
    count: KeptCount = field(default_factory=KeptCount)
    read_this_wait: bool = False
    #: How long a reading of her clock has taken, at most, this game.
    read_s: float = 0.0
    #: How many of her moves her rules have earned between looks at what the game offers before she moves
    #: (core/cognition/how_far_to_go_before_looking.py), and how many she has made since the last look.
    far: HowFarToGo = field(default_factory=HowFarToGo)
    unlooked: int = 0
    #: What she has played at a position before and seen go wrong, kept with the board (WhatFailedWhere); and this
    #: game's turns she searched: where she stood, what she played, and how she judged it.
    went_wrong: WhatFailedWhere = field(default_factory=WhatFailedWhere)
    #: What the other side answered in each position she has met, and her look on their time at what she expects next.
    they_answered: WhatTheyAnsweredWhere = field(default_factory=WhatTheyAnsweredWhere)
    pondering: Pondering = field(default_factory=Pondering)
    judged: list[tuple[str, str, int]] = field(default_factory=list)
    #: The reply her search expects to her last move, the position it leads to, and the rules it was found by; whether
    #: the reply came is counted, foreseen of foreseeable.
    expected: tuple[Move, Position, Family] | None = None
    foreseen: int = 0
    foreseeable: int = 0
    #: How long each of their answers took.
    answers_s: list[float] = field(default_factory=list)

    def timed_probe(self, seconds: float) -> None:
        self.probes_timed += 1
        self.probe_s += (seconds - self.probe_s) / self.probes_timed

    def timed_finish(self, seconds: float) -> None:
        self.finishes_timed += 1
        self.finish_s += (seconds - self.finish_s) / self.finishes_timed

    def follow(self, move: Move) -> None:
        """A move seen made: learned from, and the position taken from the board, with what the board cannot show
        (which pieces have not moved, what may be taken in passing) carried from the move."""
        allowed = a_move_her_rules_allow(self.learned.family(), self.believed, move)
        self.learned.watched(self.believed, move, allowed=allowed)
        rows, columns = self.learned.rows, self.learned.columns
        self.played.append(_square_named(move.start, self.first_is_hers, rows, columns)
                           + ("x" if move.taken_at is not None else "-")
                           + _square_named(move.end, self.first_is_hers, rows, columns))
        family = self.learned.family()
        # The move itself, made on what she held: the board may already show the answer to it as well, which is
        # followed next, as its own move.
        by_rules = after(family, self.believed, move)
        self.earlier.append(self.believed)
        self.believed = by_rules
        self.last_followed = move

    def became_on_landing(self, shown: dict) -> None:
        """The piece of the last move followed, shown on the square it came to as another kind than she holds: the
        move made it that kind there. LIVE-like 22 November 2026 her man crowned on the far row was held a man, its look
        unread, because her board already had it there."""
        move = self.last_followed
        if move is None:
            return
        held, seen = self.believed.cells.get(move.end), shown.get(move.end)
        if held is None or seen is None or seen[0] != held[0] or seen[1] in (None, held[1]):
            return
        self.learned.watched_becoming(held[1], seen[1])
        self.believed = Position(cells={**self.believed.cells, move.end: (held[0], seen[1])}, turn=self.believed.turn,
                                 unmoved=self.believed.unmoved, passable=self.believed.passable)


async def play_pieces_by_kind(seat: Seat, look: Look, click: Click, *, seconds: float,
                              say: Callable[[str], None] | None = None, thinks_s: float = THINKS_S,
                              learned: HowThePiecesMove | None = None, rehearsal: Any = None,
                              clock: Callable[[], Awaitable[float | None]] | None = None,
                              borrow: Callable[[dict, bool], Any] | None = None,
                              went_wrong: WhatFailedWhere | None = None,
                              they_answered: WhatTheyAnsweredWhere | None = None) -> dict[str, Any]:
    """Play the game at ``seat`` for ``seconds`` or until it ends, learning its rules as it goes, and what its pieces
    are worth by rehearsing it in her model while the other side thinks. ``clock`` reads the time she has left, where
    the game shows it. Where she has learned nothing of this board, ``borrow`` is asked, once she knows who moves first,
    for what she learned at a board that begins as this one does: its rules, its rehearsal and where she learned them,
    or None. What came of it, with what she learned and the rehearsal, for her to keep."""
    from concurrent.futures import ThreadPoolExecutor

    from core.agency.practising_between_games import at_a_board
    from core.agency.searching_by_kind import search

    rehearsing = _Rehearsing(rehearsal)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rehearsal")
    # Her look on their time has a thread of its own: a round of rehearsal under way does not hold it up.
    ponder_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pondering")
    the_moves: list[str] = []
    went_wrong = went_wrong if went_wrong is not None else WhatFailedWhere()
    they_answered = they_answered if they_answered is not None else WhatTheyAnsweredWhere()
    try:
        with at_a_board():
            came = await _play(seat, look, click, seconds=seconds, say=say, thinks_s=thinks_s, learned=learned,
                               rehearsing=rehearsing, pool=pool, clock=clock, search=search, borrow=borrow,
                               the_moves=the_moves, went_wrong=went_wrong, ponder_pool=ponder_pool,
                               they_answered=they_answered)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        ponder_pool.shutdown(wait=False, cancel_futures=True)
    came["rehearsal"] = rehearsing.rehearsal
    # The game as it went, for its record: every move in order, and who moved first.
    came["the_moves"] = list(the_moves)
    came["first_was_hers"] = getattr(came.get("learned"), "first_was_hers", None)
    came["went_wrong"] = went_wrong
    came["they_answered"] = they_answered
    return came


@dataclass
class _Rehearsing:
    """Her rehearsal of the game in her model: the round under way, if any, and what it has earned."""

    rehearsal: Any = None
    running: Any = None
    rounds: int = 0

    def weights_for(self, family: Family) -> Any:
        """The weights she plays by: what rehearsal earned, for the kinds it knows, and the first guess for the rest."""
        from core.agency.searching_by_kind import Weights, first_weights

        guess = first_weights(family)
        earned = self.rehearsal
        if earned is None or not earned.earned:
            return guess
        return Weights(worth={k: earned.weights.worth.get(k, guess.worth[k]) for k in family.kinds},
                       mobility={k: earned.weights.mobility.get(k, 0.0) for k in family.kinds},
                       squares={k: v for k, v in earned.weights.squares.items() if k in family.kinds},
                       tempo=earned.weights.tempo, royal_pressed=earned.weights.royal_pressed,
                       exposed={k: v for k, v in earned.weights.exposed.items() if k in family.kinds},
                       learned=getattr(earned.weights, "learned", None),
                       beyond=list(getattr(earned.weights, "beyond", None) or []))

    def collect(self) -> None:
        if self.running is not None and self.running.done():
            try:
                self.rehearsal = self.running.result()
            except Exception as exc:  # noqa: BLE001 - a round that failed leaves what she had
                logger.info("a round of rehearsal did not finish (%s: %s)", type(exc).__name__, exc)
            self.running = None

    def start(self, pool: Any, family: Family) -> None:
        """A round in her model, where none is under way and the rules she holds move something. A game in her model
        that cannot end is judged by how it stood when it stopped (core/agency/what_a_kind_is_worth.py), so she
        rehearses from her first turn rather than once she has learned which piece must be kept safe."""
        from core.agency.what_a_kind_is_worth import a_first_rehearsal, rehearse

        self.collect()
        if self.running is not None or not any(kind.steps for kind in family.kinds.values()):
            return
        from core.agency.what_a_kind_is_worth import royal_kinds

        # Weights belong to the game they were fitted in: the kinds there are, and which of them the game keeps safe.
        # A rehearsal kept before that was recorded is of a game she cannot name, and starts again.
        if (self.rehearsal is None or set(self.rehearsal.weights.worth) != set(family.kinds)
                or getattr(self.rehearsal, "royal", None) != royal_kinds(family)):
            fresh = a_first_rehearsal(family)
            if self.rehearsal is not None:
                fresh.earned = []
                # How long its games run is about the game, not about the kinds she reads in it; the positions her
                # longer look searched are kept by each kind's name, and a kind read since does not unsettle them.
                fresh.lengths = list(self.rehearsal.lengths)
                fresh.looked_at_longer = self.rehearsal.looked_at_longer
            self.rehearsal = fresh
        self.rounds += 1
        self.running = pool.submit(rehearse, family, family.the_start(), self.rehearsal, seed=self.rounds)

    def moves_left(self, made: int) -> int:
        """Her moves still to come, by the games she has rehearsed: half their median length, less hers so far, never
        less than half of that median again, and never fewer than she has made already: a game that has gone so many
        moves is as likely as not to go as many more."""
        lengths = list(getattr(self.rehearsal, "lengths", []) or [])
        if not lengths:
            return 0
        typical = sorted(lengths)[len(lengths) // 2] / 2
        return int(max(typical / 2, typical - made, made, 1))


async def _play(seat: Seat, look: Look, click: Click, *, seconds: float, say: Callable[[str], None] | None,
                thinks_s: float, learned: HowThePiecesMove | None, rehearsing: _Rehearsing, pool: Any,
                clock: Callable[[], Awaitable[float | None]] | None, search: Any,
                borrow: Callable[[dict, bool], Any] | None = None,
                the_moves: list[str] | None = None, went_wrong: WhatFailedWhere | None = None,
                ponder_pool: Any = None, they_answered: WhatTheyAnsweredWhere | None = None) -> dict[str, Any]:
    from core.agency.searching_by_kind import Found
    from core.cognition.when_the_situation_decides import nothing_to_decide

    tell = say or (lambda _line: None)
    began = time.monotonic()
    # The start is the board as it stood when she sat down: the other side may have moved since.
    start = dict(seat.start)
    if not start:
        still = await _seen_still(seat, look)
        if still is None:
            return {"ended": "the board went out of sight", "moves": 0, "learned": learned}
        start = seat.filled(still[1], {})
    game = _Game(learned=learned or HowThePiecesMove.from_the_start(seat.rows, seat.columns, start),
                 believed=Position(cells=dict(start), turn=FIRST, unmoved=frozenset(start)),
                 played=the_moves if the_moves is not None else [],
                 went_wrong=went_wrong if went_wrong is not None else WhatFailedWhere(),
                 they_answered=they_answered if they_answered is not None else WhatTheyAnsweredWhere())
    tell("Two sides facing each other on a board. I'll take the side nearest me, find out how the pieces move by "
         "picking them up and watching, and search my moves.")
    # Who goes first: the other side, if it moves while she waits.
    waited = time.monotonic()
    while time.monotonic() - waited < WAITS_FOR_THE_FIRST_MOVE_S:
        now = await _seen_still(seat, look, within_s=2.0)
        if now is not None and not _agrees(seat.as_rules(now[1]), game.believed.cells) and moves_seen(
                Position(cells=game.believed.cells, turn=SECOND, unmoved=game.believed.unmoved),
                seat.as_rules(now[1]), SECOND, game.learned.family()):
            game.believed = Position(cells=game.believed.cells, turn=SECOND, unmoved=game.believed.unmoved)
            game.first_is_hers = False
            break
        await asyncio.sleep(0.3)
    tell("They went first." if not game.first_is_hers else "They're waiting on me, so I go first.")
    if learned is None and borrow is not None:
        # Which way round she sees the board is which side she sits at, so what a board like it lends waits for that.
        lent = await asyncio.to_thread(borrow, start, game.first_is_hers)
        if lent is not None:
            game.learned, lent_rehearsal, where = lent
            game.learned.start = dict(start)
            if lent_rehearsal is not None:
                rehearsing.rehearsal = lent_rehearsal
            tell(f"This board begins as the one in {where} does, so I'll start from what I learned there.")
    game.learned.first_was_hers = game.first_is_hers
    misread = 0
    unexplained: dict | None = None
    while time.monotonic() - began < seconds:
        still = await _seen_still(seat, look)
        if still is None:
            return {"ended": "the board went out of sight", "moves": game.made, "theirs": game.theirs,
                    "learned": game.learned}
        picture, seen = still
        others = _only_the_others(game.learned)
        shown = _read_against(seat.as_rules(seen), game.believed.cells, others)
        landed = game.last_followed.end if game.last_followed is not None else None
        shown = _what_arrived_changed(seat, picture, shown, game.believed.cells, others, landed)
        game.became_on_landing(shown)
        if not _agrees(shown, game.believed.cells):
            game.learned.knows_of({kind for _side, kind in shown.values() if kind is not None})
            seen_moves = moves_seen(game.believed, shown, game.believed.turn, game.learned.family())
            if seen_moves is None:
                # A look she cannot read on a square she held empty is a mark as likely as a piece (a glow round her
                # piece under attack spilling onto the next square): where the board is explained without it, it was
                # a mark. A piece that came there by a move leaves the square it came from empty, so it is never
                # explained away. LIVE 22 November 2026, game 19, a look of nothing read on h1 beside her king in check
                # left her board unexplained.
                unmarked = _without_marks_where_nothing_stood(shown, game.believed.cells)
                if unmarked != shown:
                    seen_moves = ([] if _agrees(unmarked, game.believed.cells) else
                                  moves_seen(game.believed, unmarked, game.believed.turn, game.learned.family()))
                    if seen_moves is not None:
                        shown = unmarked
            if seen_moves is None:
                misread += 1
                settled = unexplained is not None and shown == unexplained
                unexplained = shown
                if misread == 1:
                    logger.info("the board shows what no move she can read explains: %s", _what_differs(
                        game.believed.cells, shown, seat))
                # Read the same twice, each reading itself of a board standing still, it has settled: looking again
                # will not make it explain itself, and her clock is running. LIVE 22 November 2026, game 19, twenty
                # looks took 24 s of a turn whose share was 0.6 s.
                if misread >= MISREAD_AT_MOST or settled:
                    # The board reads, and no move she can read explains it: she goes by the board as it stands, as a
                    # person who has lost the thread of the moves looks at the position, rather than stopping with her
                    # clock running. LIVE 20 November 2026 she stopped 22 seconds after her tenth move against
                    # Stockfish, and her clock ran out seven seconds later on a board she could read square by square.
                    turn = await _whose_turn_by_the_clock(clock)
                    tell("I've lost the thread of the moves, so I'll go by the board as it stands.")
                    held = game.believed
                    cells = seat.filled(seen, held.cells)
                    # A look read as the kind it is most like may be one she has never seen move.
                    game.learned.knows_of({kind for _side, kind in cells.values()})
                    game.earlier.append(held)
                    game.believed = Position(cells=cells, turn=turn, unmoved=frozenset(
                        sq for sq in held.unmoved if cells.get(sq) == held.cells.get(sq)))
                    game.waiting_since = time.monotonic()
                    misread, unexplained = 0, None
                    continue
                await asyncio.sleep(0.3)
                continue
            if misread:
                # What no move explained, the next look did: the board was believed still while it was still changing.
                seat.settles.too_soon()
            misread, unexplained = 0, None
            for move in seen_moves:
                if game.believed.turn != FIRST:
                    family = game.learned.family()
                    foreseen = _was_expected(game, move)
                    tell("They played " + said_of_a_move(family, game.believed, move, game.learned,
                                                         game.first_is_hers, whose="their")
                         + (", as I expected." if foreseen else "."))
                    game.theirs += 1
                    game.answers_s.append(time.monotonic() - game.waiting_since)
                    game.longest_answer_s = max(game.longest_answer_s, time.monotonic() - game.waiting_since)
                    game.their_move_seen_at = time.monotonic()
                    game.read_this_wait = False
                    game.they_answered.they_answered(_a_place(game.believed), _named(move))
                game.follow(move)
                game.waiting_since = time.monotonic()
            if seen_moves:
                continue
            # Nothing moved: what differed was a mark. The board is as she holds it, and the turn goes on from there.
        if game.believed.turn != FIRST:
            # While the other side thinks, she rehearses the game in her model, and looks at what she expects to face.
            rehearsing.start(pool, game.learned.family())
            _think_on_their_time(game, rehearsing, ponder_pool, thinks_s)
            if clock is not None and not game.read_this_wait:
                # Her clock is read, where her count of it may have drifted, while it stands still on their time.
                game.read_this_wait = True
                await _read_her_clock_while_it_stands(clock, rehearsing, game)
            # A side that has not answered for ten times as long as it ever took is not playing: a board that took her
            # move before the game was begun, or a game ended by something she cannot see. Given back, not waited on.
            if time.monotonic() - game.waiting_since > max(NO_ANSWER_AFTER_S, 10 * game.longest_answer_s):
                tell("They haven't answered my move, so I don't think the game has started; I'll look at the screen.")
                return {"ended": "not answered", "moves": game.made, "theirs": game.theirs, "learned": game.learned}
            await asyncio.sleep(0.2)
            continue
        rehearsing.collect()
        game.pondering.settle()
        # Her turn, all of it on her clock where the game shows one: picking pieces up, thinking and moving share the
        # turn's part of the time she has left. LIVE 20 November 2026 only the thinking was held to it, and picking up
        # six pieces a turn and seeing each move shown took a third of a three-minute game.
        turn_began = time.monotonic()
        # Her clock has run since their move was seen, or since her turn began where nothing of theirs came first.
        clock_began = game.their_move_seen_at if game.theirs else turn_began
        share, on_her_clock = await _time_for_a_move(clock, rehearsing, game.made, thinks_s, game, clock_began)
        if on_her_clock:
            game.last_share = share
        elif clock is not None and game.last_share:
            # Her clock went on running while it could not be read. LIVE 21 November 2026, game 10, four late turns
            # read no clock and were given four seconds of thought where the last reading left her one or two.
            share, on_her_clock = game.last_share, True
        # Learn from a few pieces she has not seen enough of, then think.
        offered: dict[tuple[int, int], set] = {}
        probes = 0
        # Pieces with room around them first: one boxed in by its own side shows nothing.
        mine = [sq for sq, (side, _kind) in game.believed.cells.items() if side == FIRST]
        for square in sorted(mine, key=lambda sq: -_room_around(game.believed, sq)):
            kind = game.believed.cells[square][1]
            if probes >= PROBES_A_TURN:
                break
            if _times_offered(game.learned, kind) >= SEEN_ENOUGH and game.made:
                continue
            if on_her_clock and not _room_to_learn(game, time.monotonic() - turn_began, share, probes):
                break
            probe_began = time.monotonic()
            squares = await _offers_for(seat, look, click, picture, square)
            game.timed_probe(time.monotonic() - probe_began)
            probes += 1
            if squares is not None:
                offered[square] = squares
                game.learned.offered(game.believed, square, squares, family=game.learned.family())
        if not game.made and not game.theirs and probes and not any(offered.values()):
            # Nothing she picked up was offered anywhere: the board is not taking moves yet.
            tell("The board isn't letting me move anything yet.")
            return {"ended": "not taking moves", "moves": 0, "theirs": 0, "learned": game.learned}
        family = game.learned.family(unseen_go_every_way=True)
        legal = moves(family, game.believed)
        ended = (outcome(family, game.believed, legal)
                 if any(k.royal or k.wins_arriving for k in family.kinds.values()) else "")
        if ended or not legal:
            # An ending her model infers is asked of the game before it is believed: where the game still offers her a
            # move, the game is not over and her reading of the board is what is wrong. LIVE 21 November 2026 a mark on
            # her king turned it, as she read it, into a piece of the other side; she said "I lost" with the game going
            # on, and closed the application.
            offers_left = await _what_the_game_still_offers(seat, look, click, picture, game.believed)
            if not offers_left:
                return _the_end(ended or "no move", game, tell)
            logger.info("her model says the game is over (%s) and the game still offers %d move(s)",
                        ended or "no move", sum(len(v) for v in offers_left.values()))
            for start, ends in offers_left.items():
                game.learned.offered(game.believed, start, ends, family=family)
            family = game.learned.family(unseen_go_every_way=True)
            if not moves(family, game.believed):
                start, ends = next(iter(offers_left.items()))
                end = sorted(ends)[0]
                taken = end if game.believed.cells.get(end, (FIRST,))[0] != FIRST else None
                move = Move(start, end, taken_at=taken)
                tell(f"I'll move {said_of_a_move(family, game.believed, move, game.learned, game.first_is_hers)}, "
                     "the move the game offers me.")
                await _by_hand(seat, click, picture, move, OFFERS_SHOW_S)
                if await _comes_to_show(seat, look, game, move):
                    game.made += 1
                    game.waiting_since = time.monotonic()
                continue
        weights = rehearsing.weights_for(family)
        set_aside: list[Move] = []
        here = _a_place(game.believed)
        failed_here = game.went_wrong.already_failed_at(here)
        if failed_here:
            again = [one for one in legal if _named(one) in failed_here]
            if again and len(again) < len(legal):
                set_aside = list(again)
                tell(f"I've played {said_of_a_move(family, game.believed, again[0], game.learned, game.first_is_hers)}"
                     " from here before and it went wrong, so I'll look at something else.")
        found = None
        learned_until, searched_s = time.monotonic(), 0.0
        for _ in range(CHECKS_A_TURN):
            # What is left of the turn, less what making the move and seeing it shown has taken her this game.
            thinks = (max(0.2, share - (time.monotonic() - turn_began) - game.finish_s) if on_her_clock
                      else share)
            # A move that is the only one is made without weighing (core/cognition/when_the_situation_decides.py).
            only = nothing_to_decide([m for m in legal if m not in set_aside]) if not set_aside else None
            if only is not None:
                found = Found(only, 0, 0, 0, [only], by="the only move")
            else:
                search_began = time.monotonic()
                found = search(family, game.believed, seconds=thinks, weights=weights, earlier=game.earlier,
                               set_aside=set_aside, threads=_cores_free())
                searched_s += time.monotonic() - search_began
            if found.move is None:
                break
            # Before she moves it, she picks it up: the move is made only where the game offers it. Each look that
            # finds the game offering just what her rules said lets her go one move further without it; the first
            # that does not, or a move the game refuses, has her looking before every move again. LIVE 21 November
            # 2026, game 10: the look took 47 s of a 232 s game, where her thinking had 60.
            if found.move.start not in offered:
                pressed = any(k.royal for k in family.kinds.values()) and in_check(family, game.believed, FIRST)
                unseen = not game.learned.knows_how_it_goes(game.believed.cells[found.move.start][1])
                if not _looks_first(game, pressed=pressed, unseen=unseen):
                    break
                squares = await _offers_for(seat, look, click, picture, found.move.start)
                if squares is not None:
                    offered[found.move.start] = squares
                    game.learned.offered(game.believed, found.move.start, squares, family=family)
                    said = {one.end for one in legal if one.start == found.move.start}
                    if set(squares) == said:
                        game.far.it_was_where_she_said(1)
                    else:
                        game.far.it_was_not(1)
            if found.move.start not in offered or _landing(found.move) in offered[found.move.start]:
                break
            game.learned.refused(game.believed, found.move, family=family)
            set_aside.append(found.move)
            family = game.learned.family(unseen_go_every_way=True)
            weights = rehearsing.weights_for(family)
        else:
            # Every move she looked at was refused: she picks up every piece of hers. Where the game offers none of
            # them anything, it has stopped her: the game is over.
            for square in mine:
                if square not in offered:
                    squares = await _offers_for(seat, look, click, picture, square)
                    if squares is not None:
                        offered[square] = squares
            if offered and not any(offered.values()):
                return _the_end("no move", game, tell)
        move = found.move if found is not None else None
        if move is None:
            return _the_end("no move", game, tell)
        if found.by == "the only move":
            how = "It's the only move I have"
        else:
            _judged(game, here, move, found)
            how = f"{_how_it_stands(found, weights, game.learned.start)}. (looked {found.depth} moves deep, " \
                  f"{found.nodes:,} positions)"
        tell(f"I'll move {said_of_a_move(family, game.believed, move, game.learned, game.first_is_hers)}. {how}")
        made = False
        finish_began = time.monotonic()
        for attempt in range(TRIES_OF_A_MOVE):
            if attempt and move.start not in offered:
                # Made without a look and not shown: a look says whether the game offers it at all, before her clock
                # pays for making it blind again. LIVE 21 November 2026, game 12, a crossing tried while her piece that
                # must be kept safe was attacked was made twice and waited on twice, 14 s of a 4 s turn.
                squares = await _offers_for(seat, look, click, picture, move.start)
                if squares is not None:
                    offered[move.start] = squares
                    game.learned.offered(game.believed, move.start, squares, family=family)
                    if _landing(move) not in squares:
                        break
            await _by_hand(seat, click, picture, move, OFFERS_SHOW_S * (1 + attempt))
            moved_at = time.monotonic()
            made = await _comes_to_show(seat, look, game, move)
            if made:
                break
        if made:
            game.timed_finish(time.monotonic() - finish_began)
            # Her clock stopped when her move was made: what it ran this turn is counted off.
            game.count.step(-(moved_at - clock_began))
            _expect_back(game, family, move, found)
            # Where her turn went, for knowing what her clock pays for besides thinking.
            seen = game.their_move_seen_at or turn_began
            logger.info("her turn %d took %.1f s: %.1f s before it began, %.1f s learning, %.1f s thinking, %.1f s "
                        "between, %.1f s moving; her share of the clock was %s", game.made + 1,
                        time.monotonic() - seen, turn_began - seen, learned_until - turn_began, searched_s,
                        finish_began - learned_until - searched_s, time.monotonic() - finish_began,
                        f"{share:.1f} s" if on_her_clock else "not on a clock")
        if not made:
            game.learned.refused(game.believed, move, family=family)
            if game.unlooked:
                game.far.it_was_not(game.unlooked)
                game.unlooked = 0
            tell("The game didn't take that move; I've learned it isn't one.")
            continue
        game.made += 1
        game.waiting_since = time.monotonic()
    return {"ended": "out of time", "moves": game.made, "theirs": game.theirs, "learned": game.learned,
            "foresaw": (game.foreseen, game.foreseeable)}


def _expect_back(game: _Game, family: Family, move: Move, found: Any) -> None:
    """What she expects back from the move she has made: what they answered before in the position it leaves, where
    she has met it, else the reply her search found best for them; and where it leaves her. Nothing when she has not
    met the position and her search did not look past her own move."""
    game.expected = None
    mine = after(family, game.believed, move)
    named = game.they_answered.most_often(_a_place(mine))
    before = next((m for m in moves(family, mine) if _named(m) == named), None) if named else None
    if before is not None:
        game.expected = (before, after(family, mine, before), family)
        return
    line = list(getattr(found, "line", None) or [])
    if getattr(found, "by", "") == "the only move" or len(line) < 2 or line[0] != move:
        return
    game.expected = (line[1], after(family, mine, line[1]), family)


def _think_on_their_time(game: _Game, rehearsing: _Rehearsing, pool: Any, thinks_s: float) -> None:
    """While they think, a look at the position she expects to face after their answer
    (core/agency/thinking_on_their_time.py), with the weights and rules her own turn will search with, so that what it
    finds is in her search's table when her turn comes."""
    if pool is None or game.pondering.busy() or game.expected is None:
        return
    expected, facing, _family = game.expected
    family = game.learned.family(unseen_go_every_way=True)
    if facing.turn != FIRST or not moves(family, facing):
        return
    game.pondering.start(pool, family, facing, seconds=game.pondering.look_for(game.answers_s, game.last_share or thinks_s),
                         weights=rehearsing.weights_for(family), earlier=[*game.earlier, game.believed],
                         threads=max(1, _cores_free() - 1))


def _was_expected(game: _Game, move: Move) -> bool:
    """Whether their move is the one she expected, counted."""
    expected, game.expected = game.expected, None
    if expected is None:
        return False
    game.foreseeable += 1
    if _named(expected[0]) == _named(move):
        game.foreseen += 1
        return True
    return False


def _cores_free() -> int:
    """How many of the machine's cores nothing else is using now, to think with: its cores less its load over the last
    minute, and one at the least. LIVE 21 November 2026 her search ran on one of eighteen; in a test world four times
    the time to think took her from 0.396 to 0.708 against Stockfish held to depth 4 (24 games each)."""
    import os

    cores = os.cpu_count() or 1
    try:
        busy = os.getloadavg()[0]
    except OSError:
        return 1
    return max(1, int(cores - busy))


def _room_to_learn(game: _Game, spent: float, share: float, probes: int) -> bool:
    """Whether her turn has room to pick up one more piece to learn from: learning and thinking split the turn evenly,
    since nothing says which of the two this move needs more. The first piece of a turn too, once she knows what picking
    one up costs: LIVE 21 November 2026, game 11, late turns with 2.4 s of clock spent 1.6 s of it on one piece."""
    if not probes and not game.probes_timed:
        return True
    return spent + game.probe_s <= share / 2


def _looks_first(game: _Game, *, pressed: bool = False, unseen: bool = False) -> bool:
    """Whether she picks the piece up to see what the game offers before she moves it: on every move until her rules
    have earned going without (core/cognition/how_far_to_go_before_looking.py), and then on every move they have not.
    Always while her piece that must be kept safe is attacked (``pressed``): that is where a game narrows what it
    offers most, and where her rules have seen it least. Always for a kind she has never seen go (``unseen``): what
    she searched it with is a guess that it goes every way, and what her rules earned elsewhere says nothing of it."""
    if not pressed and not unseen and game.unlooked + 1 < game.far.how_many(trusted=game.far.how_often_it_held):
        game.unlooked += 1
        return False
    game.unlooked = 0
    return True


async def _time_for_a_move(clock: Callable[[], Awaitable[float | None]] | None, rehearsing: _Rehearsing, made: int,
                           otherwise: float, game: _Game | None = None,
                           clock_began: float | None = None) -> tuple[float, bool]:
    """The part of her clock this turn may take, and whether it came from her clock: the time she has left over the
    moves she expects still to make, where the game shows her clock and her rehearsal says how long its games run;
    ``otherwise``, to think for, where it does not.

    The time she has left is her own count of her clock, where she keeps one: the clock is read on her turn only where
    she has no count yet, or her count may be off by more than a turn's share and their answers came too fast to read
    it on their time. LIVE 21 November 2026, game 16, reading it took half a second of every turn, and by her
    thirtieth move her turns had a second and a half to think in."""
    if clock is None:
        return otherwise, False
    moves_left = rehearsing.moves_left(made)
    if moves_left <= 0:
        return otherwise, False
    began = time.monotonic() if clock_began is None else clock_began
    if game is not None and game.count.value is not None and not game.count.must_read(game.count.value / moves_left):
        return max(0.2, (game.count.value - (time.monotonic() - began)) / moves_left), True
    try:
        left = await clock()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("her clock was not read (%s)", exc)
        return otherwise, False
    if left is None:
        return otherwise, False
    if game is not None:
        # Her count begins from it, as the clock showed when it began to run this turn.
        game.count.read(left + (time.monotonic() - began))
    return max(0.2, left / moves_left), True


async def _read_her_clock_while_it_stands(clock: Callable[[], Awaitable[float | None]], rehearsing: _Rehearsing,
                                          game: _Game) -> None:
    """Her clock read on their time, where her count of it may by now be off by more than a turn's share of it: a
    clock that stands still reads exactly, and the reading costs her nothing of it. Only where the reading will be
    done before their answer comes, as their answers have come: an answer that comes while she reads is seen late, on
    her clock."""
    count = game.count
    moves_left = max(1, rehearsing.moves_left(game.made))
    if count.value is not None and not count.must_read(count.value / moves_left):
        return
    waited = time.monotonic() - game.waiting_since
    if game.answers_s and waited + game.read_s > min(game.answers_s):
        return
    began = time.monotonic()
    try:
        left = await clock()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("her clock was not read (%s)", exc)
        return
    finally:
        game.read_s = max(game.read_s, time.monotonic() - began)
    if left is not None:
        if count.value is not None:
            logger.info("her clock read %.1f s where her count said %.1f s", left, count.value)
        count.read(left)


#: The coarsest a game's clock shows its time: a whole second.
A_CLOCK_SHOWS_S = 1.0


async def _whose_turn_by_the_clock(clock: Callable[[], Awaitable[float | None]] | None) -> int:
    """Whose turn it is, where the game shows her clock: hers where it runs. Hers where there is no clock to read, since
    a turn that is not hers is found out the moment the game offers her nothing."""
    if clock is None:
        return FIRST
    try:
        one = await clock()
        # A second apart, and a reading's own time on top: a running clock has shown a change by then.
        await asyncio.sleep(A_CLOCK_SHOWS_S)
        two = await clock()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("her clock was not read (%s)", exc)
        return FIRST
    if one is None or two is None:
        return FIRST
    return FIRST if two < one else SECOND


def _what_arrived_changed(seat: Seat, picture: Any, shown: dict, believed: dict,
                          others: dict[int, set[str]] | None = None, landed: tuple[int, int] | None = None) -> dict:
    """A piece that has just come to its far row with a look like no kind she knows is a kind of its own, the look
    kept as one: a piece the game changed there. LIVE-like 22 November 2026, a man crowned on the far row was read as
    the man it had been, and every move it made after was learned as a man's, backwards and taking backwards.
    ``others`` holds, for each side, the kinds only the other side has held, which a piece of that side is not;
    ``landed`` is where the last move she followed came to, whose piece has just arrived though her board has it."""
    out = dict(shown)
    for square, (side, kind) in shown.items():
        far = seat.rows - 1 if side == FIRST else 0
        if kind is None and square[0] == far and (believed.get(square, (None,))[0] != side or square == landed):
            row, column = seat.seen_square(square)
            among = None
            if others is not None:
                among = {i for i in range(len(seat.looks.kinds)) if str(i) not in others.get(side, set())}
            learned = seat.looks.learn_at(picture, seat.lattice, row, column, among)
            if learned is not None:
                out[square] = (side, str(learned))
    return out


def _only_the_others(learned: HowThePiecesMove) -> dict[int, set[str]]:
    """For each side, the kinds only the other side has held. In a game whose two sides look different, a look of
    the other side's on a piece is a misreading of it."""
    first, second = learned.kinds_of(FIRST), learned.kinds_of(SECOND)
    return {FIRST: second - first, SECOND: first - second}


def _without_marks_where_nothing_stood(shown: dict, believed: dict) -> dict:
    """The board read without the looks she could not read on squares she held empty."""
    return {square: held for square, held in shown.items() if held[1] is not None or square in believed}


def _read_against(shown: dict, believed: dict, others: dict[int, set[str]] | None = None) -> dict:
    """The board read, where a piece's look was unread taking its side from the board she holds: a mark that changes
    how a piece looks (a glow round a piece under attack) changes its colour as well, and a piece does not change side
    where it stands. A piece read as a kind only the other side has held (``others``) is read as unread: LIVE-like
    22 November 2026 a dark man crowned on its far row was read as a light man, and the backward moves it made after
    taught her own men to step back."""
    out = dict(shown)
    for square, (side, kind) in shown.items():
        if others is not None and kind is not None and kind in others.get(side, set()):
            out[square] = (side, None)
            kind = None
        if kind is None and square in believed:
            out[square] = (believed[square][0], None)
    return out


async def _what_the_game_still_offers(seat: Seat, look: Look, click: Click, picture: Any,
                                      position: Position) -> dict[tuple[int, int], set]:
    """Her pieces picked up in turn, those with room around them first, until the game offers one of them anything:
    one is enough to know the game goes on, and her clock is running."""
    offered: dict[tuple[int, int], set] = {}
    mine = [square for square, (side, _kind) in position.cells.items() if side == FIRST]
    for square in sorted(mine, key=lambda sq: -_room_around(position, sq)):
        squares = await _offers_for(seat, look, click, picture, square)
        if squares:
            offered[square] = squares
            break
    return offered


def _agrees(shown: dict, believed: dict) -> bool:
    """Whether the board seen is the board held: the same squares, each with the same side, and the same kind where
    the kind was read."""
    return set(shown) == set(believed) and all(
        shown[sq][0] == believed[sq][0] and shown[sq][1] in (None, believed[sq][1]) for sq in shown)


def _what_differs(believed: dict, shown: dict, seat: Seat) -> str:
    """The squares where the board she holds and the board she sees part, named as the board names them."""
    differ = sorted(sq for sq in set(believed) | set(shown) if believed.get(sq) != shown.get(sq))
    return "; ".join(f"{_square_named(sq, True, seat.rows, seat.columns)} held {believed.get(sq)} seen {shown.get(sq)}"
                     for sq in differ)


def _room_around(position: Position, square: tuple[int, int]) -> int:
    """Empty squares within two of a square."""
    r, c = square
    return sum(1 for dr in range(-2, 3) for dc in range(-2, 3)
               if (dr or dc) and (r + dr, c + dc) not in position.cells)


def _times_offered(learned: HowThePiecesMove, kind: str) -> int:
    ways = learned.seen.get(kind, {})
    return sum(w.moved_to_empty + w.took for w in ways.values())


async def _comes_to_show(seat: Seat, look: Look, game: _Game, move: Move) -> bool:
    """Whether the board comes to show her move made, alone or answered already. It is followed by whoever reads the
    board next, as moves seen."""
    ends_at = time.monotonic() + MOVE_SHOWS_S
    while time.monotonic() < ends_at:
        # One look that shows her move's squares as her move leaves them is enough; a board still moving cannot show
        # that. Waiting for it to read the same twice first took 2.1 s a move, a third of game 11's clock.
        picture = await look()
        seen = None if picture is None else seat.looks.read(picture, seat.lattice, settled=True)
        if seen is not None and _shows_her_move(_read_against(seat.as_rules(seen), game.believed.cells,
                                                              _only_the_others(game.learned)), move):
            return True
        still = await _seen_still(seat, look, within_s=MOVE_SHOWS_S)
        if still is None:
            return False
        shown = _read_against(seat.as_rules(still[1]), game.believed.cells, _only_the_others(game.learned))
        if not _agrees(shown, game.believed.cells):
            game.learned.knows_of({kind for _side, kind in shown.values() if kind is not None})
            seen_moves = moves_seen(game.believed, shown, FIRST, game.learned.family())
            if seen_moves and (seen_moves[0].start, seen_moves[0].end) == (move.start, move.end):
                return True
            # The squares of her move as her move leaves them: made, whatever else the board shows that she cannot
            # read yet. The rest is followed as the board is followed. Taken for a refusal, a move the game made
            # would teach her a rule that is false.
            if _shows_her_move(shown, move):
                return True
        await asyncio.sleep(0.15)
    return False


def _landing(move: Move) -> tuple[int, int]:
    """The first square a move lands on, which is what a game offers for it: for a take over again and again, the
    first landing, not the end."""
    return move.via[0] if move.via else move.end


async def _by_hand(seat: Seat, click: Click, picture: Any, move: Move, wait_s: float) -> None:
    """A move made as a person makes it: the piece picked up, then each square it lands on in turn."""
    await click(*seat.centre_share(move.start, picture))
    for square in (*move.via, move.end):
        await asyncio.sleep(wait_s)
        await click(*seat.centre_share(square, picture))


def _shows_her_move(shown: dict, move: Move) -> bool:
    """Whether the board shows her move's squares as her move leaves them: the square it left not hers, the square it
    went to hers."""
    return shown.get(move.start, (None,))[0] != FIRST and shown.get(move.end, (None,))[0] == FIRST


def _a_place(position: Position) -> str:
    """A position as a place she may come back to: what stands on each square, and whose turn it is."""
    return ";".join(f"{r},{c}:{side}{kind}" for (r, c), (side, kind) in sorted(position.cells.items())) + \
        f"|{position.turn}"


def _named(move: Move) -> str:
    return f"{move.start[0]},{move.start[1]}-{move.end[0]},{move.end[1]}{move.becomes}"


def _judged(game: _Game, place: str, move: Move, found: Any) -> None:
    """A turn she searched, kept for finding where a lost game went wrong: only a judgement not yet decided. A fall
    into a mate found is the end being seen, not the answer she least saw coming: LIVE 21 November 2026, game 15, it
    ruled out the move before the mate, at a place no game comes back to, where her judgement had fallen from even to
    4.5 pieces behind at move 9."""
    if not getattr(found, "mate", False):
        game.judged.append((place, _named(move), int(found.score)))


def _where_it_went_wrong(game: _Game) -> None:
    """In a game she lost, the move after which her own judgement fell furthest by her next turn: the answer to it she
    least saw coming. It is ruled out at the place she made it (WhatFailedWhere), so a game that comes back to that
    place goes another way. Nothing where her judgement never fell."""
    drops = [(game.judged[i][2] - game.judged[i + 1][2], i) for i in range(len(game.judged) - 1)]
    if not drops:
        return
    drop, at = max(drops)
    if drop <= 0:
        return
    place, move, _score = game.judged[at]
    game.went_wrong.failed_at(place, move)
    logger.info("where it went wrong: her move %s, after which she judged herself %d lower by her next turn", move,
                drop)


def _the_end(ended: str, game: _Game, tell: Callable[[str], None]) -> dict[str, Any]:
    if ended.startswith("won:"):
        result = "won" if int(ended.split(":", 1)[1]) == FIRST else "lost"
        tell("I won." if result == "won" else "I lost. They found the better line.")
        if result == "lost":
            _where_it_went_wrong(game)
    elif ended == "no move":
        family = game.learned.family()
        if not any(k.royal for k in family.kinds.values()):
            # The game stopped her: what of hers stood attacked is what it would not let her leave so. Kept as she
            # keeps a refusal, so the next game is played knowing it.
            game.learned.stopped_with_open(game.believed, family)
            family = game.learned.family()
        royal_attacked = any(k.royal for k in family.kinds.values()) and in_check(family, game.believed, FIRST)
        result = "lost" if royal_attacked else "over"
        if royal_attacked:
            _where_it_went_wrong(game)
        tell("I have no move left, with my piece that must be kept safe attacked: I lost." if royal_attacked
             else "The game won't let me move any more, so it's over.")
    else:
        result = "drawn"
        tell(f"It's a draw ({ended.split(':', 1)[-1].strip()}).")
    if game.foreseeable:
        logger.info("she foresaw %d of their %d answers", game.foreseen, game.foreseeable)
    return {"ended": result, "moves": game.made, "theirs": game.theirs, "learned": game.learned,
            "foresaw": (game.foreseen, game.foreseeable), "final": game.believed, "last_move": game.last_followed}
