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

from core.agency.how_the_pieces_move import HowThePiecesMove, moves_seen
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
from core.perception.pieces_by_their_look import Lattice, Looks, PiecesSeen, squares_of_one_ground

logger = logging.getLogger("Aura.PlayingPiecesByKind")

__all__ = ["Seat", "a_board_of_two_sides_at_its_start", "play_pieces_by_kind", "said_of_a_move"]

#: How long the board must look the same before it is believed: a piece being drawn across it is not a position.
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

    def rules_square(self, row: int, column: int) -> tuple[int, int]:
        """Rows counted from her back row, columns from her left: her side is the rules' first side."""
        return self.rows - 1 - row, column

    def seen_square(self, square: tuple[int, int]) -> tuple[int, int]:
        return self.rows - 1 - square[0], square[1]

    def as_rules(self, seen: PiecesSeen) -> dict[tuple[int, int], tuple[int, str]]:
        cells: dict[tuple[int, int], tuple[int, str]] = {}
        for row, line in enumerate(seen.cells):
            for column, cell in enumerate(line):
                if cell is not None:
                    side, look = cell
                    cells[self.rules_square(row, column)] = (FIRST if side == self.her_seen_side else SECOND, str(look))
        return cells

    def centre_share(self, square: tuple[int, int], picture: Any) -> tuple[float, float]:
        row, column = self.seen_square(square)
        x, y = self.lattice.centre(row, column)
        return x / picture.shape[1], y / picture.shape[0]


def a_board_of_two_sides_at_its_start(picture: Any, looks: Looks | None = None) -> Seat | None:
    """A board with two sides of pieces facing each other, each in the rows at its own end, nothing between: a game of
    pieces about to begin. Her seat is at the side nearest the bottom of the picture. ``looks`` are the looks she knows
    from an earlier sitting, so each kind keeps its name. None where the picture shows none."""
    lattice = squares_of_one_ground(picture)
    if lattice is None or lattice.rows < 4:
        return None
    looks = looks if looks is not None else Looks()
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
    words = f"{whose} {kind_named(letter).removeprefix('the ')} from {named(move.start)} to {named(move.end)}"
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


def _how_it_stands(found: Any, weights: Any) -> str:
    if getattr(found, "mate", False):
        return "I can force a win from here" if found.score > 0 else "They can force a win from here"
    worths = sorted(w for w in (weights.worth.values() if weights else []) if w > 0)
    unit = worths[0] if worths else 100.0
    ahead = found.score / unit
    if abs(ahead) < 0.3:
        return "It looks about even"
    return f"I judge myself {'ahead' if ahead > 0 else 'behind'} by about {abs(ahead):.1f} of my least piece"


async def _seen_still(seat: Seat, look: Look, *, within_s: float = 10.0) -> tuple[Any, PiecesSeen] | None:
    """The board once it has stood still: two pictures a moment apart that read the same."""
    last = ""
    began = time.monotonic()
    while time.monotonic() - began < within_s:
        picture = await look()
        if picture is None:
            return None
        seen = seat.looks.read(picture, seat.lattice)
        if seen is None:
            lattice = squares_of_one_ground(picture)
            if lattice is not None:
                seat.lattice = lattice
            await asyncio.sleep(0.3)
            continue
        said = seen.says()
        if said == last:
            return picture, seen
        last = said
        await asyncio.sleep(STILL_S)
    return None


def _marked(seat: Seat, before: Any, now: Any, picked: tuple[int, int]) -> set[tuple[int, int]]:
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
    await click(*seat.centre_share(square, picture))
    await asyncio.sleep(OFFERS_SHOW_S)
    if marked_picture is None or marked_picture.shape != picture.shape:
        return None
    return _marked(seat, picture, marked_picture, square)


@dataclass
class _Game:
    learned: HowThePiecesMove
    believed: Position
    earlier: list[Position] = field(default_factory=list)
    first_is_hers: bool = True
    made: int = 0
    theirs: int = 0
    #: Since when she has waited on the other side, and the longest it has taken to answer.
    waiting_since: float = field(default_factory=time.monotonic)
    longest_answer_s: float = 0.0

    def follow(self, move: Move) -> None:
        """A move seen made: learned from, and the position taken from the board, with what the board cannot show
        (which pieces have not moved, what may be taken in passing) carried from the move."""
        self.learned.watched(self.believed, move)
        family = self.learned.family()
        # The move itself, made on what she held: the board may already show the answer to it as well, which is
        # followed next, as its own move.
        by_rules = after(family, self.believed, move)
        self.earlier.append(self.believed)
        self.believed = by_rules


async def play_pieces_by_kind(seat: Seat, look: Look, click: Click, *, seconds: float,
                              say: Callable[[str], None] | None = None, thinks_s: float = THINKS_S,
                              learned: HowThePiecesMove | None = None, rehearsal: Any = None,
                              clock: Callable[[], Awaitable[float | None]] | None = None) -> dict[str, Any]:
    """Play the game at ``seat`` for ``seconds`` or until it ends, learning its rules as it goes, and what its pieces
    are worth by rehearsing it in her model while the other side thinks. ``clock`` reads the time she has left, where
    the game shows it. What came of it, with what she learned and the rehearsal, for her to keep."""
    from concurrent.futures import ThreadPoolExecutor

    from core.agency.searching_by_kind import search

    rehearsing = _Rehearsing(rehearsal)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rehearsal")
    try:
        came = await _play(seat, look, click, seconds=seconds, say=say, thinks_s=thinks_s, learned=learned,
                           rehearsing=rehearsing, pool=pool, clock=clock, search=search)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    came["rehearsal"] = rehearsing.rehearsal
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
                       tempo=earned.weights.tempo, royal_pressed=earned.weights.royal_pressed)

    def collect(self) -> None:
        if self.running is not None and self.running.done():
            try:
                self.rehearsal = self.running.result()
            except Exception as exc:  # noqa: BLE001 - a round that failed leaves what she had
                logger.info("a round of rehearsal did not finish (%s: %s)", type(exc).__name__, exc)
            self.running = None

    def start(self, pool: Any, family: Family) -> None:
        """A round in her model, where none is under way and the rules she holds can end a game."""
        from core.agency.what_a_kind_is_worth import a_first_rehearsal, rehearse

        self.collect()
        if self.running is not None or not any(kind.royal for kind in family.kinds.values()):
            return
        if self.rehearsal is None or set(self.rehearsal.weights.worth) != set(family.kinds):
            fresh = a_first_rehearsal(family)
            if self.rehearsal is not None:
                fresh.earned = []
            self.rehearsal = fresh
        self.rounds += 1
        self.running = pool.submit(rehearse, family, family.the_start(), self.rehearsal, seed=self.rounds)

    def moves_left(self, made: int) -> int:
        """Her moves still to come, by the games she has rehearsed: half their median length, less hers so far, and
        never less than half of that median again."""
        lengths = list(getattr(self.rehearsal, "lengths", []) or [])
        if not lengths:
            return 0
        typical = sorted(lengths)[len(lengths) // 2] / 2
        return int(max(typical / 2, typical - made, 1))


async def _play(seat: Seat, look: Look, click: Click, *, seconds: float, say: Callable[[str], None] | None,
                thinks_s: float, learned: HowThePiecesMove | None, rehearsing: _Rehearsing, pool: Any,
                clock: Callable[[], Awaitable[float | None]] | None, search: Any) -> dict[str, Any]:

    tell = say or (lambda _line: None)
    began = time.monotonic()
    # The start is the board as it stood when she sat down: the other side may have moved since.
    start = dict(seat.start)
    if not start:
        still = await _seen_still(seat, look)
        if still is None:
            return {"ended": "the board went out of sight", "moves": 0, "learned": learned}
        start = seat.as_rules(still[1])
    game = _Game(learned=learned or HowThePiecesMove.from_the_start(seat.rows, seat.columns, start),
                 believed=Position(cells=dict(start), turn=FIRST, unmoved=frozenset(start)))
    tell("Two sides facing each other on a board. I'll take the side nearest me, find out how the pieces move by "
         "picking them up and watching, and search my moves.")
    # Who goes first: the other side, if it moves while she waits.
    waited = time.monotonic()
    while time.monotonic() - waited < WAITS_FOR_THE_FIRST_MOVE_S:
        now = await _seen_still(seat, look, within_s=2.0)
        if now is not None and seat.as_rules(now[1]) != game.believed.cells and moves_seen(
                Position(cells=game.believed.cells, turn=SECOND, unmoved=game.believed.unmoved),
                seat.as_rules(now[1]), SECOND, game.learned.family()):
            game.believed = Position(cells=game.believed.cells, turn=SECOND, unmoved=game.believed.unmoved)
            game.first_is_hers = False
            break
        await asyncio.sleep(0.3)
    tell("They went first." if not game.first_is_hers else "They're waiting on me, so I go first.")
    misread = 0
    while time.monotonic() - began < seconds:
        still = await _seen_still(seat, look)
        if still is None:
            return {"ended": "the board went out of sight", "moves": game.made, "theirs": game.theirs,
                    "learned": game.learned}
        picture, seen = still
        shown = seat.as_rules(seen)
        if shown != game.believed.cells:
            game.learned.knows_of({kind for _side, kind in shown.values()})
            seen_moves = moves_seen(game.believed, shown, game.believed.turn, game.learned.family())
            if seen_moves is None:
                misread += 1
                if misread >= MISREAD_AT_MOST:
                    tell("I've lost track of the board; I'll stop here.")
                    return {"ended": "lost track of the board", "moves": game.made, "theirs": game.theirs,
                            "learned": game.learned}
                await asyncio.sleep(0.3)
                continue
            misread = 0
            for move in seen_moves:
                if game.believed.turn != FIRST:
                    family = game.learned.family()
                    tell("They played " + said_of_a_move(family, game.believed, move, game.learned,
                                                         game.first_is_hers, whose="their") + ".")
                    game.theirs += 1
                    game.longest_answer_s = max(game.longest_answer_s, time.monotonic() - game.waiting_since)
                game.follow(move)
                game.waiting_since = time.monotonic()
            continue
        if game.believed.turn != FIRST:
            # While the other side thinks, she rehearses the game in her model.
            rehearsing.start(pool, game.learned.family())
            # A side that has not answered for ten times as long as it ever took is not playing: a board that took her
            # move before the game was begun, or a game ended by something she cannot see. Given back, not waited on.
            if time.monotonic() - game.waiting_since > max(NO_ANSWER_AFTER_S, 10 * game.longest_answer_s):
                tell("They haven't answered my move, so I don't think the game has started; I'll look at the screen.")
                return {"ended": "not answered", "moves": game.made, "theirs": game.theirs, "learned": game.learned}
            await asyncio.sleep(0.2)
            continue
        rehearsing.collect()
        # Her turn: learn from a few pieces she has not seen enough of, then think.
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
            squares = await _offers_for(seat, look, click, picture, square)
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
        ended = outcome(family, game.believed, legal) if any(k.royal for k in family.kinds.values()) else ""
        if ended or not legal:
            return _the_end(ended or "no move", game, tell)
        weights = rehearsing.weights_for(family)
        thinks = await _time_for_a_move(clock, rehearsing, game.made, thinks_s)
        set_aside: list[Move] = []
        found = None
        for _ in range(CHECKS_A_TURN):
            found = search(family, game.believed, seconds=thinks, weights=weights, earlier=game.earlier,
                           set_aside=set_aside)
            if found.move is None:
                break
            # Before she moves it, she picks it up: the move is made only where the game offers it.
            if found.move.start not in offered:
                squares = await _offers_for(seat, look, click, picture, found.move.start)
                if squares is not None:
                    offered[found.move.start] = squares
                    game.learned.offered(game.believed, found.move.start, squares, family=family)
            if found.move.start not in offered or found.move.end in offered[found.move.start]:
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
        tell(f"I'll move {said_of_a_move(family, game.believed, move, game.learned, game.first_is_hers)}. "
             f"{_how_it_stands(found, weights)}. (looked {found.depth} moves deep, {found.nodes:,} positions)")
        made = False
        for attempt in range(TRIES_OF_A_MOVE):
            await click(*seat.centre_share(move.start, picture))
            await asyncio.sleep(OFFERS_SHOW_S * (1 + attempt))
            await click(*seat.centre_share(move.end, picture))
            made = await _comes_to_show(seat, look, game, move)
            if made:
                break
        if not made:
            game.learned.refused(game.believed, move, family=family)
            tell("The game didn't take that move; I've learned it isn't one.")
            continue
        game.made += 1
        game.waiting_since = time.monotonic()
    return {"ended": "out of time", "moves": game.made, "theirs": game.theirs, "learned": game.learned}


async def _time_for_a_move(clock: Callable[[], Awaitable[float | None]] | None, rehearsing: _Rehearsing, made: int,
                           otherwise: float) -> float:
    """How long this move is thought about: the time she has left over the moves she expects still to make, where the
    game shows her clock and her rehearsal says how long its games run; ``otherwise`` where it does not."""
    if clock is None:
        return otherwise
    try:
        left = await clock()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("her clock was not read (%s)", exc)
        return otherwise
    moves_left = rehearsing.moves_left(made)
    if left is None or moves_left <= 0:
        return otherwise
    return max(0.2, left / moves_left)


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
        still = await _seen_still(seat, look, within_s=MOVE_SHOWS_S)
        if still is None:
            return False
        shown = seat.as_rules(still[1])
        if shown != game.believed.cells:
            game.learned.knows_of({kind for _side, kind in shown.values()})
            seen_moves = moves_seen(game.believed, shown, FIRST, game.learned.family())
            if seen_moves and (seen_moves[0].start, seen_moves[0].end) == (move.start, move.end):
                return True
        await asyncio.sleep(0.15)
    return False


def _the_end(ended: str, game: _Game, tell: Callable[[str], None]) -> dict[str, Any]:
    if ended.startswith("won:"):
        result = "won" if int(ended.split(":", 1)[1]) == FIRST else "lost"
        tell("I won." if result == "won" else "I lost. They found the better line.")
    elif ended == "no move":
        family = game.learned.family()
        if not any(k.royal for k in family.kinds.values()):
            # The game stopped her: what of hers stood attacked is what it would not let her leave so. Kept as she
            # keeps a refusal, so the next game is played knowing it.
            game.learned.stopped_with_open(game.believed, family)
            family = game.learned.family()
        royal_attacked = any(k.royal for k in family.kinds.values()) and in_check(family, game.believed, FIRST)
        result = "lost" if royal_attacked else "over"
        tell("I have no move left, with my piece that must be kept safe attacked: I lost." if royal_attacked
             else "The game won't let me move any more, so it's over.")
    else:
        result = "drawn"
        tell(f"It's a draw ({ended.split(':', 1)[-1].strip()}).")
    return {"ended": result, "moves": game.made, "theirs": game.theirs, "learned": game.learned}
