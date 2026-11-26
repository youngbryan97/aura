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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.agency.how_the_pieces_move import (
    SEEN_TO_BELIEVE,
    HowThePiecesMove,
    a_move_her_rules_allow,
    moves_seen,
)
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
from core.agency.what_a_small_ending_comes_to import worked_to_the_end
from core.agency.what_she_did import ACCEPTED, WhatSheDid
from core.cognition.how_far_her_judgement_holds import HowFarItHolds
from core.cognition.how_far_to_go_before_looking import HowFarToGo
from core.cognition.how_long_a_change_takes_to_settle import HowLongItSettles
from core.cognition.keeping_her_own_count import KeptCount
from core.cognition.the_furthest_she_has_got import WhatFailedWhere
from core.perception.pieces_by_their_look import Lattice, Looks, PiecesSeen, squares_of_one_ground
from core.runtime.wall_clock import monotonic_time

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
#: How far past its share a turn may think where its answer keeps changing: SPENDS_AT_MOST shares, and never more than
#: LEFT_AT_MOST of the time left on her clock. The shape of an anytime search's time (a soft and a hard limit) as
#: engines that manage a clock use it; the numbers are habits to be measured.
SPENDS_AT_MOST = 3.0
LEFT_AT_MOST = 0.25
#: The least she waits on an answer before taking it that none is coming, whatever answers have taken so far.
NO_ANSWER_AFTER_S = 90.0
#: How long one wait for the board to stand still runs before it says the board did not.
SETTLES_WITHIN_S = 10.0
#: Pictures in a row that no move explains before she says she has lost track.
MISREAD_AT_MOST = 20

Look = Callable[[], Awaitable[Any]]
Click = Callable[[float, float], Awaitable[bool]]
EndReading = Callable[[], Awaitable[tuple[str, str]]]


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


def _sided_by_where_they_began(looks: Looks, picture: Any, lattice: Lattice, seen: PiecesSeen) -> PiecesSeen:
    """At a game's start each side holds the rows at its own end: a piece among one side's rows whose colour reads as
    the other side's is that side's, and its look is held to be that side's from then on. LIVE 11 October 2026 a figure
    standing on a piece of the top side was drawn in colours nearer the bottom side's."""
    half = lattice.rows // 2
    # A look held to be one side's that stands at both ends of a board at its start is a kind both sides have: what
    # held it to one side was a misreading, and its colour is believed again. LIVE 25 November 2026 her kept looks held
    # the shapes of two kinds to the lighter side, the darker side's pieces of those kinds read as the lighter side's,
    # and a new game's board was not seen as a game at its start, six runs running.
    top_kinds = {cell[1] for line in seen.cells[:half] for cell in line if cell is not None and cell[1] is not None}
    bottom_kinds = {cell[1] for line in seen.cells[half:] for cell in line if cell is not None and cell[1] is not None}
    mistaken = [kind for kind in looks.side_of_kind if kind in top_kinds and kind in bottom_kinds]
    if mistaken:
        for kind in mistaken:
            del looks.side_of_kind[kind]
        logger.info("looks held to one side but standing at both ends of a board at its start are both sides': %s",
                    sorted(mistaken))
        seen = looks.read(picture, lattice)
    ends = {"top": [cell[0] for line in seen.cells[:half] for cell in line if cell is not None],
            "bottom": [cell[0] for line in seen.cells[half:] for cell in line if cell is not None]}
    if not ends["top"] or not ends["bottom"]:
        return seen
    top = max(set(ends["top"]), key=ends["top"].count)
    bottom = max(set(ends["bottom"]), key=ends["bottom"].count)
    if top == bottom:
        return seen
    changed = False
    for row, line in enumerate(seen.cells):
        belongs = top if row < half else bottom
        for cell in line:
            if cell is not None and cell[0] != belongs and cell[1] is not None:
                # A look seen on both ends is a kind both sides have, whose colour is to be believed.
                elsewhere = [other for r, others in enumerate(seen.cells) for other in others
                             if other is not None and other[1] == cell[1] and (r < half) != (row < half)]
                if not elsewhere:
                    looks.side_of_kind[cell[1]] = belongs
                    changed = True
    return looks.read(picture, lattice) if changed else seen


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
    seen = _sided_by_where_they_began(looks, picture, lattice, seen)
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
    unit, noun = _her_unit(worth, start)
    ahead = found.score / unit
    if abs(ahead) < 0.3:
        return "It looks about even"
    return f"I judge myself {'ahead' if ahead > 0 else 'behind'} by about {abs(ahead):.1f} of {noun}"


def _her_unit(worth: dict, start: dict | None) -> tuple[float, str]:
    """What she counts worth in, and what she calls it: her commonest piece at the start, or a middling one."""
    mine = [kind for side, kind in (start or {}).values() if side == FIRST]
    commonest = max(set(mine), key=mine.count) if mine else ""
    unit = worth.get(commonest, 0.0)
    if unit > 0:
        return unit, "my commonest piece"
    positive = sorted(w for w in worth.values() if w > 0)
    return (positive[len(positive) // 2], "a middling piece of mine") if positive else (100.0, "a piece")


def _the_areas(game: _Game, family: Family, weights: Any) -> Any:
    """The board read area by area (core/agency/what_each_area_of_a_board_comes_to.py): each fight alone, two moves
    deep, played as one game, and her moves that keep everything but weaken her told apart by her whole judgement. None
    where it cannot be read."""
    from core.agency.what_each_area_of_a_board_comes_to import read_board

    try:
        return read_board(family, game.believed, dict(weights.worth) if weights else {}, plies=2, weights=weights)
    # not a failure: a board her rules cannot read area by area is played by her search alone.
    except (KeyError, ValueError, TypeError) as why:
        logger.info("the board was not read area by area: %s", why)
        return None


def _expected_with_the_areas(game: _Game, areas: Any) -> dict | None:
    """What she expects of them at their turns, for her search to look at first: the moves they have made, as shares of
    all they have made, beside the moves that would cost her most in the areas of the board, as shares of what all
    those would cost. Each counts half where both are known."""
    theirs = sum(game.they_went.values())
    made = {went: n / theirs for went, n in game.they_went.items()} if theirs else {}
    hurts = [threat for threat in (areas.threats if areas is not None else ()) if threat.costs_her > 0]
    cost = sum(threat.costs_her for threat in hurts)
    feared: dict = {}
    for threat in hurts:
        key = (threat.act.start, threat.act.end)
        feared[key] = feared.get(key, 0.0) + threat.costs_her / cost
    if not made and not feared:
        return None
    if not made or not feared:
        return made or feared
    return {key: (made.get(key, 0.0) + feared.get(key, 0.0)) / 2 for key in set(made) | set(feared)}


def _the_areas_said(game: _Game, family: Family, areas: Any, weights: Any) -> str:
    """What the areas say, in a sentence: the fight that most needs a move, and the worst they could do to her."""
    if areas is None:
        return ""
    unit, noun = _her_unit(dict(weights.worth) if weights else {}, game.learned.start)
    said = []
    hottest = max(areas.readings, key=lambda reading: reading.urgency, default=None)
    if hottest is not None and hottest.urgency * 2 / unit >= 0.5:
        mine = [sq for sq in hottest.part if game.believed.cells.get(sq, (None,))[0] == FIRST]
        if mine:
            biggest = max(mine, key=lambda sq: weights.worth.get(game.believed.cells[sq][1], 0.0) if weights else 0.0)
            what = family.kinds[game.believed.cells[biggest][1]].name or game.learned.what_it_does(
                game.believed.cells[biggest][1])
            said.append(f"The fight around {_owned(what, 'my')} on "
                        f"{_square_named(biggest, game.first_is_hers, family.rows, family.columns)} is the one that most "
                        f"needs a move: whoever moves there first gains about {hottest.urgency * 2 / unit:.1f} of {noun}")
    worst = next(iter(areas.threats), None)
    if worst is not None and worst.costs_her / unit >= 0.5:
        said.append("left alone, the worst they could do is "
                    + said_of_a_move(family, game.believed.__class__(cells=game.believed.cells, turn=1 - FIRST,
                                                                     unmoved=game.believed.unmoved,
                                                                     passable=game.believed.passable),
                                     worst.act, game.learned, game.first_is_hers, whose="their")
                    + f", costing me about {worst.costs_her / unit:.1f}")
    if game.their_areas:
        going = _where_they_are_going(game)
        if going:
            said.append(going)
    if not said:
        return ""
    sentence = "; ".join(said)
    return sentence[0].upper() + sentence[1:] + "."


def _what_it_came_at(game: _Game, family: Family, move: Move) -> str:
    """The piece of hers their move came at, in words: the one she holds worth most in the area of the board the moved
    piece went to (core/agency/what_each_area_of_a_board_comes_to.py); "" where none of hers is there."""
    from core.agency.what_each_area_of_a_board_comes_to import area_of, areas_of

    try:
        made = after(family, game.believed, move)
        area = area_of(areas_of(family, made), move.end) or frozenset()
    # not a failure: a move her rules cannot follow comes at nothing she can name.
    except (KeyError, ValueError, TypeError):
        return ""
    worth = game.weights_seen or {}
    mine = [sq for sq in area if made.cells.get(sq, (None,))[0] == FIRST]
    if not mine:
        return ""
    biggest = max(mine, key=lambda sq: worth.get(made.cells[sq][1], 0.0))
    letter = made.cells[biggest][1]
    what = family.kinds[letter].name if letter in family.kinds and family.kinds[letter].name else \
        game.learned.what_it_does(letter)
    return f"{_owned(what, 'my')} on {_square_named(biggest, game.first_is_hers, family.rows, family.columns)}"


def _where_they_are_going(game: _Game) -> str:
    """Where their recent moves keep going, as the piece of hers they keep coming at: said once they have come at the
    same one in most of their last moves."""
    from core.cognition.what_each_area_comes_to import where_they_work

    recent = game.their_areas[-3:]
    going = where_they_work(recent, (), ())
    if len(recent) >= 3 and going and going[0][1] > 0.5:
        return f"they keep coming at {going[0][0]}"
    return ""


async def _seen_still(seat: Seat, look: Look, *, within_s: float | None = None) -> tuple[Any, PiecesSeen] | None:
    """The board once it has stood still: two pictures that read the same, the second as long after the first as
    changes on this board have kept going. A change watched to its end is learned from. LIVE 21 November 2026, game
    16, seeing each of their moves stand still took 0.6 s of her clock, every turn. None where no picture can be
    taken, or the board did not stand still within ``within_s`` (SETTLES_WITHIN_S): what kept it from standing still
    is said, so that a board that would not settle can be told from one that went."""
    within_s = SETTLES_WITHIN_S if within_s is None else within_s
    # Standing still is two readings alike: a wait that cannot hold two looks can never see it.
    within_s = max(within_s, 2 * seat.settles.wait() + 0.3)
    last = ""
    began = monotonic_time()
    first_change: float | None = None
    last_change = 0.0
    unread, looks = 0, 0
    before_cells: list | None = None
    changing: dict[tuple[int, int], int] = {}
    while monotonic_time() - began < within_s:
        picture = await look()
        if picture is None:
            return None
        looks += 1
        seen = seat.looks.read(picture, seat.lattice, settled=True)
        if seen is None:
            unread += 1
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
        if before_cells is not None:
            for r, row in enumerate(seen.cells):
                for c, cell in enumerate(row):
                    if r < len(before_cells) and c < len(before_cells[r]) and before_cells[r][c] != cell:
                        changing[(r, c)] = changing.get((r, c), 0) + 1
        before_cells = seen.cells
        # A reading that differs from the one before it, or from the board as it last stood still, is a change seen.
        # LIVE 22 November 2026 her wait never learned: each of their moves came between her looks, and only a
        # change seen within one wait was counted.
        before = last or seat.last_still
        if before and said != before:
            now = monotonic_time()
            first_change = now if first_change is None else first_change
            last_change = now
        last = said
        await asyncio.sleep(seat.settles.wait())
    most = sorted(changing.items(), key=lambda item: -item[1])[:8]
    logger.info("the board did not stand still in %.0f s: %d look(s), %d not read as a board, squares that kept "
                "changing (row, column: times) %s", within_s, looks, unread, most)
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


#: How many more looks an answer is read again for, at most, before a display still changing is taken to say nothing.
READS_TO_SETTLE = 3


async def _offers_for(seat: Seat, look: Look, click: Click, picture: Any, square: tuple[int, int]) -> set | None:
    """Pick the piece on ``square`` up, see which squares the game marks, and put it down again: None where the board
    gave no answer that can be read.

    An answer is read from the board as it was, and once it holds still. LIVE 23 November 2026, game 31: of 55 answers
    read, 31 squares were ones the game never offered, most of them the marks of the piece looked at before (her king
    on f1 "offered" a2, c2, b3 and a4, the squares of the piece she had just put down). A piece not put down, or a
    display not yet changed, was read as the next piece's answer, and her rules learned ways no piece has."""
    if not await _as_it_was(seat, look, picture):
        # Another piece's marks are still showing: put it down on a square the game marks nothing on (a marked square
        # would be a move), and read nothing until the board is as it was.
        now = await look()
        showing = _marked(seat, picture, now, None) if now is not None and now.shape == picture.shape else set()
        if not showing or not await _put_down(seat, look, click, picture, square, showing, picked=False):
            return None
    if await click(*seat.centre_share(square, picture)) is False:
        return None
    await asyncio.sleep(OFFERS_SHOW_S)
    marked_picture = await look()
    if marked_picture is None or marked_picture.shape != picture.shape:
        await _put_down(seat, look, click, picture, square, set())
        return None
    marked = _marked(seat, picture, marked_picture, square)
    for _ in range(READS_TO_SETTLE):
        # Read again: the same answer twice is the game's, a changing one is still arriving.
        await asyncio.sleep(OFFERS_SHOW_S / 2)
        again = await look()
        if again is None or again.shape != picture.shape:
            continue
        settled = _marked(seat, picture, again, square)
        if settled == marked:
            break
        marked = settled
    else:
        await _put_down(seat, look, click, picture, square, marked)
        logger.info("the answer for the piece on %s did not hold still; read as nothing", square)
        return None
    return marked if await _put_down(seat, look, click, picture, square, marked) else None


async def _put_down(seat: Seat, look: Look, click: Click, picture: Any, square: tuple[int, int],
                    marked: set[tuple[int, int]], *, picked: bool = True) -> bool:
    """Put a piece picked up down again, and see the board as it was before. A second click on it puts it down in some
    applications and leaves it picked up in others, where a click may begin a drag; there, a click on an empty square
    the game did not mark puts it down. Left picked up, its marks are read as the next piece's offers. With ``picked``
    False, the piece held is not the one on ``square``, and only the click away is tried."""
    if picked:
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
    #: A piece read as another kind on the square it came to: (square, the kind it was, the kind it was read as),
    #: learned only once a second look still shows it so.
    becoming: tuple | None = None
    first_is_hers: bool = True
    made: int = 0
    theirs: int = 0
    #: Since when she has waited on the other side, and the longest it has taken to answer.
    waiting_since: float = field(default_factory=monotonic_time)
    longest_answer_s: float = 0.0
    #: When the board last stood still for her to read.
    stood_still_at: float = field(default_factory=monotonic_time)
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
    #: The moves the other side has made this game, by their squares, and how often: her search tries them early at
    #: their turns, where they could make them again.
    they_went: dict[tuple[tuple[int, int], tuple[int, int]], int] = field(default_factory=dict)
    #: What each of their moves came at: the piece of hers in the area of the board it went to (in words), latest last.
    their_areas: list[str] = field(default_factory=list)
    #: What she holds each kind worth, as her last look used it, for saying which of her pieces a fight is about.
    weights_seen: dict[str, float] = field(default_factory=dict)
    #: What each side can do from the position she left them, worked out on their time: the position's key, and the
    #: work under way or done (core/agency/what_each_area_of_a_board_comes_to.py ``possible_on_a_board``).
    mapping: tuple[Any, Any] | None = None
    #: How many moves deep the answer she is playing was worked out on their time.
    counter_deep: int = 0
    #: The kind of choice the board read area by area made, where she is playing it ("wait" or "neutral"), and whether
    #: she can wait longer than they can.
    area_kind: str = ""
    area_outlasts: bool | None = None
    #: Her moves as the game took them: proposed, attempted, and made or refused (core/agency/what_she_did.py).
    did: WhatSheDid = field(default_factory=WhatSheDid)
    #: Moves she made by hand that the board did not show made, each with the position and rules it was tried under:
    #: kept from what she learns until the game is seen taking a move after them. A board that has stopped taking
    #: anything has not refused the move; her hands did not reach it. LIVE 24 November 2026 the board stopped taking
    #: her moves at move 14, d2-d3 (legal) was learned as no move of a pawn's, and every piece picked up after it
    #: offered nothing.
    untaken: list[tuple[Any, Move, Any]] = field(default_factory=list)
    #: Pieces she picked up that the board offered nothing, held the same way: position, square, rules.
    unshown: list[tuple[Any, Any, Any]] = field(default_factory=list)
    #: What her judgements of this board have gone on to come to (core/cognition/how_far_her_judgement_holds.py).
    held: HowFarItHolds = field(default_factory=HowFarItHolds)
    pondering: Pondering = field(default_factory=Pondering)
    judged: list[tuple[str, str, int]] = field(default_factory=list)
    #: The reply her search expects to her last move, the position it leads to, and the rules it was found by; whether
    #: the reply came is counted, foreseen of foreseeable.
    expected: tuple[Move, Position, Family] | None = None
    #: Their strongest answer as her search found it, where it is not the one she expects of them.
    also_expected: tuple[Move, Position, Family] | None = None
    foreseen: int = 0
    foreseeable: int = 0
    #: How long each of their answers took.
    answers_s: list[float] = field(default_factory=list)

    def start_still_unconfirmed(self, now: float) -> bool:
        """Whether a board that has never answered has exhausted its start wait."""
        return self.theirs == 0 and now - self.waiting_since > NO_ANSWER_AFTER_S

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
        if self.becoming is not None:
            # A change of kind is learned once a second look still shows it: one look read wrong (a mark over the
            # piece, a piece still arriving) taught her, LIVE to 24 November 2026, that two kinds become a third that
            # neither does. Shown as it was, or as something else, it was a misreading.
            square, was, became = self.becoming
            now = shown.get(square)
            if now is not None and now[1] == became:
                self.learned.watched_becoming(was, became,
                                              self.learned._row_from_its_back(now[0], square))
            if now is None or now[1] is not None:
                self.becoming = None
        move = self.last_followed
        if move is None:
            return
        held, seen = self.believed.cells.get(move.end), shown.get(move.end)
        if held is None or seen is None or seen[0] != held[0] or seen[1] in (None, held[1]):
            return
        self.becoming = (move.end, held[1], seen[1])
        self.believed = Position(cells={**self.believed.cells, move.end: (held[0], seen[1])}, turn=self.believed.turn,
                                 unmoved=self.believed.unmoved, passable=self.believed.passable)


@dataclass
class BoardSession:
    """A game retained between turns on the same playing surface."""

    game: _Game | None = None
    rehearsal: Any = None
    before_input: Any = None
    confirmed_move: Move | None = None

    @property
    def in_progress(self) -> bool:
        return self.game is not None and bool(self.game.made or self.game.theirs)


async def play_pieces_by_kind(seat: Seat, look: Look, click: Click, *, seconds: float,
                              say: Callable[[str], None] | None = None, thinks_s: float = THINKS_S,
                              learned: HowThePiecesMove | None = None, rehearsal: Any = None,
                              clock: Callable[[], Awaitable[float | None]] | None = None,
                              borrow: Callable[[dict, bool], Any] | None = None,
                              went_wrong: WhatFailedWhere | None = None,
                              they_answered: WhatTheyAnsweredWhere | None = None,
                              prepared: Any = None, held: Any = None, endings: Any = None,
                              read_end: EndReading | None = None,
                              session: BoardSession | None = None) -> dict[str, Any]:
    """Play the game at ``seat`` for ``seconds`` or until it ends, learning its rules as it goes, and what its pieces
    are worth by rehearsing it in her model while the other side thinks. ``clock`` reads the time she has left, where
    the game shows it. Where she has learned nothing of this board, ``borrow`` is asked, once she knows who moves first,
    for what she learned at a board that begins as this one does: its rules, its rehearsal and where she learned them,
    or None. Where supplied, ``read_end`` confirms the result from the playing surface before it is reported or
    learned from. What came of it, with what she learned and the rehearsal, for her to keep."""
    from concurrent.futures import ThreadPoolExecutor

    from core.agency.practising_between_games import at_a_board
    from core.agency.searching_by_kind import search

    prior = session.game if session is not None else None
    rehearsing = _Rehearsing(session.rehearsal if prior is not None and session is not None else rehearsal)
    # What her eyes and hands cost, each look and each click timed: her turn's log says how long moving took, and not
    # which of the two it went on. LIVE 23 November 2026, game 21, moving took 52 s of her 194 and thinking 92.
    timed = _Timed()
    look, click = timed.looking(look), timed.clicking(click)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rehearsal")
    # Her look on their time has a thread of its own: a round of rehearsal under way does not hold it up.
    ponder_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pondering")
    # And what each side can do, worked out on their time in a process of its own: in a thread it shared one lock with
    # her eyes, and slowed her looks while they thought (core/agency/what_can_happen_worked_out_apart.py).
    from core.agency.what_can_happen_worked_out_apart import worked_out_apart

    map_pool = worked_out_apart()
    the_moves: list[str] = prior.played if prior is not None else []
    if prior is not None:
        went_wrong, they_answered = prior.went_wrong, prior.they_answered
    went_wrong = went_wrong if went_wrong is not None else WhatFailedWhere()
    they_answered = they_answered if they_answered is not None else WhatTheyAnsweredWhere()
    try:
        with at_a_board():
            came = await _play(seat, look, click, seconds=seconds, say=say, thinks_s=thinks_s, learned=learned,
                               rehearsing=rehearsing, pool=pool, clock=clock, search=search, borrow=borrow,
                               the_moves=the_moves, went_wrong=went_wrong, ponder_pool=ponder_pool,
                               they_answered=they_answered, prepared=prepared, held=held, endings=endings,
                               map_pool=map_pool, read_end=read_end, session=session)
    finally:
        # Nothing worked out for this game goes on after it, however it ended.
        map_pool.stop()
        pool.shutdown(wait=False, cancel_futures=True)
        ponder_pool.shutdown(wait=False, cancel_futures=True)
        if session is not None:
            session.rehearsal = rehearsing.rehearsal
            if session.game is not None:
                if session.confirmed_move is not None:
                    session.game.follow(session.confirmed_move)
                    session.confirmed_move = None
                    session.before_input = None
                session.game.mapping = None
                session.game.pondering = Pondering()
    if came.get("ended") == "resume unavailable":
        return came
    came["rehearsal"] = rehearsing.rehearsal
    # The game as it went, for its record: every move in order, and who moved first.
    came["the_moves"] = list(the_moves)
    came["first_was_hers"] = getattr(came.get("learned"), "first_was_hers", None)
    came["went_wrong"] = went_wrong
    came["they_answered"] = they_answered
    came["eyes_and_hands"] = timed.said()
    logger.info("her eyes and hands this game: %s", came["eyes_and_hands"])
    return came


@dataclass
class _Timed:
    """How long each look at the board and each click took her this game."""

    looks: list[float] = field(default_factory=list)
    clicks: list[float] = field(default_factory=list)

    def looking(self, look: Look) -> Look:
        async def timed() -> Any:
            began = monotonic_time()
            try:
                return await look()
            finally:
                self.looks.append(monotonic_time() - began)
        return timed

    def clicking(self, click: Click) -> Click:
        async def timed(x: float, y: float) -> bool:
            began = monotonic_time()
            try:
                return await click(x, y)
            finally:
                self.clicks.append(monotonic_time() - began)
        return timed

    def said(self) -> dict[str, Any]:
        import statistics

        def of(times: list[float]) -> dict[str, Any]:
            return {"n": len(times), "median_s": round(statistics.median(times), 3) if times else None,
                    "total_s": round(sum(times), 1)}
        return {"looks": of(self.looks), "clicks": of(self.clicks)}


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
                       beyond=list(getattr(earned.weights, "beyond", None) or []),
                       emptying={k: v for k, v in (getattr(earned.weights, "emptying", None) or {}).items()
                                 if k in family.kinds},
                       habits=dict(getattr(earned.weights, "habits", None) or {}))

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
                ponder_pool: Any = None, they_answered: WhatTheyAnsweredWhere | None = None,
                prepared: Any = None, held: Any = None, endings: Any = None, map_pool: Any = None,
                read_end: EndReading | None = None, session: BoardSession | None = None) -> dict[str, Any]:
    from core.agency.searching_by_kind import Found
    from core.cognition.when_the_situation_decides import nothing_to_decide

    tell = say or (lambda _line: None)
    began = monotonic_time()
    game = session.game if session is not None else None
    if game is not None:
        still = await _seen_still(seat, look)
        if still is None or not _matches_the_retained_game(seat, still[1], game):
            return {"ended": "resume unavailable", "moves": 0, "theirs": 0}
        before_input = session.before_input
        if game.believed.turn == FIRST and before_input is not None and (
                before_input.shape != still[0].shape or _marked(seat, before_input, still[0], None)):
            return {"ended": "resume unavailable", "moves": 0, "theirs": 0}
        tell(f"I found the same board. I'll pick up after my {game.made} confirmed moves.")
        game.stood_still_at = game.waiting_since = monotonic_time()
        game.count = KeptCount()
        game.last_share = game.their_move_seen_at = 0.0
        game.read_this_wait = False
    else:
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
        waited = monotonic_time()
        while monotonic_time() - waited < WAITS_FOR_THE_FIRST_MOVE_S:
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
        if held is not None:
            game.held = held
        if game.learned.shown is None:
            from core.agency.what_the_game_showed import WhatTheGameShowed

            game.learned.shown = WhatTheGameShowed()
        if session is not None:
            session.game = game
    misread = 0
    unexplained: dict | None = None
    while monotonic_time() - began < seconds:
        still = await _seen_still(seat, look)
        if still is None:
            # Keep the position and wait for a fresh reading of an active game. A missing frame is also possible
            # when the playing surface is temporarily unavailable, so don't hand back the controls on that alone.
            unsettled_s = monotonic_time() - game.stood_still_at
            if (unsettled_s < max(NO_ANSWER_AFTER_S, 10 * game.longest_answer_s)
                    and (game.made or game.theirs or await look() is not None)):
                if unsettled_s >= SETTLES_WITHIN_S:
                    logger.debug("the board has not been readable for %.0f s; she watches on", unsettled_s)
                await asyncio.sleep(0.3)
                continue
            return {"ended": "the board went out of sight", "moves": game.made, "theirs": game.theirs,
                    "learned": game.learned}
        game.stood_still_at = monotonic_time()
        picture, seen = still
        others = _only_the_others(game.learned)
        landed = game.last_followed.end if game.last_followed is not None else None
        shown = _read_against(seat.as_rules(seen), game.believed.cells, others, _held_by_each(game.learned), landed)
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
                    if session is not None:
                        session.confirmed_move = None
                        session.before_input = None
                    game.waiting_since = monotonic_time()
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
                    game.answers_s.append(monotonic_time() - game.waiting_since)
                    game.longest_answer_s = max(game.longest_answer_s, monotonic_time() - game.waiting_since)
                    game.their_move_seen_at = monotonic_time()
                    game.read_this_wait = False
                    game.they_answered.they_answered(_a_place(game.believed, game.learned.start, family),
                                                     _named(move))
                    game.they_went[(move.start, move.end)] = game.they_went.get((move.start, move.end), 0) + 1
                    came_at = _what_it_came_at(game, family, move)
                    if came_at:
                        game.their_areas.append(came_at)
                game.follow(move)
                game.waiting_since = monotonic_time()
            if seen_moves:
                if session is not None:
                    session.before_input = None
                    session.confirmed_move = None
                continue
            # Nothing moved: what differed was a mark. The board is as she holds it, and the turn goes on from there.
        if game.believed.turn != FIRST:
            # While the other side thinks, she rehearses the game in her model, and looks at what she expects to face.
            rehearsing.start(pool, game.learned.family())
            _think_on_their_time(game, rehearsing, ponder_pool, thinks_s)
            _work_out_what_they_can_do(game, rehearsing, map_pool, thinks_s)
            if clock is not None and not game.read_this_wait:
                # Her clock is read, where her count of it may have drifted, while it stands still on their time.
                game.read_this_wait = True
                await _read_her_clock_while_it_stands(clock, rehearsing, game)
            # With no answer yet, a setup or analysis board may have accepted
            # a move. Once the opponent has played, a long search can take
            # the rest of the task's budget without erasing that evidence.
            if game.start_still_unconfirmed(monotonic_time()):
                tell("They haven't answered my move, so I don't think the game has started; I'll look at the screen.")
                return {"ended": "not answered", "moves": game.made, "theirs": game.theirs, "learned": game.learned}
            await asyncio.sleep(0.2)
            continue
        rehearsing.collect()
        game.pondering.settle()
        # Their time is over: what each side can do stops being worked out, and what it had finished is hers to use.
        # The time it was given was a guess at how long they would take.
        if map_pool is not None:
            map_pool.stop()
        # Her turn, all of it on her clock where the game shows one: picking pieces up, thinking and moving share the
        # turn's part of the time she has left. LIVE 20 November 2026 only the thinking was held to it, and picking up
        # six pieces a turn and seeing each move shown took a third of a three-minute game.
        turn_began = monotonic_time()
        # Her clock has run since their move was seen, or since her turn began where nothing of theirs came first.
        clock_began = game.their_move_seen_at if game.theirs else turn_began
        share, on_her_clock = await _time_for_a_move(clock, rehearsing, game.made, thinks_s, game, clock_began)
        if on_her_clock:
            game.last_share = share
        elif clock is not None and game.last_share:
            # Her clock went on running while it could not be read. LIVE 21 November 2026, game 10, four late turns
            # read no clock and were given four seconds of thought where the last reading left her one or two.
            share, on_her_clock = game.last_share, True
        if session is not None:
            session.before_input = picture
        # Learn from a few pieces she has not seen enough of, then think.
        offered: dict[tuple[int, int], set] = {}
        probes = 0
        # Pieces whose answer would settle a way her rules are unsure of first (one look at what the game offers says
        # whether it goes there); then pieces with room around them: one boxed in by its own side shows nothing.
        mine = [sq for sq, (side, _kind) in game.believed.cells.items() if side == FIRST]
        settles = {sq: _would_settle(game, sq) for sq in mine}
        for square in sorted(mine, key=lambda sq: (-settles[sq], -_room_around(game.believed, sq))):
            kind = game.believed.cells[square][1]
            if probes >= PROBES_A_TURN:
                break
            if _times_offered(game.learned, kind) >= SEEN_ENOUGH and game.made and not settles[square]:
                continue
            if on_her_clock and not _room_to_learn(game, monotonic_time() - turn_began, share, probes):
                break
            probe_began = monotonic_time()
            squares = await _offers_for(seat, look, click, picture, square)
            game.timed_probe(monotonic_time() - probe_began)
            probes += 1
            if squares is not None:
                offered[square] = squares
                if squares:
                    _the_untaken_were_refused(game)
                    game.learned.offered(game.believed, square, squares, family=game.learned.family())
                else:
                    # As a pick-up in her turn that offers nothing: held until the board is seen answering. LIVE 24
                    # November 2026 her first looks were at a game's setup page, whose board takes nothing until a
                    # game is begun, and every piece she picked up there was learned as going nowhere.
                    game.unshown.append((game.believed, square, game.learned.family()))
        if not game.made and not game.theirs and probes and not any(offered.values()):
            game.unshown.clear()
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
            # The offers of this turn's looks first: one already seen is the game going on.
            offers_left = {sq: ends for sq, ends in offered.items() if ends}
            if not offers_left:
                offers_left = await _what_the_game_still_offers(seat, look, click, picture, game.believed)
                for start, ends in offers_left.items():
                    game.learned.offered(game.believed, start, ends, family=family)
            if not offers_left:
                return await _finish_game(ended or "no move", game, tell, read_end)
            logger.info("her model says the game is over (%s) and the game still offers %d move(s)",
                        ended or "no move", sum(len(v) for v in offers_left.values()))
            family = game.learned.family(unseen_go_every_way=True)
            legal = moves(family, game.believed)
            if not legal:
                await _made_as_offered(seat, look, click, picture, game, family, offers_left, tell, session)
                continue
        weights = rehearsing.weights_for(family)
        set_aside: list[Move] = []
        here = _a_place(game.believed, game.learned.start, family)
        failed_here = game.went_wrong.already_failed_at(here)
        if failed_here:
            again = [one for one in legal if _named(one) in failed_here]
            if again and len(again) < len(legal):
                set_aside = list(again)
                tell(f"I've played {said_of_a_move(family, game.believed, again[0], game.learned, game.first_is_hers)}"
                     " from here before and it went wrong, so I'll look at something else.")
        # A move the board did not take here is not tried again here, though it is not yet learned as no move.
        set_aside += [m for at, m, _rules in game.untaken
                      if at.key() == game.believed.key() and m in legal and m not in set_aside]
        found = None
        areas = None
        learned_until, searched_s = monotonic_time(), 0.0
        search_deadline = min(began + seconds, learned_until + _the_most_a_look_may_take(
            share, on_her_clock, game.count.value,
        ))
        for _ in range(CHECKS_A_TURN):
            # What is left of the turn, less what making the move and seeing it shown has taken her this game. After a
            # move the game refused, her model was wrong where she was about to act: the look that replaces it gets a
            # turn's share again, not what the refusal left. LIVE 23 November 2026 the look after a refusal had 0.2 s
            # and 7,000 positions, and chose the move that lost the game.
            thinks = (max(0.2, share - (monotonic_time() - turn_began) - game.finish_s) if on_her_clock
                      else share)
            if set_aside and on_her_clock:
                thinks = max(thinks, share)
            # Retries and comparisons spend from the same allowance as the first search.
            most = min(_the_most_a_look_may_take(thinks, on_her_clock, game.count.value),
                       search_deadline - monotonic_time())
            if most <= 0:
                found = None
                break
            # A move that is the only one is made without weighing (core/cognition/when_the_situation_decides.py).
            only = nothing_to_decide([m for m in legal if m not in set_aside]) if not set_aside else None
            # A situation she worked out ahead, among the choices it offers her now, is answered from what
            # she worked out (core/agency/thinking_before_it_comes.py), and her clock keeps the time.
            ready = _worked_out_ahead(prepared, game, family, weights, legal) if not set_aside else None
            if ready is None and not set_aside:
                # Her answer to the move they made, where she worked it out on their time: checked by her look now.
                ready = _answer_worked_out_on_their_time(game, family, legal)
            # An ending she has worked out to the end, won or lost, under the rules she plays by
            # (core/agency/what_a_small_ending_comes_to.py): its answer rests on every situation that can follow.
            ended = (worked_to_the_end(endings, family, game.believed, [m for m in legal if m not in set_aside])
                     if endings else None)
            if only is not None:
                found = Found(only, 0, 0, 0, [only], by="the only move")
            elif ended is not None:
                found = Found(ended, 0, 0, 0, [ended], by="worked out to the end")
            elif ready is not None and ready.by == "worked out ahead":
                found = ready
            else:
                search_began = monotonic_time()
                # The board read area by area: what they could do to her in each fight is what her search looks at
                # first at their turns, beside what they have done.
                areas = await asyncio.to_thread(_the_areas, game, family, weights)
                game.weights_seen = dict(weights.worth) if weights else {}
                expected = _expected_with_the_areas(game, areas)
                # A long native search must leave the event loop available for
                # chat delivery, health checks and cancellation.
                most = min(most, search_deadline - monotonic_time())
                if most <= 0:
                    found = None
                    break
                found = await asyncio.to_thread(
                    search, family, game.believed, seconds=most, weights=weights, earlier=game.earlier,
                    set_aside=set_aside, threads=_cores_free(), wary=FIRST,
                    soft=thinks if most > thinks else None, expected=expected, endings=endings,
                )

                def search_only(aside: list[Move]) -> Found:
                    left = min(max(0.2, thinks / 2), search_deadline - monotonic_time())
                    if left <= 0:
                        return Found(None, 0, 0, 0, [])
                    return search(family, game.believed, seconds=left, weights=weights,
                                  earlier=game.earlier, set_aside=list(dict.fromkeys([*set_aside, *aside])),
                                  threads=_cores_free(), wary=FIRST,
                                  expected=expected, endings=endings)

                if ready is not None:
                    found = await asyncio.to_thread(_checked, ready, found, legal, search_only)
                # What can still change after the move, proved without looking ahead.
                found = await asyncio.to_thread(
                    _held_to_what_can_change, game, family, legal, set_aside, found, search_only,
                )
                found = await asyncio.to_thread(
                    _held_to_the_areas, game, areas, found,
                    [m for m in legal if m not in set_aside], search_only,
                )
                searched_s += monotonic_time() - search_began
            if found.move is None:
                break
            # Before she moves it, she picks it up: the move is made only where the game offers it. Each look that
            # finds the game offering just what her rules said lets her go one move further without it; the first
            # that does not, or a move the game refuses, has her looking before every move again. LIVE 21 November
            # 2026, game 10: the look took 47 s of a 232 s game, where her thinking had 60.
            if found.move.start not in offered:
                pressed = any(k.royal for k in family.kinds.values()) and in_check(family, game.believed, FIRST)
                unseen = (not game.learned.knows_how_it_goes(game.believed.cells[found.move.start][1])
                          or game.learned.seen_going_there(game.believed, found.move) < SEEN_TO_BELIEVE
                          or not game.learned.sure_of(game.believed, found.move, family))
                if not _looks_first(game, pressed=pressed, unseen=unseen):
                    break
                squares = await _offers_for(seat, look, click, picture, found.move.start)
                if squares is not None:
                    offered[found.move.start] = squares
                    if squares:
                        # The board answered a pick-up: it is taking what she does.
                        _the_untaken_were_refused(game)
                        game.learned.offered(game.believed, found.move.start, squares, family=family)
                    else:
                        # Nothing offered at all is what a board that has stopped answering shows too: held, as a
                        # move the board did not take is, until the board is seen answering.
                        game.unshown.append((game.believed, found.move.start, family))
                    said = {one.end for one in legal if one.start == found.move.start}
                    if set(squares) == said:
                        game.far.it_was_where_she_said(1)
                    else:
                        game.far.it_was_not(1)
            if found.move.start not in offered or _landing(found.move) in offered[found.move.start]:
                break
            if offered[found.move.start]:
                game.learned.refused(game.believed, found.move, family=family)
            else:
                game.untaken.append((game.believed, found.move, family))
            set_aside.append(found.move)
            family = game.learned.family(unseen_go_every_way=True)
            weights = rehearsing.weights_for(family)
        else:
            # Every move she looked at was refused: she picks up every piece of hers to check if play can continue.
            for square in mine:
                if square not in offered:
                    squares = await _offers_for(seat, look, click, picture, square)
                    if squares is not None:
                        offered[square] = squares
            if set(mine) <= offered.keys() and not any(offered.values()):
                return await _finish_game("no move", game, tell, read_end)
            found = None
        move = found.move if found is not None else None
        if move is None:
            # Her rules leave her nothing the game has not refused, and what the game offers decides, not her rules.
            # LIVE 23 November 2026, the start of a game: her rules gave her three moves, the game refused all three,
            # and she said she had lost with the game offering her pawns' moves in the same turn's looks.
            offers_left = {sq: ends for sq, ends in offered.items() if ends}
            if not offers_left:
                offers_left = await _what_the_game_still_offers(seat, look, click, picture, game.believed)
                for start, ends in offers_left.items():
                    game.learned.offered(game.believed, start, ends, family=family)
            if not offers_left:
                return await _finish_game("no move", game, tell, read_end)
            logger.info("her rules leave her no move and the game offers %d", sum(len(v) for v in offers_left.values()))
            await _made_as_offered(seat, look, click, picture, game, game.learned.family(unseen_go_every_way=True),
                                   offers_left, tell, session)
            continue
        if found.by == "the only move":
            how = "It's the only move I have"
        elif found.by == "worked out ahead":
            how = f"I worked this one out before the game (looked {found.depth} moves deep)"
        elif found.by == "worked out to the end":
            how = "I worked this ending out to the end before the game: every position it can come to"
        elif found.by == "nothing can change after it":
            how = ("After it nothing either side can do changes the position, so it can only be drawn, and a sure "
                   "draw is worth more than how my other moves have turned out from positions like this")
        elif found.by == "worked out area by area, checked":
            how = ("I held my position and left them to commit first" if game.area_kind == "wait" else
                   "I held my position while I set up the rest")
            how += f". (looked {found.depth} moves deep)"
        elif found.by == "worked out on their time, checked":
            how = (f"I'd worked out my answer to that while they were thinking ({game.counter_deep} moves deep), and a "
                   f"look now ({found.depth} moves deep) still finds it best")
        elif found.by == "worked out ahead, checked":
            how = (f"I worked this one out before the game, and a look now ({found.depth} moves deep), with what I "
                   "have learned of the pieces since, still finds it best")
        else:
            how = f"{_how_it_stands(found, weights, game.learned.start)}. (looked {found.depth} moves deep, " \
                  f"{found.nodes:,} positions)"
            areas_said = _the_areas_said(game, family, areas, weights)
            if areas_said:
                how += f" {areas_said}"
            # What judgements like it have come to here, where she has seen enough to say: her judgement read as the
            # chance it has turned out to be.
            held_so = game.held.says(found.score)
            if held_so:
                how += f" {held_so[0].upper()}{held_so[1:]}."
        small = (getattr(found, "accounting", None) or {}).get("small_solver")
        if small:
            how += f" My small solver compared {small['moves']} modeled moves through {small['depth']} plies."
            how += (" This was among its best." if small["rank"] == 1 else
                    f" My deeper search backed this move, which the small solver ranked {small['rank']}.")
        if not await _as_it_was(seat, look, picture):
            continue
        act = game.did.proposed(_as_played(game, move))
        game.did.attempted(act)
        made = False
        finish_began = monotonic_time()
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
            if not await _by_hand(seat, click, picture, move, OFFERS_SHOW_S * (1 + attempt)):
                tell("I couldn't send the move to the game window.")
                return await _finish_game("no move", game, tell, read_end)
            game.did.dispatched(act)
            moved_at = monotonic_time()
            made = await _comes_to_show(seat, look, game, move)
            if made:
                break
        if made:
            game.did.settled(act, ACCEPTED, known_by="the board matched the move after my input was sent")
            if found.by not in ("the only move", "worked out to the end", "nothing can change after it"):
                _judged(game, here, move, found)
            tell(f"I played {said_of_a_move(family, game.believed, move, game.learned, game.first_is_hers)}. {how}")
            # What making a move costs her clock: until it is made, for her clock stops there, and the look that sees it
            # shown runs on theirs. It is held back from her thinking on every turn.
            game.timed_finish(moved_at - finish_began)
            # Her clock stopped when her move was made: what it ran this turn is counted off.
            game.count.step(-(moved_at - clock_began))
            _expect_back(game, family, move, found)
            # Where her turn went, for knowing what her clock pays for besides thinking.
            seen = game.their_move_seen_at or turn_began
            logger.info("her turn %d took %.1f s: %.1f s before it began, %.1f s learning, %.1f s thinking, %.1f s "
                        "between, %.1f s moving; her share of the clock was %s", game.made + 1,
                        monotonic_time() - seen, turn_began - seen, learned_until - turn_began, searched_s,
                        finish_began - learned_until - searched_s, monotonic_time() - finish_began,
                        f"{share:.1f} s" if on_her_clock else "not on a clock")
            looked = getattr(found, "accounting", None) or {}
            if looked:
                # What the look did, for knowing what it cost and where its time went (suggestions of 23 November 2026).
                steps = list(looked.get("iterations") or [])
                logger.info("her look at turn %d: %d deep, furthest ply %s, %s of %s positions settled by her table, %s "
                            "past the horizon, its best move changed at %d of %d depths", game.made + 1, found.depth,
                            looked.get("seldepth"), looked.get("tt_hits"), found.nodes, looked.get("qnodes"),
                            sum(1 for step in steps if step[2]), len(steps))
        if not made:
            game.untaken.append((game.believed, move, family))
            if game.unlooked:
                game.far.it_was_not(game.unlooked)
                game.unlooked = 0
            tell("I sent a move but couldn't confirm it on the board.")
            continue
        _the_untaken_were_refused(game)
        if session is not None:
            session.confirmed_move = move
        game.made += 1
        game.waiting_since = monotonic_time()
    return {"ended": "out of time", "moves": game.made, "theirs": game.theirs, "learned": game.learned,
            "foresaw": (game.foreseen, game.foreseeable), "acts": game.did.as_record()}


def _matches_the_retained_game(seat: Seat, seen: PiecesSeen, game: _Game) -> bool:
    """Check the retained position or one unambiguous reply before sending input."""
    position = game.believed
    shown = _read_against(seat.as_rules(seen), position.cells, _only_the_others(game.learned),
                          _held_by_each(game.learned))
    if _agrees(_without_marks_where_nothing_stood(shown, position.cells), position.cells):
        return True
    if position.turn != SECOND:
        return False
    family = game.learned.family()
    successors = (after(family, position, move) for move in moves(family, position))
    return sum(_agrees(_without_marks_where_nothing_stood(shown, next_position.cells), next_position.cells)
               for next_position in successors) == 1


def _as_played(game: _Game, move: Move) -> str:
    """A move named as her record of the game names it: its squares, and an x where it takes."""
    rows, columns = game.learned.rows, game.learned.columns
    return (_square_named(move.start, game.first_is_hers, rows, columns) + ("x" if move.taken_at is not None else "-")
            + _square_named(move.end, game.first_is_hers, rows, columns))


def _worked_out_ahead(prepared: Any, game: _Game, family: Family, weights: Any, legal: list[Move]) -> Any:
    """Her answer to this situation, where she worked it out ahead among the choices it offers her now."""
    if prepared is None:
        return None
    from core.agency.searching_by_kind import Found
    from core.agency.thinking_before_it_comes import among

    held = prepared.answer(_a_place(game.believed, game.learned.start, family), among(legal),
                           count=game.believed.count, limit=family.count_to_draw)
    move = next((m for m in legal if held is not None and _named(m) == held.get("move")), None)
    if move is None:
        return None
    # A move that brings back a situation of this game was worked out without the game's history, which a repeat
    # can end: it is checked like any hint (suggestions of 23 November 2026: a key that leaves out repetitions).
    there = after(family, game.believed, move)
    repeats = any(key == there.key() for key in there.seen)
    by = "worked out ahead" if prepared.certified(held, family, weights) and not repeats else \
        "worked out ahead, to check"
    return Found(move, int(held.get("score", 0)), int(held.get("depth", 0)), 0, [move], by=by)


def _held_to_what_can_change(game: _Game, family: Family, legal: list[Move], set_aside: list[Move], found: Any,
                             search_only: Callable[[list[Move]], Any]) -> Any:
    """Her search's move, held to what can still change after it. A position after which nothing that decides a game
    can ever happen is a draw, whatever her judgement of it: worth a half for certain. Where her judgement of the
    position has turned out worth less than a half (her own curve, or, before she has one, her judgement below even),
    a move after which nothing can change is taken in place of the search's; where it has turned out worth more, a
    move of the search's after which nothing can change is set aside for the best of the rest, kept where that is
    still worth more than a half (suggestions of 23 November 2026, item 18)."""
    from core.agency.searching_by_kind import Found
    from core.agency.what_cannot_change_on_a_board import what_can_change

    if found.move is None:
        return found
    open_moves = [m for m in legal if m not in set_aside]

    def worth(score: int) -> float:
        chance = game.held.chance(score)
        return chance if chance is not None else 0.5 + 0.5 * ((score > 0) - (score < 0))

    def settled_after(move: Move) -> bool:
        return not what_can_change(family, after(family, game.believed, move)).can

    if not what_can_change(family, game.believed).can:
        return found
    here = worth(int(found.score))
    if here < 0.5:
        for move in sorted(open_moves, key=lambda m: m != found.move):
            if settled_after(move):
                logger.info("a move after which nothing can change, a sure draw, over a judgement worth %.2f", here)
                return Found(move, 0, int(found.depth), int(found.nodes), [move], by="nothing can change after it")
    elif here > 0.5 and settled_after(found.move) and len(open_moves) > 1:
        other = search_only([*set_aside, found.move])
        if other.move is not None and worth(int(other.score)) > 0.5:
            logger.info("her move would leave nothing that can change, a sure draw, from a judgement worth %.2f: "
                        "another kept", here)
            return other
    return found


def _checked(hint: Any, now: Any, legal: list[Move], search_only: Callable[[list[Move]], Any]) -> Any:
    """Keep a proposed move when the current search agrees or measures it at the same depth.

    The restricted search can finish at a different depth. Compare its score
    at the main search's completed depth, within the main search's last
    score change. When the depths cannot be aligned, use the main result.
    """
    from dataclasses import replace

    origin = str(getattr(hint, "by", "") or "worked out ahead").removesuffix(", to check")
    if now.move is None:
        return replace(hint, by=f"{origin}, not checked")
    if now.move == hint.move:
        return replace(now, by=f"{origin}, checked")
    alone = search_only([m for m in legal if m != hint.move])
    if alone.move != hint.move:
        return replace(now, by=f"{origin}, not checked at the same depth")
    comparable = alone.score_at_depth(now.depth)
    if comparable is None:
        return replace(now, by=f"{origin}, not checked at the same depth")
    if now.score - comparable <= max(0, int(now.unsteady)):
        return replace(alone, by=f"{origin}, checked")
    return replace(now, by=f"{origin}, and a look now found better")


def _expect_back(game: _Game, family: Family, move: Move, found: Any) -> None:
    """What she expects back from the move she has made: what they answered before in the position it leaves, where
    she has met it, else the reply her search found best for them; and where it leaves her. Nothing when she has not
    met the position and her search did not look past her own move."""
    game.expected = None
    game.also_expected = None
    mine = after(family, game.believed, move)
    named = game.they_answered.most_often(_a_place(mine, game.learned.start, family))
    before = next((m for m in moves(family, mine) if _named(m) == named), None) if named else None
    line = list(getattr(found, "line", None) or [])
    strongest = line[1] if getattr(found, "by", "") != "the only move" and len(line) >= 2 and line[0] == move else None
    if before is not None:
        game.expected = (before, after(family, mine, before), family)
        if strongest is not None and _named(strongest) != _named(before):
            # Their best answer, as her search found it, where it is not the one they gave before: looked at too.
            game.also_expected = (strongest, after(family, mine, strongest), family)
        return
    if strongest is None:
        return
    game.expected = (strongest, after(family, mine, strongest), family)


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
    also = game.also_expected[1] if game.also_expected is not None else None
    if also is not None and (also.turn != FIRST or not moves(family, also)):
        also = None
    game.pondering.start(pool, family, facing, seconds=game.pondering.look_for(game.answers_s, game.last_share or thinks_s),
                         weights=rehearsing.weights_for(family), earlier=[*game.earlier, game.believed],
                         threads=max(1, _cores_free() - 1), also=also)


def _held_to_the_areas(game: _Game, areas: Any, found: Any, legal: list[Move],
                       search_only: Callable[[list[Move]], Any]) -> Any:
    """Give the exact finite-horizon composition a voice in any decision it covered. A better move in its ranking
    is checked against the deep search at the same depth. The approximate area reading can still suggest waiting."""
    from dataclasses import replace

    from core.agency.searching_by_kind import Found

    if areas is None or found.move is None or found.by == "nothing can change after it":
        return found
    if areas.played is not None:
        hers = areas.played[0]
        candidates = set(legal)
        scores = {move: score for _part, move, score in hers.each if move in candidates}
        if hers.complete and hers.deep > 0 and scores.keys() == candidates and found.move in scores:
            best = max(scores, key=scores.__getitem__)
            if scores[best] > scores[found.move]:
                hint = Found(best, 0, hers.deep, 0, [best], by="worked out by the small solver, to check")
                checked = _checked(hint, found, legal, search_only)
                if checked.by == "worked out by the small solver, checked":
                    found = checked
            rank = 1 + sum(score > scores[found.move] for score in scores.values())
            logger.info("small solver compared %d moves at %d plies, best %s (%.2f); played candidate %s (%.2f), "
                        "rank %d", len(scores), hers.deep, _named(best), scores[best], _named(found.move),
                        scores[found.move], rank)
            return replace(found, accounting={**found.accounting,
                                              "small_solver": {"moves": len(scores), "depth": hers.deep, "rank": rank}})

    if areas.choice.weighed is None:
        return found
    from core.cognition.what_each_area_comes_to import changes_nothing, weigh_acts

    choice = areas.choice
    if choice.kind not in ("wait", "neutral"):
        return found
    move = next((m for m in legal if m == choice.weighed.act), None)
    if move is None or move == found.move:
        return found
    # Her own look already chose a move that changes nothing: which of two such moves, the reading cannot say.
    if any(weighed.act == found.move and changes_nothing(weighed, areas.readings)
           for weighed in weigh_acts(areas.readings)):
        return found
    hint = Found(move, int(found.score), int(found.depth), 0, [move], by="worked out area by area, to check")
    held = _checked(hint, found, legal, search_only)
    if held.by == "worked out area by area, checked":
        game.area_kind, game.area_outlasts = choice.kind, choice.outlasts
        logger.info("the board read area by area chose to %s, and her look found it no worse: %s", choice.kind,
                    _named(move))
    return held


def _work_out_what_they_can_do(game: _Game, rehearsing: _Rehearsing, pool: Any, thinks_s: float) -> None:
    """While they think: every move they can make in each area of the board she left them, and her answer to each,
    worked out as deep as the time they are likely to take allows (core/cognition/what_each_area_comes_to.py
    ``what_can_happen``). Not what they will do: what they can, so that whichever they choose, her answer is already
    worked out."""
    if pool is None or game.believed.turn == FIRST:
        return
    key = game.believed.key()
    if game.mapping is not None and game.mapping[0] == key:
        return
    family = game.learned.family(unseen_go_every_way=True)
    worth = dict(rehearsing.weights_for(family).worth)
    seconds = game.pondering.look_for(game.answers_s, game.last_share or thinks_s)
    position = game.believed
    game.mapping = (key, pool.submit(family, position, worth, plies_at_most=10, seconds=seconds))


def _answer_worked_out_on_their_time(game: _Game, family: Family, legal: list[Move]) -> Any:
    """Her answer to the move they made, where she worked it out on their time from the position before it: a hint for
    her look now to check, as an answer worked out before the game is."""
    from core.agency.searching_by_kind import Found

    if game.mapping is None or game.last_followed is None:
        return None
    key, work = game.mapping
    # Worked out from the position their move was made in, and that move the last one followed: else it answers
    # something else.
    if not game.earlier or game.earlier[-1].key() != key or game.believed.turn != FIRST:
        return None
    if not work.done() or work.cancelled():
        return None
    if work.exception() is not None:
        # Worked out in a process of its own: a failure there is said here, or nothing would say it.
        logger.info("what each side could do was not worked out on their time (%s: %s)",
                    type(work.exception()).__name__, work.exception())
        return None
    theirs = game.last_followed
    counters, _found = work.result()
    held = counters.get((theirs.start, theirs.end, theirs.becomes))
    if held is None or held[0] is None:
        return None
    answer, comes_to, deep = held
    move = next((m for m in legal if (m.start, m.end, m.becomes) == (answer.start, answer.end, answer.becomes)), None)
    if move is None:
        return None
    game.counter_deep = deep
    return Found(move, int(comes_to), deep, 0, [move], by="worked out on their time, to check")


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


#: The span the machine's load is averaged over, in seconds: its one-minute load average.
LOAD_SPAN_S = 60.0
#: Her own use of the machine, averaged over the same span as its load: the processor time she had used when last
#: asked, when that was, and how many cores she has kept busy.
_HER_OWN: dict[str, float | None] = {"used": None, "at": None, "cores": 0.0}


def _her_own_cores() -> float:
    """How many cores her own process has kept busy, averaged as the machine's one-minute load is: each span since she
    last asked weighs in by how much of the minute it is."""
    import math
    import time

    used, now = time.process_time(), monotonic_time()
    was_used, was_at = _HER_OWN["used"], _HER_OWN["at"]
    _HER_OWN["used"], _HER_OWN["at"] = used, now
    if was_used is None or was_at is None or now <= was_at:
        return float(_HER_OWN["cores"] or 0.0)
    kept = math.exp(-(now - was_at) / LOAD_SPAN_S)
    _HER_OWN["cores"] = float(_HER_OWN["cores"] or 0.0) * kept + max(0.0, used - was_used) / (now - was_at) * (1 - kept)
    return float(_HER_OWN["cores"])


def _cores_free() -> int:
    """How many of the machine's cores nothing else is using now, to think with: its cores less what others keep busy
    over the last minute (its load less her own use), and one at the least. LIVE 21 November 2026 her search ran on one
    of eighteen; in a test world four times the time to think took her from 0.396 to 0.708 against Stockfish held to
    depth 4 (24 games each). LIVE 23 November 2026, game 26, her own searchers' load was taken for someone else's: a
    turn on fifteen threads looked at 9.4 million positions, and three turns later one on a single thread at 0.6
    million."""
    import os

    cores = os.cpu_count() or 1
    try:
        busy = os.getloadavg()[0]
    except OSError:
        return 1
    return max(1, min(cores, int(cores - max(0.0, busy - _her_own_cores()))))


def _would_settle(game: _Game, square: tuple[int, int]) -> int:
    """How many of the ways her rules are unsure of, for the piece of hers on ``square``, one look at what the game
    offers it would settle: those whose first landing from there is on the board and not held by her own side, so the
    game either offers it or withholds it. A belief she cannot rely on is asked of the world where asking is cheap."""
    position = game.believed
    _side, kind = position.cells[square]
    steps = game.learned.family().kinds.get(kind)
    if steps is None:
        return 0
    rows, columns = game.learned.rows, game.learned.columns
    settled = 0
    for step in steps.steps:
        if step.sure or (step.only_unmoved and square not in position.unmoved):
            continue
        for forward, right in step.offsets:
            there = (square[0] + forward, square[1] + right)
            held = position.cells.get(there)
            if 0 <= there[0] < rows and 0 <= there[1] < columns and (held is None or held[0] != FIRST):
                settled += 1
    return settled


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
    offers most, and where her rules have seen it least. Always for a kind she has never seen go, or a move her rules
    allow on fewer than SEEN_TO_BELIEVE sightings of the kind going that way (``unseen``): what she searched it with is
    a guess, and what her rules earned elsewhere says nothing of it."""
    if not pressed and not unseen and game.unlooked + 1 < game.far.how_many(trusted=game.far.how_often_it_held):
        game.unlooked += 1
        return False
    game.unlooked = 0
    return True


def _the_most_a_look_may_take(thinks: float, on_her_clock: bool, left: float | None) -> float:
    """The most a turn's look may take where another depth is still likely to change her answer. On her clock,
    SPENDS_AT_MOST of what the turn means to take, and never more than LEFT_AT_MOST of what is left on it. With no
    clock to keep, as long as she would wait on theirs: held to what a clock would allow, LIVE 25 November 2026 46 of
    her 173 turns against Stockfish thought for that bound or longer, in games no clock was kept in."""
    if on_her_clock and left is not None:
        return max(thinks, min(SPENDS_AT_MOST * thinks, LEFT_AT_MOST * max(0.0, left)))
    if not on_her_clock:
        return max(SPENDS_AT_MOST * thinks, NO_ANSWER_AFTER_S)
    return thinks


def _their_pace(game: _Game | None, otherwise: float) -> float:
    """With no clock to share, as long as the other side takes, and never less than ``otherwise``: the middle of the
    times it has taken to answer her this game. What they take says how long a turn may run here, not how long hers
    must, since without a clock a slow turn costs her nothing. LIVE 24 November 2026, an untimed game against
    Stockfish 19: it took about five seconds a move and she four, and the moves that lost were tactics thirteen and
    fourteen moves deep that eight seconds of her looking finds. The next game it answered in about a second and a
    half, and her turns, held to that, thought for a fifth of a second."""
    answered = sorted(game.answers_s) if game is not None else []
    if not answered:
        return otherwise
    middle = len(answered) // 2
    pace = answered[middle] if len(answered) % 2 else (answered[middle - 1] + answered[middle]) / 2
    return max(otherwise, pace)


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
        return _their_pace(game, otherwise), False
    moves_left = rehearsing.moves_left(made)
    if moves_left <= 0:
        return _their_pace(game, otherwise), False
    began = monotonic_time() if clock_began is None else clock_began
    if game is not None and game.count.value is not None and not game.count.must_read(game.count.value / moves_left):
        return max(0.2, (game.count.value - (monotonic_time() - began)) / moves_left), True
    try:
        left = await clock()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("her clock was not read (%s)", exc)
        return _their_pace(game, otherwise), False
    if left is None:
        return _their_pace(game, otherwise), False
    if game is not None:
        # Her count begins from it, as the clock showed when it began to run this turn.
        game.count.read(left + (monotonic_time() - began))
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
    waited = monotonic_time() - game.waiting_since
    if game.answers_s and waited + game.read_s > min(game.answers_s):
        return
    began = monotonic_time()
    try:
        left = await clock()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.info("her clock was not read (%s)", exc)
        return
    finally:
        game.read_s = max(game.read_s, monotonic_time() - began)
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


def _held_by_each(learned: HowThePiecesMove) -> dict[int, set[str]]:
    """For each side, the kinds it has surely held: those it began with, and what they were seen to become as often
    as a way is believed on. A becoming seen once may be the very mark this tells apart from a piece: LIVE 25 November
    2026 the glow round her royal piece, read once on a square a move had landed on, was counted as her royal kind
    becoming the glow's look, and would have let the glow through again."""
    from core.agency.how_the_pieces_move import SEEN_TO_BELIEVE

    held = {}
    for side in (FIRST, SECOND):
        kinds = {kind for owner, kind in learned.start.values() if owner == side}
        more = set(kinds)
        while more:
            far = learned.rows - 1
            more = {other for kind in more for other, seen in learned.became_at.get(kind, {}).get(far, {}).items()
                    if seen >= SEEN_TO_BELIEVE} - kinds
            kinds |= more
        held[side] = kinds
    return held


def _without_marks_where_nothing_stood(shown: dict, believed: dict) -> dict:
    """The board read without the looks she could not read on squares she held empty."""
    return {square: held for square, held in shown.items() if held[1] is not None or square in believed}


def _read_against(shown: dict, believed: dict, others: dict[int, set[str]] | None = None,
                  held: dict[int, set[str]] | None = None, landed: tuple[int, int] | None = None) -> dict:
    """The board read, where a piece's look was unread taking its side from the board she holds: a mark that changes
    how a piece looks (a glow round a piece under attack) changes its colour as well, and a piece does not change side
    where it stands. A piece read as a kind only the other side has held (``others``) is read as unread: LIVE-like
    22 November 2026 a dark man crowned on its far row was read as a light man, and the backward moves it made after
    taught her own men to step back.

    Nor does a piece become another kind where it stands: a piece of a side, on a square where she holds a piece of
    that side that did not just land there (``landed``), read as a kind its side has never held (``held``: at the start
    or by becoming), is a mark over it and read as unread. LIVE 25 November 2026 her royal piece, under attack, was
    drawn with a glow her kept looks held as a look of its own; read as that, it was no longer the piece the game
    keeps safe, and twice she moved as if it were not attacked."""
    out = dict(shown)
    for square, (side, kind) in shown.items():
        if others is not None and kind is not None and kind in others.get(side, set()):
            out[square] = (side, None)
            kind = None
        elif (held is not None and kind is not None and square != landed and square in believed
              and believed[square][0] == side and believed[square][1] != kind and kind not in held.get(side, set())):
            out[square] = (side, None)
            kind = None
        if kind is None and square in believed:
            out[square] = (believed[square][0], None)
    return out


async def _made_as_offered(seat: Seat, look: Look, click: Click, picture: Any, game: _Game, family: Family,
                           offers: dict[tuple[int, int], set], tell: Callable[[str], None],
                           session: BoardSession | None = None) -> None:
    """A move the game offers, made where her rules give her none: what the game offers is what the game allows."""
    start, ends = next(iter(offers.items()))
    end = sorted(ends)[0]
    taken = end if game.believed.cells.get(end, (FIRST,))[0] != FIRST else None
    move = Move(start, end, taken_at=taken)
    if not await _as_it_was(seat, look, picture):
        return
    act = game.did.proposed(_as_played(game, move))
    game.did.attempted(act)
    if not await _by_hand(seat, click, picture, move, OFFERS_SHOW_S):
        return
    game.did.dispatched(act)
    if await _comes_to_show(seat, look, game, move):
        game.did.settled(act, ACCEPTED, known_by="the board matched the move after my input was sent")
        tell(f"I played {said_of_a_move(family, game.believed, move, game.learned, game.first_is_hers)}, "
             "the move the game offers me.")
        _the_untaken_were_refused(game)
        if session is not None:
            session.confirmed_move = move
        game.made += 1
        game.waiting_since = monotonic_time()


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
    ends_at = monotonic_time() + MOVE_SHOWS_S
    family = game.learned.family(unseen_go_every_way=True)
    expected = after(family, game.believed, move)
    footprint = {sq: expected.cells.get(sq) for sq in expected.cells.keys() | game.believed.cells.keys()
                 if expected.cells.get(sq) != game.believed.cells.get(sq)}
    while monotonic_time() < ends_at:
        # One look that shows her move's squares as her move leaves them is enough; a board still moving cannot show
        # that. Waiting for it to read the same twice first took 2.1 s a move, a third of game 11's clock.
        picture = await look()
        seen = None if picture is None else seat.looks.read(picture, seat.lattice, settled=True)
        if seen is not None:
            shown = _read_against(seat.as_rules(seen), expected.cells, _only_the_others(game.learned),
                                  _held_by_each(game.learned), move.end)
            if _agrees(_without_marks_where_nothing_stood(shown, expected.cells), expected.cells):
                return True
        still = await _seen_still(seat, look, within_s=MOVE_SHOWS_S)
        if still is None:
            await asyncio.sleep(0.15)
            continue
        shown = _read_against(seat.as_rules(still[1]), game.believed.cells, _only_the_others(game.learned),
                              _held_by_each(game.learned), move.end)
        shown = _without_marks_where_nothing_stood(shown, expected.cells | game.believed.cells)
        if not _agrees(shown, game.believed.cells):
            seen_moves = moves_seen(game.believed, shown, FIRST, family)
            if seen_moves and seen_moves[0] == move:
                return True
            # The game may change elsewhere before a stable observation. This still requires every changed square
            # of the dispatched move to agree, including captures, transformations and any partner movement.
            if footprint and all(shown.get(sq) == held for sq, held in footprint.items()):
                return True
        await asyncio.sleep(0.15)
    return False


def _landing(move: Move) -> tuple[int, int]:
    """The first square a move lands on, which is what a game offers for it: for a take over again and again, the
    first landing, not the end."""
    return move.via[0] if move.via else move.end


async def _by_hand(seat: Seat, click: Click, picture: Any, move: Move, wait_s: float) -> bool:
    """A move made as a person makes it: the piece picked up, then each square it lands on in turn."""
    if not await click(*seat.centre_share(move.start, picture)):
        return False
    for square in (*move.via, move.end):
        await asyncio.sleep(wait_s)
        if not await click(*seat.centre_share(square, picture)):
            return False
    return True


def _a_place(position: Position, start: dict | None = None, family: Family | None = None) -> str:
    """A position as a situation she may come back to: what stands on each square and whose turn it is, and, where
    its history is not the ordinary one, the history that changes what may happen next.

    Two situations with the same pieces can differ in what may be done from them: a piece standing where it began may
    have moved and come back, and a square may be passable for one move. A situation known by its pieces alone shares
    a name with one it is not, and what was worked out or seen in one is used in the other (suggestions of 23 November
    2026). The ordinary history, given ``start``, is every piece standing on its own starting square unmoved, no other
    unmoved, and nothing passable that a move of the side to move can take in passing (given ``family``; without it,
    anything passable counts): there the name is the pieces' alone, as it has always been, and everything kept under
    it stays hers. Any other history is named."""
    placed = ";".join(f"{r},{c}:{side}{kind}" for (r, c), (side, kind) in sorted(position.cells.items())) + \
        f"|{position.turn}"
    if start is None:
        return placed
    ordinary = {square for square, held in position.cells.items() if start.get(square) == held}
    passable = position.passable is not None and (
        family is None or any(m.taken_at not in (None, m.end) and not m.via for m in moves(family, position)))
    if set(position.unmoved) == ordinary and not passable:
        return placed
    unmoved = ",".join(f"{r}.{c}" for r, c in sorted(position.unmoved))
    passing = f"{position.passable[0][0]}.{position.passable[0][1]}" if passable else ""
    return f"{placed}|unmoved:{unmoved}|passable:{passing}"


def _named(move: Move) -> str:
    return f"{move.start[0]},{move.start[1]}-{move.end[0]},{move.end[1]}{move.becomes}"


def _judged(game: _Game, place: str, move: Move, found: Any) -> None:
    """A turn she searched, kept for finding where a lost game went wrong: only a judgement not yet decided. A fall
    into a mate found is the end being seen, not the answer she least saw coming: LIVE 21 November 2026, game 15, it
    ruled out the move before the mate, at a place no game comes back to, where her judgement had fallen from even to
    4.5 pieces behind at move 9."""
    if not getattr(found, "mate", False):
        game.judged.append((place, _named(move), int(found.score)))


#: Her moves in a lost game suspected of where it went wrong, at most: those after which her judgement fell furthest.
SUSPECTS_A_GAME = 3


def _where_it_went_wrong(game: _Game) -> None:
    """In a game she lost, the moves after which her own judgement fell furthest by her next turn, suspected of where
    it went wrong (WhatFailedWhere): looked into between games, by a longer look at the move against what else was
    open there, before any is ruled out. A fall after a move is a reason to look, not proof the move was the mistake
    (suggestions of 23 November 2026): the move that lost may come earlier, or there may have been nothing better.
    Nothing where her judgement never fell."""
    drops = sorted(((game.judged[i][2] - game.judged[i + 1][2], i) for i in range(len(game.judged) - 1)),
                   reverse=True)
    for drop, at in drops[:SUSPECTS_A_GAME]:
        if drop <= 0:
            break
        place, move, _score = game.judged[at]
        game.went_wrong.suspect(place, move, drop)
        logger.info("suspected of where it went wrong: her move %s, after which she judged herself %d lower by her "
                    "next turn", move, drop)


def _position_from_place(place: str, start: dict) -> Position | None:
    """A situation read back from its name, given where the pieces began: the pieces and whose turn it is, and the
    history its name holds where it is not the ordinary one (``_a_place``), the ordinary one where it is."""
    try:
        parts = place.split("|")
        cells: dict[tuple[int, int], tuple[int, str]] = {}
        for one in filter(None, parts[0].split(";")):
            square, held = one.split(":", 1)
            row, column = square.split(",")
            cells[(int(row), int(column))] = (int(held[0]), held[1:])
        turn = int(parts[1])
        named = {key: value for key, value in (part.split(":", 1) for part in parts[2:])}
        if "unmoved" in named:
            unmoved = frozenset(tuple(int(v) for v in one.split(".")) for one in filter(None, named["unmoved"].split(",")))
        else:
            unmoved = frozenset(square for square, held in cells.items() if start.get(square) == held)
        passable = None
        if named.get("passable"):
            row, column = (int(v) for v in named["passable"].split("."))
            # The square passed over, and the square of the piece that passed: next to it, away from its own side.
            beyond = row + (1 if cells.get((row + 1, column), (None,))[0] is not None else -1)
            passable = ((row, column), (beyond, column))
        return Position(cells=cells, turn=turn, unmoved=unmoved, passable=passable)
    # not a failure: a name that does not read back as a situation is one she cannot look at again.
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _the_untaken_were_refused(game: _Game) -> None:
    """The board has answered after the moves it did not take and the pick-ups it offered nothing: it was answering,
    and those were its answers."""
    for at, square, rules in game.unshown:
        game.learned.offered(at, square, set(), family=rules)
    for at, move, rules in game.untaken:
        game.learned.refused(at, move, family=rules)
    game.unshown.clear()
    game.untaken.clear()


async def _finish_game(ended: str, game: _Game, tell: Callable[[str], None],
                       read_end: EndReading | None) -> dict[str, Any]:
    observed = await read_end() if read_end is not None else None
    return _the_end(ended, game, tell, observed=observed)


def _the_end(ended: str, game: _Game, tell: Callable[[str], None], *,
             observed: tuple[str, str] | None = None) -> dict[str, Any]:
    if game.untaken or game.unshown:
        # The board answered nothing after these: it had stopped answering, and they say nothing of the rules.
        logger.info("%d move(s) the board did not take and %d pick-up(s) it offered nothing, before it answered nothing "
                    "more, are not learned: %s", len(game.untaken), len(game.unshown),
                    ", ".join(_named(move) for _at, move, _rules in game.untaken))
        game.untaken.clear()
        game.unshown.clear()
    if observed is not None:
        result, evidence = observed
        if result in ("won", "lost", "drawn") and evidence:
            tell({"won": "The window says I won.", "lost": "The window says I lost.",
                  "drawn": "The window says it was drawn."}[result])
            if result == "lost":
                _where_it_went_wrong(game)
            if ended == "no move":
                game.learned.the_window_said(result, game.believed, game.last_followed, game.earlier)
        else:
            result = "result unconfirmed"
            tell("I couldn't get another move to register or verify a result. I'll leave the board open.")
    elif ended.startswith("won:"):
        result = "won" if int(ended.split(":", 1)[1]) == FIRST else "lost"
        tell("I won." if result == "won" else "I lost. They found the better line.")
        if result == "lost":
            _where_it_went_wrong(game)
    elif ended == "no move":
        result = "result unconfirmed"
        tell("I couldn't get another move to register or verify a result.")
    else:
        result = "drawn"
        tell(f"It's a draw ({ended.split(':', 1)[-1].strip()}).")
    if game.foreseeable:
        logger.info("she foresaw %d of their %d answers", game.foreseen, game.foreseeable)
    ended_with = {"ended": result, "moves": game.made, "theirs": game.theirs, "learned": game.learned,
                  "foresaw": (game.foreseen, game.foreseeable), "final": game.believed, "last_move": game.last_followed,
                  "acts": game.did.as_record(), "held": game.held,
                  # Every position she held before the last, for what the window says of how the game ended.
                  "positions": list(game.earlier)}
    if observed is not None and result != "result unconfirmed":
        ended_with["read_from_the_window"] = observed[1]
    outcome = {"won": 1.0, "lost": 0.0, "drawn": 0.5}.get(result)
    if outcome is not None:
        game.held.came_to([score for _place, _move, score in game.judged], outcome)
        error = game.held.error()
        if error is not None:
            logger.info("what her judgements of this board come to: %d games, error %.3f, %.3f on games each was not "
                        "fitted on", game.held.undertakings, error, game.held.held_out or float("nan"))
    if game.learned.shown is not None:
        # Her rules as they stand, checked against every answer the game has given: what they would have it offer that
        # it did not, and what it offered that they would not. Both should be none.
        checked = game.learned.shown.checked(game.learned.family())
        logger.info("her rules against what the game showed: %s", checked.says())
        ended_with["rules_checked"] = checked.as_dict()
    return ended_with
