"""Games of pieces that move by their kind: the vocabulary she holds how a game's pieces move in, and its moves.

However a game of pieces on a grid goes, each kind moves in a few ways:

* a leap: to a square at a fixed offset, whatever stands between;
* a ride: along a line, square after square, until something stands in the way, as far as it goes or a set number;
* a step that only moves, or only takes;
* a step only a piece that has not moved yet may make;
* a step that leaves the square it passed over open to be taken on, for one move, and the steps that may take there;
* a royal kind: a side may not leave it where it can be taken, and a side that cannot keep it safe has lost;
* a royal that, unmoved, crosses with an unmoved partner;
* a kind that, reaching the far row, becomes another.

Which of these a game's kinds have is not written here for any game. She learns it from the game itself: the squares
the game offers when she picks a piece up, the moves she watches made, the moves the game refuses
(core/agency/how_the_pieces_move.py). What she has learned is a ``Family`` in this vocabulary, and this module makes its
moves, so her search (core/agency/searching_by_kind.py) looks ahead by the rules as she understands them. Where they
are wrong the game says so, and she learns again.

Squares are (row, column) in the first side's frame: row 0 is the first side's back row, column 0 its left.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

__all__ = ["FIRST", "SECOND", "Family", "Kind", "Move", "Position", "Step", "after", "all_ways", "in_check", "moves",
           "outcome"]

Square = tuple[int, int]
#: Each side's forward direction in the rules' frame: the first side moves toward higher rows.
FIRST, SECOND = 0, 1


def all_ways(*offsets: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Every turn and reflection of the offsets given: a knight's (1, 2) gives its eight."""
    found: list[tuple[int, int]] = []
    for f, s in offsets:
        for a, b in ((f, s), (s, f)):
            for one in ((a, b), (-a, b), (a, -b), (-a, -b)):
                if one not in found:
                    found.append(one)
    return tuple(found)


@dataclass(frozen=True)
class Step:
    """One way a kind moves: by offsets of (forward, rightward) as its own side faces, once or along a line."""

    offsets: tuple[tuple[int, int], ...]
    #: How many times the offset repeats: 1 a leap, 0 as far as the way is clear, 2 at most twice.
    rides: int = 1
    #: "either": to an empty square or taking; "moves": only to an empty square; "takes": only taking.
    may: str = "either"
    only_unmoved: bool = False
    #: The squares it passes over may be taken on by the other side's very next move, as if it stood there.
    leaves_passable: bool = False
    #: It may take on a passable square, taking the piece that passed over it.
    takes_in_passing: bool = False


@dataclass(frozen=True)
class Kind:
    letter: str
    steps: tuple[Step, ...]
    royal: bool = False
    #: The kinds it may become on the far row; () where it never does.
    becomes: tuple[str, ...] = ()
    #: A move of it, like a take, starts the count toward a draw again: it cannot be undone.
    cannot_go_back: bool = False
    #: The kind of partner a royal crosses with, both unmoved, the way between them empty and not attacked.
    crosses_with: str = ""
    #: What its players call it.
    name: str = ""


@dataclass(frozen=True)
class Family:
    name: str
    rows: int
    columns: int
    kinds: dict[str, Kind]
    #: Where each piece stood at the start: square to (side, kind).
    start: dict[Square, tuple[int, str]] = field(default_factory=dict)
    #: Moves without a take or an irreversible move after which the game is drawn; 0 where there is no such count.
    count_to_draw: int = 0
    repeats_to_draw: int = 0
    #: A side with no move, its royal not attacked: "draw", or "loses".
    no_move_unattacked: str = "draw"

    def kind(self, letter: str) -> Kind:
        return self.kinds[letter]

    def the_start(self) -> Position:
        return Position(cells=dict(self.start), turn=FIRST, unmoved=frozenset(self.start))


@dataclass(frozen=True)
class Move:
    start: Square
    end: Square
    becomes: str = ""
    #: The square of the piece taken, where it is not ``end`` (in passing); None where the move takes nothing there.
    taken_at: Square | None = None
    #: The partner's start and end, for a royal crossing.
    partner: tuple[Square, Square] | None = None


@dataclass(frozen=True)
class Position:
    cells: dict[Square, tuple[int, str]]
    turn: int
    unmoved: frozenset[Square] = frozenset()
    #: A square that may be taken on in passing this move, and the square of the piece that passed over it.
    passable: tuple[Square, Square] | None = None
    count: int = 0
    seen: tuple[tuple, ...] = field(default=(), compare=False)

    def key(self) -> tuple:
        return tuple(sorted(self.cells.items())), self.turn, tuple(sorted(self.unmoved)), self.passable


def _forward(side: int) -> int:
    return 1 if side == FIRST else -1


def _inside(family: Family, square: Square) -> bool:
    return 0 <= square[0] < family.rows and 0 <= square[1] < family.columns


def _reach(family: Family, position: Position, square: Square, side: int, kind: Kind,
           only_takes: bool = False) -> Iterable[tuple[Square, Step, list[Square]]]:
    """The squares a piece of ``kind`` of ``side`` on ``square`` reaches by its steps, each with the step and the
    squares it passed over."""
    ahead = _forward(side)
    for step in kind.steps:
        if step.only_unmoved and square not in position.unmoved:
            continue
        if only_takes and step.may == "moves":
            continue
        for forward, right in step.offsets:
            dr, dc = forward * ahead, right * ahead
            passed: list[Square] = []
            r, c = square
            times = 0
            while True:
                r, c = r + dr, c + dc
                times += 1
                if not _inside(family, (r, c)):
                    break
                yield (r, c), step, list(passed)
                if (r, c) in position.cells:
                    break
                passed.append((r, c))
                if step.rides and times >= step.rides:
                    break


def attacked(family: Family, position: Position, square: Square, by: int) -> bool:
    """Whether a piece of side ``by`` could take on ``square``."""
    for there, (side, letter) in position.cells.items():
        if side != by:
            continue
        for end, step, _passed in _reach(family, position, there, side, family.kind(letter), only_takes=True):
            if end == square and step.may != "moves":
                return True
    return False


def _royal_of(family: Family, position: Position, side: int) -> list[Square]:
    return [sq for sq, (s, letter) in position.cells.items() if s == side and family.kind(letter).royal]


def in_check(family: Family, position: Position, side: int | None = None) -> bool:
    side = position.turn if side is None else side
    return any(attacked(family, position, sq, 1 - side) for sq in _royal_of(family, position, side))


def _possible(family: Family, position: Position) -> list[Move]:
    side = position.turn
    far = family.rows - 1 if side == FIRST else 0
    found: list[Move] = []
    for square, (owner, letter) in list(position.cells.items()):
        if owner != side:
            continue
        kind = family.kind(letter)
        for end, step, _passed in _reach(family, position, square, side, kind):
            there = position.cells.get(end)
            taken_at = None
            if there is not None:
                if there[0] == side or step.may == "moves":
                    continue
                taken_at = end
            elif step.may == "takes":
                if not (step.takes_in_passing and position.passable and position.passable[0] == end):
                    continue
                taken_at = position.passable[1]
            if kind.becomes and end[0] == far:
                found.extend(Move(square, end, becomes=b, taken_at=taken_at) for b in kind.becomes)
            else:
                found.append(Move(square, end, taken_at=taken_at))
        if kind.crosses_with and square in position.unmoved:
            found.extend(_crossings(family, position, square, kind))
    unique: dict[tuple, Move] = {}
    for move in found:
        unique.setdefault((move.start, move.end, move.becomes, move.partner), move)
    return list(unique.values())


def _crossings(family: Family, position: Position, square: Square, kind: Kind) -> list[Move]:
    """A royal and an unmoved partner on its row: the royal moves two squares toward it, the partner to the square it
    crossed, if the way between is empty and the royal neither stands, passes nor lands where it could be taken."""
    side = position.turn
    found: list[Move] = []
    row = square[0]
    for partner_at, (owner, letter) in position.cells.items():
        if owner != side or letter != kind.crosses_with or partner_at[0] != row or partner_at not in position.unmoved:
            continue
        way = 1 if partner_at[1] > square[1] else -1
        between = range(square[1] + way, partner_at[1], way)
        if any((row, c) in position.cells for c in between):
            continue
        end = (row, square[1] + 2 * way)
        if not _inside(family, end):
            continue
        crossed = [(row, square[1]), (row, square[1] + way), end]
        if any(attacked(family, position, sq, 1 - side) for sq in crossed):
            continue
        found.append(Move(square, end, partner=(partner_at, (row, square[1] + way))))
    return found


def after(family: Family, position: Position, move: Move) -> Position:
    """The position a move makes."""
    cells = dict(position.cells)
    side, letter = cells.pop(move.start)
    took = move.taken_at is not None
    if took:
        cells.pop(move.taken_at, None)
    cells[move.end] = (side, move.becomes or letter)
    unmoved = set(position.unmoved) - {move.start, move.end}
    if move.taken_at is not None:
        unmoved.discard(move.taken_at)
    if move.partner is not None:
        start, end = move.partner
        cells[end] = cells.pop(start)
        unmoved.discard(start)
    passable = None
    kind = family.kind(letter)
    distance = max(abs(move.end[0] - move.start[0]), abs(move.end[1] - move.start[1]))
    if distance > 1 and move.partner is None and any(s.leaves_passable for s in kind.steps):
        dr = (move.end[0] - move.start[0]) // distance
        dc = (move.end[1] - move.start[1]) // distance
        passable = ((move.start[0] + dr, move.start[1] + dc), move.end)
    count = 0 if (took or kind.cannot_go_back) else position.count + 1
    made = Position(cells=cells, turn=1 - side, unmoved=frozenset(unmoved), passable=passable, count=count,
                    seen=(*position.seen, position.key()))
    return made


def moves(family: Family, position: Position) -> list[Move]:
    """Every move the side to move may make: its possible moves that do not leave its royal where it can be taken."""
    side = position.turn
    legal = []
    for move in _possible(family, position):
        made = after(family, position, move)
        if not in_check(family, made, side):
            legal.append(move)
    return legal


def outcome(family: Family, position: Position, legal: list[Move] | None = None) -> str:
    """"" while the game goes on; "won:0", "won:1" for the side that won; "drawn" and why."""
    legal = moves(family, position) if legal is None else legal
    if not legal:
        if in_check(family, position):
            return f"won:{1 - position.turn}"
        return "drawn: no move" if family.no_move_unattacked == "draw" else f"won:{1 - position.turn}"
    if family.count_to_draw and position.count >= family.count_to_draw:
        return "drawn: the count"
    if family.repeats_to_draw and sum(1 for k in position.seen if k == position.key()) + 1 >= family.repeats_to_draw:
        return "drawn: repeated"
    return ""


def with_turn(position: Position, turn: int) -> Position:
    return replace(position, turn=turn)
