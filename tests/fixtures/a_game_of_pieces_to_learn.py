"""Games of pieces she is never told, for tests: each is the hidden world a test plays her against.

Her runtime holds no game's rules (core/agency/pieces_by_their_kind.py is a vocabulary, and
core/agency/how_the_pieces_move.py learns a game into it). A test needs a world that knows its own rules, as an
application does, to offer squares, refuse moves and answer: these are those worlds. Chess is here because published
move counts (perft) check that the vocabulary and the move maker can hold a real game exactly; the others are games
nobody plays, so a learner that has quietly come to know chess shows it.
"""
from __future__ import annotations

from core.agency.pieces_by_their_kind import FIRST, SECOND, Family, Kind, Position, Step, all_ways

_ORTHOGONAL = all_ways((1, 0))
_DIAGONAL = all_ways((1, 1))


def _start(rows: int, back: str, front: str) -> dict:
    cells = {}
    for column, letter in enumerate(back):
        if letter != ".":
            cells[(0, column)] = (FIRST, letter)
            cells[(rows - 1, column)] = (SECOND, letter)
    for column, letter in enumerate(front):
        if letter != ".":
            cells[(1, column)] = (FIRST, letter)
            cells[(rows - 2, column)] = (SECOND, letter)
    return cells


CHESS = Family(
    name="chess", rows=8, columns=8,
    kinds={
        "P": Kind("P", name="pawn", steps=(
            Step(((1, 0),), may="moves"),
            Step(((1, 0),), rides=2, may="moves", only_unmoved=True, leaves_passable=True),
            Step(((1, -1), (1, 1)), may="takes", takes_in_passing=True),
        ), becomes=("Q", "R", "B", "N"), cannot_go_back=True),
        "N": Kind("N", name="knight", steps=(Step(all_ways((1, 2))),)),
        "B": Kind("B", name="bishop", steps=(Step(_DIAGONAL, rides=0),)),
        "R": Kind("R", name="rook", steps=(Step(_ORTHOGONAL, rides=0),)),
        "Q": Kind("Q", name="queen", steps=(Step(_ORTHOGONAL + _DIAGONAL, rides=0),)),
        "K": Kind("K", name="king", steps=(Step(_ORTHOGONAL + _DIAGONAL),), royal=True, crosses_with="R"),
    },
    start=_start(8, "RNBQKBNR", "PPPPPPPP"),
    count_to_draw=100, repeats_to_draw=3,
)

#: A game nobody plays: on seven by seven, a piece that leaps (2, 2), one that rides straight but only two squares, a
#: forward stepper that takes straight ahead, and a royal that steps diagonally only.
NOBODYS_GAME = Family(
    name="nobody's game", rows=7, columns=7,
    kinds={
        "S": Kind("S", steps=(Step(((1, 0),)),), cannot_go_back=True),
        "A": Kind("A", steps=(Step(all_ways((2, 2))),)),
        "W": Kind("W", steps=(Step(_ORTHOGONAL, rides=2),)),
        "K": Kind("K", steps=(Step(_DIAGONAL),), royal=True),
    },
    start=_start(7, "WAWKWAW", ".SSSSS."),
)


#: Draughts as English players play it, for tests of the vocabulary's reach only: men on the dark squares step forward
#: on the diagonal and take by leaping over, again and again while they can; a man reaching the far row becomes a king,
#: which goes both ways; a side that can take must; a side with no move has lost.
_MAN_STEPS = (Step(((1, 1), (1, -1)), may="moves"), Step(((2, 2), (2, -2)), may="takes over"))
_KING_STEPS = (Step(((1, 1), (1, -1), (-1, 1), (-1, -1)), may="moves"),
               Step(((2, 2), (2, -2), (-2, 2), (-2, -2)), may="takes over"))
DRAUGHTS = Family(
    name="draughts", rows=8, columns=8,
    kinds={
        "M": Kind("M", name="man", steps=_MAN_STEPS, becomes=("K",), cannot_go_back=True, takes_again=True),
        "K": Kind("K", name="king", steps=_KING_STEPS, takes_again=True),
    },
    start={**{(r, c): (FIRST, "M") for r in range(3) for c in range(8) if (r + c) % 2 == 0},
           **{(r, c): (SECOND, "M") for r in range(5, 8) for c in range(8) if (r + c) % 2 == 0}},
    count_to_draw=80, no_move_unattacked="loses", takes_first=True,
)

def from_fen(fen: str) -> Position:
    """A chess position written the usual way, for the published counts."""
    board, turn, rights, passing = fen.split()[:4]
    cells = {}
    for i, line in enumerate(board.split("/")):
        row, column = 7 - i, 0
        for ch in line:
            if ch.isdigit():
                column += int(ch)
                continue
            cells[(row, column)] = (FIRST if ch.isupper() else SECOND, ch.upper())
            column += 1
    unmoved = {sq for sq, (side, k) in cells.items() if k == "P" and sq[0] == (1 if side == FIRST else 6)}
    for letter, squares in (("K", {(0, 4), (0, 7)}), ("Q", {(0, 4), (0, 0)}), ("k", {(7, 4), (7, 7)}),
                            ("q", {(7, 4), (7, 0)})):
        if letter in rights:
            unmoved |= squares
    passable = None
    if passing != "-":
        column, row = ord(passing[0]) - 97, int(passing[1]) - 1
        passable = ((row, column), (row - 1, column) if row == 5 else (row + 1, column))
    return Position(cells=cells, turn=FIRST if turn == "w" else SECOND, unmoved=frozenset(unmoved & set(cells)),
                    passable=passable)
