"""Room to move as a thing to want, and taking somebody else's away.

The position at the end of the recorded game: four pieces on the near back
row, never moved, one crowned piece loose in the middle, and the other side
with eight. Losing on material by three, and won — because a man that only
moves forward, facing a full row, with no empty square behind it to land on,
has nothing it may legally do.

Nothing in the module knows any of that. It is handed a way of acting and
counts what each side can still bring about.
"""

from __future__ import annotations

from core.cognition.what_is_still_open import (
    acts_by_what_they_leave_open,
    what_is_still_open,
)

SIDE = 8
MINE, THEIRS = "mine", "theirs"


def _board(pieces: dict[tuple[int, int], tuple[str, bool]]) -> frozenset:
    return frozenset((row, col, who, crowned) for (row, col), (who, crowned) in pieces.items())


def _at(board: frozenset) -> dict[tuple[int, int], tuple[str, bool]]:
    return {(row, col): (who, crowned) for row, col, who, crowned in board}


def _every_act() -> list[tuple[int, int, int, int]]:
    acts = []
    for row in range(SIDE):
        for col in range(SIDE):
            for down in (-1, 1):
                for right in (-1, 1):
                    for reach in (1, 2):
                        to_row, to_col = row + down * reach, col + right * reach
                        if 0 <= to_row < SIDE and 0 <= to_col < SIDE:
                            acts.append((row, col, to_row, to_col))
    return acts


ACTS = _every_act()


def _stepper(who: str, forward: int):
    """Ordinary draughts movement, written once and handed over as a rule."""

    def step(board: frozenset, act: tuple[int, int, int, int]) -> frozenset | None:
        row, col, to_row, to_col = act
        here = _at(board)
        mine = here.get((row, col))
        if mine is None or mine[0] != who:
            return None
        if (to_row, to_col) in here:
            return None
        crowned = mine[1]
        going = to_row - row
        if not crowned and (going > 0) != (forward > 0):
            return None
        if abs(to_row - row) == 1:
            after = dict(here)
            del after[(row, col)]
            after[(to_row, to_col)] = (who, crowned or to_row in (0, SIDE - 1))
            return _board(after)
        over = ((row + to_row) // 2, (col + to_col) // 2)
        jumped = here.get(over)
        if jumped is None or jumped[0] == who:
            return None
        after = dict(here)
        del after[(row, col)]
        del after[over]
        after[(to_row, to_col)] = (who, crowned or to_row in (0, SIDE - 1))
        return _board(after)

    return step


MINE_MOVES = _stepper(MINE, forward=-1)
THEIR_MOVES = _stepper(THEIRS, forward=+1)


def test_a_full_back_row_leaves_them_nothing_to_do() -> None:
    board = _board(
        {
            (7, 0): (MINE, False),
            (7, 2): (MINE, False),
            (7, 4): (MINE, False),
            (7, 6): (MINE, False),
            (5, 4): (MINE, True),
            (6, 1): (THEIRS, False),
            (6, 3): (THEIRS, False),
            (6, 5): (THEIRS, False),
            (6, 7): (THEIRS, False),
        }
    )
    open_ = what_is_still_open(
        board,
        acts=ACTS,
        step=MINE_MOVES,
        their_acts=ACTS,
        their_step=THEIR_MOVES,
        named=lambda one: frozenset(one),
    )
    assert open_.nothing_left_for_them, open_.describe()
    assert open_.hers > 0
    # Down on material and winning, which is the point.
    at = _at(board)
    assert sum(1 for who, _ in at.values() if who == MINE) == 5
    assert sum(1 for who, _ in at.values() if who == THEIRS) == 4


def test_she_will_give_a_piece_away_to_keep_what_she_is_winning_with() -> None:
    """A sacrifice is not a bad move, it is a move that costs the wrong thing."""
    board = _board(
        {
            (7, 0): (MINE, False),
            (7, 2): (MINE, False),
            (7, 4): (MINE, False),
            (7, 6): (MINE, False),
            (4, 3): (MINE, False),
            (5, 4): (MINE, True),
            (3, 2): (THEIRS, False),
            (6, 1): (THEIRS, False),
            (6, 5): (THEIRS, False),
        }
    )

    def back_row_whole(after: frozenset) -> bool:
        at = _at(after)
        return all(at.get((7, col), ("", False))[0] == MINE for col in (0, 2, 4, 6))

    ranked = acts_by_what_they_leave_open(
        board,
        acts=ACTS,
        step=MINE_MOVES,
        their_acts=ACTS,
        their_step=THEIR_MOVES,
        named=lambda one: frozenset(one),
        keeps=back_row_whole,
    )
    assert ranked
    best, _ = ranked[0]
    after = MINE_MOVES(board, best)
    assert after is not None and back_row_whole(after)
    # Every act that breaks the held thing sorts below every act that keeps it.
    kept = [act for act, _ in ranked if back_row_whole(MINE_MOVES(board, act))]
    assert [act for act, _ in ranked][: len(kept)] == kept


def test_it_counts_where_it_can_get_to_not_how_many_moves_it_has() -> None:
    """Twenty moves that all lead to one place is one option, not twenty."""

    def nowhere(state: int, act: int) -> int:
        return 0

    same = what_is_still_open(1, acts=list(range(20)), step=nowhere, named=int)
    assert same.hers == 1


def test_it_is_not_about_boards() -> None:
    """A thing with doors. Shutting theirs is the same reasoning."""
    doors = {"hall": ("kitchen", "study"), "kitchen": ("hall",), "study": ("hall",)}

    def walk(where: str, door: str) -> str | None:
        return door if door in doors.get(where, ()) else None

    open_ = what_is_still_open(
        "hall", acts=list(doors), step=walk, their_acts=["hall"], their_step=walk, ahead=2
    )
    assert open_.hers == 3
    assert open_.theirs == 0
