"""One game with two sides, so the reasoning under test has something to act in.

Ordinary draughts, written once. Nothing here is imported by anything under
``core``; it exists so that a claim about reasoning can be checked against a
world with real rules rather than against a mock that agrees with it.
"""

from __future__ import annotations

SIDE = 8
MINE, THEIRS = "mine", "theirs"

Board = frozenset
Act = tuple[int, int, int, int]


def board_of(pieces: dict[tuple[int, int], tuple[str, bool]]) -> Board:
    return frozenset(
        (row, col, who, crowned) for (row, col), (who, crowned) in pieces.items()
    )


def pieces_on(board: Board) -> dict[tuple[int, int], tuple[str, bool]]:
    return {(row, col): (who, crowned) for row, col, who, crowned in board}


def every_act() -> list[Act]:
    acts: list[Act] = []
    for row in range(SIDE):
        for col in range(SIDE):
            for down in (-1, 1):
                for right in (-1, 1):
                    for reach in (1, 2):
                        to_row, to_col = row + down * reach, col + right * reach
                        if 0 <= to_row < SIDE and 0 <= to_col < SIDE:
                            acts.append((row, col, to_row, to_col))
    return acts


ACTS = every_act()


def stepper(who: str, forward: int):
    def step(board: Board, act: Act) -> Board | None:
        row, col, to_row, to_col = act
        here = pieces_on(board)
        mine = here.get((row, col))
        if mine is None or mine[0] != who:
            return None
        if (to_row, to_col) in here:
            return None
        crowned = mine[1]
        if not crowned and ((to_row - row) > 0) != (forward > 0):
            return None
        after = dict(here)
        if abs(to_row - row) == 2:
            over = ((row + to_row) // 2, (col + to_col) // 2)
            jumped = here.get(over)
            if jumped is None or jumped[0] == who:
                return None
            del after[over]
        del after[(row, col)]
        after[(to_row, to_col)] = (who, crowned or to_row in (0, SIDE - 1))
        return board_of(after)

    return step


MINE_MOVES = stepper(MINE, forward=-1)
THEIR_MOVES = stepper(THEIRS, forward=+1)


def parts_of(board: Board) -> list[tuple[int, int, str, bool]]:
    return sorted(board)


def without(board: Board, part: tuple[int, int, str, bool]) -> Board:
    return board - {part}


def hers(board: Board) -> list:
    return [one for one in board if one[2] == MINE]


def theirs(board: Board) -> list:
    return [one for one in board if one[2] == THEIRS]


def back_row_whole(board: Board) -> bool:
    at = pieces_on(board)
    return all(at.get((SIDE - 1, col), ("", False))[0] == MINE for col in (0, 2, 4, 6))
