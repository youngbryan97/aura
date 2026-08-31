"""The board, out of the page it sits on.

LIVE 2026-08-31 on 2048game.com: she read the whole page as the thing —
forty-four columns by thirty-seven rows — so of thirty moves only five could be
compared with each other, the rule that governs the board scored nought out of
five, and every sentence she said about the position was narration over a
reading that was not of the board. She was playing correctly and learning
nothing from it.
"""

from __future__ import annotations

from core.perception.what_is_there import arranged, the_part_laid_out_regularly

#: A real reading, captured from 2048game.com.
THE_BOARD = [
    (229, 453, "2"), (342, 453, "4"), (456, 453, "16"), (569, 453, "2"),
    (229, 566, "8"), (342, 566, "8"), (456, 566, "8"),
    (229, 680, "4"), (342, 680, "16"), (229, 793, "64"), (569, 793, "2"),
]
THE_PAGE = [
    (14, 60, "When to meet"), (29, 55, "Spin The Wheel"),
    (44, 62, "Free Tetris Online"), (59, 42, "Poll Maker"), (33, 307, "2048"),
    (24, 490, "SCORE"), (38, 490, "0"), (24, 536, "BEST"), (38, 536, "460"),
    (75, 300, "Play 2048 Game Online"), (89, 296, "Join the numbers"),
    (82, 517, "New Game"), (14, 640, "Language"), (405, 400, "Advert"),
    (430, 410, "Book Now"),
]


def test_the_board_is_found_in_the_page_around_it():
    kept = the_part_laid_out_regularly(THE_PAGE + THE_BOARD)
    assert set(kept) == set(THE_BOARD)


def test_and_it_then_reads_as_the_shape_it_is():
    board = arranged(the_part_laid_out_regularly(THE_PAGE + THE_BOARD))
    assert (board.rows, board.columns) == (4, 4)
    grid = [["."] * 4 for _ in range(4)]
    for cell in board.cells:
        grid[cell.row][cell.column] = cell.says
    assert grid[0] == ["2", "8", "4", "64"]
    assert grid[3] == ["2", ".", ".", "2"]


def test_read_whole_the_page_is_not_a_board():
    """What she was doing before: the thing and its furniture as one grid."""
    whole = arranged(THE_PAGE + THE_BOARD)
    assert (whole.rows, whole.columns) != (4, 4)


def test_neither_axis_decides_alone():
    """The strongest rhythm among the x positions is a spurious ninety-three
    shared by three pieces of furniture, beating the board's own hundred and
    thirteen. It is the pair of axes, scored by how full the block is, that
    picks the board out."""
    across = sorted({x for _y, x, _said in the_part_laid_out_regularly(THE_PAGE + THE_BOARD)})
    assert across == [453, 566, 680, 793]


def test_a_reading_with_no_block_in_it_is_left_alone():
    """Cropping to nothing is not a reading."""
    prose = [(10, 20, "one"), (40, 90, "two"), (95, 300, "three"), (200, 55, "four")]
    assert set(the_part_laid_out_regularly(prose)) == set(prose)


def test_too_little_to_judge_is_left_alone():
    few = [(10, 20, "a"), (10, 40, "b")]
    assert set(the_part_laid_out_regularly(few)) == set(few)


def test_a_wider_thing_is_found_the_same_way():
    """Nothing here knows what a board is: a twelve-column timetable is found
    by the same rule, with no number written for either."""
    timetable = [
        (100 + 30 * row, 200 + 45 * column, f"{row}{column}")
        for row in range(5)
        for column in range(12)
    ]
    noise = [(7, 13, "title"), (900, 17, "footer"), (33, 611, "link")]
    kept = the_part_laid_out_regularly(timetable + noise)
    assert set(kept) == set(timetable)
    laid = arranged(kept)
    assert (laid.rows, laid.columns) == (5, 12)
