"""A goal is measurable when it is a layout to make, as well as when it is a number to reach.

Asked to put a puzzle in order, every future used to be equally far from
done, and her search ranked them by everything except what she was asked.
"""

from __future__ import annotations

from core.agency.what_she_is_after import goal_in
from core.perception.what_is_there import Arrangement, Cell


def board(rows: list[list[str]]) -> Arrangement:
    cells = tuple(
        Cell(r, c, said, (0.0, 0.0))
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    )
    return Arrangement(len(rows), len(rows[0]), cells, places_seen=True)


SOLVED = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", ""]]
ONE_AWAY = [["1", "2", "3"], ["4", "5", "6"], ["7", "", "8"]]
SCRAMBLED = [["8", "7", "6"], ["5", "", "4"], ["3", "2", "1"]]
PUZZLE = "1 2 3 / 4 5 6 / 7 8 _"


def test_a_number_is_still_a_number():
    goal = goal_in("2,048")
    assert goal.number == 2048.0 and not goal.layout
    assert goal.nearness(board([["1024", "2"]])) < 1.0
    assert goal.reached(board([["2048", "2"]]))


def test_a_layout_written_as_rows_is_read_as_one():
    goal = goal_in(PUZZLE)
    assert goal.layout == (("1", "2", "3"), ("4", "5", "6"), ("7", "8", ""))
    assert goal.names_something()


def test_a_layout_can_be_written_one_row_to_a_line():
    assert goal_in("1 2 3\n4 5 6\n7 8 _").layout == goal_in(PUZZLE).layout


def test_the_layout_itself_is_the_goal_reached():
    goal = goal_in(PUZZLE)
    assert goal.nearness(board(SOLVED)) == 1.0
    assert goal.reached(board(SOLVED))


def test_one_step_away_is_nearer_than_scrambled_and_not_there():
    goal = goal_in(PUZZLE)
    near, far = goal.nearness(board(ONE_AWAY)), goal.nearness(board(SCRAMBLED))
    assert far < near < 1.0
    assert not goal.reached(board(ONE_AWAY))


def test_a_board_of_another_shape_is_nowhere_near():
    assert goal_in(PUZZLE).nearness(board([["1", "2"], ["3", ""]])) == 0.0


def test_words_that_name_nothing_measurable_say_so():
    goal = goal_in("win the game")
    assert not goal.names_something()
    assert goal.nearness(board(SOLVED)) == 0.0
