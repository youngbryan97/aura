"""A grid written into a request is read as the finish, whole, and met on the board.

Read clause by clause, "until it reads 1 2 3 / 4 5 6 / 7 8 _" finished at
"1", the first number in it, and no text on any screen says a layout is made.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.perception.what_is_there import Arrangement, Cell
from core.runtime.watched_goal import read_watched_goal
from core.utils.written_layout import written_layout

PUZZLE = "1 2 3 / 4 5 6 / 7 8 _"


@pytest.mark.parametrize(
    ("said", "layout"),
    [
        ("solve the puzzle until it reads 1 2 3 / 4 5 6 / 7 8 _", PUZZLE),
        ("1 2 3 / 4 5 6 / 7 8 _", PUZZLE),
        ("make it a b / c _ please", "a b / c _"),
        ("rows:\n1 2\n3 _", "1 2 / 3 _"),
    ],
)
def test_a_grid_is_read_out_of_the_words_around_it(said, layout):
    assert written_layout(said) == layout


@pytest.mark.parametrize(
    "said",
    ["play until 2048", "read and/or write the file", "open 1/2 of the page", "win the game"],
)
def test_text_that_is_not_a_grid_is_not_read_as_one(said):
    assert written_layout(said) == ""


def test_the_request_finishes_at_the_whole_layout_not_its_first_number():
    from core.runtime.watched_goal import _best_finishing_test

    said = "Play the sliding puzzle until it reads 1 2 3 / 4 5 6 / 7 8 _"
    assert _best_finishing_test(said) == PUZZLE


def test_and_a_number_is_still_the_finish_where_one_is_named():
    assert read_watched_goal("play it until 128").success_when == "128"


def _board(rows: list[list[str]]) -> Arrangement:
    cells = tuple(
        Cell(r, c, said, (0.0, 0.0)) for r, row in enumerate(rows) for c, said in enumerate(row) if said
    )
    return Arrangement(len(rows), len(rows[0]), cells, places_seen=True)


def test_the_live_loop_sees_a_made_layout_on_the_board():
    from core.skills.screen_pursuit_decision import _the_layout_is_made

    knows = SimpleNamespace(rules=None)
    made = _board([["1", "2", "3"], ["4", "5", "6"], ["7", "8", ""]])
    short = _board([["1", "2", "3"], ["4", "5", "6"], ["7", "", "8"]])
    assert _the_layout_is_made(PUZZLE, knows, made)
    assert not _the_layout_is_made(PUZZLE, knows, short)
    assert not _the_layout_is_made("2048", knows, made)


def test_and_a_made_layout_is_success_where_the_run_checks_for_it():
    import inspect

    from core.skills import screen_pursuit, screen_pursuit_decision

    assert 'pending.get("the_layout_is_made")' in inspect.getsource(screen_pursuit)
    body = inspect.getsource(screen_pursuit_decision)
    assert 'pending["the_layout_is_made"] = True' in body
    assert "a made layout is not moved out of" in body
