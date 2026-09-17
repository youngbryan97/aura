"""Each move is said as it is made: what it does, and why that one.

Asked to narrate every move, what she had to say for most of them was "Going
left" and "Going up" — a log of keystrokes. The rule she learned says what each
move makes of the board and her measure says why it beat the next best, so the
line is read off those.
"""
from __future__ import annotations

from core.agency.saying_what_a_move_does import _a, what_a_move_does, why_this_one
from core.perception.how_it_moves import shifted_and_combined
from core.perception.what_is_there import Arrangement, Cell


def _board(values: list[int]) -> Arrangement:
    return Arrangement(
        4, 4, tuple(Cell(i // 4, i % 4, str(v), (0.0, 0.0)) for i, v in enumerate(values) if v)
    )


def test_what_came_together_is_said_with_where_it_went():
    before = _board([2, 2, 0, 0] + [0] * 12)
    line = what_a_move_does(before, "left", shifted_and_combined(before, "left"))
    assert line == "Left — two 2s make a 4 in the top-left corner."


def test_two_joins_in_one_push_are_both_counted():
    """Two 2s into a 4 and two 4s into an 8 at once leave the count of 4s unchanged."""
    before = _board([2, 2, 4, 4] + [0] * 12)
    line = what_a_move_does(before, "left", shifted_and_combined(before, "left"))
    assert "two 4s make an 8" in line
    assert "two 2s make a 4" in line


def test_many_joins_are_summed_up_rather_than_listed():
    before = _board([4, 4, 2, 2, 8, 8, 8, 8, 0, 0, 0, 0, 16, 0, 0, 16])
    line = what_a_move_does(before, "left", shifted_and_combined(before, "left"))
    assert "plus 3 more pairs" in line


def test_a_move_that_joins_nothing_says_why_it_was_chosen():
    before = _board([2, 4, 0, 0] + [0] * 12)
    because = why_this_one(
        {"room": 0.5, "order": 0.9}, {"room": 0.5, "order": 0.6}, {"room": 1.0, "order": 1.0},
        runner_up_name="up",
    )
    line = what_a_move_does(before, "right", shifted_and_combined(before, "right"), because=because)
    assert line == "Right — it keeps things more in order than up would."


def test_nothing_to_say_but_the_move_is_just_the_move():
    before = _board([2, 4, 0, 0] + [0] * 12)
    assert what_a_move_does(before, "down", shifted_and_combined(before, "down")) == "Down."


def test_a_nearly_full_board_says_how_little_room_is_left():
    before = _board([2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 0, 0])
    line = what_a_move_does(before, "right", shifted_and_combined(before, "right"))
    assert "only 2 places left" in line


def test_the_reason_is_the_term_that_separated_them_most_by_weight():
    said = why_this_one(
        {"room": 0.9, "smoothness": 0.5}, {"room": 0.1, "smoothness": 0.4}, {"room": 0.1, "smoothness": 1.0}
    )
    # 0.8 * 0.1 for room against 0.1 * 1.0 for smoothness.
    assert said == "it keeps neighbours close in value"


def test_numbers_take_the_article_they_are_said_with():
    assert [_a(n) for n in ("8", "11", "18", "16", "80", "1024", "2048", "11000")] == [
        "an 8", "an 11", "an 18", "a 16", "an 80", "a 1024", "a 2048", "an 11000",
    ]
