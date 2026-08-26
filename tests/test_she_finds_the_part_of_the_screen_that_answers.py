"""The part of the screen that answers to her is the part her task is about.

A reading of a screen is everything on it. On the page holding a game that is
the board, the score, two advertising rails, a cookie footer and a copyright
line — so what she recalls about "a situation like this one" is dominated by
whichever advertisement was loaded, and two readings of the same board look
like different situations because the advertising rotated under her.

Nothing on the page says which part is the task. Something about her does.
"""
from __future__ import annotations

from core.perception.where_it_responds import (
    ENOUGH_ACTS,
    Responsive,
    describe,
    noticed,
    within,
)


def _screen(values, rail: str) -> dict:
    layout = [
        {"text": str(value), "center_x": 0.40 + 0.07 * i, "center_y": 0.40,
         "x": 0.40 + 0.07 * i, "y": 0.40}
        for i, value in enumerate(values)
    ]
    layout.append({"text": rail, "center_x": 0.92, "center_y": 0.20, "x": 0.92, "y": 0.20})
    layout.append({"text": "© 2014-2026", "center_x": 0.50, "center_y": 0.95,
                   "x": 0.50, "y": 0.95})
    return {"ok": True, "text": " ".join([*(str(v) for v in values), rail, "© 2014-2026"]),
            "layout": layout}


def _played(moves: int = 9) -> Responsive:
    """Her acting on the board while the advertising rotates on its own clock."""
    state = Responsive()
    boards = [[2, 4, 8], [4, 8, 16], [8, 16, 32], [2, 16, 32], [4, 32, 64],
              [8, 32, 64], [2, 64, 64], [4, 64, 128], [8, 128, 4], [2, 128, 8]]
    for i in range(min(moves, len(boards) - 1)):
        # The rail turns over once in the whole run; the board answers to
        # every move.
        before = _screen(boards[i], "BuyNow" if i < 6 else "Sale")
        after = _screen(boards[i + 1], "BuyNow" if i + 1 < 6 else "Sale")
        state = noticed(state, before, after)
    return state


def test_the_board_is_found_and_the_advertising_is_not():
    state = _played()
    band = state.band()
    assert band is not None
    left, top, right, bottom = band
    assert left <= 0.40 and right >= 0.54, "the board is not inside the answer"
    assert right < 0.92, "an advertising rail was counted as part of the task"
    assert bottom < 0.95, "the copyright line was counted as part of the task"


def test_the_situation_she_reasons_about_is_the_part_that_answers():
    state = _played()
    situation = within(_screen([2, 128, 8], "Sale"), state.band())
    assert "128" in situation
    assert "Sale" not in situation
    assert "2014-2026" not in situation


def test_one_coincidence_is_a_coincidence():
    """Before enough acts have answered, there is no answer — and the whole
    reading is used, because a guess about where the task is would be worse."""
    state = _played(moves=ENOUGH_ACTS - 2)
    assert state.band() is None
    whole = _screen([2, 4, 8], "BuyNow")
    assert within(whole, state.band()) == whole["text"]


def test_a_screen_where_nothing_answers_gives_no_band():
    state = Responsive()
    still = _screen([2, 4, 8], "BuyNow")
    for _ in range(ENOUGH_ACTS + 2):
        state = noticed(state, still, still)
    assert state.band() is None


def test_the_answer_can_be_said_out_loud():
    said = describe(_played().band())
    assert "responds to me" in said
    assert describe(None) == ""


def test_the_loop_reasons_from_the_part_that_answers():
    """Held where it matters: the situation handed to a decision, and the one
    written to the consequence graph, are the same string."""
    import inspect

    from core.skills import screen_pursuit

    source = inspect.getsource(screen_pursuit.pursue_on_screen)
    assert "seen = within(observation, band)" in source
    assert "noticed(responds[" in source
