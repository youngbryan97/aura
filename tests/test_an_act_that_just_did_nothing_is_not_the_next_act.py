"""What just happened outranks what has worked here before.

"It has worked here before" is a lifetime record; the board in front of her is
now. LIVE 2026-09-04: fifteen presses of up into a board with nothing above
anything, each one chosen because up had worked earlier in the same game.

The record of which acts have done nothing clears the moment anything answers,
so it is the freshest evidence there is about the next move. It never empties
the choice: every act failing is the thing having ended, and that is judged
somewhere else.
"""

from __future__ import annotations

import inspect

from core.perception.where_it_responds import Responsive, noticed
from core.skills import screen_pursuit

SOURCE = inspect.getsource(screen_pursuit.pursue_on_screen)


def _screen(**cells: str) -> dict:
    return {
        "layout": [
            {"text": text, "center_x": n / 10.0, "center_y": 0.5}
            for n, text in enumerate(cells.values())
        ]
    }


def test_the_record_names_the_acts_that_did_nothing():
    state = Responsive()
    still = _screen(a="2", b="4")
    noticed(state, still, still, worked=False, acting="up")
    noticed(state, still, still, worked=False, acting="up")
    assert state.unanswered_by == {"up"}


def test_it_clears_the_moment_something_answers():
    state = Responsive()
    still = _screen(a="2", b="4")
    noticed(state, still, still, worked=False, acting="up")
    noticed(state, still, _screen(a="4", b="2"), worked=True, acting="left")
    assert not state.unanswered_by


def test_the_pursuit_drops_those_acts_from_the_choice():
    at = SOURCE.index("doing_nothing = responds[\"state\"].unanswered_by")
    nearby = SOURCE[at : at + 400]
    assert "one.name not in doing_nothing" in nearby


def test_it_never_empties_the_choice():
    at = SOURCE.index("doing_nothing = responds[\"state\"].unanswered_by")
    nearby = SOURCE[at : at + 400]
    assert "if still_worth:" in nearby


def test_it_is_applied_after_what_she_is_holding_rules_out():
    """Both narrow; neither may leave her nothing."""
    holding = SOURCE.index("available = [one for one in available if one.name not in wont]")
    fresh = SOURCE.index("doing_nothing = responds[\"state\"].unanswered_by")
    assert holding < fresh
