"""A direction word in prose about the board is not a key to press.

Measured live on 2026-08-26: "1. down (merge 4+4=8) 2. right (shift remaining
tiles to the right edge to keep the left open) 3. down" was read as the plan
down, right, LEFT, down — and she pressed a key her own plan never called for.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption, choose_sequence

MOVES = [ActionOption(name=name) for name in ("up", "down", "left", "right")]


def _plan(said: str, limit: int = 4) -> list[str]:
    return [option.name for option in choose_sequence(said, MOVES, limit)]


def test_a_word_about_the_board_is_not_a_move():
    said = (
        "I choose **down**. Next moves in order: 1. **down** (merge 4+4=8) "
        "2. **right** (shift remaining tiles to the right edge to keep the left open) "
        "3. **down** (merge any new vertical matches)."
    )
    assert _plan(said) == ["down", "right", "down"]


def test_a_sentence_of_prose_naming_one_move_is_a_plan_of_one():
    assert _plan("Go right; the left column is full.") == ["right"]


@pytest.mark.parametrize(
    "said",
    ["up left up", "up, then left, then up", "I'll press up. Then left, then up again."],
)
def test_a_bare_list_is_still_a_list(said):
    assert _plan(said) == ["up", "left", "up"]


def test_the_caller_s_limit_still_holds():
    assert len(_plan("up, then left, then down, then right, then up", limit=2)) == 2


def test_nothing_named_is_no_plan():
    assert _plan("The board is nearly full.") == []
