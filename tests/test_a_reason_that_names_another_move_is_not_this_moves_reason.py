"""The sentence found around a move's name can be about a different move.

Measured live on 2026-08-26: "Going left — I choose down because the two 4s
in column 1 will merge into an 8, consolidating the left side and freeing
space." The word "left" sat inside "the left side", so the sentence pulled
around it was the one where she chose down.
"""

from __future__ import annotations

from core.agency.deliberate_action import ActionOption, _rationale

MOVES = [ActionOption(name=name) for name in ("up", "down", "left", "right")]


def _move(name: str) -> ActionOption:
    return next(option for option in MOVES if option.name == name)


REPLY = (
    "I choose down because the two 4s in column 1 will merge into an 8, "
    "consolidating the left side and freeing space."
)


def test_a_sentence_that_chose_another_move_is_not_this_ones_reason():
    assert _rationale(REPLY, _move("left"), MOVES) == ""


def test_the_move_it_did_choose_keeps_the_sentence():
    assert "merge into an 8" in _rationale(REPLY, _move("down"), MOVES)


def test_a_reason_that_names_nothing_still_counts():
    said = "The bottom row is nearly full, so consolidate before it locks up. Left."
    assert _rationale(said, _move("left"), MOVES)


def test_with_no_options_to_compare_the_sentence_stands():
    assert _rationale(REPLY, _move("left")) != ""
