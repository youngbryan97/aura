"""A plan she mentions is not the plan she names.

LIVE 2026-09-24 her voice answered how she would go about the game, and the
reader took any "plan" or "approach" followed by anything as the label of her
line. "The previous approach of consolidating the center is no longer viable"
was narrated as "Plan: of consolidating the center is no longer viable", and a
line set in bold came out as "Plan: ** I am going to prioritize **".
"""
from __future__ import annotations

from core.agency.deliberate_action import ActionOption, Expectation
from core.agency.standing_strategy import read_strategy

MOVES = [
    ActionOption(name=name, expectation=Expectation(changed=True, describes=f"the view after {name}"))
    for name in ("up", "down", "left", "right")
]
BOARD = "4 . . 2 / 16 8 . . / 2 16 8 2 / 16 4 . ."

# Her answer at 08:54 UTC, as far as the line.
PREVIOUS_THEN_NEW = (
    "I am looking at the board as it stands right now. The previous plan to "
    "consolidate the center has failed, and my repeated attempts to push left "
    "have only shuffled things around without making progress. My new approach "
    "is to clear out the center by pushing everything toward the left edge, so "
    "the 16s have room to merge."
)


def test_the_line_she_names_is_taken_over_the_one_she_mentions():
    held = read_strategy(PREVIOUS_THEN_NEW, MOVES, situation=BOARD)
    assert held is not None
    assert held.approach.startswith("to clear out the center"), held.approach
    assert "has failed" not in held.approach


def test_a_line_mentioned_in_passing_is_not_read_as_named():
    said = (
        "The previous approach of consolidating the center is no longer viable. "
        "I will keep the 16 in the bottom-left corner and slide everything toward it."
    )
    held = read_strategy(said, MOVES, situation=BOARD)
    assert held is not None
    assert not held.approach.startswith("of "), held.approach


def test_emphasis_is_not_part_of_what_she_said():
    said = (
        "**Plan:** I am going to keep **the largest tile in the bottom-left corner** "
        "and build the bottom row toward it."
    )
    held = read_strategy(said, MOVES, situation=BOARD)
    assert held is not None
    assert "*" not in held.approach and "*" not in held.narrate()


def test_a_label_with_a_word_before_its_colon_is_still_a_label():
    held = read_strategy(
        "My plan is simple: keep the largest tile in a corner and slide everything toward it.",
        MOVES,
        situation=BOARD,
    )
    assert held is not None
    assert held.approach.startswith("keep the largest tile"), held.approach


def test_a_numbered_plan_is_read_from_its_first_step():
    """LIVE 2026-09-24: "Plan: help you play toward the goal: My Approach: 1."."""
    said = (
        "I will help you play toward the goal. My Approach: 1. Keep the largest tile "
        "in the bottom-left corner and build the bottom row toward it. 2. Never press up."
    )
    held = read_strategy(said, MOVES, situation=BOARD)
    assert held is not None
    assert held.approach.startswith("Keep the largest tile"), held.approach
