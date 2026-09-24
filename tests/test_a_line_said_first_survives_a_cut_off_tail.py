"""A line said first is read even when the answer ran past its budget.

LIVE 2026-09-24: "a simple, repeatable plan: keep the largest tile anchored in
a corner, slide everything toward it, and only change direction when a move
would trap a big tile", then two hundred words of reading the board aloud until
the budget ran out, and the whole answer was dropped for its last half-sentence.
"""

from __future__ import annotations

from core.agency.standing_strategy import read_strategy
from core.utils.an_answer import what_was_finished

LIVE = (
    "I'm playing 2048 with a simple, repeatable plan: keep the largest tile anchored "
    "in a corner, slide everything toward it, and only change direction when a move "
    "would trap a big tile or block future merges. I'm not reacting to one random "
    "shuffle; I'm steering the board into a state where high tiles can keep doubling. "
    "What would change this: - If a move leaves the 256 with two or more non-matching "
    "neighbors on"
)


def test_what_was_finished_keeps_the_whole_sentences():
    kept = what_was_finished(LIVE)
    assert kept.endswith("keep doubling.")
    assert "non-matching neighbors on" not in kept


def test_a_finished_answer_is_left_alone():
    assert what_was_finished("Keep the big tile in a corner.") == "Keep the big tile in a corner."


def test_nothing_finished_is_nothing():
    assert what_was_finished("Keep it. and then the") == "Keep it."
    assert what_was_finished("Keep it (and then the") == ""


def test_the_line_she_said_first_is_held():
    held = read_strategy(LIVE, situation="2 8 4 16 2 8 64 4 64 256 4 2 8 2 4 16")
    assert held is not None
    assert "corner" in held.approach


def test_the_line_is_said_once_and_what_is_watched_is_its_anchor():
    held = read_strategy(LIVE, situation="2 8 4 16 2 8 64 4 64 256 4 2 8 2 4 16")
    said = held.narrate()
    assert said.count("keep the largest tile anchored") == 1
    assert said.endswith("block future merges, while the 256 is still there")
