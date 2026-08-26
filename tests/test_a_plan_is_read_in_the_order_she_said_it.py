"""Her plan is read where she said it, cut at words, and said once.

All four faults measured live on 2026-08-26 in one narrated line: the pivot
taken as the approach, a clause ending mid-word, the reason quoted twice, and
what she is watching for running on into what she would do instead.
"""

from __future__ import annotations

from core.agency.deliberate_action import ActionOption
from core.agency.standing_strategy import read_strategy

MOVES = [ActionOption(name=name) for name in ("up", "down", "left", "right")]

SPOKEN = (
    "I choose down because the two 4s in column 1 will merge into an 8, "
    "consolidating the left side and freeing space for new tiles without "
    "disturbing the bottom row 4-16-32 stack. I am watching to see if this "
    "creates a safe pocket on the left for future merges, and I will switch "
    "to right if a new tile appears that blocks the bottom row growth."
)


def _held():
    held = read_strategy(SPOKEN, MOVES, situation="2 4 8 64")
    assert held is not None
    return held


def test_what_she_would_do_instead_is_not_the_line_she_is_taking():
    held = _held()
    assert "switch to right" not in held.approach
    assert held.otherwise
    assert "right" in held.otherwise[0]


def test_watching_survives_a_phrasing_that_is_not_watching_for():
    assert "safe pocket" in _held().holds_while.describes


def test_what_she_is_watching_for_stops_before_what_she_would_do_instead():
    assert "switch" not in _held().holds_while.describes


def test_nothing_is_cut_in_the_middle_of_a_word():
    held = _held()
    words = set(SPOKEN.replace(".", " ").replace(",", " ").split())
    clauses = (held.approach, held.because, *held.otherwise)
    for clause in clauses:
        assert not clause or clause.split()[-1].strip(".,;") in words


def test_the_narration_says_the_reason_once():
    said = _held().narrate()
    assert said.count("consolidating the left side") == 1


def test_one_alternative_is_not_listed_as_two():
    held = read_strategy(
        "Plan: build in the bottom-left corner while the 64 stays there. "
        "Otherwise switch to the right edge.",
        MOVES,
    )
    assert held is not None
    assert held.otherwise == ("switch to the right edge",)


NAMED_LATE = (
    "I choose to press left. The board shows two adjacent 16s in the bottom row "
    "that will merge into a 32, so left is the only move that creates a merge. "
    "My approach: I'll prioritize moves that create merges, watching for any row "
    "or column with adjacent equal tiles."
)


def test_the_approach_she_named_beats_the_move_she_opened_with():
    held = read_strategy(NAMED_LATE, MOVES, situation="16 16 4 32")
    assert held is not None
    assert "prioritize moves that create merges" in held.approach
    assert "press left" not in held.approach


def test_a_clause_too_short_to_be_an_approach_is_passed_over():
    held = read_strategy(
        "I choose left. My plan is to feed the bottom-left corner.", MOVES, situation="2 4 64"
    )
    assert held is not None
    assert "bottom-left corner" in held.approach


def test_a_move_on_its_own_is_still_not_an_approach():
    assert read_strategy("I choose to press left.", MOVES) is None


def test_a_reason_caught_on_its_own_is_not_the_line_she_is_taking():
    said = (
        "I'll press up because the board is sparse, so maximizing the size of the "
        "main stack before new tiles spawn is safer than chasing small merges."
    )
    held = read_strategy(said, MOVES, situation="2 4 64")
    assert held is None or not held.approach.lower().startswith("because")
