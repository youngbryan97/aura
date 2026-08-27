"""A marker has to be a word, not letters that happen to be there.

"til" sits inside "tiles". "to a" sits inside "onto a". "reach" sits inside
"outreach". Matched anywhere at all, a request that merely mentions tiles gets
its finishing condition sliced out of the middle of a word — and what she is
left waiting for is whatever noise falls out of the cut.

LIVE 2026-08-27: "the classic one with numbered tiles and a single empty
space" cut at "numbered ti|les", and the thing she waited to see appear on
screen was the word "are", taken from "how sure you are" at the far end of the
message. Nothing on screen ever offered a move, and the run ended having made
none.
"""

from __future__ import annotations

import pytest

from core.runtime.watched_goal import _best_finishing_test, _condition_clauses


# ── the cut that started it ──────────────────────────────────────────────

def test_a_word_inside_a_word_does_not_start_a_condition():
    said = (
        "Find a sliding puzzle online — the classic one with numbered tiles "
        "and a single empty space — open it in the browser, and work out how "
        "it moves. Tell me the rule you worked out and how sure you are."
    )
    assert _best_finishing_test(said) == ""


def test_and_nothing_is_cut_out_of_the_middle_of_one():
    for clause in _condition_clauses("the classic one with numbered tiles"):
        assert not clause.startswith("es ")


@pytest.mark.parametrize(
    "said",
    [
        "drag the tiles into place",
        "count the tiles on the board",
        "put the utilities back",
        "read it onto a fresh page",
        "check community outreach numbers",
    ],
)
def test_a_request_that_merely_mentions_one_names_no_finish(said):
    assert _best_finishing_test(said) == ""


# ── and the conditions that are really there still read ──────────────────

@pytest.mark.parametrize(
    ("said", "test"),
    [
        ("Find 2048 online, play it, and get to a 256 tile.", "256"),
        ("play until you get a 512 tile", "512"),
        ("keep going till the board is full", "full"),
        ("go up to a 1,024 tile", "1024"),
        ("carry on until it says 'Game over'", "Game over"),
        ("open the puzzle and reach a solved state", "state"),
    ],
)
def test_a_marker_that_is_a_word_still_starts_one(said, test):
    assert _best_finishing_test(said) == test


def test_the_marker_is_found_wherever_the_case_falls():
    assert _best_finishing_test("Play it. UNTIL you get a 128 tile.") == "128"


# ── and an aside does not swallow the name of the place ──────────────────

from core.runtime.watched_goal import _where_it_happens  # noqa: E402


@pytest.mark.parametrize(
    ("said", "place"),
    [
        (
            "Find a sliding puzzle online — the classic one with numbered "
            "tiles and a single empty space — open it in the browser",
            "sliding puzzle",
        ),
        ("Open 2048 (the browser one) and play", "2048"),
        ("find the crossword; then solve it", "crossword"),
        ("pull up the tide table: the one for Tuesday", "tide table"),
        ("go find a 2048 game online", "2048 game"),
        ("Find 2048 online, play it, and get to a 256 tile.", "2048"),
    ],
)
def test_whatever_sets_an_aside_off_ends_the_name(said, place):
    """A comma is not the only thing that closes a noun phrase."""
    assert _where_it_happens(said) == place


def test_a_thing_already_open_is_still_not_a_place_to_go():
    assert _where_it_happens("2048 is open in Chrome already") == ""
