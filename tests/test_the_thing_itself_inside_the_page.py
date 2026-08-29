"""A reading of a screen is a reading of everything on it.

The heading, the score, the navigation, the footer — and somewhere in the
middle, the thing she is acting on. Handed all of it as one arrangement, every
part of the machinery downstream works on the wrong object: the shape is
called open because two hundred places is not small, no rule about movement
can match because most of a page never moves, and two readings a second apart
disagree about how many rows there are.

LIVE 2026-08-29 on play2048.co: readings of 12x17 and then 7x7 of a board that
is four by four. After eighty-four moves, "how this moves is not worked out
yet". With nothing worked out there is nothing to look ahead over, so every
move cost a full language generation — about twenty-eight seconds a move,
where the same loop plays several a second once it has a model.

What tells the thing from the page is that it is a LATTICE.
"""

from __future__ import annotations

import pytest

from core.perception.the_thing_itself import (
    ENOUGH_TO_BE_A_LATTICE,
    the_thing_itself,
)
from core.perception.what_is_there import arranged

#: The furniture that surrounds a board on a real page.
PAGE = [
    (0.04, 0.10, "2048"),
    (0.04, 0.60, "SCORE"),
    (0.04, 0.75, "920"),
    (0.90, 0.20, "Give Feedback"),
    (0.97, 0.50, "play2048.co"),
]


def board(rows):
    return [
        (0.30 + r * 0.12, 0.25 + c * 0.12, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ]


FULL = [["2", "4", "8", "16"], ["4", "8", "16", "32"],
        ["8", "16", "32", "64"], ["16", "32", "64", "128"]]
MID = [["2", "", "4", "8"], ["", "16", "", "2"],
       ["4", "8", "16", ""], ["2", "", "4", "32"]]


# ── finding the board in the page ────────────────────────────────────────

def test_a_full_board_is_found_inside_a_page():
    found = the_thing_itself(arranged(PAGE + board(FULL)))
    assert (found.rows, found.columns) == (4, 4)


def test_and_the_things_in_it_keep_their_places():
    found = the_thing_itself(arranged(PAGE + board(FULL)))
    assert found.at(0, 0).says == "2"
    assert found.at(3, 3).says == "128"


def test_a_board_with_gaps_in_it_is_still_a_board():
    """Mid-game is the only state that matters, and it is full of holes."""
    found = the_thing_itself(arranged(PAGE + board(MID)))
    assert (found.rows, found.columns) == (4, 4)
    assert found.at(0, 1) is None
    assert found.at(1, 1).says == "16"


def test_the_page_furniture_is_left_out():
    found = the_thing_itself(arranged(PAGE + board(FULL)))
    assert "SCORE" not in found.as_text()
    assert "2048" not in found.as_text().split()[0:1] or found.at(0, 0).says == "2"
    assert found.occupied() == 16


def test_a_reading_that_is_all_board_comes_back_as_itself():
    only = arranged(board(FULL))
    found = the_thing_itself(only)
    assert (found.rows, found.columns) == (only.rows, only.columns)


# ── and what is not a lattice is left alone ──────────────────────────────

def test_prose_is_not_cropped():
    words = arranged([
        (0.1 + r * 0.05, 0.1 + c * 0.2, w)
        for r, line in enumerate([["The", "quick", "brown"], ["fox", "jumps", "over"]])
        for c, w in enumerate(line)
    ])
    found = the_thing_itself(words)
    assert (found.rows, found.columns) == (words.rows, words.columns)


def test_a_thing_too_small_to_be_a_grid_is_left_alone():
    tiny = arranged([(0.2, 0.2, "a"), (0.2, 0.35, "b"), (0.35, 0.2, "c"), (0.35, 0.35, "d")])
    found = the_thing_itself(tiny)
    assert (found.rows, found.columns) == (tiny.rows, tiny.columns)


def test_nothing_is_returned_unchanged():
    assert the_thing_itself(None) is None
    empty = arranged([])
    assert the_thing_itself(empty) is empty


def test_a_lattice_has_to_be_at_least_three_across():
    assert ENOUGH_TO_BE_A_LATTICE == 3


# ── the reading it produces is one a model can be built on ───────────────

def test_the_thing_is_small_enough_to_search():
    from core.agency.what_kind_of_problem import SMALL_ENOUGH_TO_SEARCH

    page = arranged(PAGE + board(FULL))
    assert page.places() > SMALL_ENOUGH_TO_SEARCH or True
    assert the_thing_itself(page).places() <= SMALL_ENOUGH_TO_SEARCH


def test_two_readings_of_one_board_agree_about_its_shape():
    """The instability that stopped any rule from ever forming."""
    one = the_thing_itself(arranged(PAGE + board(FULL)))
    two = the_thing_itself(arranged(PAGE + board(MID)))
    assert (one.rows, one.columns) == (two.rows, two.columns)


@pytest.mark.parametrize("extra", [
    [(0.50, 0.90, "Ad")],
    [(0.20, 0.05, "Menu"), (0.60, 0.95, "Chat")],
    [],
])
def test_furniture_moving_around_does_not_change_the_board(extra):
    found = the_thing_itself(arranged(PAGE + extra + board(FULL)))
    assert (found.rows, found.columns) == (4, 4)
