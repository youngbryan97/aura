"""Keeping the biggest tile in a corner is a plan she can now hold.

Bryan, 2026-08-26, watching her play: "she never employs the best and super
common strategy of keep your highest block in the corner." She could not.
Her reading of the board was "2 4 8 64" — the contents with the arrangement
thrown away — so a corner was not a thing she could see, name, or check.

With the arrangement kept, a plan about a place is checkable: it holds while
the tile is in the corner, and ends when it drifts out.
"""
from __future__ import annotations

import pytest

from core.agency.standing_strategy import (
    place_named_in,
    read_strategy,
    still_holds,
    where_it_sits,
)

IN_THE_CORNER = "2 4 8 4\n16 32 2 8\n4 8 2 2\n128 2 4 16"
IN_THE_MIDDLE = "2 4 8 4\n16 32 2 8\n4 128 2 2\n8 2 4 16"


@pytest.mark.parametrize(
    ("said", "place"),
    [
        ("keep the largest tile in the bottom-left corner", "bottom-left"),
        ("hold the big one in the upper right", "top-right"),
        ("keep it in a corner", "corner"),
        ("build along the left column", "left"),
        ("stack everything on the bottom row", "bottom"),
    ],
)
def test_a_condition_can_name_a_place(said, place):
    assert place_named_in(said) == place


def test_a_condition_that_names_no_place_is_left_alone():
    assert place_named_in("keep merging the small tiles") == ""
    assert place_named_in("") == ""


def test_where_a_value_sits_is_read_from_the_arrangement():
    sits = where_it_sits("128", IN_THE_CORNER)
    assert "bottom-left" in sits
    assert "corner" in sits
    assert "bottom" in sits and "left" in sits
    # In the middle it is at no edge at all.
    assert where_it_sits("128", IN_THE_MIDDLE) == set()


def test_a_plan_about_a_corner_holds_while_it_is_in_the_corner():
    held = read_strategy(
        "Keep the largest tile in the bottom-left corner and merge downward.",
        situation=IN_THE_CORNER,
    )
    assert held is not None
    assert held.holds_while.contains == ("128",)
    holds, why = still_holds(held, IN_THE_CORNER)
    assert holds and not why


def test_the_same_plan_ends_when_it_drifts_out_of_the_corner():
    """This is what could not be noticed before: the tile is still on the
    board, so every content-based check stays true while the plan quietly
    stops being the plan."""
    held = read_strategy(
        "Keep the largest tile in the bottom-left corner and merge downward.",
        situation=IN_THE_CORNER,
    )
    holds, why = still_holds(held, IN_THE_MIDDLE)
    assert not holds
    assert "128" in why and "bottom-left" in why


def test_a_value_that_is_gone_is_still_reported_as_gone():
    held = read_strategy(
        "Keep the largest tile in the bottom-left corner.", situation=IN_THE_CORNER
    )
    holds, why = still_holds(held, "2 4 8\n16 32 2")
    assert not holds
    assert "gone" in why


def test_a_plan_with_no_place_in_it_is_unaffected():
    held = read_strategy(
        "Keep merging upward while the 128 is still around.", situation=IN_THE_CORNER
    )
    assert held is not None
    holds, _why = still_holds(held, IN_THE_MIDDLE)
    assert holds, "a plan that never mentioned a place must not fail on one"


def test_a_reading_with_no_arrangement_is_not_asked_about_places():
    """A place is a fact about a layout. A reading that arrived as one line of
    prose has no rows, no columns and no corners, and asking where something
    sits in it produces an answer about nothing."""
    held = read_strategy(
        "Keep the largest tile in the bottom-left corner.",
        situation="tiles 2 4 8 128 with 128 bottom left",
    )
    assert held is not None
    holds, why = still_holds(held, "tiles 2 4 8 128 somewhere on the board")
    assert holds, why
