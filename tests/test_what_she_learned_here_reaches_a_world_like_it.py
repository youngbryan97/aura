"""What she works out about one world reaching the next world of its kind.

Everything she learns is filed under the thing she learned it in — an
application and an address. So a second world that moves in exactly the same
way started as ignorant as the first, and the fortieth was no better off than
the second. She recognised the kind of world she was in and said so out loud,
and then did nothing whatever with it.
"""

from __future__ import annotations

import pytest

from core.agency.what_kind_of_problem import Shape, recognise
from core.perception.how_it_moves import ENOUGH_TO_TRUST
from core.perception.where_it_responds import what_is_there
from core.skills.screen_pursuit import (
    CARRIES_TO_A_WORLD_LIKE_IT,
    _no_more_than_a_fresh_one_is_worth,
    _the_kind_of_world_this_is,
)


def _board(text: str):
    return what_is_there({"ok": True, "text": text, "layout": [], "bounds": []}, None)


FOUR_BY_FOUR = "2 4 8 2\n16 32 64 4\n2 8 4 2\n4 2 8 16"


def test_two_worlds_of_the_same_size_and_acts_share_a_name():
    one = _the_kind_of_world_this_is(_board(FOUR_BY_FOUR), ["up", "down", "left", "right"], "2048")
    two = _the_kind_of_world_this_is(
        _board("8 2 4 16\n4 2 2 8\n. . 4 2\n2 8 16 4"), ["up", "down", "left", "right"], "512"
    )
    assert one and one == two


def test_a_different_size_is_a_different_kind():
    small = _the_kind_of_world_this_is(_board(FOUR_BY_FOUR), ["up", "down"], "x")
    big = _the_kind_of_world_this_is(
        _board("\n".join(" ".join("2" for _ in range(6)) for _ in range(6))), ["up", "down"], "x"
    )
    assert small and big and small != big


def test_a_different_act_set_is_a_different_kind():
    four = _the_kind_of_world_this_is(_board(FOUR_BY_FOUR), ["up", "down", "left", "right"], "x")
    two = _the_kind_of_world_this_is(_board(FOUR_BY_FOUR), ["up", "down"], "x")
    assert four != two


def test_the_kind_does_not_depend_on_what_she_has_worked_out():
    """Keyed on the transition, only worlds she has already solved would match."""
    unknown = Shape(acts=4, transition_known=False, discrete=True, across=4, down=4)
    known = Shape(acts=4, transition_known=True, world_moves_too=True, discrete=True, across=4, down=4)
    assert unknown.of_this_kind() == known.of_this_kind()


def test_a_world_with_no_size_has_no_kind():
    """Nothing to carry, and nothing that would wrongly match everything."""
    assert Shape(acts=4, discrete=True).of_this_kind() == ""
    assert Shape(across=4, down=4).of_this_kind() == ""


def test_another_worlds_evidence_arrives_worth_what_a_fresh_rule_is_worth():
    """The conclusion carries; the confidence does not."""
    hard_won = {"tried": {"shifted": 200, "composed": 200}}
    share = _no_more_than_a_fresh_one_is_worth(hard_won)
    assert 200 * share == pytest.approx(ENOUGH_TO_TRUST)


def test_a_little_evidence_carries_whole():
    assert _no_more_than_a_fresh_one_is_worth({"tried": {"shifted": 2}}) == 1.0
    assert _no_more_than_a_fresh_one_is_worth({}) == 1.0
    assert _no_more_than_a_fresh_one_is_worth(None) == 1.0


def test_where_it_is_drawn_does_not_carry_to_another_world():
    """Two worlds of a kind move alike. They are not drawn alike."""
    for about_this_screen in ("responds", "lattice", "moves_within", "reaches"):
        assert about_this_screen not in CARRIES_TO_A_WORLD_LIKE_IT
    for about_the_kind in ("moves", "acts", "skill", "world"):
        assert about_the_kind in CARRIES_TO_A_WORLD_LIKE_IT


def test_the_kind_is_named_from_the_reading_she_has():
    shape = recognise(
        acts=["up", "down", "left", "right"],
        knows_how_it_moves=None,
        state=_board(FOUR_BY_FOUR),
        toward="reach 2048",
    ).shape
    assert (shape.across, shape.down) == (4, 4)
    assert "4x4" in shape.of_this_kind()
