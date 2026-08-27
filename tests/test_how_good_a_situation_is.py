"""Futures she has not visited, compared — so imagining one is worth something.

Two things make a situation good and neither is about any particular kind of
screen: nearness to what she was asked for, and whether the line she said she
was taking is still true of it. Scoring a future by her own stated approach is
what makes a plan cause the moves rather than accompany them.
"""

from __future__ import annotations

import pytest

from core.agency.how_good_is_this import how_good, rank, why, worth_comparing
from core.perception.how_it_moves import shifted_and_combined
from core.perception.what_is_there import arranged


def board(rows: list[list[str]]):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


CORNER = board([
    ["2", "4", "", "8"],
    ["", "2", "4", ""],
    ["4", "", "", "2"],
    ["64", "2", "", "4"],
])
LINE = "keep the 64 in the bottom-left corner"


# ── whether there is anything to go on ───────────────────────────────────

def test_with_neither_a_goal_nor_a_line_there_is_nothing_to_compare_by():
    assert not worth_comparing("", "")


def test_a_goal_that_names_something_is_enough():
    assert worth_comparing("256", "")


def test_a_line_she_is_holding_is_enough_on_its_own():
    assert worth_comparing("open the settings", LINE)


# ── nearness to what she was asked for ───────────────────────────────────

def test_a_situation_that_has_it_scores_the_most_it_can_for_nearness():
    there = board([["256", "2"], ["4", "8"]])
    nearly = board([["128", "2"], ["4", "8"]])
    assert how_good(there, toward="256") > how_good(nearly, toward="256")


def test_nearness_is_counted_in_doublings_not_in_ratio():
    """Halfway to 256 is a 16, not a 128, in anything built by combining.

    Asked of the term rather than of the total, which carries the affordances
    too and moves whenever one of them is added.
    """
    from core.agency.how_good_is_this import _nearness

    assert _nearness(board([["16", "2"], ["4", "8"]]), "256") == pytest.approx(0.5)
    assert _nearness(board([["128", "2"], ["4", "8"]]), "256") == pytest.approx(0.875)


def test_a_goal_that_names_nothing_measurable_scores_no_nearness():
    plain = board([["8", "2"], ["4", ""]])
    assert how_good(plain, toward="open the settings") == pytest.approx(how_good(plain))


# ── her own line doing the choosing ──────────────────────────────────────

def test_the_line_she_is_holding_picks_the_move():
    futures = {name: shifted_and_combined(CORNER, name) for name in ("up", "down", "left", "right")}
    best = [name for name, _score in rank(futures, toward="256", approach=LINE)]
    assert set(best[:2]) == {"down", "left"}
    for name in best[:2]:
        assert "bottom-left" in futures[name].places_of("64")


def test_without_the_line_those_moves_are_not_preferred():
    futures = {name: shifted_and_combined(CORNER, name) for name in ("up", "down", "left", "right")}
    with_line = dict(rank(futures, toward="256", approach=LINE))
    without = dict(rank(futures, toward="256"))
    assert with_line["down"] - without["down"] == pytest.approx(1.0)


def test_a_line_that_says_nothing_specific_changes_nothing():
    state = board([["2", "4"], ["8", "16"]])
    assert how_good(state, toward="256", approach="play well") == how_good(state, toward="256")


def test_a_situation_is_asked_only_about_the_content_of_the_line():
    """Asked of a state against itself, a claim that something DIFFERS is false
    of every state, including the good ones."""
    kept = board([["2", "4"], ["64", "8"]])
    assert how_good(kept, approach="keep the 64 in the bottom-left corner") > 0.9


# ── room to act ──────────────────────────────────────────────────────────

def test_room_left_is_worth_something_and_not_much():
    roomy = board([["2", "", "", "4"], ["", "8", "", ""], ["", "", "2", ""], ["4", "", "", "8"]])
    full = board([["2", "4", "8", "2"], ["4", "8", "2", "4"], ["8", "2", "4", "8"], ["2", "4", "8", "2"]])
    assert how_good(roomy) > how_good(full)
    assert how_good(roomy) - how_good(full) < 0.2


# ── saying why ───────────────────────────────────────────────────────────

def test_the_reason_is_something_she_could_say():
    said = why(CORNER, toward="256", approach=LINE)
    assert "largest is 64" in said
    assert "keeps the line I am taking" in said
    assert "place(s) left" in said


def test_a_situation_that_has_the_target_says_so():
    assert "has the 256" in why(board([["256", "2"], ["4", "8"]]), toward="256")


def test_something_that_is_not_an_arrangement_is_not_scored_wrongly():
    assert how_good(object(), toward="256", approach=LINE) == 0.0


# ── order, the other affordance ──────────────────────────────────────────

TIDY = board([["64", "32", "16", "8"], ["32", "16", "8", "4"],
              ["16", "8", "4", "2"], ["8", "4", "2", "2"]])
JUMBLED = board([["8", "64", "4", "32"], ["32", "2", "16", "4"],
                 ["4", "16", "2", "64"], ["16", "8", "32", "2"]])


def test_a_thing_that_runs_in_order_is_easier_to_act_in():
    assert how_good(TIDY) > how_good(JUMBLED)


def test_order_is_read_off_the_thing_rather_than_assumed():
    from core.agency.how_good_is_this import _order

    assert _order(TIDY) == pytest.approx(1.0)
    assert _order(JUMBLED) < 1.0


def test_which_end_the_big_end_is_makes_no_difference():
    """Ascending and descending are both order. Which end is hers to say."""
    from core.agency.how_good_is_this import _order

    other_way = board([["2", "4", "8", "16"], ["4", "8", "16", "32"],
                       ["8", "16", "32", "64"], ["16", "32", "64", "128"]])
    assert _order(other_way) == pytest.approx(_order(TIDY))


def test_order_is_worth_about_what_room_is_worth():
    from core.agency.how_good_is_this import ORDER_MATTERS, ROOM_MATTERS

    assert ORDER_MATTERS == ROOM_MATTERS


def test_neither_of_them_outweighs_the_line_she_is_holding():
    from core.agency.how_good_is_this import LINE_MATTERS, ORDER_MATTERS, ROOM_MATTERS

    assert LINE_MATTERS > ORDER_MATTERS + ROOM_MATTERS


def test_something_with_no_numbers_in_it_is_not_scored_for_order():
    from core.agency.how_good_is_this import _order

    words = board([["Mon", "Tue"], ["Wed", "Thu"]])
    assert _order(words) == 0.0
