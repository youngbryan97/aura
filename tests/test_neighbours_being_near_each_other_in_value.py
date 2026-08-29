"""A thing can be perfectly ordered and impossible to work with.

2, 32, 4, 64 runs one way along no line. 2, 4, 512, 1024 runs one way along
every line and offers nothing that could ever combine. What makes a situation
workable is that the things beside each other are CLOSE — one step apart
rather than eight — because that is what lets them come together at all.

She had monotonicity, free space and a corner through her line, and nothing
that measured this. Measured 2026-08-29, six games at each weight run to a
dead board, same seeds, line held and world model on:

    smoothness    median best tile    total
           0.0                1024     2133
          0.15                1536     2670
           0.4                2048     4020
           1.0                1536     2463

The median best tile doubles. Nothing else moved it: depth did not (six with
four times the budget gave the same 1024), and the other weights did not (a
whole sweep came back flat).
"""

from __future__ import annotations

import math

import pytest

from core.agency.how_good_is_this import (
    SMOOTHNESS_MATTERS,
    _smoothness,
    how_good,
    terms,
)
from core.perception.what_is_there import arranged


def board(rows):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


SMOOTH = [["2", "4", "8", "16"], ["4", "8", "16", "32"],
          ["8", "16", "32", "64"], ["16", "32", "64", "128"]]
ROUGH = [["2", "512", "4", "1024"], ["256", "8", "128", "16"],
         ["2", "64", "4", "32"], ["512", "16", "1024", "8"]]


# ── what it measures ─────────────────────────────────────────────────────

def test_a_board_whose_neighbours_are_close_scores_higher():
    assert _smoothness(board(SMOOTH)) > _smoothness(board(ROUGH))


def test_a_board_of_one_value_is_as_close_as_it_gets():
    same = board([["4"] * 4] * 4)
    assert _smoothness(same) == pytest.approx(1.0)


def test_it_is_counted_in_doublings_not_in_differences():
    """A gap among small things is the same gap among large ones."""
    small = board([["2", "8"]])
    large = board([["512", "2048"]])
    assert _smoothness(small) == pytest.approx(_smoothness(large))


def test_a_thing_with_nothing_in_it_is_not_smooth_or_rough():
    assert _smoothness(arranged([])) == 0.0
    assert _smoothness(None) == 0.0


def test_words_are_not_measured_for_closeness():
    assert _smoothness(arranged([(0.2, 0.2, "Mon"), (0.2, 0.35, "Tue")])) == 0.0


# ── and it is one of the things a situation is judged by ─────────────────

def test_it_is_among_terms_a_situation():
    assert "smoothness" in terms(board(SMOOTH))


def test_a_workable_board_is_judged_better_than_an_unworkable_one():
    """The same numbers, arranged well and arranged badly.

    Compared against a board holding LARGER numbers, nearness rightly wins and
    says nothing about this — so the comparison has to hold the numbers fixed
    and move only where they sit.
    """
    laid = ["2", "4", "8", "16", "4", "8", "16", "32",
            "8", "16", "32", "64", "16", "32", "64", "128"]
    tidy = board([laid[n : n + 4] for n in range(0, 16, 4)])
    # The same sixteen values, put back in an order that puts nothing near
    # anything like it.
    shaken = [laid[n] for n in (15, 0, 11, 1, 8, 2, 12, 3, 9, 4, 13, 5, 10, 6, 14, 7)]
    scattered = board([shaken[n : n + 4] for n in range(0, 16, 4)])
    assert sorted(tidy.numbers()) == sorted(scattered.numbers())
    assert how_good(tidy, toward="2048") > how_good(scattered, toward="2048")


def test_the_weight_is_the_one_that_was_measured():
    """0.4 is the peak; 1.0 turns back. See the module docstring."""
    assert SMOOTHNESS_MATTERS == pytest.approx(0.4)


def test_ordered_and_unworkable_is_told_from_ordered_and_workable():
    """The case monotonicity alone cannot see."""
    from core.agency.how_good_is_this import _order

    laddered = board([["2", "4", "512", "1024"]])
    stepped = board([["2", "4", "8", "16"]])
    assert _order(laddered) == pytest.approx(_order(stepped))
    assert _smoothness(stepped) > _smoothness(laddered)


def test_what_can_honestly_be_said_about_her_play_is_written_down():
    """2048 as a rule, 4096 on a good run — not "she can reach 4096"."""
    import inspect

    from core.agency import how_good_is_this

    source = inspect.getsource(how_good_is_this)
    assert "4096" in source
    assert "on a good run" in source
