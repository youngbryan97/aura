"""A property that explains the past is a story. One that improves the play is a finding.

Her measure of a good situation is a handful of properties added up. When two
situations score the same and one goes on to turn out much better, the measure
has nothing to say about a difference that is demonstrably real. That gap is
the only honest place a new property can come from — not from somebody noticing
one, but from her own failures to account for what happened.

What this pins is the discipline, not the answer: a candidate is chosen on half
the pairs and judged on the half it never saw, it has to beat a coin by a
margin, a measure that says nothing about the pairs gets no credit for its
silence, and nothing is promoted on explanatory power alone.
"""

from __future__ import annotations

import random

import pytest

from core.agency.inventing_a_measure import Measure
from core.agency.what_i_cannot_explain import (
    ENOUGH_PAIRS_TO_LOOK,
    MUST_BEAT_CHANCE_BY,
    WhatICannotExplain,
)
from core.perception.what_is_there import arranged

SMOOTHNESS = Measure("neighbours", "the gap between them", "on average", True)


def a_situation(rng, size=4):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, str(2 ** rng.randint(1, 7)))
        for r in range(size)
        for c in range(size)
    ])


def lived(*, blind_to, count=120, noise=0.05, seed=0):
    """Situations whose outcomes depend on a property her measure cannot see."""
    rng = random.Random(seed)
    what = WhatICannotExplain()
    for _ in range(count):
        situation = a_situation(rng)
        scored = max(situation.numbers() or (1.0,)) / 128.0
        what.been_here(situation, scored, blind_to.read(situation) + rng.gauss(0, noise))
    return what


# ── what needs explaining ────────────────────────────────────────────────

def test_pairs_it_calls_equal_that_did_not_turn_out_equal():
    assert len(lived(blind_to=SMOOTHNESS).unexplained()) > ENOUGH_PAIRS_TO_LOOK


def test_situations_it_already_scored_differently_need_no_explaining():
    """However badly it scored them, it said something. This is about silence."""
    what = lived(blind_to=SMOOTHNESS)
    for one, other in what.unexplained():
        assert abs(one.scored - other.scored) < 0.5


def test_and_neither_do_situations_that_turned_out_alike():
    what = lived(blind_to=SMOOTHNESS)
    for one, other in what.unexplained():
        assert abs(one.turned_out - other.turned_out) > 0.0


def test_nothing_lived_needs_nothing_explained():
    assert WhatICannotExplain().unexplained() == []


# ── what would explain them ──────────────────────────────────────────────

def test_she_finds_the_property_her_measure_was_blind_to():
    found = lived(blind_to=SMOOTHNESS).what_would_explain()
    assert found is not None
    measure, held_back, _pairs = found
    assert measure == SMOOTHNESS
    assert held_back > 0.5 + MUST_BEAT_CHANCE_BY


def test_it_is_judged_on_pairs_it_was_not_chosen_on():
    """A measure fitted to everything has been tested against nothing."""
    found = lived(blind_to=SMOOTHNESS).what_would_explain()
    assert found is not None
    assert found[1] <= 1.0


def test_a_different_blindness_finds_a_different_property():
    """Nothing about this is aimed at one answer."""
    edges = Measure("at an edge", "its size in doublings", "on average")
    found = lived(blind_to=edges, seed=3).what_would_explain()
    assert found is not None
    assert found[0] != SMOOTHNESS


def test_too_little_lived_proposes_nothing():
    """Below the bar, whichever measure happens to fit is fitting noise."""
    assert lived(blind_to=SMOOTHNESS, count=8).what_would_explain() is None


def test_outcomes_that_are_pure_noise_explain_nothing():
    rng = random.Random(1)
    what = WhatICannotExplain()
    for _ in range(140):
        situation = a_situation(rng)
        what.been_here(situation, 0.5, rng.random())
    found = what.what_would_explain()
    assert found is None or found[1] > 0.5 + MUST_BEAT_CHANCE_BY


def test_a_measure_that_says_nothing_gets_no_credit_for_it():
    """Silence is not agreement, or a constant wins every time."""
    from core.agency.what_i_cannot_explain import Lived, _agrees_on

    flat = Measure("everything", "how big it is", "at most")
    same = arranged([(0.2, 0.2, "8")])
    pairs = [(Lived(same, 0.5, 1.0), Lived(same, 0.5, 0.0)) for _ in range(10)]
    assert _agrees_on(flat, pairs) == 0.0


# ── and the evaluator she can add it to ──────────────────────────────────

def test_a_promoted_property_is_read_beside_the_authored_ones():
    from core.agency.how_good_is_this import forget, promote, terms

    situation = a_situation(random.Random(0))
    before = set(terms(situation))
    name = promote(SMOOTHNESS, 0.3)
    try:
        assert name and name in terms(situation)
        assert before < set(terms(situation))
    finally:
        forget(name)
    assert set(terms(situation)) == before


def test_what_earned_its_place_can_lose_it():
    from core.agency.how_good_is_this import forget, promote

    name = promote(SMOOTHNESS, 0.3)
    assert forget(name) is True
    assert forget(name) is False


def test_something_that_is_not_a_measure_is_not_promoted():
    from core.agency.how_good_is_this import promote

    assert promote(object(), 0.3) == ""
    assert promote(None, 0.3) == ""
