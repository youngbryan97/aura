"""Three things are called "the language grew" and they are not the same.

Conservativity and eliminability are different properties, and conflating them
is what produced the wrong claim this file exists to prevent. A word DEFINED as
a term can be substituted away and adds no meanings. A word admitted WITHOUT a
defining term — a correspondence read off examples — need not be eliminable,
and can add a distinction the old language could not draw.

Measured over her five given words: six random behaviours were each unreachable
after some four and a half million terms, so the strong kind is genuinely
available here and not a formality.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.what_an_invention_buys import the_horizon_of
from core.cognition.which_kind_of_growth import (
    A_NEW_DISTINCTION,
    A_SHORTER_NAME,
    UNDECIDED,
    which_kind_of_growth,
)
from core.cognition.widening_the_language import DerivedAddressing, widen_with_addressing

#: A renamed "one along": a term three symbols long already says it.
A_MACRO = {4: (1, 2, 3, 0), 5: (1, 2, 3, 4, 0)}
#: A correspondence nothing short says.
AN_AWKWARD_ONE = {4: (2, 3, 1, 0), 5: (3, 0, 4, 1, 2)}


@pytest.fixture(autouse=True)
def _left_as_found():
    was = dict(kinds.WHERE_FROM)
    yield
    kinds.WHERE_FROM.clear()
    kinds.WHERE_FROM.update(was)


def _says(at):
    def says_it(word):
        try:
            return all(
                int(word(place, size)) % size == at[size][place]
                for size in at
                for place in range(len(at[size]))
            )
        except (ArithmeticError, IndexError, KeyError, TypeError, ValueError):
            return False

    return says_it


def test_a_renamed_old_word_is_called_a_shorter_name():
    """What she found was a search she had not run, not a new concept."""
    found = which_kind_of_growth(
        _says(A_MACRO),
        the_old_language=dict(kinds.WHERE_FROM),
        horizon=the_horizon_of(4),
        certify_to=5,
        holes=1,
        within=20.0,
    )
    assert found.kind == A_SHORTER_NAME
    assert found.without == 3
    assert not found.is_a_new_distinction


def test_a_correspondence_nothing_says_is_called_a_new_distinction():
    found = which_kind_of_growth(
        _says(AN_AWKWARD_ONE),
        the_old_language=dict(kinds.WHERE_FROM),
        horizon=the_horizon_of(4),
        certify_to=5,
        holes=1,
        within=60.0,
    )
    assert found.kind == A_NEW_DISTINCTION
    assert found.finished
    assert found.is_a_new_distinction


def test_the_claim_carries_the_length_it_was_certified_to():
    """"Not at this length", never "not at any length"."""
    found = which_kind_of_growth(
        _says(AN_AWKWARD_ONE),
        the_old_language=dict(kinds.WHERE_FROM),
        horizon=the_horizon_of(4),
        certify_to=5,
        holes=1,
        within=60.0,
    )
    assert found.certified_to == 5
    assert "up to 5" in found.describes()


def test_a_search_that_ran_out_of_time_never_claims_a_new_distinction():
    """A search that ran out has said nothing about the language, only about
    the clock — which is the whole discipline."""
    found = which_kind_of_growth(
        _says(AN_AWKWARD_ONE),
        the_old_language=dict(kinds.WHERE_FROM),
        horizon=the_horizon_of(4),
        certify_to=9,
        holes=2,
        within=0.0,
    )
    assert found.kind == UNDECIDED
    assert not found.finished
    assert not found.is_a_new_distinction


def test_admitting_a_word_records_which_kind_of_growth_it_was():
    """`widen_with_addressing` admitted every word the same way and called all
    of it growth. A correspondence read off examples has no defining term, so
    it CAN be the strong kind — and the same word may equally be something a
    three-symbol term already says."""
    macro = DerivedAddressing(name="renamed one along", at=A_MACRO)
    widen_with_addressing("renamed one along", macro)
    assert macro.growth == A_SHORTER_NAME

    awkward = DerivedAddressing(name="an awkward one", at=AN_AWKWARD_ONE)
    widen_with_addressing("an awkward one", awkward)
    assert awkward.growth == A_NEW_DISTINCTION
    assert awkward.growth_evidence.finished


def test_a_word_with_nothing_observed_is_not_judged():
    empty = DerivedAddressing(name="nothing seen", at={})
    widen_with_addressing("nothing seen", empty)
    assert empty.growth == ""
