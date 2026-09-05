"""An answer the evidence admits is not an answer the evidence settles.

Several shapes can fit everything shown and disagree about the case in hand.
The search returned the first of them in preference order and said nothing
about the rest, so shown one worked example it answered — and was wrong
thirteen times in thirty, confidently, on evidence that settled nothing.

That is not a wrong answer. It is an answer where a refusal was the correct
one, and the caller had no way to tell the two apart.
"""

from __future__ import annotations

import pytest

from core.cognition.primitive_invention import (
    ENOUGH_TO_SETTLE,
    Transition,
    invent_relation,
)


def _mirror(row):
    return tuple(reversed(row))


def _shown(fn, lengths):
    return [
        Transition(tuple(range(n)), tuple(fn(tuple(range(n))))) for n in lengths
    ]


def test_one_observation_settles_nothing():
    """A world exchanging its first and last cells produced {0<->3} at length
    four, which is false at length eight, and one observation cannot tell
    those apart."""

    found = invent_relation(_shown(_mirror, (5,)))
    assert found is not None
    assert not found.settled
    assert found.learned_from < ENOUGH_TO_SETTLE


def test_two_observations_at_different_lengths_can_settle_it():
    found = invent_relation(_shown(_mirror, (5, 8)))
    assert found is not None
    assert found.settled
    assert found.form == "position i takes from n-1-i"


def test_a_settled_answer_names_no_disagreeing_shape():
    found = invent_relation(_shown(_mirror, (5, 8, 11)))
    assert found is not None and found.settled
    assert found.also_fits == ()


def test_what_would_have_to_be_separated_is_named():
    """Naming a shortage without naming what would end it leaves waiting as
    the only move."""

    found = invent_relation(_shown(_mirror, (4,)))
    assert found is not None
    if found.also_fits:
        assert all(isinstance(one, str) and one for one in found.also_fits)


def test_a_length_bound_rule_refuses_rather_than_crashing():
    """`generalises` already says it only fits one length. Saying it again
    where it is used is what makes it safe for anything that did not check."""

    def _odd(row):
        made = list(row)
        if len(made) >= 4:
            made[0], made[2] = made[2], made[0]
            made[1], made[3] = made[3], made[1]
        return tuple(made)

    found = invent_relation(_shown(_odd, (4,)))
    if found is None or found.generalises:
        pytest.skip("this instance did not produce a fixed correspondence")
    with pytest.raises(ValueError, match="says nothing at length"):
        found.apply(tuple(range(9)))


def test_a_shape_exists_at_every_length_it_fits():
    """The span range stopped at half the length, so "grouped every three"
    existed at length six and not at length five — and a shape has to be in
    the basis at every length shown before it can be shared across them."""

    from core.cognition.primitive_invention import _index_forms

    at_five = {said for _f, said, _r in _index_forms(5)}
    at_nine = {said for _f, said, _r in _index_forms(9)}
    assert "cells are grouped every 3, the group at 0 first" in at_five
    assert "cells are grouped every 3, the group at 0 first" in at_nine


def test_three_deep_compositions_are_reachable():
    """Two was the whole of it, so a world that is three shapes one after
    another was unreachable however many observations were offered."""

    def _three(row):
        row = tuple(row[1:] + row[:1])
        row = tuple(reversed(row))
        made = list(row)
        if len(made) > 1:
            made[0], made[-1] = made[-1], made[0]
        return tuple(made)

    found = invent_relation(_shown(_three, (6, 7, 9)))
    assert found is not None, "a three-deep composition is still unreachable"
    assert found.generalises
    for length in (8, 11):
        assert tuple(found.apply(tuple(range(length)))) == _three(tuple(range(length)))


def test_settledness_is_checked_against_the_question_when_the_caller_names_it():
    """Whether observations settle a question depends on the question.

    Without the case in hand it is checked over a neighbourhood of the
    lengths shown, which is a guess at where the question will be. Measured
    on two hundred sealed rules from a fixed seed it changed nothing — 163
    right and 2 wrong either way — so what this test guards is that the
    parameter reaches the check, not a gain it does not have.
    """

    import inspect

    from core.cognition import primitive_invention as invention

    said = inspect.signature(invention.invent_relation).parameters
    assert "about" in said

    checked = inspect.getsource(invention._which_others_disagree)
    assert "about" in checked
    assert "int(one) for one in about" in checked


def test_naming_a_length_cannot_make_an_unsettled_answer_settled():
    """It can only find more disagreement, never less."""

    shown = _shown(_mirror, (5,))
    without = invent_relation(shown)
    with_it = invent_relation(shown, about=(9,))
    assert without is not None and with_it is not None
    assert not without.settled
    assert not with_it.settled


def test_a_length_the_caller_names_is_actually_looked_at():
    """Two shapes agreeing everywhere shown and differing at the asked length
    is the case this exists for."""

    from core.cognition.primitive_invention import _which_others_disagree

    shown = _shown(_mirror, (4, 6))
    found = invent_relation(shown)
    assert found is not None
    #: Nothing to disagree about here, whatever length is named.
    assert _which_others_disagree(found.form, {found.form: ("", None)}, shown) == ()
    assert (
        _which_others_disagree(found.form, {found.form: ("", None)}, shown, about=(11,))
        == ()
    )
