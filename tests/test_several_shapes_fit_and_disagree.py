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
