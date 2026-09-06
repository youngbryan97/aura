"""The search composes over what she has learned, not only over primitives.

``every_code`` says it in its own docstring: ``also`` is "the only channel by
which a long term becomes reachable — shortest-first over a universal
language reaches a few dozen symbols and no further, which is Levin's bound
rather than a defect here, and a library is what moves the horizon rather
than a bigger budget."

Almost every caller passed ``also=()``. The autonomous operator search walked
the same 380 terms at depth three forever, over a computationally universal
language — which is the gap an external review named between being able to
REPRESENT an improvement and being able to FIND one. The cause was not
budget. It was that the documented channel for moving the horizon was fed
nothing.
"""

from __future__ import annotations

import ast
import inspect
import itertools

import pytest

from core.cognition.the_floor_she_stands_on import (
    L,
    MINUS,
    PLUS,
    V,
    build,
    every_code,
    how_long,
)
from core.cognition.what_she_already_knows_how_to_say import (
    how_far_the_search_reaches,
    what_she_already_knows_how_to_say,
)


def _reach(also, *, walk: int = 4000) -> int:
    """The longest term the search produces from these leaves, at depth three."""
    return max(
        how_long(one)
        for one in itertools.islice(
            every_code(deepest=3, variables=1, constants=(0, 1, 2), also=also), walk
        )
    )


def test_a_library_reaches_further_than_the_primitives_alone():
    """The measurement the whole idea rests on."""
    bare = _reach(())
    with_one = _reach((build(L("a", L("b", PLUS(V("a"), MINUS(V("b"), V("a")))))),))
    assert with_one > bare, f"a learned leaf bought nothing: {with_one} vs {bare}"
    assert bare == 3


@pytest.mark.parametrize(
    ("term", "leaf_symbols"),
    [
        (L("a", PLUS(V("a"), V("a"))), 4),
        (L("a", L("b", PLUS(V("a"), MINUS(V("b"), V("a"))))), 7),
    ],
)
def test_reach_is_twice_the_longest_leaf_plus_one(term, leaf_symbols):
    """The reported number is the measured one, not an authored one.

    The first version of `how_far_the_search_reaches` said n*4+3 and would
    have reported 31 where the search reaches 15.
    """
    built = build(term)
    assert how_long(built) == leaf_symbols
    assert _reach((built,)) == 2 * leaf_symbols + 1


def test_the_report_matches_what_the_search_does():
    said = how_far_the_search_reaches()
    assert said["reach_at_depth_three"] == 2 * said["longest_leaf_symbols"] + 1
    assert said["reach_with_no_library"] == _reach(())


def test_the_library_is_bounded():
    """Enumeration is combinatorial in the leaf count.

    An unbounded library turns a bounded search into an unbounded one, which
    is the same defect as a fixed horizon arriving from the other side.
    """
    assert len(what_she_already_knows_how_to_say(most=3)) <= 3
    assert what_she_already_knows_how_to_say(most=0) == ()


def test_the_library_is_longest_first():
    """Reach is the reason to offer them, and length is what buys reach."""
    lengths = [how_long(one) for one in what_she_already_knows_how_to_say()]
    assert lengths == sorted(lengths, reverse=True)


def test_the_operator_search_is_offered_the_library():
    """The call that walked 380 terms forever."""
    from core.cognition import an_operator_she_invents

    source = inspect.getsource(an_operator_she_invents._a_candidate_for)
    tree = ast.parse(source.lstrip())
    assert "also=what_she_already_knows_how_to_say()" in ast.unparse(tree)


def test_the_action_writer_is_offered_the_library():
    from core.cognition import an_action_she_writes_for_a_gap

    source = inspect.getsource(an_action_she_writes_for_a_gap)
    assert "also=what_she_already_knows_how_to_say()" in source


def test_an_empty_library_is_not_an_error():
    """At boot she has learned nothing, and the search must still run."""
    assert _reach(what_she_already_knows_how_to_say()) >= 3
