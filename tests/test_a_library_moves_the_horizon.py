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
    MINUS,
    PLUS,
    L,
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


def _also_comes_from_her_library(function) -> bool:
    """Does this function pass her library as ``also``, however it is written?

    Asked of the tree rather than of the text. The first version of this looked
    for the literal ``also=what_she_already_knows_how_to_say()`` and went red
    when the call was hoisted into a variable so the search could also report
    how many of the leaves came from her — a refactor that changed nothing about
    what reaches ``also``. A test that fails on the shape of the source rather
    than on what the source does will fail again on the next honest edit.
    """
    tree = ast.parse(inspect.getsource(function).lstrip())
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        called = node.value.func
        name = called.id if isinstance(called, ast.Name) else getattr(called, "attr", "")
        if name != "what_she_already_knows_how_to_say":
            continue
        bound.update(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "also":
                continue
            value = keyword.value
            if isinstance(value, ast.Name) and value.id in bound:
                return True
            if isinstance(value, ast.Call):
                called = value.func
                name = (
                    called.id
                    if isinstance(called, ast.Name)
                    else getattr(called, "attr", "")
                )
                if name == "what_she_already_knows_how_to_say":
                    return True
    return False


def test_the_operator_search_is_offered_the_library():
    """The call that walked 380 terms forever."""
    from core.cognition import an_operator_she_invents

    assert _also_comes_from_her_library(an_operator_she_invents._a_candidate_for)


def test_the_operator_search_says_how_far_it_reached():
    """A search that reports only what it examined implies it looked at what
    mattered. Over the bare floor at depth three there are 380 terms against a
    cap of four thousand, so the walk is exhaustive and the horizon is the
    constraint — which is only visible if the leaf count is reported."""
    from core.cognition.an_operator_she_invents import _NOTE_THE_REACH, _a_candidate_for

    list(itertools.islice(_a_candidate_for("fam", ()), 1))
    assert _NOTE_THE_REACH.get("leaves", 0) >= 5
    assert "from_her_library" in _NOTE_THE_REACH


def test_the_action_writer_is_offered_the_library():
    from core.cognition import an_action_she_writes_for_a_gap

    source = inspect.getsource(an_action_she_writes_for_a_gap)
    assert "also=what_she_already_knows_how_to_say()" in source


def test_an_empty_library_is_not_an_error():
    """At boot she has learned nothing, and the search must still run."""
    assert _reach(what_she_already_knows_how_to_say()) >= 3
