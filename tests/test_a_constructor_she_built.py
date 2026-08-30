"""Building a way of making words, rather than activating one from the source.

Deciding that a dormant designer-written constructor deserves admission grows
what she uses and never what she has. These pin the difference: what comes out
here is described by a recipe she composed, and the source does not contain it.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition import widening_the_language as widening
from core.cognition.a_constructor_she_built import (
    IN_SEQUENCE,
    OVER_AND_OVER,
    UNDONE,
    Recipe,
    a_constructor_she_built,
    build,
    every_recipe,
    read_back,
    written_down,
)


def _chain(size: int) -> tuple[int, ...]:
    """Far end, then one back, then one back — three words deep."""
    return tuple(((size - 1 - index) - 2) % size for index in range(size))


STATES = [
    (1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3),
    (5, 1, 9, 3, 7), (8, 2, 6, 4, 9), (3, 7, 1, 5, 2),
]
THREE_DEEP = [
    (state, tuple(state[where] for where in _chain(len(state)))) for state in STATES
]


@pytest.fixture(autouse=True)
def _ways_left_as_found():
    ways = dict(kinds.WAYS_TO_BUILD)
    kinds.WAYS_TO_BUILD.clear()
    try:
        yield
    finally:
        kinds.WAYS_TO_BUILD.clear()
        kinds.WAYS_TO_BUILD.update(ways)


def _sayable() -> bool:
    return kinds.induce_from(THREE_DEEP) is not None


def test_she_builds_a_constructor_that_makes_the_family_sayable():
    assert not _sayable()
    built = a_constructor_she_built(THREE_DEEP, now_sayable=_sayable)
    assert built is not None
    assert _sayable()


def test_what_she_built_is_not_in_the_source_registry():
    """The claim under test. A name can only ever resolve to what source has."""
    built = a_constructor_she_built(THREE_DEEP, now_sayable=_sayable)
    assert built is not None
    assert built.name not in widening.CONSTRUCTORS
    assert f"a way she built: {built.name}" not in widening.CONSTRUCTORS


def test_the_depth_is_read_off_the_problem_and_not_chosen():
    """States five long cannot need a chain longer than five."""
    deepest = max(recipe.depth for recipe in every_recipe(5))
    assert deepest == 5
    assert max(recipe.depth for recipe in every_recipe(3)) == 3


def test_nothing_is_built_when_the_family_was_already_sayable():
    already = [(state, tuple(reversed(state))) for state in STATES]
    assert (
        a_constructor_she_built(
            already, now_sayable=lambda: kinds.induce_from(already) is not None
        )
        is None
    )


def test_a_recipe_that_helps_nothing_leaves_the_language_as_it_was():
    nothing_says_this = [
        ((1, 2), (5, 5)), ((3, 4), (9, 9)), ((5, 6), (1, 1)),
        ((7, 8), (2, 2)), ((9, 1), (3, 3)), ((2, 3), (7, 7)),
    ]
    built = a_constructor_she_built(
        nothing_says_this,
        now_sayable=lambda: kinds.induce_from(nothing_says_this) is not None,
    )
    assert built is None
    assert kinds.WAYS_TO_BUILD == {}


def test_a_recipe_survives_being_written_down_and_read_back():
    recipe = Recipe(kind=IN_SEQUENCE, depth=3, then=Recipe(kind=UNDONE))
    assert read_back(written_down(recipe)) == recipe


def test_nothing_outside_the_three_ways_can_be_read_back():
    assert read_back({"kind": "os.system", "depth": 2}) is None
    assert read_back({"kind": IN_SEQUENCE, "depth": "not a number"}) is None
    assert read_back("not a recipe") is None


def test_undoing_a_word_is_worked_out_at_whatever_size_it_is_asked():
    made = build(Recipe(kind=UNDONE))(dict(kinds.WHERE_FROM))
    undone = made["one along, undone"]
    for size in (3, 4, 7, 11):
        assert all(
            undone(kinds.WHERE_FROM["one along"](index, size) % size, size) == index
            for index in range(size)
        )


def test_undoing_refuses_a_word_that_does_not_move_things_one_for_one():
    def everything_to_here(_index: int, _size: int) -> int:
        return 0

    made = build(Recipe(kind=UNDONE))({"everything to here": everything_to_here})
    with pytest.raises(ValueError):
        made["everything to here, undone"](1, 4)


def test_a_recipe_may_be_followed_by_another_so_the_space_is_a_closure():
    plain = len(build(Recipe(kind=IN_SEQUENCE, depth=2))(dict(kinds.WHERE_FROM)))
    onward = len(
        build(Recipe(kind=IN_SEQUENCE, depth=2, then=Recipe(kind=UNDONE)))(
            dict(kinds.WHERE_FROM)
        )
    )
    assert onward > plain


def test_over_and_over_applies_a_word_to_its_own_output():
    made = build(Recipe(kind=OVER_AND_OVER, depth=3))(dict(kinds.WHERE_FROM))
    thrice = made["one along, 3 times over"]
    once = kinds.WHERE_FROM["one along"]
    for size in (4, 5, 9):
        for index in range(size):
            at = index
            for _ in range(3):
                at = once(at, size) % size
            assert thrice(index, size) == at
