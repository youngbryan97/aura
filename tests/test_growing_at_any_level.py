"""Adding a way of building ways of building, through the code that adds a word.

The claim under test is that there is no tower. If level 2 needs its own
mechanism, the next level needs another, and it never ends — so every test here
is really the same test: the level is a number and not a special case.
"""

from __future__ import annotations

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition import growing_at_any_level as levels
from core.cognition import widening_the_language as widening


def _three_in_a_row(size: int) -> tuple[int, ...]:
    """Far end, then one back, then one back — three words deep."""
    return tuple(((size - 1 - index) - 2) % size for index in range(size))


STATES = [
    (1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3),
    (5, 1, 9, 3, 7), (8, 2, 6, 4, 9), (3, 7, 1, 5, 2),
]
THREE_DEEP = [
    (state, tuple(state[where] for where in _three_in_a_row(len(state))))
    for state in STATES
]


@pytest.fixture(autouse=True)
def _levels_left_as_found():
    ways, registry = dict(kinds.WAYS_TO_BUILD), dict(levels.REGISTRY)
    kinds.WAYS_TO_BUILD.clear()
    levels.REGISTRY.clear()
    try:
        yield
    finally:
        for holds, was in ((kinds.WAYS_TO_BUILD, ways), (levels.REGISTRY, registry)):
            holds.clear()
            holds.update(was)


def _sayable() -> bool:
    return kinds.induce_from(THREE_DEEP) is not None


def test_three_words_deep_is_unsayable_as_given():
    assert not _sayable()


def test_one_level_of_growth_is_not_enough_for_a_three_deep_family():
    kinds.WAYS_TO_BUILD["one after another"] = widening.one_after_another
    assert not _sayable()


def test_growing_two_levels_at_once_makes_it_sayable():
    kept = levels.grow_until_sayable(
        [
            (1, "one after another", widening.one_after_another),
            (2, "twice over", levels.twice_over),
        ],
        now_sayable=_sayable,
    )
    assert _sayable()
    assert levels.how_far_up_it_goes() == 2
    assert {one.level for one in kept} == {1, 2}


def test_what_it_reached_answers_a_state_it_never_saw():
    levels.grow_until_sayable(
        [
            (1, "one after another", widening.one_after_another),
            (2, "twice over", levels.twice_over),
        ],
        now_sayable=_sayable,
    )
    found = kinds.induce_from(THREE_DEEP)
    assert found is not None
    unseen = (9, 8, 7, 6, 5)
    assert found.read(unseen) == tuple(
        unseen[where] for where in _three_in_a_row(len(unseen))
    )


def test_a_stack_that_does_not_help_is_rolled_back_whole():
    """Nothing is kept when the top of the stack still cannot say it."""
    nothing_helps = [((1, 2), (5, 5)), ((3, 4), (9, 9)), ((5, 6), (1, 1)),
                     ((7, 8), (2, 2)), ((9, 1), (3, 3)), ((2, 3), (7, 7))]

    kept = levels.grow_until_sayable(
        [
            (1, "one after another", widening.one_after_another),
            (2, "twice over", levels.twice_over),
        ],
        now_sayable=lambda: kinds.induce_from(nothing_helps) is not None,
    )
    assert kept == ()
    assert levels.REGISTRY == {}
    assert kinds.WAYS_TO_BUILD == {}


def test_a_level_that_was_not_needed_is_cut_back_out():
    """Two levels go in together; only the ones the answer needs stay."""
    two_deep = [
        (state, tuple(state[(len(state) - 1 - index - 1) % len(state)]
                      for index in range(len(state))))
        for state in STATES
    ]
    kept = levels.grow_until_sayable(
        [
            (1, "one after another", widening.one_after_another),
            (2, "twice over", levels.twice_over),
        ],
        now_sayable=lambda: kinds.induce_from(two_deep) is not None,
    )
    assert kinds.induce_from(two_deep) is not None
    assert 2 not in {one.level for one in kept}


def test_nothing_is_admitted_when_it_was_already_sayable():
    already = [
        (state, tuple(reversed(state))) for state in STATES
    ]
    kept = levels.grow_until_sayable(
        [(1, "one after another", widening.one_after_another)],
        now_sayable=lambda: kinds.induce_from(already) is not None,
    )
    assert kept == ()


def test_a_level_two_maker_multiplies_what_level_zero_reaches():
    plain = levels.what_it_reaches(0)
    levels.REGISTRY["one after another"] = levels.Maker(
        "one after another", 1, widening.one_after_another
    )
    levels._publish(1)
    once = levels.what_it_reaches(0)
    levels.REGISTRY["twice over"] = levels.Maker("twice over", 2, levels.twice_over)
    levels._publish(2)
    assert levels.what_it_reaches(0) > once > plain
