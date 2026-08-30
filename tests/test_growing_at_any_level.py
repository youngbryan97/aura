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


def test_a_bare_run_of_numbers_is_read_as_a_sequence():
    """How a person writes one when they are not writing code."""
    from core.cognition.sequence_induction import read_sequence_question

    asked = read_sequence_question(
        "1 2 3 4 5 becomes 5 4 3 2 1. 6 7 8 9 1 becomes 1 9 8 7 6. "
        "What does 9 8 7 6 5 become?"
    )
    assert asked is not None
    assert asked.shown[0].before == (1, 2, 3, 4, 5)
    assert asked.shown[0].after == (5, 4, 3, 2, 1)
    assert asked.asked == (9, 8, 7, 6, 5)


def test_the_last_number_before_a_full_stop_is_not_eaten():
    from core.cognition.sequence_induction import read_sequence_question

    asked = read_sequence_question(
        "1 2 3 becomes 3 2 1. 4 5 6 becomes 6 5 4. What does 7 8 9 become?"
    )
    assert asked is not None
    assert all(len(one.before) == len(one.after) == 3 for one in asked.shown)


@pytest.mark.parametrize(
    "prose",
    [
        "I paid 12 50 for lunch and 3 40 for coffee, then walked 2 miles home.",
        "There were 3 4 5 people in the room over three days.",
        "I walked 3.5 6 7 miles and paid 2.25 for a coffee.",
    ],
)
def test_prose_with_numbers_in_it_is_not_read_as_examples(prose):
    from core.cognition.sequence_induction import read_sequence_question

    assert read_sequence_question(prose) is None


def test_the_answering_path_grows_two_levels_when_the_family_needs_them():
    """End to end: unsayable, grown, answered, and said out loud."""
    from core.cognition.sequence_induction import answer_sequence_question

    def rule(state):
        size = len(state)
        return tuple(
            max(state[(index + 2) % size], state[size - 1 - index])
            for index in range(size)
        )

    asked = (9, 8, 7, 6, 5)
    text = " ".join(
        f"{' '.join(map(str, state))} becomes {' '.join(map(str, rule(state)))}."
        for state in STATES
    )
    text += f" What does {' '.join(map(str, asked))} become?"

    said = answer_sequence_question(text)
    assert str(list(rule(asked))) in said
    assert levels.how_far_up_it_goes() == 2
