"""Nothing fitting is not evidence that a word is missing.

It is equally what a search that went badly looks like, and writing a new way
of building words in answer to that is an expensive way of being wrong. More
compute against a hypothesis space that does contain the answer is the right
response; more compute against one that does not buys nothing.

What tells them apart is what is left over. If something the language already
says accounts for exactly the cases the best reading missed, then both halves
are sayable and the whole is not — and what is missing is a way to join them.
"""

from __future__ import annotations

import random

import pytest

from core.cognition import an_invented_kind as kinds
from core.cognition.what_the_failures_have_in_common import (
    A_REPRESENTATION_FAILURE,
    A_SEARCH_FAILURE,
    NOTHING_FAILED,
    why_nothing_fits,
)

FOURS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 6), (4, 7, 2, 8)]
FIVES = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 1), (2, 4, 6, 8, 3), (5, 1, 9, 3, 7)]


@pytest.fixture(autouse=True)
def _the_language_she_was_given():
    was = dict(kinds.WAYS_TO_BUILD)
    kinds.WAYS_TO_BUILD.clear()
    try:
        yield
    finally:
        kinds.WAYS_TO_BUILD.clear()
        kinds.WAYS_TO_BUILD.update(was)


def test_something_the_language_says_is_not_a_failure_at_all():
    said = why_nothing_fits([(one, tuple(reversed(one))) for one in FOURS])
    assert said.because == NOTHING_FAILED


def test_two_halves_she_can_say_and_a_whole_she_cannot_is_the_language():
    far = kinds.WHERE_FROM["the far end"]
    along = kinds.WHERE_FROM["one along"]

    def either_way(at, size):
        return far(at, size) if size % 2 == 0 else along(at, size)

    family = [
        (one, tuple(one[either_way(at, len(one)) % len(one)] for at in range(len(one))))
        for one in FOURS + FIVES
    ]
    said = why_nothing_fits(family)
    assert said.because == A_REPRESENTATION_FAILURE
    assert said.together_on > 0


def test_leftovers_nothing_accounts_for_are_not_a_missing_word():
    random.seed(7)
    noise = [
        (one, tuple(random.sample(list(one), len(one)))) for one in FOURS + FIVES
    ]
    assert why_nothing_fits(noise).because == A_SEARCH_FAILURE


def test_nothing_to_read_says_so_rather_than_guessing():
    from core.cognition.what_the_failures_have_in_common import UNDECIDED

    assert why_nothing_fits([]).because == UNDECIDED


def test_the_answering_path_asks_before_it_writes():
    """Writing a way of building words in answer to bad luck is expensive."""
    import inspect

    from core.cognition import sequence_induction

    source = inspect.getsource(sequence_induction)
    # By name, not by argument list. Both calls took a bare ``pairs`` when this
    # was written and both take a field off a situation object now, which is
    # the same contract with a different spelling — and pinning the spelling
    # reported the refactor as the writer having been removed.
    asked = source.index("why_nothing_fits(")
    wrote = source.index("a_maker_she_wrote(")
    assert asked < wrote
    assert "if not why.is_the_language" in source


# --- and a second opinion, from compression ------------------------------


def test_failures_that_repeat_score_above_a_shuffled_control():
    from core.cognition.what_the_failures_have_in_common import (
        how_much_the_failures_share,
    )

    assert how_much_the_failures_share([-8] * 8) > 0


@pytest.mark.parametrize("how_many", [8, 16, 24])
def test_failures_that_share_nothing_score_nothing(how_many):
    """A compressor finds savings in the digits alone; the control removes them.

    Over several draws, because one unlucky sample proves nothing either way.
    """
    from core.cognition.what_the_failures_have_in_common import (
        how_much_the_failures_share,
    )

    drawn = random.Random(how_many)
    scored = [
        how_much_the_failures_share(
            [drawn.randint(-999, 999) for _ in range(how_many)]
        )
        for _ in range(20)
    ]
    # The floor the null actually reaches is what it is measured against, so
    # a stray positive is possible and a run of them is not.
    assert sum(1 for one in scored if one > 1) <= 1, scored


@pytest.mark.parametrize("leftovers", [[-8], []])
def test_too_few_leftovers_to_compare_score_nothing(leftovers):
    from core.cognition.what_the_failures_have_in_common import (
        how_much_the_failures_share,
    )

    assert how_much_the_failures_share(leftovers) == 0


def test_a_ramp_is_structured_and_scores_nothing_here():
    """Stated as a limit rather than hidden: this sees repetition, not algebra.

    A ramp is perfectly structured and no two of its cases are alike, so a
    compressor finds nothing shared. That is why this is a second opinion and
    the coverage test is the verdict.
    """
    from core.cognition.what_the_failures_have_in_common import (
        how_much_the_failures_share,
    )

    assert how_much_the_failures_share(list(range(8))) <= 0


def test_the_verdict_does_not_rest_on_it():
    """The coverage test decides; compression is reported beside it."""
    import inspect

    from core.cognition import what_the_failures_have_in_common as telling

    source = inspect.getsource(telling)
    decided = source.index("because=A_REPRESENTATION_FAILURE if joins_up")
    scored = source.index("shared=how_much_the_failures_share")
    assert decided < scored
