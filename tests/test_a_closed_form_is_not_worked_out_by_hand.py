"""A statistic with a closed form is computed, not transcribed.

LIVE 2026-08-19: "I have 17 experimental runs. 12 succeeded. What's the exact
95% Wilson score interval for the success rate?" The reply wrote the formula
out, substituted by hand across nine lines, arrived at (0.37, 1.00), capped
the upper bound "since a probability can't exceed 1", and closed by saying
both that it had computed the interval and that the interval was an estimate.

The answer is 0.4687 to 0.8672. The Wilson bounds lie inside [0, 1] by
construction, so a bound above 1 is arithmetic that went wrong rather than a
bound that needs capping — and the process it ran in has Python.
"""

from __future__ import annotations

import math
import statistics as stdlib_statistics

import pytest

from core.conversation.computable_statistics import (
    computed_statistic,
    computed_statistic_result,
    statistic_form_failures,
    wilson_interval,
)


def test_every_declared_form_answers_its_own_examples():
    """The registry checks itself; this is what makes the values trustworthy."""
    assert statistic_form_failures() == []


def test_the_live_question_is_answered_exactly():
    question = (
        "I have 17 experimental runs. 12 succeeded. What's the exact 95% "
        "Wilson score interval for the success rate?"
    )
    low, high = wilson_interval(12, 17)

    assert computed_statistic(question) == f"{round(low, 4)} to {round(high, 4)}"
    assert (round(low, 4), round(high, 4)) == (0.4687, 0.8672)


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 10), (10, 10), (1, 3), (12, 17), (500, 1000), (1, 100000)],
)
def test_the_interval_stays_inside_zero_and_one(successes: int, trials: int):
    """The property the hand-worked version lost, over the whole range."""
    low, high = wilson_interval(successes, trials)

    assert 0.0 <= low <= high <= 1.0


@pytest.mark.parametrize(
    ("successes", "trials", "level"),
    [(12, 17, 95), (5, 10, 90), (30, 40, 99), (2, 7, 80)],
)
def test_the_interval_matches_the_formula_computed_independently(
    successes: int, trials: int, level: int
):
    """Recomputed here from the definition rather than copied from the module."""
    z = {80: 1.2815515655446004, 90: 1.6448536269514722,
         95: 1.959963984540054, 99: 2.5758293035489004}[level]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = (z / denominator) * math.sqrt(
        p * (1 - p) / trials + z * z / (4.0 * trials * trials)
    )

    low, high = wilson_interval(successes, trials, level)

    assert low == pytest.approx(max(0.0, centre - spread))
    assert high == pytest.approx(min(1.0, centre + spread))


def test_the_counts_are_read_however_the_sentence_arranges_them():
    """"12 of 17" and "17 runs, 12 succeeded" are the same question."""
    direct = computed_statistic("wilson interval for 12 of 17")
    apart = computed_statistic(
        "I ran 17 trials and 12 succeeded — wilson score interval please"
    )

    assert direct == apart == "0.4687 to 0.8672"


def test_a_question_about_the_concept_computes_nothing():
    assert computed_statistic("what is a wilson score interval") is None
    assert computed_statistic("explain what standard deviation measures") is None


def test_a_spread_says_which_spread_it_is():
    """Sample and population differ by more than rounding at this size."""
    values = [2, 4, 4, 4, 5, 5, 7, 9]
    answer = computed_statistic("standard deviation of 2, 4, 4, 4, 5, 5, 7, 9")

    assert answer is not None
    assert str(round(stdlib_statistics.stdev(values), 4)) in answer
    assert str(round(stdlib_statistics.pstdev(values), 4)) in answer
    assert "sample" in answer and "population" in answer


def test_the_statistic_names_what_produced_it():
    result = computed_statistic_result("what is the mean of 2, 4, 4, 4, 5, 5, 7, 9")

    assert result is not None
    assert result.function == "_mean"
    assert "not generated" in result.provenance()


def test_one_registration_is_the_only_wiring_step():
    """The reader reaches turn ownership, self-knowledge and grounding.

    Statistics were added by declaring one reader. If any of these three has
    to be edited per capability, the next one will be half-wired.
    """
    from core.brain.observable_registry import observable_names
    from core.conversation.turn_ownership import owning_readers
    from core.self.capability_lexicon import capabilities_named_in

    assert "statistics" in owning_readers("wilson interval for 12 of 17")
    named = [m.skill for m in capabilities_named_in("can you compute a standard deviation")]
    assert "statistics" in named
    assert "computed_statistic" in observable_names()
