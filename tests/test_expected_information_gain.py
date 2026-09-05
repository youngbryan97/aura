"""Choosing what to look at, rather than describing what arrived.

Aura's perception is receptive: a camera sends an image and something
describes it. The observations she gets are the ones that happened to arrive.
The version worth having chooses — she is unsure whether X is true, some
observations would discriminate and most would not, and she takes the one that
would.

The property that matters most is the one a salience-based system gets wrong:
an observation every hypothesis predicts identically has zero gain however
interesting it looks.
"""

from __future__ import annotations

import pytest

from core.perception.expected_information_gain import (
    SETTLED_BELOW_BITS,
    Observation,
    Recommendation,
    best,
    choose,
    entropy,
    expected_information_gain,
    posterior,
    score,
)

THREE_WAY = {"mug": 1 / 3, "phone": 1 / 3, "nothing": 1 / 3}

CAMERA = Observation(
    "move the camera left",
    {
        "sees_cylinder": {"mug": 0.9, "phone": 0.05, "nothing": 0.02},
        "sees_flat": {"mug": 0.05, "phone": 0.9, "nothing": 0.02},
        "sees_bare": {"mug": 0.05, "phone": 0.05, "nothing": 0.96},
    },
    cost=0.1,
)
TEMPERATURE = Observation(
    "check the room temperature",
    {
        "warm": {"mug": 0.5, "phone": 0.5, "nothing": 0.5},
        "cool": {"mug": 0.5, "phone": 0.5, "nothing": 0.5},
    },
    cost=0.01,
)
SCAN = Observation(
    "take a full 3D scan",
    {
        "sees_cylinder": {"mug": 0.99, "phone": 0.005, "nothing": 0.005},
        "sees_flat": {"mug": 0.005, "phone": 0.99, "nothing": 0.005},
        "sees_bare": {"mug": 0.005, "phone": 0.005, "nothing": 0.99},
    },
    cost=5.0,
)


# ── the arithmetic ───────────────────────────────────────────────────────


def test_entropy_is_zero_when_one_hypothesis_has_the_mass():
    assert entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0, abs=1e-6)


def test_entropy_is_maximal_when_nothing_is_known():
    assert entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert entropy(THREE_WAY) == pytest.approx(1.585, abs=0.01)


def test_a_posterior_moves_toward_what_the_outcome_favours():
    after = posterior({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1})
    assert after["a"] > 0.8


def test_no_hypothesis_is_ever_driven_to_exactly_zero():
    """One that cannot be revived by evidence is not a belief state."""
    after = posterior({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0})
    assert after["b"] > 0.0


# ── the property a salience-based system gets wrong ──────────────────────


def test_an_observation_every_hypothesis_predicts_alike_gains_nothing():
    assert expected_information_gain(THREE_WAY, TEMPERATURE) == pytest.approx(0.0, abs=1e-9)
    assert score(THREE_WAY, TEMPERATURE).recommendation is Recommendation.UNINFORMATIVE


def test_a_discriminating_observation_gains_most_of_the_entropy():
    gain = expected_information_gain(THREE_WAY, CAMERA)
    assert gain > 1.0
    assert gain < entropy(THREE_WAY)


def test_a_more_precise_instrument_gains_more():
    assert expected_information_gain(THREE_WAY, SCAN) > expected_information_gain(
        THREE_WAY, CAMERA
    )


# ── gain is not worth ────────────────────────────────────────────────────


def test_an_observation_that_settles_it_and_costs_too_much_is_refused():
    verdict = score(THREE_WAY, SCAN)
    assert verdict.expected_bits > 1.4
    assert verdict.recommendation is Recommendation.TOO_EXPENSIVE


def test_the_same_observation_flips_on_what_the_question_is_worth():
    assert score(THREE_WAY, SCAN, value_per_bit=1.0).recommendation is (
        Recommendation.TOO_EXPENSIVE
    )
    assert score(THREE_WAY, SCAN, value_per_bit=100.0).recommendation is (
        Recommendation.TAKE
    )


def test_the_best_observation_is_the_one_worth_most_not_the_one_that_gains_most():
    chosen = best(THREE_WAY, [CAMERA, TEMPERATURE, SCAN])
    assert chosen.observation == "move the camera left"
    assert expected_information_gain(THREE_WAY, SCAN) > chosen.expected_bits


# ── settled ──────────────────────────────────────────────────────────────


def test_a_settled_question_is_not_worth_looking_at():
    settled = {"mug": 0.9999, "phone": 5e-5, "nothing": 5e-5}
    assert entropy(settled) < SETTLED_BELOW_BITS
    assert score(settled, CAMERA).recommendation is Recommendation.SETTLED


def test_settled_is_reachable_and_distinct_from_uninformative():
    """It was unreachable: the noise floor was reused for a real question, and
    a belief at 99.9999% still carries more entropy than floating-point error."""
    nearly = {"x": 0.999, "y": 0.001}
    assert entropy(nearly) < SETTLED_BELOW_BITS
    assert score(nearly, CAMERA).recommendation is Recommendation.SETTLED
    assert score(THREE_WAY, TEMPERATURE).recommendation is Recommendation.UNINFORMATIVE


def test_nothing_worth_looking_at_returns_none():
    assert best(THREE_WAY, [TEMPERATURE]) is None
    assert best(THREE_WAY, []) is None


# ── ranking hygiene ──────────────────────────────────────────────────────


def test_the_ranking_does_not_depend_on_the_order_they_were_written_down():
    forward = [s.observation for s in choose(THREE_WAY, [CAMERA, TEMPERATURE, SCAN])]
    backward = [s.observation for s in choose(THREE_WAY, [SCAN, TEMPERATURE, CAMERA])]
    assert forward == backward


def test_every_recommendation_reports_why():
    for candidate in choose(THREE_WAY, [CAMERA, TEMPERATURE, SCAN]):
        assert candidate.because
