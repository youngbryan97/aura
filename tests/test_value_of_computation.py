"""More thinking is worth something only if it can change what she does.

Uncertainty on its own buys nothing. If the leading option is ahead by more
than further deliberation could plausibly close, thinking longer produces a
better-argued version of the same act, and the argument is not the
deliverable.

That turns an unanswerable question — how much better would the answer be? —
into a measurable one: how far has extra spend actually moved the leader
before. The record needs a null arm, because movement without spend is drift.
"""

from __future__ import annotations

import random

import pytest

from core.cognition.value_of_computation import (
    MIN_OBSERVATIONS,
    Swing,
    Worth,
    observe_deliberation,
    reset_swings,
    swing_record,
    worth_continuing,
    worth_learning,
)


def _record(seed=4, spent=30, idle=20, moves=0.10, drift=0.01, change_rate=0.3):
    rng = random.Random(seed)
    swing = Swing("test")
    for _ in range(spent):
        swing.observe(
            spend=1.0,
            movement=abs(rng.gauss(moves, moves * 0.4)),
            changed_decision=rng.random() < change_rate,
        )
    for _ in range(idle):
        swing.observe(spend=0.0, movement=abs(rng.gauss(drift, drift)))
    return swing


# ── the null arm ─────────────────────────────────────────────────────────


def test_with_no_record_it_says_unmeasured_not_no():
    judgement = worth_continuing(margin=0.05, cost=0.1, swing=Swing())
    assert judgement.worth is Worth.UNMEASURED
    assert judgement.spend is False
    assert judgement.plausible_swing is None


def test_a_record_with_no_null_arm_cannot_claim_the_movement_was_thinking():
    swing = Swing("no-null")
    for _ in range(MIN_OBSERVATIONS * 3):
        swing.observe(spend=1.0, movement=0.2)
    assert swing.plausible() is None, (
        "movement was reported as caused by thinking with nothing to compare "
        "it against"
    )
    assert worth_continuing(margin=0.01, cost=0.01, swing=swing).worth is Worth.UNMEASURED


def test_movement_that_matches_the_drift_is_not_movement():
    """Spending and not spending moved things equally. That is the wobble."""
    swing = _record(moves=0.02, drift=0.02)
    assert swing.plausible() is None
    assert worth_continuing(margin=0.001, cost=0.001, swing=swing).worth is Worth.UNMEASURED


def test_too_few_observations_cannot_be_a_distribution():
    swing = _record(spent=MIN_OBSERVATIONS - 1, idle=10)
    assert swing.plausible() is None


# ── the verdicts ─────────────────────────────────────────────────────────


def test_a_runaway_leader_settles_the_question():
    swing = _record()
    judgement = worth_continuing(margin=0.4, cost=0.01, stakes=10.0, swing=swing)
    assert judgement.worth is Worth.SETTLED
    assert judgement.spend is False
    assert judgement.expected_value == 0.0


def test_a_near_tie_worth_paying_for_is_worth_paying_for():
    swing = _record()
    judgement = worth_continuing(margin=0.03, cost=0.05, stakes=1.0, swing=swing)
    assert judgement.worth is Worth.WORTH
    assert judgement.spend is True


def test_a_near_tie_that_costs_more_than_it_is_worth_is_refused():
    swing = _record()
    judgement = worth_continuing(margin=0.03, cost=0.9, stakes=1.0, swing=swing)
    assert judgement.worth is Worth.TOO_EXPENSIVE
    assert judgement.expected_value < 0.0


def test_the_same_near_tie_flips_on_what_the_decision_is_worth():
    """Cost is only meaningful against stakes in the same unit."""
    swing = _record()
    cheap_stakes = worth_continuing(margin=0.03, cost=0.5, stakes=0.1, swing=swing)
    high_stakes = worth_continuing(margin=0.03, cost=0.5, stakes=100.0, swing=swing)
    assert cheap_stakes.worth is Worth.TOO_EXPENSIVE
    assert high_stakes.worth is Worth.WORTH


def test_only_worth_means_spend():
    for worth in Worth:
        assert worth.spend is (worth is Worth.WORTH)


# ── whether to learn at all ──────────────────────────────────────────────


def test_a_real_skill_with_a_real_gain_needed_twice_is_not_worth_learning():
    """A system that cannot ask this studies whatever is in front of it."""
    judgement = worth_learning(cost=5.0, gain_per_use=0.3, expected_uses=2)
    assert judgement.worth is Worth.TOO_EXPENSIVE
    assert judgement.expected_value < 0.0


def test_the_same_skill_needed_constantly_is_worth_learning():
    judgement = worth_learning(cost=5.0, gain_per_use=0.3, expected_uses=50)
    assert judgement.worth is Worth.WORTH


def test_something_forgotten_before_the_second_use_repays_once():
    kept = worth_learning(cost=1.0, gain_per_use=0.5, expected_uses=10, retention=1.0)
    lost = worth_learning(cost=1.0, gain_per_use=0.5, expected_uses=10, retention=0.0)
    assert kept.expected_value > lost.expected_value
    assert lost.expected_value == pytest.approx(-0.5)


def test_retention_compounds_rather_than_applying_once():
    high = worth_learning(cost=0.0, gain_per_use=1.0, expected_uses=10, retention=0.9)
    low = worth_learning(cost=0.0, gain_per_use=1.0, expected_uses=10, retention=0.5)
    assert high.expected_value > low.expected_value * 2


def test_a_fractional_use_count_is_handled():
    judgement = worth_learning(cost=0.0, gain_per_use=1.0, expected_uses=2.5, retention=0.5)
    assert 1.0 < judgement.expected_value < 2.5


# ── recording from a real deliberation ───────────────────────────────────


def test_a_deliberation_that_flipped_the_leader_is_recorded_as_a_change():
    reset_swings()
    observe_deliberation(
        scores_before=[0.50, 0.52], scores_after=[0.80, 0.30], spend=1.0, name="d"
    )
    spent, _idle = swing_record("d")._split()
    assert spent[0].changed_decision is True
    reset_swings()


def test_a_deliberation_that_only_widened_the_lead_is_not_a_change():
    reset_swings()
    observe_deliberation(
        scores_before=[0.80, 0.30], scores_after=[0.95, 0.20], spend=1.0, name="d"
    )
    spent, _idle = swing_record("d")._split()
    assert spent[0].changed_decision is False
    assert spent[0].movement > 0.0
    reset_swings()


def test_records_are_named_and_separate():
    reset_swings()
    swing_record("a").observe(spend=1.0, movement=0.5)
    assert swing_record("b").snapshot()["observations"] == 0
    assert swing_record("a") is swing_record("a")
    reset_swings()
