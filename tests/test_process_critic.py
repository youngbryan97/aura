"""Process-critic contracts (CP221): grade the step, not just the answer.

Final-answer loss says whether a trajectory ended correctly; it cannot say
which internal step helped. Every mechanism built so far (branches, depth
ladder, protected memory) moves state without any signal that the movement
was progress -- which is how a period-2 oscillation and a fixed-point
collapse both went unnoticed.

These tests pin the disciplines that keep a critic from becoming another
unverified signal: it must beat the constant predictor it replaces, refuse
to speak when undertrained, and detect its own drift.
"""
from __future__ import annotations

import pytest

from core.learning.process_critic import (
    MIN_OBSERVATIONS,
    CriticObservation,
    ProcessCritic,
)


def _separable(n: int = 60) -> list[CriticObservation]:
    """Trajectories where a single feature really predicts success."""
    rows = []
    for index in range(n):
        good = index % 2 == 0
        signal = 0.9 if good else 0.1
        rows.append(
            CriticObservation(
                features=(1.0, signal, 0.0, 0.0, index / n),
                step=index % 8,
                verified_correct=good,
            )
        )
    return rows


def _noise(n: int = 60) -> list[CriticObservation]:
    """Outcomes uncorrelated with features: no critic should claim skill."""
    return [
        CriticObservation(
            features=(1.0, 0.5, 0.5, 0.5, 0.5),
            step=index % 8,
            verified_correct=index % 3 == 0,
        )
        for index in range(n)
    ]


# ── Refusing to speak without evidence ──────────────────────────────────


def test_untrained_critic_returns_honest_ignorance():
    critic = ProcessCritic()
    assert critic.predict((1.0, 0.9, 0.0, 0.0, 0.5)) == 0.5


def test_too_few_observations_refuses_to_fit():
    critic = ProcessCritic()
    report = critic.fit(_separable(MIN_OBSERVATIONS - 1))
    assert report["fitted"] is False
    assert report["reason"] == "insufficient_graded_observations"
    assert critic.predict((1.0, 0.9, 0.0, 0.0, 0.5)) == 0.5


def test_single_outcome_class_refuses_to_fit():
    """All-correct or all-wrong data teaches nothing but a constant."""
    rows = [
        CriticObservation(features=(1.0, 0.5, 0.0, 0.0, 0.0), step=i,
                          verified_correct=True)
        for i in range(MIN_OBSERVATIONS + 5)
    ]
    report = ProcessCritic().fit(rows)
    assert report["fitted"] is False
    assert report["reason"] == "single_outcome_class"


def test_observations_require_real_graded_outcomes():
    with pytest.raises(ValueError, match="real graded outcome"):
        CriticObservation(features=(1.0,), step=0, verified_correct=0.8)
    with pytest.raises(ValueError, match="at least one feature"):
        CriticObservation(features=(), step=0, verified_correct=True)


# ── Earning trust by beating the constant predictor ─────────────────────


def test_learnable_signal_produces_a_trustworthy_calibrated_critic():
    critic = ProcessCritic()
    rows = _separable()
    report = critic.fit(rows, epochs=400, learning_rate=0.5)
    assert report["fitted"] is True
    assert report["beats_constant_predictor"] is True
    assert report["trustworthy"] is True
    assert report["brier"] < 0.22
    assert critic.predict((1.0, 0.9, 0.0, 0.0, 0.5)) > critic.predict(
        (1.0, 0.1, 0.0, 0.0, 0.5)
    )


def test_unlearnable_data_yields_an_untrustworthy_verdict():
    """A critic with no signal must SAY so rather than emit confident noise."""
    critic = ProcessCritic()
    report = critic.fit(_noise(), epochs=400, learning_rate=0.5)
    assert report["fitted"] is False
    assert report["trustworthy"] is False, (
        "a critic that cannot beat the base rate must not be trusted"
    )
    assert critic.predict((1.0, 0.5, 0.5, 0.5, 0.5)) == 0.5


def test_reliability_bins_expose_where_it_is_miscalibrated():
    critic = ProcessCritic()
    report = critic.fit(_separable(), epochs=400, learning_rate=0.5)
    assert report["reliability"], "calibration must be inspectable, not scalar"
    for row in report["reliability"]:
        assert 0.0 <= row["mean_prediction"] <= 1.0
        assert 0.0 <= row["observed_rate"] <= 1.0
        assert row["n"] > 0


# ── Drift detection (the TEMPO failure mode) ────────────────────────────


def test_drift_is_detected_when_the_world_changes_under_the_critic():
    critic = ProcessCritic()
    critic.fit(_separable(), epochs=400, learning_rate=0.5)
    assert critic.drift(_separable(40))["drifted"] is False
    # Same features, INVERTED outcomes: the critic is now systematically wrong.
    inverted = [
        CriticObservation(
            features=row.features,
            step=row.step,
            verified_correct=not row.verified_correct,
        )
        for row in _separable(40)
    ]
    report = critic.drift(inverted)
    assert report["drifted"] is True
    assert report["recalibration_required"] is True


# ── Per-step credit: the signal outcome-only loss cannot give ───────────


def test_step_credit_rewards_progress_not_mere_movement():
    critic = ProcessCritic()
    critic.fit(_separable(), epochs=400, learning_rate=0.5)
    improving = [
        (1.0, 0.1, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.0, 0.0, 0.5),
        (1.0, 0.9, 0.0, 0.0, 1.0),
    ]
    credit = critic.step_credit(improving)
    assert len(credit) == 2
    assert all(value > 0 for value in credit), "progress must earn credit"

    churning = [(1.0, 0.5, 0.0, 0.0, 0.0)] * 3
    assert all(
        abs(value) < 1e-6 for value in critic.step_credit(churning)
    ), "movement without progress must earn nothing"


def test_step_credit_needs_a_trajectory():
    assert ProcessCritic().step_credit([(1.0, 0.5, 0.0, 0.0, 0.0)]) == []


def test_state_features_are_bounded_and_descriptive():
    mx = pytest.importorskip("mlx.core")

    from core.learning.process_critic import state_features

    state = mx.random.normal((1, 4, 16), key=mx.random.key(0))
    features = state_features(state, step=3, max_depth=8)
    assert len(features) == 5
    assert features[0] == 1.0
    assert features[4] == pytest.approx(0.375)
    assert all(abs(value) < 1e4 for value in features)
