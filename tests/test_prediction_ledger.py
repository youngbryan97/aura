"""Tests for the prediction ledger and Brier/calibration scorers."""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from core.runtime.prediction_ledger import (
    PredictionLedger,
    PredictionLedgerError,
    _binary_brier,
    _categorical_brier,
)


@pytest.fixture
def ledger(tmp_path: Path) -> PredictionLedger:
    return PredictionLedger(tmp_path / "predictions.db")


# ---------------------------------------------------------------------------
# pure scoring helpers
# ---------------------------------------------------------------------------
def test_binary_brier_perfect_prediction_is_zero():
    assert _binary_brier(1.0, True) == pytest.approx(0.0)
    assert _binary_brier(0.0, False) == pytest.approx(0.0)


def test_binary_brier_max_error_is_one():
    assert _binary_brier(0.0, True) == pytest.approx(1.0)
    assert _binary_brier(1.0, False) == pytest.approx(1.0)


def test_binary_brier_uniform_is_quarter():
    assert _binary_brier(0.5, True) == pytest.approx(0.25)
    assert _binary_brier(0.5, False) == pytest.approx(0.25)


def test_categorical_brier_correct_class_one_hot():
    dist = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert _categorical_brier(dist, "a") == pytest.approx(0.0)


def test_categorical_brier_uniform_three_classes():
    dist = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    # observed=a -> (1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2 = 4/9 + 1/9 + 1/9 = 6/9
    assert _categorical_brier(dist, "a") == pytest.approx(6 / 9)


def test_categorical_brier_observed_class_missing_from_prior():
    dist = {"a": 0.6, "b": 0.4}
    # priors contribute 0.36 + 0.16 = 0.52, then (0 - 1)^2 = 1.0 for the
    # missing observed class -> 1.52 total.
    assert _categorical_brier(dist, "c") == pytest.approx(1.52)


# ---------------------------------------------------------------------------
# write/read path
# ---------------------------------------------------------------------------
def test_register_and_resolve_binary(ledger: PredictionLedger):
    pid = ledger.register(
        belief="rain_today",
        modality="weather",
        action="check_radar",
        expected={"rain": True},
        prior_prob=0.7,
        agent_state={"location": "Boston"},
    )
    record = ledger.get(pid)
    assert record is not None
    assert record.resolved is False
    assert record.prior_prob == pytest.approx(0.7)
    assert record.agent_state == {"location": "Boston"}

    resolved = ledger.resolve(pid, observed={"rain": True}, observed_truth=True, posterior_prob=0.9)
    assert resolved.resolved is True
    assert resolved.brier == pytest.approx((0.7 - 1.0) ** 2)
    assert resolved.error == pytest.approx(0.3)
    assert resolved.posterior_prob == pytest.approx(0.9)


def test_register_and_resolve_categorical(ledger: PredictionLedger):
    pid = ledger.register(
        belief="dominant_emotion",
        modality="text",
        action="classify_sentiment",
        expected={"top": "joy"},
        prior_dist={"joy": 0.6, "anger": 0.3, "fear": 0.1},
    )
    resolved = ledger.resolve(pid, observed={"top": "anger"}, observed_class="anger")
    expected = (0.6 - 0.0) ** 2 + (0.3 - 1.0) ** 2 + (0.1 - 0.0) ** 2
    assert resolved.brier == pytest.approx(expected)
    assert resolved.error == pytest.approx(1.0 - 0.3)


def test_register_rejects_no_prior(ledger: PredictionLedger):
    with pytest.raises(PredictionLedgerError):
        ledger.register(
            belief="x",
            modality="m",
            action="a",
            expected={},
        )


def test_register_rejects_out_of_range_prob(ledger: PredictionLedger):
    with pytest.raises(PredictionLedgerError):
        ledger.register(
            belief="x", modality="m", action="a", expected={}, prior_prob=1.5
        )


def test_register_rejects_unnormalized_dist(ledger: PredictionLedger):
    with pytest.raises(PredictionLedgerError):
        ledger.register(
            belief="x",
            modality="m",
            action="a",
            expected={},
            prior_dist={"a": 0.4, "b": 0.4},  # sums to 0.8
        )


def test_resolve_unknown_id_raises(ledger: PredictionLedger):
    with pytest.raises(PredictionLedgerError):
        ledger.resolve("pred-does-not-exist", observed={}, observed_truth=True)


def test_double_resolve_raises(ledger: PredictionLedger):
    pid = ledger.register(
        belief="x", modality="m", action="a", expected={}, prior_prob=0.5
    )
    ledger.resolve(pid, observed={}, observed_truth=False)
    with pytest.raises(PredictionLedgerError):
        ledger.resolve(pid, observed={}, observed_truth=True)


def test_binary_resolve_requires_truth(ledger: PredictionLedger):
    pid = ledger.register(
        belief="x", modality="m", action="a", expected={}, prior_prob=0.5
    )
    with pytest.raises(PredictionLedgerError):
        ledger.resolve(pid, observed={})


def test_categorical_resolve_requires_class(ledger: PredictionLedger):
    pid = ledger.register(
        belief="x",
        modality="m",
        action="a",
        expected={},
        prior_dist={"a": 0.5, "b": 0.5},
    )
    with pytest.raises(PredictionLedgerError):
        ledger.resolve(pid, observed={})


# ---------------------------------------------------------------------------
# durability / persistence
# ---------------------------------------------------------------------------
def test_persists_across_reopen(tmp_path: Path):
    db = tmp_path / "predictions.db"
    led = PredictionLedger(db)
    pid = led.register(
        belief="x", modality="m", action="a", expected={}, prior_prob=0.4
    )
    led.resolve(pid, observed={"truth": True}, observed_truth=True)

    led2 = PredictionLedger(db)
    rec = led2.get(pid)
    assert rec is not None
    assert rec.resolved is True
    assert rec.brier == pytest.approx((0.4 - 1.0) ** 2)


def test_iter_unresolved_skips_resolved(ledger: PredictionLedger):
    a = ledger.register(belief="a", modality="m", action="x", expected={}, prior_prob=0.1)
    b = ledger.register(belief="b", modality="m", action="x", expected={}, prior_prob=0.2)
    ledger.resolve(a, observed={}, observed_truth=False)
    pending = list(ledger.iter_unresolved())
    assert [p.prediction_id for p in pending] == [b]


def test_recent_returns_newest_first(ledger: PredictionLedger):
    ts = time.time()
    a = ledger.register(
        belief="a", modality="m", action="x", expected={}, prior_prob=0.1, created_at=ts
    )
    b = ledger.register(
        belief="b", modality="m", action="x", expected={}, prior_prob=0.2, created_at=ts + 1
    )
    out = ledger.recent(limit=2)
    assert [r.prediction_id for r in out] == [b, a]


# ---------------------------------------------------------------------------
# scoring queries
# ---------------------------------------------------------------------------
def test_score_brier_empty_ledger(ledger: PredictionLedger):
    score = ledger.score_brier()
    assert score == {"count": 0, "mean_brier": None}


def test_score_brier_mean_over_window(ledger: PredictionLedger):
    p1 = ledger.register(
        belief="a", modality="m", action="x", expected={}, prior_prob=1.0
    )
    p2 = ledger.register(
        belief="b", modality="m", action="x", expected={}, prior_prob=0.0
    )
    p3 = ledger.register(
        belief="c", modality="m", action="x", expected={}, prior_prob=0.5
    )
    ledger.resolve(p1, observed={}, observed_truth=True)   # brier 0
    ledger.resolve(p2, observed={}, observed_truth=True)   # brier 1
    ledger.resolve(p3, observed={}, observed_truth=True)   # brier 0.25

    score = ledger.score_brier()
    assert score["count"] == 3
    assert score["mean_brier"] == pytest.approx((0.0 + 1.0 + 0.25) / 3)


def test_calibration_perfectly_calibrated_predictor(ledger: PredictionLedger):
    """If a predictor says 0.7 and is right 70% of the time, ECE -> 0."""
    # 10 predictions at p=0.7, 7 true / 3 false.
    pids = [
        ledger.register(belief=f"x{i}", modality="m", action="a", expected={}, prior_prob=0.7)
        for i in range(10)
    ]
    for i, pid in enumerate(pids):
        ledger.resolve(pid, observed={"truth": i < 7}, observed_truth=i < 7)
    cal = ledger.calibration(bins=10)
    # The bin covering 0.7 should have actual_rate = 0.7 = mean_predicted.
    bin_for_07 = cal["bins"][7]
    assert bin_for_07["count"] == 10
    assert bin_for_07["mean_predicted"] == pytest.approx(0.7)
    assert bin_for_07["actual_rate"] == pytest.approx(0.7)
    assert cal["ece"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_overconfident_predictor_has_nonzero_ece(ledger: PredictionLedger):
    """Predictor says 0.9 but is right only 50% -> calibration error 0.4."""
    pids = [
        ledger.register(belief=f"x{i}", modality="m", action="a", expected={}, prior_prob=0.9)
        for i in range(10)
    ]
    for i, pid in enumerate(pids):
        ledger.resolve(pid, observed={"truth": i < 5}, observed_truth=i < 5)
    cal = ledger.calibration(bins=10)
    assert cal["total"] == 10
    assert cal["ece"] == pytest.approx(0.4)


def test_calibration_with_no_resolved_predictions(ledger: PredictionLedger):
    cal = ledger.calibration(bins=5)
    assert cal["total"] == 0
    assert cal["ece"] == pytest.approx(0.0)
    assert all(b["count"] == 0 for b in cal["bins"])


def test_calibration_window_excludes_old_predictions(ledger: PredictionLedger):
    old = ledger.register(
        belief="old", modality="m", action="a", expected={}, prior_prob=1.0,
        created_at=1000.0,
    )
    new = ledger.register(
        belief="new", modality="m", action="a", expected={}, prior_prob=0.0,
        created_at=2000.0,
    )
    ledger.resolve(old, observed={"truth": False}, observed_truth=False, resolved_at=1001.0)
    ledger.resolve(new, observed={"truth": False}, observed_truth=False, resolved_at=2001.0)

    score_recent = ledger.score_brier(since=1500.0)
    assert score_recent["count"] == 1
    assert score_recent["mean_brier"] == pytest.approx(0.0)

    score_all = ledger.score_brier()
    assert score_all["count"] == 2


def test_count_tracks_writes(ledger: PredictionLedger):
    assert ledger.count() == 0
    ledger.register(belief="x", modality="m", action="a", expected={}, prior_prob=0.5)
    ledger.register(belief="y", modality="m", action="a", expected={}, prior_prob=0.5)
    assert ledger.count() == 2
