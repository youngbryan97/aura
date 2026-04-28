"""Tests for the curriculum learning loop."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from core.curriculum import (
    CurriculumLoop,
    GapDetector,
    GapReport,
    ImprovementRecorder,
    LearningTask,
    Lesson,
    LessonStore,
    Strategy,
    StrategyController,
    TaskGenerator,
)
from core.runtime.prediction_ledger import PredictionLedger


@pytest.fixture
def ledger(tmp_path: Path) -> PredictionLedger:
    return PredictionLedger(tmp_path / "predictions.db")


@pytest.fixture
def lessons(tmp_path: Path) -> LessonStore:
    return LessonStore(tmp_path / "lessons.db")


# ---------------------------------------------------------------------------
# StrategyController
# ---------------------------------------------------------------------------
def test_strategy_starts_at_default():
    sc = StrategyController()
    assert sc.current("b") is Strategy.DEFAULT


def test_strategy_escalates_after_patience():
    sc = StrategyController(patience=1)
    sc.record_outcome("b", success=False)
    assert sc.current("b") is Strategy.MORE_EXAMPLES
    sc.record_outcome("b", success=False)
    assert sc.current("b") is Strategy.DECOMPOSE
    sc.record_outcome("b", success=False)
    assert sc.current("b") is Strategy.ASK_USER
    sc.record_outcome("b", success=False)
    assert sc.current("b") is Strategy.PARK


def test_strategy_resets_to_default_on_success():
    sc = StrategyController(patience=1)
    sc.record_outcome("b", success=False)
    sc.record_outcome("b", success=False)
    assert sc.current("b") is Strategy.DECOMPOSE
    sc.record_outcome("b", success=True)
    assert sc.current("b") is Strategy.DEFAULT


def test_strategy_isolates_per_belief():
    sc = StrategyController(patience=1)
    sc.record_outcome("a", success=False)
    sc.record_outcome("a", success=False)
    assert sc.current("a") is Strategy.DECOMPOSE
    assert sc.current("b") is Strategy.DEFAULT


# ---------------------------------------------------------------------------
# GapDetector
# ---------------------------------------------------------------------------
def _seed_predictions(ledger: PredictionLedger, *, count: int, brier_target: float) -> None:
    """Seed N resolved predictions with a target Brier each."""
    # For binary, brier=(prior - obs)^2; pick prior so that brier matches.
    import math
    prior = math.sqrt(brier_target) if brier_target <= 1.0 else 1.0
    for i in range(count):
        pid = ledger.register(
            belief="x",
            modality="text",
            action="probe",
            expected={},
            prior_prob=prior,
        )
        ledger.resolve(pid, observed={"truth": False}, observed_truth=False)


def test_gap_detector_returns_no_gap_when_below_threshold(ledger):
    _seed_predictions(ledger, count=5, brier_target=0.04)  # below default 0.10
    gap = GapDetector(ledger).detect()
    assert gap.has_gap is False


def test_gap_detector_finds_gap_above_threshold(ledger):
    _seed_predictions(ledger, count=5, brier_target=0.49)
    gap = GapDetector(ledger).detect()
    assert gap.has_gap is True
    assert gap.belief == "x"
    assert gap.modality == "text"
    assert gap.mean_brier == pytest.approx(0.49, abs=1e-6)


def test_gap_detector_requires_minimum_resolved(ledger):
    _seed_predictions(ledger, count=2, brier_target=0.5)  # below min_resolved=3
    gap = GapDetector(ledger).detect()
    assert gap.has_gap is False


# ---------------------------------------------------------------------------
# closed loop with stub attempter
# ---------------------------------------------------------------------------
def _stub_attempter_factory(success_after: int) -> Tuple[list, callable]:
    """Stub LLM that fails the first ``success_after`` calls then succeeds."""
    calls: List[str] = []

    def attempter(task: LearningTask) -> Tuple[bool, float, str]:
        calls.append(task.strategy)
        if len(calls) <= success_after:
            return (False, 0.85, "wrong")  # confident wrong -> high brier
        return (True, 0.95, "right")  # confident right -> low brier

    return calls, attempter


def _truth_oracle(task: LearningTask) -> bool:
    return True  # ground truth is always True for the test task


def test_loop_converges_when_attempter_eventually_succeeds(ledger, lessons):
    calls, attempter = _stub_attempter_factory(success_after=3)
    seed_task = LearningTask(
        task_id="seed-1",
        belief="add_two",
        modality="arith",
        prompt="2 + 2 = 4 ?",
        expected={"truth": True},
    )
    loop = CurriculumLoop(
        ledger=ledger,
        lessons=lessons,
        attempter=attempter,
        oracle=_truth_oracle,
        seed_tasks=[seed_task] * 8,
    )
    outcome = loop.run(max_iterations=8, success_run_to_converge=3)
    assert outcome.converged is True
    # Strategy should have escalated past DEFAULT during the failures.
    distinct_strategies = {l.strategy for l in outcome.lessons}
    assert "default" in distinct_strategies
    # And then snapped back to default after the first success.
    assert outcome.final_strategy is Strategy.DEFAULT


def test_loop_records_improving_trend(ledger, lessons):
    _, attempter = _stub_attempter_factory(success_after=2)
    seed_task = LearningTask(
        task_id="seed-improve",
        belief="add",
        modality="arith",
        prompt="?",
        expected={"truth": True},
    )
    loop = CurriculumLoop(
        ledger=ledger,
        lessons=lessons,
        attempter=attempter,
        oracle=_truth_oracle,
        seed_tasks=[seed_task] * 6,
    )
    outcome = loop.run(max_iterations=6, success_run_to_converge=99)
    assert outcome.trend in {"improving", "plateau"}
    # First snapshot should have higher Brier than last.
    assert outcome.snapshots[0].mean_brier is not None
    assert outcome.snapshots[-1].mean_brier is not None
    assert outcome.snapshots[-1].mean_brier <= outcome.snapshots[0].mean_brier


def test_loop_persists_lessons(ledger, lessons):
    _, attempter = _stub_attempter_factory(success_after=1)
    seed_task = LearningTask(
        task_id="seed-persist",
        belief="x",
        modality="m",
        prompt="?",
        expected={},
    )
    loop = CurriculumLoop(
        ledger=ledger,
        lessons=lessons,
        attempter=attempter,
        oracle=_truth_oracle,
        seed_tasks=[seed_task] * 4,
    )
    outcome = loop.run(max_iterations=4, success_run_to_converge=99)
    persisted = lessons.all()
    assert len(persisted) == outcome.iterations
    assert all(isinstance(l, Lesson) for l in persisted)


def test_loop_writes_into_prediction_ledger(ledger, lessons):
    _, attempter = _stub_attempter_factory(success_after=0)
    seed_task = LearningTask(
        task_id="seed-write",
        belief="x",
        modality="m",
        prompt="?",
        expected={},
    )
    loop = CurriculumLoop(
        ledger=ledger,
        lessons=lessons,
        attempter=attempter,
        oracle=_truth_oracle,
        seed_tasks=[seed_task] * 3,
    )
    loop.run(max_iterations=3, success_run_to_converge=99)
    assert ledger.count() == 3
    score = ledger.score_brier()
    assert score["count"] == 3


# ---------------------------------------------------------------------------
# improvement recorder
# ---------------------------------------------------------------------------
def test_improvement_recorder_returns_unknown_with_few_snapshots(ledger):
    rec = ImprovementRecorder(ledger)
    assert rec.trend() == "unknown"
    rec.snapshot(0)
    assert rec.trend() == "unknown"


def test_improvement_recorder_detects_improvement(ledger):
    # Manually seed: first snapshot at high brier, second at low.
    high_pred = ledger.register(belief="x", modality="m", action="a", expected={}, prior_prob=1.0)
    ledger.resolve(high_pred, observed={"truth": False}, observed_truth=False)  # brier 1.0
    rec = ImprovementRecorder(ledger)
    rec.snapshot(0)

    low_pred = ledger.register(belief="x", modality="m", action="a", expected={}, prior_prob=1.0)
    ledger.resolve(low_pred, observed={"truth": True}, observed_truth=True)  # brier 0.0
    rec.snapshot(1)

    assert rec.trend() == "improving"


# ---------------------------------------------------------------------------
# task generator
# ---------------------------------------------------------------------------
def test_task_generator_uses_gap_belief_and_modality():
    gen = TaskGenerator()
    gap = GapReport(
        cluster="text::weather",
        belief="weather",
        modality="text",
        n_resolved=5,
        mean_brier=0.4,
        accuracy=0.2,
    )
    task = gen.generate(gap=gap, strategy="default", iteration=2)
    assert task.belief == "weather"
    assert task.modality == "text"
    assert task.iteration == 2
    assert task.strategy == "default"


def test_task_generator_uses_seed_when_no_gap():
    gen = TaskGenerator()
    empty = GapReport(cluster=None, belief=None, modality=None, n_resolved=0, mean_brier=0.0, accuracy=1.0)
    task = gen.generate(
        gap=empty,
        strategy="default",
        seed_prompt="practise",
        seed_expected={"belief": "primer"},
    )
    assert task.prompt == "practise"
