from __future__ import annotations

from pathlib import Path

import pytest

from core.learning.recursive_self_improvement import (
    ImprovementScorecard,
    RecursiveSelfImprovementLoop,
)
from core.self_modification.structural_improver import StructuralImprover


class FakeLearner:
    def __init__(self, train_results):
        self.train_results = list(train_results)
        self.force_train_calls = 0
        self.rollback_calls = 0

    def get_learning_stats(self):
        return {
            "buffer_size": 12,
            "session_avg_quality": 0.4,
            "training_policy": {
                "fine_tune_type": "full",
                "full_weights_unlocked": True,
            },
        }

    async def force_train(self):
        self.force_train_calls += 1
        return self.train_results.pop(0) if self.train_results else True

    def rollback_adapter(self):
        self.rollback_calls += 1
        return True


@pytest.mark.asyncio
async def test_recursive_loop_reenters_after_verified_gain(tmp_path: Path):
    learner = FakeLearner([True, True])
    scores = iter([0.2, 0.35, 0.35, 0.5])

    loop = RecursiveSelfImprovementLoop(
        live_learner=learner,
        evaluator=lambda: ImprovementScorecard(score=next(scores)),
        ledger_path=tmp_path / "rsi.jsonl",
        max_depth=2,
        min_score_delta=0.05,
        require_will_authorization=False,
    )
    loop.record_signal("test", "training_data_ready", severity=0.7)

    result = await loop.run_cycle("improve reasoning", force=True)

    assert result.promoted is True
    assert len(result.child_results) == 1
    assert learner.force_train_calls == 2
    assert learner.rollback_calls == 0
    assert (tmp_path / "rsi.jsonl").exists()


@pytest.mark.asyncio
async def test_recursive_loop_rolls_back_weight_update_without_gain(tmp_path: Path):
    learner = FakeLearner([True])
    scores = iter([0.5, 0.49])
    loop = RecursiveSelfImprovementLoop(
        live_learner=learner,
        evaluator=lambda: ImprovementScorecard(score=next(scores)),
        ledger_path=tmp_path / "rsi.jsonl",
        max_depth=2,
        min_score_delta=0.01,
        require_will_authorization=False,
    )
    loop.record_signal("test", "training_data_ready", severity=0.7)

    result = await loop.run_cycle("avoid regression", force=False)

    assert result.promoted is False
    assert result.rollback_performed is True
    assert learner.rollback_calls == 1


def test_structural_improver_finds_and_repairs_missing_os_import(tmp_path: Path):
    source = tmp_path / "mod.py"
    source.write_text(
        "def enabled():\n"
        "    return os.environ.get('AURA_FLAG') == '1'\n",
        encoding="utf-8",
    )
    improver = StructuralImprover(tmp_path, ledger_path=tmp_path / "ledger.jsonl")

    issues = improver.scan()
    result = improver.apply_known_repair(issues[0])

    assert result.success is True
    assert "import os" in source.read_text(encoding="utf-8")


def test_structural_improver_repairs_generated_gateway_mkdir(tmp_path: Path):
    source = tmp_path / "mod.py"
    source.write_text(
        "from pathlib import Path\n\n"
        "def make(root):\n"
        "    target = root / 'x'\n"
        "    get_task_tracker().create_task(get_storage_gateway().create_dir(target, cause='test'))\n",
        encoding="utf-8",
    )
    improver = StructuralImprover(tmp_path, ledger_path=tmp_path / "ledger.jsonl")

    issue = next(i for i in improver.scan() if i.kind == "unsafe_async_gateway_mkdir")
    result = improver.apply_known_repair(issue)

    text = source.read_text(encoding="utf-8")
    assert result.success is True
    assert "Path(target).mkdir(parents=True, exist_ok=True)" in text
    assert "get_storage_gateway" not in text
