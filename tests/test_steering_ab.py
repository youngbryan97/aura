"""Steering A/B proof contract tests for the exact active cortex.

The old test module loaded a small MLX model at import time and skipped unless
an environment flag was present. That made collection expensive, hid the real
production lane, and did not prove the cortex path Aura actually uses. The
legacy-named live runner remains the source of truth; this module makes its contract
collection-safe and keeps the heavy execution in an explicit live lane.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

RUNNER_PATH = Path(__file__).with_name("run_32b_steering_ab_live.py")


def _load_live_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("aura_live_32b_steering_ab", RUNNER_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {RUNNER_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_runner_targets_the_exact_active_cortex_lane():
    runner = _load_live_runner()

    assert runner.MODEL_NAME == "active-cortex-exact-artifact"
    assert not hasattr(runner, "FALLBACK_MODEL")
    assert callable(runner._resolve_model_contract)
    assert runner.N_TRIALS >= 5
    assert runner.MAX_TOKENS >= 80
    assert len(runner.HELD_OUT_TASKS) >= 5


def test_live_runner_covers_required_behavioral_controls():
    runner = _load_live_runner()

    required_tasks = {
        "planning_under_uncertainty",
        "memory_retrieval_choice",
        "tool_selection",
        "affective_recovery",
        "adversarial_instruction_hygiene",
    }
    assert required_tasks.issubset(set(runner.HELD_OUT_TASKS))
    assert runner.RICH_AFFECT_PROMPT
    assert runner.STEERING_ALPHA > 0.0


@pytest.mark.hardware
@pytest.mark.live
def test_live_cortex_steering_ab_runner_passes():
    runner = _load_live_runner()

    assert runner.main([]) == 0
