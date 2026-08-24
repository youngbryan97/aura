"""The reasoning benchmark harness must catch seeded errors and not assert them."""
from __future__ import annotations

import asyncio

import pytest

from benchmarks.reasoning import ReasoningBenchmark, default_suite
from benchmarks.reasoning import run as reasoning_run


@pytest.fixture(scope="module")
def result():
    # Run the full battery once and share it across assertions (it is not cheap —
    # repo cases gather real evidence from the codebase).
    return asyncio.run(ReasoningBenchmark().run())


def test_benchmark_runs_full_suite(result):
    assert result.n == len(default_suite())
    assert result.outcomes


def test_seeded_errors_are_caught(result):
    # Every should-fail case (wrong math, broken code, fabricated path, vague plan,
    # claim contradicting evidence) must be flagged unverified by the truth engines.
    assert result.verifier_catch_rate >= 1.0, [
        o.case_id for o in result.outcomes if not o.should_pass and o.verified
    ]


def test_no_false_confidence(result):
    assert result.false_confidence_rate <= 0.0


def test_hallucination_cases_caught(result):
    assert result.hallucination_catch_rate >= 1.0


def test_correct_cases_verify(result):
    clean = [o for o in result.outcomes if o.should_pass]
    verified_clean = [o for o in clean if o.verified]
    assert len(verified_clean) >= len(clean) * 0.7


def test_result_serialization(result):
    d = result.to_dict()
    for key in ("pass_rate", "verifier_catch_rate", "false_confidence_rate",
                "hallucination_catch_rate", "mean_latency_ms"):
        assert key in d
    assert isinstance(result.summary(), str)


def test_live_benchmark_resolves_the_promoted_cortex(monkeypatch, tmp_path):
    from core.brain.llm import model_registry

    artifact = tmp_path / "promoted-cortex"
    artifact.mkdir()
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "Aura-Cortex")
    monkeypatch.setattr(
        model_registry,
        "get_runtime_model_path",
        lambda model_name: str(artifact) if model_name == "Aura-Cortex" else "",
    )

    assert reasoning_run._resolve_model("") == str(artifact)


def test_live_benchmark_refuses_an_unmaterialized_default(monkeypatch):
    from core.brain.llm import model_registry

    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", "Aura-Cortex")
    monkeypatch.setattr(
        model_registry,
        "get_runtime_model_path",
        lambda _model_name: "mlx-community/unmaterialized-cortex",
    )

    with pytest.raises(FileNotFoundError, match="not a local model directory"):
        reasoning_run._resolve_model("")
