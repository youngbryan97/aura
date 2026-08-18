"""Tests for the Reasoning Amplifier v2 orchestrator."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.calibration_gate import CalibrationGate
from core.brain.generation_provenance import attributed_text
from core.brain.reasoning_amplifier_v2 import (
    AmplificationRequest,
    ReasoningAmplifierV2,
    ReasoningBudgetPolicy,
    ReasoningMode,
    classify_task_type,
    normalize_problem,
)
from core.brain.reasoning_memory import ReasoningMemory
from core.brain.verifiers import get_verifier_registry


def _gen(answer: str):
    async def generate(prompt: str, temperature: float) -> str:
        return answer

    return generate


def _amp(generate, tmp_path: Path) -> ReasoningAmplifierV2:
    return ReasoningAmplifierV2(
        generate,
        verifier=get_verifier_registry(),
        calibration=CalibrationGate(),
        memory=ReasoningMemory(path=tmp_path / "refl.jsonl"),
    )


def test_classify_task_type():
    assert classify_task_type("fix the bug in this function") == "code"
    assert classify_task_type("compute the factorial of 6") == "math"
    assert classify_task_type("which file implements the inference gate") == "repo_audit"
    assert classify_task_type("what is the capital of France") == "factual"
    assert classify_task_type("how would you plan the migration steps") == "planning"
    assert classify_task_type("find the optimal schedule and makespan within horizon 7") == "planning"
    assert classify_task_type("infer the causal order from these interventions") == "logic"


def test_normalize_problem_attaches_verification_plan():
    p = normalize_problem("solve x for 2x = 4", task_type="math")
    assert p.task_type == "math"
    assert any("arithmetic" in s or "sympy" in s for s in p.verification_plan)


def test_budget_policy():
    assert ReasoningBudgetPolicy.choose_mode("code") is ReasoningMode.DEEP
    assert ReasoningBudgetPolicy.choose_mode("factual") is ReasoningMode.NORMAL
    assert ReasoningBudgetPolicy.choose_mode("architecture", risk_level="high") is ReasoningMode.EXTREME
    assert ReasoningBudgetPolicy.choose_mode("generic", explicit=ReasoningMode.FAST) is ReasoningMode.FAST


@pytest.mark.asyncio
async def test_fast_mode_returns_calibrated_answer(tmp_path):
    amp = _amp(_gen("Paris is definitely the capital of France."), tmp_path)
    req = AmplificationRequest(objective="capital of France?", mode=ReasoningMode.FAST)
    out = await amp.amplify(req)
    assert out.answer
    assert out.receipt.mode == "fast"
    assert out.receipt.budget_used["samples"] >= 1


@pytest.mark.asyncio
async def test_math_error_is_caught_and_lowers_confidence(tmp_path):
    amp = _amp(_gen("The result: 2 + 2 = 5"), tmp_path)
    req = AmplificationRequest(objective="what is 2 + 2", task_type="math", mode=ReasoningMode.NORMAL)
    out = await amp.amplify(req)
    assert not out.verified
    assert out.confidence < 0.6
    assert any("arithmetic" in f for f in out.receipt.known_failures)


@pytest.mark.asyncio
async def test_proof_mode_refuses_unverified(tmp_path):
    amp = _amp(_gen("3 * 3 = 10"), tmp_path)
    req = AmplificationRequest(objective="3 * 3", task_type="math", mode=ReasoningMode.PROOF)
    out = await amp.amplify(req)
    assert "can't assert" in out.answer.lower() or "did not survive" in out.answer.lower()
    assert "proof_refused_unverified" in out.receipt.fallbacks_used
    assert out.generation_metadata["model_native_output"] is False
    assert out.generation_metadata["deterministic_repair_applied"] is True
    assert any(
        item["stage"] == "reasoning_amplifier.proof_refusal"
        for item in out.text_mutations
    )


@pytest.mark.asyncio
async def test_clean_code_verifies(tmp_path):
    amp = _amp(_gen("Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```"), tmp_path)
    req = AmplificationRequest(objective="write an add function", task_type="code", mode=ReasoningMode.NORMAL)
    out = await amp.amplify(req)
    assert out.verified


@pytest.mark.asyncio
async def test_receipt_is_complete(tmp_path):
    amp = _amp(_gen("The answer is 4."), tmp_path)
    req = AmplificationRequest(objective="2+2", task_type="math", mode=ReasoningMode.NORMAL)
    out = await amp.amplify(req)
    d = out.receipt.to_dict()
    for key in ("mode", "strategy_used", "task_type", "num_candidates", "confidence",
                "epistemic_status", "budget_used"):
        assert key in d


def test_receipt_serialization_preserves_all_bounded_cognitive_operations():
    from core.brain.reasoning_amplifier_v2 import ReasoningReceipt

    receipt = ReasoningReceipt(
        mode="normal",
        strategy_used="fixture",
        task_type="math",
        num_candidates=8,
        verifiers_run=[],
        valid_candidates=0,
        winning_candidate_id=None,
        confidence=0.0,
        agreement=0.0,
        epistemic_status="unverified",
        cognitive_operations=[{"index": index} for index in range(8)],
    )

    assert receipt.to_dict()["cognitive_operations"] == [
        {"index": index} for index in range(8)
    ]


@pytest.mark.asyncio
async def test_read_only_evaluation_cannot_train_on_or_cache_its_items(
    tmp_path,
    monkeypatch,
):
    calls = {
        "episode_record": 0,
        "playbook_record": 0,
        "self_improvement": 0,
        "preference": 0,
        "cache": 0,
    }

    class EvaluationMemory:
        def as_guard_text(self, *_args, **_kwargs):
            return ""

        def record(self, **_kwargs):
            calls["episode_record"] += 1

    class EvaluationPlaybooks:
        def as_playbook_text(self, *_args, **kwargs):
            assert kwargs["record_usage"] is False
            return ""

        def record_win(self, **_kwargs):
            calls["playbook_record"] += 1

    class EvaluationSelfImprovement:
        def record_win(self, *_args, **_kwargs):
            calls["self_improvement"] += 1

    class EvaluationPreferences:
        def ingest(self, *_args, **_kwargs):
            calls["preference"] += 1

    class EvaluationCache:
        def get(self, *_args, **_kwargs):
            calls["cache"] += 1
            return None

        def put(self, *_args, **_kwargs):
            calls["cache"] += 1

    monkeypatch.setattr(
        "core.brain.procedural_memory.get_procedural_memory",
        lambda: EvaluationPlaybooks(),
    )
    monkeypatch.setattr(
        "core.brain.reasoning_self_improvement.get_reasoning_self_improvement",
        lambda: EvaluationSelfImprovement(),
    )
    monkeypatch.setattr(
        "core.learning.verifiable_preference_harness.get_verifiable_preference_harness",
        lambda: EvaluationPreferences(),
    )
    monkeypatch.setattr(
        "core.brain.reasoning_solved_cache.get_reasoning_solved_cache",
        lambda: EvaluationCache(),
    )

    amp = ReasoningAmplifierV2(
        _gen("The answer is 4."),
        verifier=_CheckedPassingVerifier(),
        calibration=CalibrationGate(),
        memory=EvaluationMemory(),
    )
    out = await amp.amplify(
        AmplificationRequest(
            objective="what is 2 + 2",
            task_type="math",
            mode=ReasoningMode.NORMAL,
            sample_budget=3,
            context={
                "disable_batched_candidates": True,
                "read_only_evaluation": True,
            },
        )
    )

    assert out.answer
    assert calls == {
        "episode_record": 0,
        "playbook_record": 0,
        "self_improvement": 0,
        "preference": 0,
        "cache": 0,
    }


@pytest.mark.asyncio
async def test_sealed_evaluation_cannot_retrieve_prior_answers(tmp_path, monkeypatch):
    class ForbiddenMemory:
        def as_guard_text(self, *_args, **_kwargs):
            raise AssertionError("sealed evaluation retrieved reasoning memory")

        def record(self, **_kwargs):
            raise AssertionError("sealed evaluation wrote reasoning memory")

    class ForbiddenPlaybooks:
        def as_playbook_text(self, *_args, **_kwargs):
            raise AssertionError("sealed evaluation retrieved procedural memory")

        def record_win(self, **_kwargs):
            raise AssertionError("sealed evaluation wrote procedural memory")

    monkeypatch.setattr(
        "core.brain.procedural_memory.get_procedural_memory",
        lambda: ForbiddenPlaybooks(),
    )
    amp = ReasoningAmplifierV2(
        _gen("The answer is 4."),
        verifier=_CheckedPassingVerifier(),
        calibration=CalibrationGate(),
        memory=ForbiddenMemory(),
    )

    def _forbidden_live_substrate_read():
        raise AssertionError("sealed evaluation read resident substrate state")

    monkeypatch.setattr(amp, "_read_substrate", _forbidden_live_substrate_read)

    out = await amp.amplify(
        AmplificationRequest(
            objective="what is 2 + 2",
            task_type="math",
            mode=ReasoningMode.NORMAL,
            sample_budget=3,
            context={
                "disable_batched_candidates": True,
                "read_only_evaluation": True,
                "sealed_evaluation": True,
                "skip_evidence": True,
            },
        )
    )

    assert out.answer
    assert "playbooks_injected" not in out.receipt.fallbacks_used


@pytest.mark.asyncio
async def test_read_only_evaluation_cannot_update_foundry_weights(monkeypatch):
    from core.brain.verifiers.base import VerificationResult
    from core.brain.verifiers.registry import VerifierRegistry

    class EvaluationVerifier:
        name = "evaluation"

        @staticmethod
        def handles(_task_type):
            return True

        async def verify(self, _candidate, *, context=None):
            return VerificationResult(
                domain="math",
                ok=True,
                checked=True,
                engine=self.name,
            )

    class RefusingFoundry:
        @staticmethod
        def record_verdict(**_kwargs):
            raise AssertionError("read-only evaluation reached Foundry mutation")

    monkeypatch.setattr(
        "core.runtime.service_access.optional_service",
        lambda name, default=None: RefusingFoundry()
        if name == "verifier_foundry"
        else default,
    )
    registry = VerifierRegistry([EvaluationVerifier()])

    result = await registry.verify(
        "4",
        task_type="math",
        context={"read_only_evaluation": True},
    )

    assert result.ok is True
    assert result.checked is True


@pytest.mark.asyncio
async def test_memory_guard_applied_second_time(tmp_path):
    # First episode fails verification → a failure-mode is recorded.
    mem = ReasoningMemory(path=tmp_path / "refl.jsonl")
    amp1 = ReasoningAmplifierV2(_gen("2 + 2 = 5"), verifier=get_verifier_registry(),
                                calibration=CalibrationGate(), memory=mem)
    await amp1.amplify(AmplificationRequest(objective="add 2 and 2 carefully", task_type="math",
                                            mode=ReasoningMode.NORMAL))
    # Second, similar episode should surface the guard.
    captured = {}

    async def capture_gen(prompt: str, temperature: float) -> str:
        captured["prompt"] = prompt
        return "2 + 2 = 4"

    amp2 = ReasoningAmplifierV2(capture_gen, verifier=get_verifier_registry(),
                                calibration=CalibrationGate(), memory=mem)
    out = await amp2.amplify(AmplificationRequest(objective="add 2 and 2 again", task_type="math",
                                                  mode=ReasoningMode.NORMAL))
    assert "Lessons from similar past reasoning" in captured.get("prompt", "")
    assert out.receipt.guards_applied


@pytest.mark.asyncio
async def test_deep_mode_math_uses_answer_preserving_path(tmp_path):
    # Math DEEP must NOT go through the courtroom (its single-line judge/simplifier
    # extraction mangles worked arithmetic); it takes the answer-preserving
    # self-consistency + sandbox path instead.
    amp = _amp(_gen("Answer: 12 * 12 = 144"), tmp_path)
    req = AmplificationRequest(objective="what is 12 times 12", task_type="math", mode=ReasoningMode.DEEP,
                               time_budget_s=20.0)
    out = await amp.amplify(req)
    assert out.receipt.mode == "deep"
    assert "courtroom" not in out.receipt.strategy_used
    assert "sandbox" in out.receipt.strategy_used or "self_consistency" in out.receipt.strategy_used


@pytest.mark.asyncio
async def test_deep_mode_prose_uses_courtroom(tmp_path):
    # A prose/architecture DEEP turn does engage the courtroom.
    amp = _amp(_gen("Answer: the gateway routes through effect governance."), tmp_path)
    req = AmplificationRequest(objective="explain how the subsystem coordinates governance",
                               task_type="architecture", mode=ReasoningMode.DEEP, time_budget_s=20.0)
    out = await amp.amplify(req)
    assert out.receipt.mode == "deep"
    assert out.receipt.strategy_used in {"courtroom", "self_consistency", "direct"}


class _CheckedPassingVerifier:
    async def verify(self, candidate, *, task_type=None, context=None):
        from core.brain.verifiers.base import VerificationResult

        return VerificationResult(
            domain=str(task_type or "generic"),
            ok=True,
            checked=True,
            engine="test_checked",
        )


@pytest.mark.asyncio
async def test_seeded_rlc_candidate_is_verified_without_regeneration(tmp_path):
    async def generation_must_not_run(_prompt: str, _temperature: float) -> str:
        raise AssertionError("a complete one-candidate seed budget regenerated")

    amp = _amp_with_verifier(
        generation_must_not_run,
        _CheckedPassingVerifier(),
        tmp_path,
    )
    out = await amp.amplify(
        AmplificationRequest(
            objective="Explain the bounded result",
            task_type="planning",
            mode=ReasoningMode.FAST,
            sample_budget=1,
            context={
                "seed_candidates": ["The bounded result is 42."],
                "read_only_evaluation": True,
                "skip_cache": True,
                "skip_evidence": True,
            },
        )
    )

    assert out.answer == "The bounded result is 42."
    assert out.verified is True
    assert out.receipt.num_candidates == 1
    assert out.receipt.budget_used["seed_candidates"] == 1
    assert "seed_candidates_admitted:1" in out.receipt.fallbacks_used


@pytest.mark.asyncio
async def test_verified_incumbent_skips_regeneration_even_with_larger_budget(tmp_path):
    async def generation_must_not_run(_prompt: str, _temperature: float) -> str:
        raise AssertionError("a verifier-clean incumbent triggered redundant generation")

    amp = _amp_with_verifier(
        generation_must_not_run,
        _CheckedPassingVerifier(),
        tmp_path,
    )
    out = await amp.amplify(
        AmplificationRequest(
            objective="Explain the verified result",
            task_type="factual",
            mode=ReasoningMode.DEEP,
            sample_budget=5,
            context={
                "seed_candidates": ["The mechanically verified result is 42."],
                "read_only_evaluation": True,
                "skip_cache": True,
            },
        )
    )

    assert out.answer == "The mechanically verified result is 42."
    assert out.receipt.strategy_used == "verified_incumbent"
    assert out.receipt.promotion_authority == "preserve_incumbent"
    assert "incumbent_verified" in out.receipt.fallbacks_used


@pytest.mark.asyncio
async def test_concurrent_winner_keeps_its_own_generation_metadata(tmp_path):
    invocation = 0

    async def generate(prompt: str, temperature: float) -> str:
        nonlocal invocation
        candidate_id = invocation
        invocation += 1
        delays = (0.04, 0.01, 0.02)
        await asyncio.sleep(delays[candidate_id])
        text = "Answer: 4" if candidate_id in {0, 2} else "Answer: 5"
        return attributed_text(text, {"candidate_id": candidate_id})

    amp = _amp_with_verifier(generate, _CheckedPassingVerifier(), tmp_path)
    out = await amp.amplify(
        AmplificationRequest(
            objective="what is 2 + 2",
            task_type="math",
            mode=ReasoningMode.NORMAL,
            context={"disable_batched_candidates": True},
        )
    )

    assert out.answer == "Answer: 4"
    assert out.generation_metadata == {
        "candidate_id": 0,
        "reasoning_candidate_index": 0,
    }
    assert out.receipt.winning_candidate_id == 0


@pytest.mark.asyncio
async def test_calibration_rewrite_cannot_retain_model_native_attribution(tmp_path):
    class _RewritingCalibration:
        def assess(self, *_args, **_kwargs):
            return SimpleNamespace(
                calibrated_answer="Calibrated answer.",
                confidence=0.81,
                overall=SimpleNamespace(value="bounded"),
                downgraded=1,
                flagged_impossible=0,
            )

    amp = ReasoningAmplifierV2(
        _gen(attributed_text("Raw candidate.", {"candidate_id": 4})),
        verifier=_CheckedPassingVerifier(),
        calibration=_RewritingCalibration(),
        memory=ReasoningMemory(path=tmp_path / "refl.jsonl"),
    )
    out = await amp.amplify(
        AmplificationRequest(
            objective="Give a bounded answer",
            task_type="generic",
            mode=ReasoningMode.FAST,
        )
    )

    assert out.answer == "Calibrated answer."
    assert out.source_answer == "Raw candidate."
    assert out.generation_metadata["candidate_id"] == 4
    assert out.generation_metadata["model_native_output"] is False
    assert out.generation_metadata["post_generation_repair_applied"] is True
    assert [item["stage"] for item in out.text_mutations] == [
        "reasoning_amplifier.calibration"
    ]


# ── Verified-answer semantics, fail-closed (July external review) ────────


class _CrashingVerifier:
    async def verify(self, candidate, *, task_type=None, context=None):
        raise RuntimeError("verifier backend unavailable")


class _VacuousVerifier:
    """ok=True but checked=False — 'no objection' without an actual check."""

    async def verify(self, candidate, *, task_type=None, context=None):
        from core.brain.verifiers.base import VerificationResult

        return VerificationResult(domain="generic", ok=True, checked=False, engine="vacuous")


def _amp_with_verifier(generate, verifier, tmp_path: Path) -> ReasoningAmplifierV2:
    return ReasoningAmplifierV2(
        generate,
        verifier=verifier,
        calibration=CalibrationGate(),
        memory=ReasoningMemory(path=tmp_path / "refl.jsonl"),
    )


@pytest.mark.asyncio
async def test_crashed_verifier_never_yields_verified(tmp_path):
    """A verdict of None (verifier crashed) is NOT a pass. verified=True was
    the old behavior — the exact 'verified-answer semantics' defect."""
    amp = _amp_with_verifier(_gen("The answer is 4."), _CrashingVerifier(), tmp_path)
    out = await amp.amplify(
        AmplificationRequest(objective="what is 2 + 2", mode=ReasoningMode.FAST)
    )
    assert out.answer
    assert out.verified is False
    assert out.confidence <= 0.55, "unverified answers keep the honest confidence cap"


@pytest.mark.asyncio
async def test_vacuous_pass_is_not_verified(tmp_path):
    """ok=True/checked=False means nothing was actually evaluated — the
    answer may be fine, but 'verified' may not be claimed."""
    amp = _amp_with_verifier(_gen("Prose with nothing checkable."), _VacuousVerifier(), tmp_path)
    out = await amp.amplify(
        AmplificationRequest(objective="describe the weather", mode=ReasoningMode.FAST)
    )
    assert out.answer
    assert out.verified is False


@pytest.mark.asyncio
async def test_proof_mode_refuses_when_verifier_unavailable(tmp_path):
    """PROOF with a crashed verifier must REFUSE, not assert. Refusal-on-
    unverifiable is the feature."""
    amp = _amp_with_verifier(_gen("The answer is 4."), _CrashingVerifier(), tmp_path)
    out = await amp.amplify(
        AmplificationRequest(objective="what is 2 + 2", mode=ReasoningMode.PROOF)
    )
    assert "can't assert" in out.answer
    assert "verifier was unavailable" in out.answer
    assert "proof_refused_unverified" in out.receipt.fallbacks_used


@pytest.mark.asyncio
async def test_proof_mode_refuses_on_vacuous_pass(tmp_path):
    amp = _amp_with_verifier(_gen("Prose with nothing checkable."), _VacuousVerifier(), tmp_path)
    out = await amp.amplify(
        AmplificationRequest(objective="describe the weather", mode=ReasoningMode.PROOF)
    )
    assert "can't assert" in out.answer
    assert "mechanically checkable" in out.answer
