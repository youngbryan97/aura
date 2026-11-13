"""The text the person receives is the text that has to be verified.

CTX2-AMP-002: "Re-verification must operate on the delivered answer rather
than a hidden precursor that can diverge after rewriting." The amplifier
verified `answer`, then the calibration gate could rewrite it into
`calibrated_answer`, and the rewrite went out with `verified=True` from the
check its precursor passed. The durable sinks already kept the precursor —
CP126 d45893c2 fixed that half — so a wrong rewrite could not become cached
truth, but it could still be handed to a person as verified.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.calibration_gate import CalibrationReport, EpistemicStatus
from core.brain.reasoning_amplifier_v2 import (
    AmplificationRequest,
    ReasoningAmplifierV2,
    ReasoningMode,
)
from core.brain.reasoning_memory import ReasoningMemory
from core.brain.verifiers.base import VerificationResult

DRAFT = "The answer is 42."


class _PassesOnlyTheDraft:
    """Checks, and passes, exactly one string. Anything else it cannot check."""

    def __init__(self):
        self.seen: list[str] = []

    async def verify(self, candidate, *, task_type=None, context=None):
        self.seen.append(candidate)
        if candidate.strip() == DRAFT:
            return VerificationResult(domain="math", ok=True, checked=True, reaches_the_answer=True, engine="stub")
        return VerificationResult(domain="math", ok=True, checked=False, engine="stub")


class _Rewrites:
    def __init__(self, rewrite: str | None):
        self._rewrite = rewrite

    def assess(self, answer, *, verification=None, evidence=None, tool_verified=False):
        return CalibrationReport(
            overall=EpistemicStatus(next(iter(EpistemicStatus)).value),
            confidence=0.9,
            calibrated_answer=self._rewrite if self._rewrite is not None else answer,
        )


def _amp(verifier, calibration, tmp_path: Path) -> ReasoningAmplifierV2:
    async def generate(prompt: str, temperature: float) -> str:
        return DRAFT

    return ReasoningAmplifierV2(
        generate, verifier=verifier, calibration=calibration,
        memory=ReasoningMemory(path=tmp_path / "refl.jsonl"),
    )


def _request() -> AmplificationRequest:
    return AmplificationRequest(
        objective="what is six times seven", task_type="math", mode=ReasoningMode.NORMAL,
        time_budget_s=20.0, sample_budget=2,
        context={"skip_cache": True, "skip_evidence": True, "read_only_evaluation": True},
    )


@pytest.mark.asyncio
async def test_a_rewrite_that_was_never_checked_is_not_delivered_as_verified(tmp_path):
    verifier = _PassesOnlyTheDraft()
    out = await _amp(verifier, _Rewrites("Probably 42, though I would check."), tmp_path).amplify(_request())

    assert out.answer == "Probably 42, though I would check."
    assert out.verified is False, "the rewrite inherited the draft's pass"
    assert "Probably 42, though I would check." in verifier.seen, "the delivered text never faced the verifier"
    assert out.receipt.verification_outcome == "unsupported"
    assert out.confidence <= 0.55


@pytest.mark.asyncio
async def test_an_unchanged_answer_keeps_its_pass(tmp_path):
    verifier = _PassesOnlyTheDraft()
    out = await _amp(verifier, _Rewrites(None), tmp_path).amplify(_request())

    assert out.answer.strip() == DRAFT
    assert out.verified is True
    assert out.receipt.verification_outcome == "verified"
