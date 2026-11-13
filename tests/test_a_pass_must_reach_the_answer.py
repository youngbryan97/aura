"""A verifier's pass is authority over an answer only where the check reached the answer itself.

LIVE 13 November 2026 (F04's typed-lane measure): an AIME draft cut off by its cap before it reached any answer was
kept as a verified incumbent in 0.006 seconds, because the logic audit found no non-sequitur in its working. A check
of the working, the citations, the form or a trace can refute an answer and cannot confirm one.
"""
from __future__ import annotations

import asyncio

from core.brain.verifiers.base import VerificationResult, combine_results

WORKING_ONLY = "Let x = 3, so 2x = 6 and the area follows from the next step, which"


class _Verifier:
    """Passes everything; says whether its pass reached the answer, and refutes one string if told to."""

    def __init__(self, *, reaches: bool, refutes: str = ""):
        self.reaches, self.refutes = reaches, refutes

    async def verify(self, candidate, *, task_type=None, context=None):
        if self.refutes and str(candidate).strip() == self.refutes:
            return VerificationResult(domain="math", ok=False, checked=True, engine="stub", issues=["wrong"])
        return VerificationResult(domain="math", ok=True, checked=True, reaches_the_answer=self.reaches,
                                  engine="stub")


def _amplify(verifier, seed: str, generated: str, tmp_path):
    from core.brain.calibration_gate import CalibrationGate
    from core.brain.reasoning_amplifier_v2 import (
        AmplificationRequest,
        ReasoningAmplifierV2,
        ReasoningMode,
    )
    from core.brain.reasoning_memory import ReasoningMemory

    asked: list[float] = []

    async def generate(prompt: str, temperature: float) -> str:
        asked.append(temperature)
        return generated

    amp = ReasoningAmplifierV2(generate, verifier=verifier, calibration=CalibrationGate(),
                               memory=ReasoningMemory(path=tmp_path / "refl.jsonl"))
    out = asyncio.run(amp.amplify(AmplificationRequest(
        objective="Find the area.", task_type="math", mode=ReasoningMode.NORMAL, sample_budget=2,
        context={"seed_candidates": [seed], "read_only_evaluation": True, "skip_cache": True})))
    return out, asked


def test_a_pass_that_did_not_reach_the_answer_does_not_keep_the_draft(tmp_path):
    out, asked = _amplify(_Verifier(reaches=False), WORKING_ONLY, WORKING_ONLY, tmp_path)
    assert out.receipt.promotion_authority != "preserve_incumbent"
    assert "incumbent_proxy_pass_not_authoritative" in out.receipt.fallbacks_used
    assert asked, "the search did not run"
    assert out.verified is False


def test_a_pass_that_reached_the_answer_keeps_it(tmp_path):
    out, asked = _amplify(_Verifier(reaches=True), "The area is 6.", "unused", tmp_path)
    assert out.receipt.promotion_authority == "preserve_incumbent"
    assert asked == []
    assert out.verified is True


def test_a_draft_proved_wrong_is_replaced_but_the_replacement_is_not_called_verified(tmp_path):
    out, _asked = _amplify(_Verifier(reaches=False, refutes="The area is 7."), "The area is 7.", "The area is 6.",
                           tmp_path)
    assert out.receipt.promotion_authority == "checked_verifier"
    assert out.answer.strip() == "The area is 6."
    assert out.verified is False


def test_the_fold_reaches_the_answer_only_through_an_engine_that_did():
    reached = VerificationResult(domain="math", ok=True, checked=True, reaches_the_answer=True, engine="math")
    audited = VerificationResult(domain="logic", ok=True, checked=True, engine="logic")
    refuted = VerificationResult(domain="logic", ok=False, checked=True, engine="logic")
    assert combine_results("math", [reached, audited]).reaches_the_answer is True
    assert combine_results("math", [audited]).reaches_the_answer is False
    assert combine_results("math", [reached, refuted]).reaches_the_answer is False  # a provable fault is final
