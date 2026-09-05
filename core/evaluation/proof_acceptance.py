"""Shared acceptance boundary for proof and evaluation runners."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.runtime.skill_contract import (
    ActionExpectation,
    SkillExecutionResult,
    SkillStatus,
    apply_action_expectation,
)


@dataclass(frozen=True)
class ProofAcceptanceResult:
    proof_id: str
    accepted: bool
    status: str
    expectation: dict[str, Any]
    verdict: dict[str, Any]
    failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "accepted": self.accepted,
            "status": self.status,
            "expectation": dict(self.expectation),
            "verdict": dict(self.verdict),
            "failure_reason": self.failure_reason,
        }


def evaluate_proof_acceptance(
    proof_id: str,
    *,
    candidate_passed: bool,
    evidence: Mapping[str, Any],
    expectation: ActionExpectation,
) -> ProofAcceptanceResult:
    """Evaluate proof evidence through the same contract used by live actions."""
    payload = dict(evidence)
    payload.setdefault("ok", bool(candidate_passed))
    checked = apply_action_expectation(
        SkillExecutionResult(
            skill=f"proof:{proof_id}",
            status=(
                SkillStatus.SUCCESS_VERIFIED
                if candidate_passed
                else SkillStatus.FAILED_RECOVERABLE
            ),
            output=payload,
            verification_evidence=payload,
            expectation=expectation,
            failure_reason=None if candidate_passed else "candidate evaluator failed",
        )
    )
    verdict = dict(checked.verification_evidence.get("expectation_verdict") or {})
    accepted = bool(
        candidate_passed
        and checked.status == SkillStatus.SUCCESS_VERIFIED
        and verdict.get("passed", False)
    )
    return ProofAcceptanceResult(
        proof_id=str(proof_id),
        accepted=accepted,
        status=checked.status.value,
        expectation=expectation.to_dict(),
        verdict=verdict,
        failure_reason=str(checked.failure_reason or ""),
    )


__all__ = ["ProofAcceptanceResult", "evaluate_proof_acceptance"]
