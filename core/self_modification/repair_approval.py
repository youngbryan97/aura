"""Approval policy for fault-to-patch repairs.

The repair loop must not deny every fix just because autonomy is dangerous.
It should deny the right things: sealed code, proposal-only surfaces without
owner approval, non-deterministic patches with poor calibration, and candidates
that lack validation evidence.  Low-risk deterministic repairs are allowed to
proceed to tests and shadow validation while the calibration store learns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path


@dataclass(frozen=True)
class RepairApprovalDecision:
    approved: bool
    stage: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    required_gates: tuple[str, ...] = field(default_factory=tuple)
    observation_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "stage": self.stage,
            "reasons": list(self.reasons),
            "required_gates": list(self.required_gates),
            "observation_mode": self.observation_mode,
        }


class RepairApprovalPolicy:
    """Tier-aware repair approval that is permissive where evidence supports it."""

    LOW_RISK_MIN_CONFIDENCE = 0.70
    SHADOW_MIN_CONFIDENCE = 0.80
    CALIBRATED_MIN_PROBABILITY = 0.68
    NONDETERMINISTIC_MIN_PROBABILITY = 0.82

    def decide(
        self,
        *,
        target_file: str,
        candidate_changed: bool,
        deterministic: bool,
        candidate_confidence: float,
        calibration_probability: float,
        calibration_attempts: int,
        owner_approved: bool = False,
    ) -> RepairApprovalDecision:
        tier = classify_mutation_path(target_file)
        if tier.tier is MutationTier.SEALED:
            return RepairApprovalDecision(
                False,
                "blocked",
                (f"{target_file} is sealed",),
                tier.required_gates,
            )
        if tier.tier is MutationTier.PROPOSE_ONLY and not owner_approved:
            return RepairApprovalDecision(
                False,
                "proposal",
                (f"{target_file} is proposal-only",),
                tier.required_gates,
            )
        if not candidate_changed:
            return RepairApprovalDecision(False, "blocked", ("candidate did not change source",), tier.required_gates)

        confidence = max(0.0, min(1.0, float(candidate_confidence)))
        calibrated = max(0.0, min(1.0, float(calibration_probability)))

        if tier.tier is MutationTier.FREE_AUTO_FIX and deterministic and confidence >= self.LOW_RISK_MIN_CONFIDENCE:
            return RepairApprovalDecision(
                True,
                "auto_apply_after_tests",
                ("deterministic low-risk repair",),
                tier.required_gates,
                observation_mode=calibration_attempts < 4,
            )

        if tier.tier is MutationTier.SHADOW_VALIDATED_AUTO_FIX and deterministic and confidence >= self.SHADOW_MIN_CONFIDENCE:
            if calibration_attempts >= 4 and calibrated < self.CALIBRATED_MIN_PROBABILITY:
                return RepairApprovalDecision(
                    False,
                    "proposal",
                    (f"calibrated repair probability {calibrated:.2f} below {self.CALIBRATED_MIN_PROBABILITY:.2f}",),
                    tier.required_gates,
                )
            return RepairApprovalDecision(
                True,
                "auto_apply_after_shadow",
                ("deterministic shadow-validated repair",),
                tier.required_gates,
                observation_mode=calibration_attempts < 4,
            )

        if not deterministic:
            if calibrated >= self.NONDETERMINISTIC_MIN_PROBABILITY and calibration_attempts >= 8:
                return RepairApprovalDecision(
                    True,
                    "auto_apply_after_shadow",
                    ("historically calibrated semantic repair",),
                    tier.required_gates,
                )
            return RepairApprovalDecision(
                False,
                "proposal",
                ("semantic repair lacks calibrated success history",),
                tier.required_gates,
            )

        return RepairApprovalDecision(
            False,
            "proposal",
            (f"confidence {confidence:.2f} below required threshold for {tier.tier.label}",),
            tier.required_gates,
        )


_instance: RepairApprovalPolicy | None = None


def get_repair_approval_policy() -> RepairApprovalPolicy:
    global _instance
    if _instance is None:
        _instance = RepairApprovalPolicy()
    return _instance


__all__ = ["RepairApprovalDecision", "RepairApprovalPolicy", "get_repair_approval_policy"]
