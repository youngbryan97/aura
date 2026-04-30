"""Promotion gate for ASA shadow candidates."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.architect.config import ASAConfig
from core.architect.errors import PromotionError
from core.architect.models import MutationTier, PromotionDecision, PromotionStatus, ProofReceipt, RefactorPlan, RollbackPacket
from core.architect.mutation_classifier import MutationClassifier
from core.architect.shadow_workspace import ShadowRun
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text


class PromotionGovernor:
    """Autonomously promote candidates only when proof obligations pass."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.classifier = MutationClassifier(self.config)

    def decide(self, plan: RefactorPlan, proof: ProofReceipt, rollback_packet: RollbackPacket) -> PromotionDecision:
        if plan.risk_tier >= MutationTier.T4_GOVERNANCE_SENSITIVE:
            return self._decision(plan, proof, PromotionStatus.PROPOSAL_ONLY, "T4/T5 surfaces are proposal-only")
        if plan.risk_tier > self.config.max_tier:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"plan tier {plan.risk_tier.name} exceeds configured max {self.config.max_tier.name}")
        if not proof.passed:
            failed = [
                result.obligation_id
                for result in proof.results
                if not result.passed
                and not (plan.risk_tier <= MutationTier.T1_CLEANUP and result.status == "BOOT_HARNESS_UNAVAILABLE")
            ]
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"proof failed: {failed[:8]}")
        if not rollback_packet.dry_run_passed:
            return self._decision(plan, proof, PromotionStatus.REJECTED, "rollback dry-run did not pass")
        sealed = [path for path in plan.changed_files if self.classifier.classify_path(path) is MutationTier.T5_SEALED]
        if sealed:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"sealed files touched: {sealed}")
        if not proof.behavior_delta.equivalent and plan.risk_tier <= MutationTier.T2_REFACTOR:
            return self._decision(plan, proof, PromotionStatus.REJECTED, f"behavior regression: {proof.behavior_delta.regressions}")
        return self._decision(plan, proof, PromotionStatus.SHADOW_PASSED, "eligible for atomic promotion")

    def promote(self, plan: RefactorPlan, shadow: ShadowRun, proof: ProofReceipt, rollback_packet: RollbackPacket) -> PromotionDecision:
        decision = self.decide(plan, proof, rollback_packet)
        if decision.status is not PromotionStatus.SHADOW_PASSED:
            self._persist_decision(decision)
            return decision
        promoted: list[str] = []
        for rel in plan.changed_files:
            candidate = Path(shadow.candidate_files.get(rel, ""))
            if not candidate.exists():
                raise PromotionError(f"candidate snapshot missing for {rel}")
            target = self.config.repo_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(target, candidate.read_bytes())
            promoted.append(rel)
        promoted_decision = PromotionDecision(
            run_id=proof.run_id,
            plan_id=plan.id,
            status=PromotionStatus.PROMOTED,
            reason="atomic file promotion completed after proof pass",
            receipt_hash=proof.decision_hash,
            promoted_files=tuple(promoted),
        )
        self._persist_decision(promoted_decision)
        return promoted_decision

    def _decision(self, plan: RefactorPlan, proof: ProofReceipt, status: PromotionStatus, reason: str) -> PromotionDecision:
        return PromotionDecision(
            run_id=proof.run_id,
            plan_id=plan.id,
            status=status,
            reason=reason,
            receipt_hash=proof.decision_hash,
        )

    def _persist_decision(self, decision: PromotionDecision) -> None:
        decisions = self.config.artifacts / "decisions"
        decisions.mkdir(parents=True, exist_ok=True)
        atomic_write_text(decisions / f"{decision.run_id}.json", json.dumps(asdict(decision), indent=2, sort_keys=True, default=str))
