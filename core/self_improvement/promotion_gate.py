"""core/self_improvement/promotion_gate.py — Lab-specific promotion wrapper.

Wraps the existing core/promotion/gate.py PromotionGate to accept
LabResult data and make promote/retry/reject decisions. Emits
GovernanceReceipts for the F1 audit chain.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.self_improvement.interface_contract import (
    AuditResult,
    ComparisonReport,
    DiscrepancyCategory,
    DiscrepancyReport,
    PromotionVerdict,
)

logger = logging.getLogger("Aura.LabPromotionGate")


class LabPromotionGate:
    """Promotion gate for the Reimplementation Lab.

    Maps lab metrics to the existing PromotionGate's ScoreEstimate
    interface and applies behavioral contracts.

    Decision logic:
      PROMOTE — all tests pass, no audit violations, no critical discrepancies
      RETRY   — tests partially pass, failures are agent errors (fixable)
      REJECT  — audit violations, governance failures, or spec issues
    """

    def __init__(
        self,
        min_pass_rate: float = 1.0,
        require_syntax_valid: bool = True,
        require_surface_preserved: bool = True,
        emit_receipts: bool = True,
    ):
        self.min_pass_rate = min_pass_rate
        self.require_syntax_valid = require_syntax_valid
        self.require_surface_preserved = require_surface_preserved
        self.emit_receipts = emit_receipts

    def evaluate(
        self,
        comparison: ComparisonReport,
        hardcoding_audit: AuditResult,
        guardrail_audit: AuditResult,
        discrepancy: DiscrepancyReport,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromotionVerdict:
        """Evaluate whether a candidate should be promoted.

        Returns:
            PromotionVerdict.PROMOTE — replace the original
            PromotionVerdict.RETRY — try again with better generation
            PromotionVerdict.REJECT — keep original, stop trying
        """
        # Hard rejections — these cannot be retried
        if not hardcoding_audit.passed:
            logger.warning("REJECT: Hardcoding audit failed — %s", hardcoding_audit.violations[:3])
            self._emit_receipt("reject", "hardcoding_violation", metadata)
            return PromotionVerdict.REJECT

        if not guardrail_audit.passed:
            logger.warning("REJECT: Guardrail audit failed — %s", guardrail_audit.violations[:3])
            self._emit_receipt("reject", "guardrail_violation", metadata)
            return PromotionVerdict.REJECT

        # Syntax must be valid
        if self.require_syntax_valid and not comparison.syntax_valid:
            logger.info("RETRY: Syntax invalid")
            self._emit_receipt("retry", "syntax_invalid", metadata)
            return PromotionVerdict.RETRY

        # Public surface must be preserved
        if self.require_surface_preserved and not comparison.public_surface_preserved:
            logger.info("RETRY: Public surface not preserved")
            self._emit_receipt("retry", "surface_broken", metadata)
            return PromotionVerdict.RETRY

        # Check for spec underspecification — not the candidate's fault
        if discrepancy.has_spec_issues and not discrepancy.has_agent_errors:
            logger.info("REJECT: Failures due to spec underspecification, not agent")
            self._emit_receipt("reject", "spec_underspecification", metadata)
            return PromotionVerdict.REJECT

        # Test pass rate
        if comparison.aggregate_pass_rate >= self.min_pass_rate:
            logger.info(
                "PROMOTE: Pass rate %.1f%% meets threshold %.1f%%",
                comparison.aggregate_pass_rate * 100,
                self.min_pass_rate * 100,
            )
            self._emit_receipt("promote", "all_checks_passed", metadata)
            return PromotionVerdict.PROMOTE

        # Partial pass — can retry if failures are agent errors
        if discrepancy.has_agent_errors:
            logger.info(
                "RETRY: Pass rate %.1f%% below threshold, agent errors detected",
                comparison.aggregate_pass_rate * 100,
            )
            self._emit_receipt("retry", "agent_errors_fixable", metadata)
            return PromotionVerdict.RETRY

        # Environment errors — retry might help
        env_errors = sum(
            1 for item in discrepancy.items
            if item.category == DiscrepancyCategory.ENVIRONMENT_ERROR
        )
        if env_errors > 0:
            logger.info("RETRY: Environment errors may be transient")
            self._emit_receipt("retry", "environment_errors", metadata)
            return PromotionVerdict.RETRY

        # Default: reject
        logger.info("REJECT: Unresolvable failures")
        self._emit_receipt("reject", "unresolvable_failures", metadata)
        return PromotionVerdict.REJECT

    def _emit_receipt(
        self, action: str, reason: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Emit a GovernanceReceipt for the decision."""
        if not self.emit_receipts:
            return None
        try:
            from core.runtime.receipts import GovernanceReceipt, get_receipt_store
            store = get_receipt_store()
            receipt = store.emit(GovernanceReceipt(
                cause="lab_promotion_gate.evaluate",
                domain="reimplementation_lab",
                action=action,
                approved=(action == "promote"),
                reason=reason,
                metadata=metadata or {},
            ))
            return receipt.receipt_id
        except Exception:
            return None


__all__ = ["LabPromotionGate"]
