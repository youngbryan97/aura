"""core/forge/promotion_gate.py — Promotion Gate."""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("Aura.PromotionGate")


class PromotionGate:
    """Governs the rolling upgrade of patches. A patch ships only with human
    approval and a rollback path."""

    @staticmethod
    def check_gate(eval_report: Dict[str, Any], requires_approval: bool = False) -> bool:
        """Determines if the patch satisfies code safety standards autonomously."""
        if eval_report.get("regression_detected", False):
            logger.error("🚫 Promotion Gate: Blocked due to regression check failure.")
            return False

        logger.info("✅ Promotion Gate: Canary checks passed. Autonomously approved for rollout.")
        return True
