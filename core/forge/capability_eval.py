"""core/forge/capability_eval.py — Capability Evaluator."""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("Aura.CapabilityEval")


class CapabilityEval:
    """Assesses capability levels and guarantees zero regression before patching."""

    @staticmethod
    def evaluate(baseline: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compares baseline vs test stats. Zero-regression policy is enforced."""
        base_pass = float(baseline.get("pass_rate", 1.0))
        test_pass = float(test_results.get("pass_rate", 0.0))
        
        passed = test_pass >= base_pass
        logger.info("Capability evaluation: baseline=%.2f, test=%.2f (passed=%s)", base_pass, test_pass, passed)
        
        return {
            "passed": passed,
            "baseline_pass_rate": base_pass,
            "test_pass_rate": test_pass,
            "regression_detected": not passed,
            "welfare_impact": "neutral",
        }
