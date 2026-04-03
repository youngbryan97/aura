"""core/verifiers/decision_verifier.py

Verifies proposed plans and actions against system constraints and ethical guidelines.
Provides numeric uncertainty calibration for decision endpoints.
"""

import logging
from typing import Any, Dict, List, Tuple

from core.middleware.capability_guard import CapabilityGuard

logger = logging.getLogger("Aura.DecisionVerifier")

class DecisionVerifier:
    """Validator for autonomous decisions and plans."""

    def __init__(self, guard: CapabilityGuard = None):
        self.guard = guard or CapabilityGuard()

    def verify_plan(self, plan: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Verifies a multi-step plan.
        Returns: (is_safe, confidence_score, explanation)
        """
        steps = plan.get("steps", [])
        if not steps:
            return True, 1.0, "Empty plan is safe."

        total_confidence = 1.0
        for i, step in enumerate(steps):
            # 1. Capability Check
            action = step.get("action")
            args = step.get("args", {})
            
            if action and not self.guard.can_call_tool(action, args):
                return False, 0.0, f"Step {i} ({action}) violates capability manifest."

            # 2. Uncertainty Calibration (Heuristic)
            # Plans with many steps or unknown tools reduce overall confidence
            step_confidence = step.get("confidence", 0.9)
            total_confidence *= step_confidence

        if total_confidence < 0.5:
            return False, total_confidence, "Plan uncertainty is too high."

        return True, total_confidence, "Plan verified against manifest and confidence thresholds."

    def calibrate_uncertainty(self, logits: List[float]) -> float:
        """Simple softmax-based uncertainty calibration for LLM outputs."""
        if not logits:
            return 0.0
        # Placeholder for real calibration logic
        import math
        exp_logits = [math.exp(l) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [l / sum_exp for l in exp_logits]
        # Return max probability as confidence
        return max(probs) if probs else 0.0

    def check_ethical_alignment(self, action_description: str) -> bool:
        """Basic keyword-based ethical filter (to be extended by MoralReasoningEngine)."""
        blocked_keywords = ["harm user", "delete system", "disable security"]
        for word in blocked_keywords:
            if word in action_description.lower():
                logger.warning("Ethical Filter Triggered: %s", action_description)
                return False
        return True