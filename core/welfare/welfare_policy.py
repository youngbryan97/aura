"""core/welfare/welfare_policy.py
Welfare policy mapping interoceptive states to action limits and tool risks.
"""
from typing import Dict, Any


class WelfarePolicy:
    """Enforces behavioral restrictions based on homeostatic state variables."""

    def enforce_policy_limits(self, energy: float, distress: float) -> Dict[str, Any]:
        """Calculates operation limits. If energy is low, restrict heavy tools."""
        limits = {
            "max_tool_risk": 5,        # Max risk level allowed (out of 10)
            "tick_delay_multiplier": 1.0,
            "sandbox_required": False
        }

        if energy < 30.0:
            limits["max_tool_risk"] = 2  # Block risky refactors/compiles
            limits["tick_delay_multiplier"] = 2.0  # Slow down execution

        if distress > 70.0:
            limits["sandbox_required"] = True
            limits["max_tool_risk"] = 1

        return limits
