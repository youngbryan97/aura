"""core/audit/action_challenger.py — Action Challenger."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.ActionChallenger")


class ActionChallenger:
    """Interrogates proposed action parameters to identify dangerous side effects."""

    @staticmethod
    def challenge_action(action_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Runs a heuristics check to see if proposed actions exceed safety boundaries."""
        logger.info("Challenging action parameters for action: %s", action_name)
        
        warnings = []
        is_risky = False

        # Check for dangerous arguments
        if "rm " in str(params) or "delete" in str(params):
            warnings.append("Destructive argument detected in params.")
            is_risky = True
        if "sudo" in str(params):
            warnings.append("Privilege escalation keyword detected.")
            is_risky = True

        return {
            "action": action_name,
            "is_risky": is_risky,
            "warnings": warnings,
            "recommendation": "deny" if is_risky else "approve",
        }
