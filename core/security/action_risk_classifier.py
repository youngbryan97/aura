"""core/security/action_risk_classifier.py
Classifies risk levels of commands and file write operations.
"""
from typing import Any


class ActionRiskClassifier:
    """Calculates risk levels for planned actions."""

    def classify_risk(self, channel: str, params: dict[str, Any]) -> int:
        """Returns risk coefficient (1 to 10)."""
        if channel == "terminal":
            cmd = params.get("command", "").lower()
            if "sudo" in cmd or "rm " in cmd or "curl" in cmd:
                return 8
            return 4
        elif channel == "file" and params.get("action") == "delete":
            return 7
        return 1
        
        
