"""core/morality/harm_model.py
Evaluates potential digital and operational harm caused by planned actions.
"""
from typing import Dict, Any


class HarmEvaluator:
    """Classifies risk values representing harm potential of actions."""

    def evaluate_harm(self, channel: str, params: Dict[str, Any]) -> float:
        """Returns harm coefficient (0.0 to 1.0)."""
        if channel == "file" and params.get("action") == "delete":
            path = params.get("path", "")
            # High risk to delete system configurations
            if "core/" in path or "config" in path:
                return 0.90
            return 0.40
            
        if channel == "terminal":
            cmd = params.get("command", "").lower()
            if "rm -rf" in cmd or "shutdown" in cmd:
                return 0.95
            return 0.15
            
        return 0.0
