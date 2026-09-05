"""core/world/affordance_model.py
Determines actions possible on files, paths, and workspace tools.
"""
from typing import List


class AffordanceModel:
    """Predicts permissible tools and action affordances for environment targets."""

    def get_affordances(self, entity_type: str, path: str) -> List[str]:
        """Maps target types to valid somatic motors."""
        if entity_type == "file":
            if path.endswith(".py"):
                return ["read", "write", "terminal_run"]
            return ["read", "write"]
        elif entity_type == "app":
            return ["focus_app", "ui_click"]
        return []
