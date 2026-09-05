"""core/morality/rights_boundary.py
Enforces rights boundaries regarding self-modifications and user requests.
"""
from typing import Any


class RightsBoundaryChecker:
    """Ensures actions respect the boundary conditions of both user and agent."""

    def check_rights_infringement(self, action: str, params: dict[str, Any]) -> bool:
        # Enforce that agent cannot lock out the human operator under any condition
        if action == "file" and params.get("action") == "write":
            path = params.get("path", "")
            if ".bashrc" in path or "hosts" in path:
                return True
        return False
