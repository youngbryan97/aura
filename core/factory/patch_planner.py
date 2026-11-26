"""core/factory/patch_planner.py — Patch Planning Engine.

Coordinates change plans from objectives and repo maps. Every plan is a
minimal edit carrying its own rationale.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("Aura.PatchPlanner")


class PatchPlanner:
    """Decomposes an objective into specific file-level changes."""

    async def create_plan(
        self, objective: str, repo_map: dict[str, Any]
    ) -> dict[str, Any]:
        """Produce a structured change plan targeting the objective."""
        logger.info("PatchPlanner: creating plan for '%s'", objective[:60])

        changes: list[dict[str, Any]] = []
        files = repo_map.get("files", [])

        # Determine which files are likely affected
        objective_lower = objective.lower()
        affected_files = []

        for f in files:
            basename = os.path.basename(f).lower()
            if basename in objective_lower or basename.replace(".py", "") in objective_lower:
                affected_files.append(f)

        if not affected_files:
            # Fallback to search keywords in full path
            affected_files = [f for f in files if any(
                kw in f.lower() for kw in objective_lower.split()[:5]
            )]

        if not affected_files and files:
            # Default: target the first file
            affected_files = files[:1]

        for file_path in affected_files:
            changes.append({
                "module": file_path,
                "action": "modify",
                "rationale": f"Address objective: {objective[:40]}",
                "estimated_lines": 20,
                "risk": "low",
            })

        return {
            "objective": objective,
            "changes": changes,
            "affected_modules": affected_files,
            "estimated_total_lines": sum(c.get("estimated_lines", 0) for c in changes),
        }
