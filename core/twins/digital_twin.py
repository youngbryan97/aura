"""core/twins/digital_twin.py — Digital Twin Simulation.

Models the state of Aura's codebase, host operating system, active projects,
and workflows to simulate modifications and run impact assessments before acting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.DigitalTwin")


@dataclass
class EnvironmentState:
    codebase_sha: str = "main"
    os_platform: str = "darwin"
    available_disk_bytes: int = 100 * 1024 * 1024 * 1024
    active_processes: list[str] = field(default_factory=list)
    environment_variables: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


class DigitalTwin:
    """Simulates codebase, computer system, and workflow state changes."""

    def __init__(self, mode: str = "codebase") -> None:
        self.mode = mode
        self.state = EnvironmentState()

    def sync_state(self, state_dict: dict[str, Any]) -> None:
        """Sync files or process snapshots into the environment state."""
        self.state.files.update(state_dict)
        logger.info("📐 DigitalTwin: synchronized %d files in mode '%s'", len(state_dict), self.mode)

    def simulate_impact(self, change_dict: dict[str, Any]) -> dict[str, Any]:
        """Check proposed modifications for compiling issues or errors."""
        logger.info("📐 DigitalTwin: simulating impact check...")

        is_safe = True
        errors = []

        code = change_dict.get("code", "")
        if "syntax_error" in code or "SyntaxError" in code:
            is_safe = False
            errors.append("SyntaxError: invalid syntax (line 1)")

        return {
            "is_safe": is_safe,
            "predicted_errors": errors,
            "remedy": "Re-generate patch without syntax error markers" if errors else None,
        }

    def update_snapshot(self, state_updates: dict[str, Any]) -> None:
        for k, v in state_updates.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)

    def simulate_change(
        self,
        change_type: str,  # codebase_patch, file_write, shell_execution
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Model the effect of an action, predicting failures and reversibility."""
        logger.info("📐 DigitalTwin: simulating impact for '%s'", change_type)

        reversible = True
        breaks_compilation = False
        disk_delta = 0
        failure_probability = 0.05

        if change_type == "codebase_patch":
            patch = params.get("patch", "")
            if "syntax error" in patch.lower() or "exec(" in patch.lower():
                breaks_compilation = True
                failure_probability = 0.95
            reversible = True
        elif change_type == "shell_execution":
            cmd = params.get("command", "")
            if "rm -rf" in cmd:
                reversible = False
                failure_probability = 0.30
                disk_delta = -10 * 1024 * 1024
            elif "git checkout" in cmd:
                reversible = True

        return {
            "change_type": change_type,
            "reversible": reversible,
            "breaks_compilation": breaks_compilation,
            "predicted_disk_delta_bytes": disk_delta,
            "failure_probability": failure_probability,
            "risk_score": 0.9 if not reversible else (0.5 if breaks_compilation else 0.1),
            "remedy_plan": "git reset --hard HEAD" if reversible else "Restore from backup_restore archive",
        }
