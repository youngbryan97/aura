"""core/factory/rollback_manager.py — Rollback and Workspace Manager.

Manages git workspaces, branches, stash points, and git rollbacks
for safe self-modification and software patches.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.RollbackManager")


class RollbackManager:
    """Safely manages isolated git workspaces, checkout, and rollback operations."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, dict[str, Any]] = {}

    async def create_workspace(self, repo_path: str, branch_name: str) -> bool:
        """Create a new branch in the target repo to isolate changes."""
        logger.info("🌳 RollbackManager: creating branch '%s' in '%s'", branch_name, repo_path)
        try:
            # Check current status and checkout new branch via approved gateway
            get_subprocess_gateway().run(["git", "status"], cwd=repo_path, source="rollback_manager", accelerator_capability="none")
            get_subprocess_gateway().run(["git", "checkout", "-b", branch_name], cwd=repo_path, source="rollback_manager", accelerator_capability="none")
            return True
        except (OSError, RuntimeError) as e:
            record_degradation(
                "rollback_manager",
                e,
                action="failed to create isolated git branch",
                extra={"repo": repo_path, "branch": branch_name}
            )
            return False

    def register_rollback_point(self, repo_path: str, branch_name: str) -> str:
        """Register the current git state (SHA) as a rollback checkpoint."""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        try:
            res = get_subprocess_gateway().run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                source="rollback_manager",
                accelerator_capability="none",
            )
            sha = res.stdout.strip() if res.stdout else "HEAD"
            self._checkpoints[checkpoint_id] = {
                "repo_path": repo_path,
                "branch_name": branch_name,
                "commit_sha": sha,
                "timestamp": time.time(),
            }
            logger.info("🛡️  Registered rollback point '%s' at SHA %s", checkpoint_id, sha)
        except (OSError, RuntimeError) as e:
            logger.warning("Could not register rollback commit SHA: %s", e)
            self._checkpoints[checkpoint_id] = {
                "repo_path": repo_path,
                "branch_name": branch_name,
                "commit_sha": "HEAD",
                "timestamp": time.time(),
            }
        return checkpoint_id

    async def execute_rollback(self, checkpoint_id: str) -> bool:
        """Roll back the repository to the registered checkpoint state."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.error("🚫 RollbackManager: checkpoint '%s' not found", checkpoint_id)
            return False

        repo_path = checkpoint["repo_path"]
        sha = checkpoint["commit_sha"]
        logger.warning("🚨 RollbackManager: triggering rollback to '%s' (SHA: %s)", checkpoint_id, sha)

        try:
            # Hard reset changes in git via approved gateway
            get_subprocess_gateway().run(["git", "reset", "--hard", sha], cwd=repo_path, source="rollback_manager", accelerator_capability="none")
            get_subprocess_gateway().run(["git", "clean", "-fd"], cwd=repo_path, source="rollback_manager", accelerator_capability="none")
            logger.info("✅ Rollback successful. Repository restored.")
            return True
        except (OSError, RuntimeError) as e:
            record_degradation(
                "rollback_manager",
                e,
                action="failed git reset/clean during rollback",
                extra={"checkpoint": checkpoint_id, "sha": sha}
            )
            return False

    def get_checkpoints(self) -> list[dict[str, Any]]:
        return [{"id": k, **v} for k, v in self._checkpoints.items()]
