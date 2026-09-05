"""core/goals/project_manager.py
==============================
Manages long-running, multi-step tasks (projects).
Checkpoints project states to receipt stores and enforces safety limits/budgets.
"""

import logging
import time
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ProjectManager")


class Project:
    """Represents a tracked, long-running project with safety envelopes."""

    def __init__(self, project_id: str, name: str, limits: dict[str, Any] | None = None):
        self.project_id = project_id
        self.name = name
        self.start_time = time.time()
        self.last_checkpoint_at = self.start_time
        
        # Enforce default safety limits
        limits = limits or {}
        self.max_tool_calls = int(limits.get("max_tool_calls", 50))
        self.max_tokens = int(limits.get("max_tokens", 1000000))
        self.max_wall_time = float(limits.get("max_wall_time", 86400.0)) # 1 day max
        
        # Consumed metrics
        self.tool_calls_count = 0
        self.tokens_consumed = 0
        self.checkpoints: dict[float, dict[str, Any]] = {}
        self.current_state: dict[str, Any] = {}

    def is_valid(self) -> bool:
        """Enforces physical project limits."""
        elapsed = time.time() - self.start_time
        if self.tool_calls_count > self.max_tool_calls:
            logger.warning("Project '%s' exceeded max tool calls limit (%d/%d)", self.name, self.tool_calls_count, self.max_tool_calls)
            return False
        if self.tokens_consumed > self.max_tokens:
            logger.warning("Project '%s' exceeded max tokens limit (%d/%d)", self.name, self.tokens_consumed, self.max_tokens)
            return False
        if elapsed > self.max_wall_time:
            logger.warning("Project '%s' exceeded max wall time limit (%.1fs/%.1fs)", self.name, elapsed, self.max_wall_time)
            return False
        return True


class ProjectManager:
    """Manages active projects, state checkpointing, and safety budgets."""

    def __init__(self):
        self._projects: dict[str, Project] = {}

    def start_project(self, project_id: str, name: str, limits: dict[str, Any] | None = None) -> Project:
        """Create and start tracking a new project."""
        project = Project(project_id, name, limits)
        self._projects[project_id] = project
        logger.info("⚡ [PROJECT] Started project '%s' (ID: %s)", name, project_id)
        return project

    def get_project(self, project_id: str) -> Project | None:
        return self._projects.get(project_id)

    def checkpoint_project(self, project_id: str, state: dict[str, Any], receipt_id: str = "") -> bool:
        """Save a state checkpoint for the project."""
        project = self.get_project(project_id)
        if not project:
            logger.warning("Project '%s' not found for checkpoint.", project_id)
            return False
            
        now = time.time()
        project.last_checkpoint_at = now
        project.current_state = state
        project.checkpoints[now] = {
            "state": state,
            "receipt_id": receipt_id
        }
        
        # Mirror checkpoint metadata to ReceiptStore if available
        try:
            store = ServiceContainer.get("receipt_store", default=None)
            if store and hasattr(store, "emit"):
                from core.runtime.receipts import AutonomyReceipt
                store.emit(
                    AutonomyReceipt(
                        cause=f"Checkpoint state for project {project.name}",
                        autonomy_level=3,
                        proposed_action=f"project_checkpoint:{project_id}",
                        metadata={"project_id": project_id, "state_keys": list(state.keys())}
                    )
                )
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
            record_degradation("project_manager", e, severity="warning", action="kept project checkpoint in memory after receipt emission failure")
            logger.debug("Failed to record project checkpoint to ReceiptStore: %s", e)

        logger.info("⚡ [PROJECT] Saved checkpoint for '%s' (keys: %s)", project.name, list(state.keys()))
        return True

    def track_tool_call(self, project_id: str) -> bool:
        project = self.get_project(project_id)
        if project:
            project.tool_calls_count += 1
            return project.is_valid()
        return True

    def track_tokens(self, project_id: str, tokens: int) -> bool:
        project = self.get_project(project_id)
        if project:
            project.tokens_consumed += tokens
            return project.is_valid()
        return True

    def is_within_limits(self, project_id: str) -> bool:
        project = self.get_project(project_id)
        if project:
            return project.is_valid()
        return True
