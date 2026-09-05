"""core/agency/task_resumption.py
Task resumption logic. Restores goals and task state lists after system reboot.
"""
import logging
from typing import Any

logger = logging.getLogger("Agency.TaskResumption")


class TaskResumptionManager:
    """Handles parsing historical snapshots on boot to reload active tasks."""

    def resume_tasks(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extracts incomplete tasks from the recent episode logs."""
        if not history:
            return []
            
        logger.info("TaskResumptionManager scanning historical logs for incomplete tasks...")
        resumed = []
        for event in reversed(history):
            wanted = event.get("wanted", {})
            goals = wanted.get("goals", [])
            for g in goals:
                if g and g not in [r.get("id") for r in resumed]:
                    resumed.append({"id": g, "status": "resumed"})
                    logger.info("Resumed incomplete goal: %s", g)
        return resumed
