"""core/agency/blocked_task_manager.py
Blocked task manager tracking tasks waiting for user corrections or approval gates.
"""
from typing import Dict, List, Any, Optional
import time


class BlockedTaskManager:
    """Manages tasks gated by external events or approvals."""

    def __init__(self):
        self._blocked: Dict[str, Dict[str, Any]] = {}

    def block_task(self, task_id: str, reason: str, gate_condition: str) -> None:
        self._blocked[task_id] = {
            "task_id": task_id,
            "blocked_at": time.time(),
            "reason": reason,
            "gate": gate_condition
        }

    def unblock_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._blocked.pop(task_id, None)

    def check_gates(self, state: Any) -> List[str]:
        """Check if any gate conditions are met (e.g. user presence returned)."""
        unblocked = []
        for task_id, details in list(self._blocked.items()):
            gate = details["gate"]
            if gate == "user_active" and state.body.last_user_activity > time.time() - 30:
                unblocked.append(task_id)
                self.unblock_task(task_id)
        return unblocked
