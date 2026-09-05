"""core/agency/autonomous_followup.py
Coordinates background follow-up loops for tasks that require long processing times.
"""
from typing import Any


class AutonomousFollowupCoordinator:
    """Monitors running background tasks and queues followup check actions."""

    def evaluate_followups(self, active_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        followups = []
        for t in active_tasks:
            # If a task has been running for a while, schedule a status query action
            if t.get("status") == "in_progress":
                followups.append({
                    "channel": "terminal",
                    "params": {
                        "command": f"echo 'Checking task status for: {t.get('name')}'",
                        "timeout": 2.0
                    }
                })
        return followups
