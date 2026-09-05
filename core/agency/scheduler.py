"""core/agency/scheduler.py
Schedules recurring tasks and one-shot timer callbacks.
"""
import time
from typing import Any


class Scheduler:
    """Manages scheduled events and time triggers."""

    def __init__(self):
        self._scheduled_tasks: list[dict[str, Any]] = []

    def schedule_one_shot(self, label: str, delay_s: float, task_data: dict[str, Any]) -> None:
        self._scheduled_tasks.append({
            "label": label,
            "trigger_time": time.time() + delay_s,
            "data": task_data,
            "triggered": False
        })

    def check_and_trigger(self) -> list[dict[str, Any]]:
        """Identifies and returns tasks whose schedule trigger time has passed."""
        now = time.time()
        triggered = []
        for t in self._scheduled_tasks:
            if not t["triggered"] and now >= t["trigger_time"]:
                t["triggered"] = True
                triggered.append(t["data"])
        return triggered
