"""core/agency/project_manager.py
Manages multi-day task scopes, hierarchies, and completion checklists.
"""
from typing import Any


class ProjectManager:
    """Manages active project tasks and checklists."""

    def __init__(self):
        # Maps project_id -> list of tasks
        self._projects: dict[str, list[dict[str, Any]]] = {}

    def create_project(self, project_id: str) -> None:
        self._projects[project_id] = []

    def add_task(self, project_id: str, task_name: str) -> None:
        if project_id not in self._projects:
            self.create_project(project_id)
        self._projects[project_id].append({
            "name": task_name,
            "status": "pending"
        })

    def complete_task(self, project_id: str, task_name: str) -> None:
        if project_id in self._projects:
            for t in self._projects[project_id]:
                if t["name"] == task_name:
                    t["status"] = "completed"

    def get_project_completion_pct(self, project_id: str) -> float:
        if project_id not in self._projects or not self._projects[project_id]:
            return 0.0
        tasks = self._projects[project_id]
        completed = len([t for t in tasks if t["status"] == "completed"])
        return (completed / len(tasks)) * 100.0
