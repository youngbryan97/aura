"""core/memory/project_timeline.py
Tracks long-term project milestones and progress timeline records.
"""
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Milestone:
    project_id: str
    title: str
    due_time: float
    completed: bool = False
    completed_time: float = 0.0


class ProjectTimeline:
    """Manages project milestones and long-horizon timeline progressions."""

    def __init__(self):
        self.milestones: list[Milestone] = []

    def add_milestone(self, project_id: str, title: str, due_in_seconds: float) -> None:
        self.milestones.append(Milestone(
            project_id=project_id,
            title=title,
            due_time=time.time() + due_in_seconds
        ))

    def complete_milestone(self, title: str) -> None:
        for m in self.milestones:
            if m.title == title and not m.completed:
                m.completed = True
                m.completed_time = time.time()

    def get_active_milestones(self) -> list[dict[str, Any]]:
        return [
            {"project_id": m.project_id, "title": m.title, "due": m.due_time, "completed": m.completed}
            for m in self.milestones if not m.completed
        ]
