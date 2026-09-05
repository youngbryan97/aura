"""core/world/project_graph.py
Project dependency graph. Maps task hierarchies and milestones.
"""


class ProjectGraph:
    """Tracks tasks and task dependency linkages."""

    def __init__(self):
        self._tasks: dict[str, list[str]] = {}

    def register_task(self, task_id: str, dependencies: list[str]) -> None:
        self._tasks[task_id] = dependencies

    def get_dependencies(self, task_id: str) -> list[str]:
        return self._tasks.get(task_id, [])

    def is_executable(self, task_id: str, completed_tasks: list[str]) -> bool:
        """Confirms all dependency nodes have been satisfied."""
        deps = self.get_dependencies(task_id)
        return all(d in completed_tasks for d in deps)
