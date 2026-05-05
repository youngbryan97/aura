"""Hierarchical Task Network (HTN) planner for goal decomposition."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph


@dataclass
class TaskNode:
    """A single task or sub-goal in the HTN."""
    task_id: str
    name: str
    status: str = "pending"  # pending, active, completed, failed
    prerequisites: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


class HTNPlanner:
    """Decomposes long-horizon goals into active plan states and prerequisites."""

    def __init__(self):
        self.tasks: dict[str, TaskNode] = {}
        self.root_task_id: str | None = None

    def set_root_goal(self, name: str, context: dict[str, Any] | None = None) -> str:
        """Sets the overarching primary goal."""
        task_id = f"goal_{name.replace(' ', '_').lower()}"
        self.tasks[task_id] = TaskNode(task_id=task_id, name=name, context=context or {})
        self.root_task_id = task_id
        return task_id

    def add_subtask(self, parent_id: str, name: str, prerequisites: list[str] | None = None) -> str:
        """Adds a subtask to an existing task."""
        task_id = f"task_{len(self.tasks)}_{name.replace(' ', '_').lower()}"
        self.tasks[task_id] = TaskNode(
            task_id=task_id,
            name=name,
            prerequisites=prerequisites or []
        )
        if parent_id in self.tasks:
            self.tasks[parent_id].subtasks.append(task_id)
        return task_id

    def update(self, parsed_state: ParsedState, belief: EnvironmentBeliefGraph) -> None:
        """Updates task statuses based on current environment state and beliefs."""
        # Generic state check. In a specific domain, we'd map
        # belief.check_condition() or parsed_state.events to task completion.
        
        # Example: check if semantic events indicate a task completed
        completed_events = [e.name for e in parsed_state.semantic_events]
        
        for task in self.tasks.values():
            if task.status == "active":
                # Check if completion criteria met
                if task.context.get("completion_event") in completed_events:
                    task.status = "completed"
                    
            elif task.status == "pending":
                # Check if prerequisites are met
                prereqs_met = True
                for prereq_id in task.prerequisites:
                    prereq = self.tasks.get(prereq_id)
                    if not prereq or prereq.status != "completed":
                        prereqs_met = False
                        break
                        
                if prereqs_met:
                    # Depending on parent status, we might activate it
                    task.status = "active"

    def get_active_tasks(self) -> list[TaskNode]:
        """Returns the frontier of currently active actionable subtasks."""
        return [t for t in self.tasks.values() if t.status == "active" and not t.subtasks]

__all__ = ["TaskNode", "HTNPlanner"]
