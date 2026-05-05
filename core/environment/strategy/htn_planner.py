"""Hierarchical Task Network (HTN) planner for goal decomposition.

Supports:
- Milestone tracking with do-not-progress-until gates
- Emergency override conditions
- Phase-specific objectives
- State-based goal decomposition (not just manual seeding)
- Prerequisite graph resolution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.environment.parsed_state import ParsedState
from core.environment.belief_graph import EnvironmentBeliefGraph


@dataclass
class TaskNode:
    """A single task or sub-goal in the HTN."""
    task_id: str
    name: str
    status: str = "pending"  # pending, active, completed, failed, blocked
    prerequisites: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    phase: str = ""  # e.g., "early_exploration", "resource_stabilization"
    gate_condition: str | None = None  # do-not-progress-until condition
    emergency_override: bool = False  # bypasses normal prerequisite checks


@dataclass
class Milestone:
    """A tracked progress marker."""
    milestone_id: str
    name: str
    achieved: bool = False
    achieved_at_step: int | None = None
    required_for: list[str] = field(default_factory=list)  # task_ids that need this


@dataclass
class EmergencyCondition:
    """An emergency that overrides normal planning."""
    name: str
    check: Callable[[ParsedState, EnvironmentBeliefGraph], bool]
    forced_task_name: str
    priority: float = 100.0


class HTNPlanner:
    """Decomposes long-horizon goals into active plan states and prerequisites.
    
    Now supports:
    - Milestone-based do-not-progress gates
    - Emergency condition overrides
    - State-driven goal decomposition
    - Phase-specific objective filtering
    """

    def __init__(self):
        self.tasks: dict[str, TaskNode] = {}
        self.root_task_id: str | None = None
        self.milestones: dict[str, Milestone] = {}
        self.emergency_conditions: list[EmergencyCondition] = []
        self.current_phase: str = "orientation"
        self._step_count: int = 0

    def set_root_goal(self, name: str, context: dict[str, Any] | None = None) -> str:
        """Sets the overarching primary goal."""
        task_id = f"goal_{name.replace(' ', '_').lower()}"
        self.tasks[task_id] = TaskNode(task_id=task_id, name=name, context=context or {})
        self.root_task_id = task_id
        return task_id

    def add_subtask(
        self,
        parent_id: str,
        name: str,
        prerequisites: list[str] | None = None,
        phase: str = "",
        gate_condition: str | None = None,
        priority: float = 0.0,
    ) -> str:
        """Adds a subtask to an existing task."""
        task_id = f"task_{len(self.tasks)}_{name.replace(' ', '_').lower()}"
        self.tasks[task_id] = TaskNode(
            task_id=task_id,
            name=name,
            prerequisites=prerequisites or [],
            phase=phase,
            gate_condition=gate_condition,
            priority=priority,
        )
        if parent_id in self.tasks:
            self.tasks[parent_id].subtasks.append(task_id)
        return task_id

    def add_milestone(self, name: str, required_for: list[str] | None = None) -> str:
        """Register a progress milestone."""
        milestone_id = f"ms_{name.replace(' ', '_').lower()}"
        self.milestones[milestone_id] = Milestone(
            milestone_id=milestone_id,
            name=name,
            required_for=required_for or [],
        )
        return milestone_id

    def achieve_milestone(self, milestone_id: str, step: int) -> None:
        """Mark a milestone as achieved."""
        if milestone_id in self.milestones:
            ms = self.milestones[milestone_id]
            ms.achieved = True
            ms.achieved_at_step = step

    def register_emergency(
        self,
        name: str,
        check: Callable[[ParsedState, EnvironmentBeliefGraph], bool],
        forced_task_name: str,
        priority: float = 100.0,
    ) -> None:
        """Register an emergency condition that overrides normal planning."""
        self.emergency_conditions.append(EmergencyCondition(
            name=name, check=check, forced_task_name=forced_task_name, priority=priority,
        ))

    def check_emergencies(self, parsed: ParsedState, belief: EnvironmentBeliefGraph) -> TaskNode | None:
        """Check if any emergency condition is active. Returns forced task or None."""
        for ec in sorted(self.emergency_conditions, key=lambda e: e.priority, reverse=True):
            try:
                if ec.check(parsed, belief):
                    return TaskNode(
                        task_id=f"emergency_{ec.name}",
                        name=ec.forced_task_name,
                        status="active",
                        emergency_override=True,
                        priority=ec.priority,
                    )
            except Exception:
                continue
        return None

    def decompose_from_state(self, parsed: ParsedState, belief: EnvironmentBeliefGraph) -> None:
        """Infer sub-goals from the current state if no tasks are active.
        
        This is the key difference from manual goal seeding — the planner
        looks at what's available and creates objectives autonomously.
        """
        if self.get_active_tasks():
            return  # already have work

        # If we have a root goal but no active subtasks, try to decompose
        if not self.root_task_id:
            self.set_root_goal("survive_and_progress")

        root = self.tasks[self.root_task_id]

        # Phase-based auto-decomposition
        resources = parsed.resources
        entities = parsed.entities
        hazards = parsed.hazards

        # If critical resources are low, add stabilization task
        for rname, rstate in resources.items():
            if hasattr(rstate, 'value') and hasattr(rstate, 'max_value'):
                if rstate.value < rstate.max_value * 0.3:
                    tid = f"task_stabilize_{rname}"
                    if tid not in self.tasks:
                        self.tasks[tid] = TaskNode(
                            task_id=tid,
                            name=f"stabilize_{rname}",
                            status="active",
                            phase="resource_stabilization",
                            priority=8.0,
                            context={"resource": rname, "completion_event": f"{rname}_stabilized"},
                        )
                        root.subtasks.append(tid)

        # If frontier exists in belief graph, add exploration task
        if belief.frontiers:
            tid = "task_explore_frontier"
            if tid not in self.tasks:
                self.tasks[tid] = TaskNode(
                    task_id=tid,
                    name="explore_frontier",
                    status="active",
                    phase="exploration",
                    priority=3.0,
                    context={"completion_event": "frontier_exhausted"},
                )
                root.subtasks.append(tid)

    def update(self, parsed_state: ParsedState, belief: EnvironmentBeliefGraph) -> None:
        """Updates task statuses based on current environment state and beliefs."""
        self._step_count += 1

        # Check for milestone achievements from semantic events
        completed_events = [e.label for e in parsed_state.semantic_events]

        # Also check self_state for milestone-like conditions
        for ms in self.milestones.values():
            if not ms.achieved:
                if ms.name in completed_events:
                    self.achieve_milestone(ms.milestone_id, self._step_count)

        for task in self.tasks.values():
            if task.status == "active":
                # Check if completion criteria met
                if task.context.get("completion_event") in completed_events:
                    task.status = "completed"

            elif task.status == "pending":
                # Check do-not-progress gate
                if task.gate_condition:
                    gate_ms = self.milestones.get(task.gate_condition)
                    if gate_ms and not gate_ms.achieved:
                        task.status = "blocked"
                        continue

                # Check if prerequisites are met
                prereqs_met = True
                for prereq_id in task.prerequisites:
                    prereq = self.tasks.get(prereq_id)
                    if not prereq or prereq.status != "completed":
                        prereqs_met = False
                        break

                if prereqs_met:
                    task.status = "active"

            elif task.status == "blocked":
                # Re-check gate
                if task.gate_condition:
                    gate_ms = self.milestones.get(task.gate_condition)
                    if gate_ms and gate_ms.achieved:
                        task.status = "pending"  # will be re-evaluated next tick

        # Auto-decompose if no active tasks
        self.decompose_from_state(parsed_state, belief)

    def get_active_tasks(self) -> list[TaskNode]:
        """Returns the frontier of currently active actionable subtasks."""
        active = [t for t in self.tasks.values() if t.status == "active" and not t.subtasks]
        return sorted(active, key=lambda t: t.priority, reverse=True)

    def get_blocked_tasks(self) -> list[TaskNode]:
        """Returns tasks blocked by unmet gates."""
        return [t for t in self.tasks.values() if t.status == "blocked"]

    def get_milestone_status(self) -> dict[str, bool]:
        """Returns milestone achievement status."""
        return {ms.milestone_id: ms.achieved for ms in self.milestones.values()}


__all__ = ["TaskNode", "Milestone", "EmergencyCondition", "HTNPlanner"]

