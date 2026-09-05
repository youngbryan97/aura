"""core/mission/objective_graph.py — Milestone Dependency Graph.

Manages a DAG of subtasks and milestones with dependencies, status tracking,
and blocker detection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.ObjectiveGraph")


@dataclass
class Milestone:
    """A single milestone in the mission dependency graph."""
    milestone_id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed, blocked
    dependencies: List[str] = field(default_factory=list)
    assigned_worker: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    priority: float = 1.0
    estimated_duration_s: float = 60.0


class ObjectiveGraph:
    """Manages a directed acyclic graph of milestones with dependency resolution."""

    def __init__(self) -> None:
        self.milestones: Dict[str, Milestone] = {}

    def add_milestone(self, ms: Milestone) -> None:
        self.milestones[ms.milestone_id] = ms

    def set_status(self, milestone_id: str, status: str) -> None:
        ms = self.milestones.get(milestone_id)
        if ms:
            ms.status = status

    def is_blocked(self, milestone_id: str) -> bool:
        """A milestone is blocked if any of its dependencies are not completed."""
        ms = self.milestones.get(milestone_id)
        if not ms:
            return False
        for dep_id in ms.dependencies:
            dep = self.milestones.get(dep_id)
            if dep and dep.status != "completed":
                return True
        return False

    def get_ready_milestones(self) -> List[Milestone]:
        """Return milestones whose dependencies are all met."""
        ready = []
        for ms in self.milestones.values():
            if ms.status == "pending" and not self.is_blocked(ms.milestone_id):
                ready.append(ms)
        return sorted(ready, key=lambda m: -m.priority)

    def get_critical_path(self) -> List[str]:
        """Return the longest dependency chain (simplified critical path)."""
        def _depth(mid: str, visited: set) -> int:
            if mid in visited:
                return 0
            visited.add(mid)
            ms = self.milestones.get(mid)
            if not ms or not ms.dependencies:
                return 1
            return 1 + max(_depth(d, visited) for d in ms.dependencies)

        if not self.milestones:
            return []
        depths = {mid: _depth(mid, set()) for mid in self.milestones}
        return sorted(depths, key=lambda m: -depths[m])

    def completion_ratio(self) -> float:
        if not self.milestones:
            return 1.0
        completed = sum(1 for m in self.milestones.values() if m.status == "completed")
        return completed / len(self.milestones)

    def summary(self) -> Dict[str, int]:
        by_status: Dict[str, int] = {}
        for ms in self.milestones.values():
            by_status[ms.status] = by_status.get(ms.status, 0) + 1
        return by_status
