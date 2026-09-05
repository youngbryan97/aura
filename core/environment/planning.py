"""Bounded non-LLM search for environment navigation and subgoal execution."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from .belief_graph import EnvironmentBeliefGraph
from .command import ActionIntent


@dataclass(frozen=True)
class SearchNode:
    position: tuple[int, int]
    cost: float
    priority: float
    path: tuple[tuple[int, int], ...] = field(default_factory=tuple)


class GridPathPlanner:
    """A* over the canonical spatial belief map."""

    _NEIGHBORS = {
        "north": (0, -1),
        "south": (0, 1),
        "east": (1, 0),
        "west": (-1, 0),
        "northeast": (1, -1),
        "northwest": (-1, -1),
        "southeast": (1, 1),
        "southwest": (-1, 1),
    }

    def plan(
        self,
        belief: EnvironmentBeliefGraph,
        *,
        context_id: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_expansions: int = 512,
    ) -> list[tuple[int, int]]:
        frontier: list[tuple[float, int, SearchNode]] = []
        counter = 0
        heapq.heappush(frontier, (0.0, counter, SearchNode(start, 0.0, 0.0, (start,))))
        best_cost = {start: 0.0}
        expansions = 0
        while frontier and expansions < max_expansions:
            _, _, node = heapq.heappop(frontier)
            expansions += 1
            if node.position == goal:
                return list(node.path)
            for _, (dx, dy) in self._NEIGHBORS.items():
                nxt = (node.position[0] + dx, node.position[1] + dy)
                if not self._walkable(belief, context_id, nxt):
                    continue
                step_cost = 1.4 if dx and dy else 1.0
                new_cost = node.cost + step_cost + self._risk_cost(belief, context_id, nxt)
                if new_cost >= best_cost.get(nxt, float("inf")):
                    continue
                best_cost[nxt] = new_cost
                priority = new_cost + abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                counter += 1
                heapq.heappush(frontier, (priority, counter, SearchNode(nxt, new_cost, priority, node.path + (nxt,))))
        return []

    def next_move_intent(
        self,
        belief: EnvironmentBeliefGraph,
        *,
        context_id: str,
        goal: tuple[int, int],
    ) -> ActionIntent | None:
        start = belief.current_position(context_id)
        if start is None:
            return None
        path = self.plan(belief, context_id=context_id, start=start, goal=goal)
        if len(path) < 2:
            return None
        direction = self._direction_between(path[0], path[1])
        if not direction:
            return None
        return ActionIntent(
            name="move",
            parameters={"direction": direction, "planned_goal": goal},
            expected_effect="planned_progress",
            risk="caution",
            tags={"planned_path"},
        )

    @classmethod
    def _direction_between(cls, start: tuple[int, int], end: tuple[int, int]) -> str | None:
        delta = (max(-1, min(1, end[0] - start[0])), max(-1, min(1, end[1] - start[1])))
        for name, vector in cls._NEIGHBORS.items():
            if vector == delta:
                return name
        return None

    @staticmethod
    def _walkable(belief: EnvironmentBeliefGraph, context_id: str, pos: tuple[int, int]) -> bool:
        cell = belief.spatial.get((context_id, pos[0], pos[1]))
        if cell is None:
            return True
        if cell.get("walkable") is False:
            return False
        kind = str(cell.get("kind", "unknown"))
        return kind not in {"wall", "hazard", "trap", "hostile_entity"}

    @staticmethod
    def _risk_cost(belief: EnvironmentBeliefGraph, context_id: str, pos: tuple[int, int]) -> float:
        cell = belief.spatial.get((context_id, pos[0], pos[1]))
        if cell is None:
            return 0.2
        kind = str(cell.get("kind", "unknown"))
        confidence = float(cell.get("confidence", 0.0) or 0.0)
        if kind in {"hazard", "trap", "hostile_entity"}:
            return 10.0 * confidence
        if kind == "unknown":
            return 0.5
        return 0.0


__all__ = ["GridPathPlanner", "SearchNode"]
