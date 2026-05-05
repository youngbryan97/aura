"""Generalized topological and spatial memory mapping."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.environment.parsed_state import ParsedState


@dataclass
class MapNode:
    """A single spatial location or node (e.g. grid cell, DOM element)."""
    node_id: str
    coordinates: tuple[int, ...]  # (x, y) or (x, y, z)
    properties: dict[str, Any] = field(default_factory=dict)
    visited: bool = False
    entities_present: list[str] = field(default_factory=list)


class TopologicalMap:
    """Tracks the agent's spatial understanding of the environment."""

    def __init__(self):
        # Maps coordinate tuples to MapNodes
        self.grid: dict[tuple[int, ...], MapNode] = {}
        # Tracks current agent position
        self.current_position: tuple[int, ...] | None = None

    def update_from_state(self, parsed_state: ParsedState) -> None:
        """Reconciles observations into the persistent map."""
        # This is a generic update. Specific adapters will provide
        # normalized coordinates in parsed_state.spatial_observations
        spatial_obs = getattr(parsed_state, "spatial_observations", [])
        
        for obs in spatial_obs:
            coords = obs.get("coordinates")
            if not coords:
                continue
            
            coords_tuple = tuple(coords)
            if coords_tuple not in self.grid:
                self.grid[coords_tuple] = MapNode(
                    node_id=f"node_{'_'.join(map(str, coords_tuple))}",
                    coordinates=coords_tuple,
                )
            
            node = self.grid[coords_tuple]
            node.properties.update(obs.get("properties", {}))
            if "entities" in obs:
                node.entities_present = obs["entities"]
            
            # If the observation is marked as the agent's location
            if obs.get("is_self", False):
                self.current_position = coords_tuple
                node.visited = True

    def get_node_at(self, coordinates: tuple[int, ...]) -> MapNode | None:
        """Retrieves the map node at specific coordinates."""
        return self.grid.get(coordinates)

    def get_adjacent_nodes(self, coordinates: tuple[int, ...]) -> list[MapNode]:
        """Returns adjacent nodes (assumes 2D grid for now, but can be generalized)."""
        if len(coordinates) != 2:
            return []
            
        x, y = coordinates
        adjacent = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node = self.grid.get((x + dx, y + dy))
            if node:
                adjacent.append(node)
        return adjacent

__all__ = ["MapNode", "TopologicalMap"]
