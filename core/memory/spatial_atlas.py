"""core/memory/spatial_atlas.py — Cartesian Spatial Memory.

Provides a topological memory layer for mapping physical environments (like NetHack levels).
Tracks explored tiles, items, and features with coordinate-based persistence.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger("Aura.Memory.SpatialAtlas")

@dataclass
class EvidenceItem:
    id: str
    name: str
    appearance: str
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MapNode:
    x: int
    y: int
    kind: str = "unknown"  # wall, floor, door, altar, etc.
    explored: bool = False
    walkable: bool = True
    items: List[EvidenceItem] = field(default_factory=list)
    last_seen: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)

class DungeonLevel:
    """Represents a single 2D level in a dungeon."""
    def __init__(self, dlvl: int, width: int = 80, height: int = 24):
        self.dlvl = dlvl
        self.width = width
        self.height = height
        self.grid: Dict[Tuple[int, int], MapNode] = {}
        for y in range(height):
            for x in range(width):
                self.grid[(x, y)] = MapNode(x, y)

    def update_node(self, x: int, y: int, kind: str, walkable: bool = True):
        if (x, y) in self.grid:
            node = self.grid[(x, y)]
            node.kind = kind
            node.walkable = walkable
            node.explored = True
            node.last_seen = time.time()

    def add_item(self, x: int, y: int, item: EvidenceItem):
        if (x, y) in self.grid:
            self.grid[(x, y)].items.append(item)

    def clear_items(self, x: int, y: int):
        if (x, y) in self.grid:
            self.grid[(x, y)].items = []

class SpatialAtlas:
    """The high-level manager for multi-level spatial memory."""
    def __init__(self):
        self.levels: Dict[int, DungeonLevel] = {}
        self.current_dlvl: int = 1

    def get_level(self, dlvl: int) -> DungeonLevel:
        if dlvl not in self.levels:
            self.levels[dlvl] = DungeonLevel(dlvl)
        return self.levels[dlvl]

    def update_current(self, dlvl: int, grid_data: List[List[Dict[str, Any]]]):
        """Update the current level with new sensory data."""
        self.current_dlvl = dlvl
        level = self.get_level(dlvl)
        for y, row in enumerate(grid_data):
            for x, cell in enumerate(row):
                level.update_node(x, y, cell.get("kind", "unknown"), cell.get("walkable", True))
                if "item" in cell:
                    item_data = cell["item"]
                    level.add_item(x, y, EvidenceItem(
                        id=item_data.get("id", "unknown"),
                        name=item_data.get("name", "unknown"),
                        appearance=item_data.get("appearance", "unknown")
                    ))

    def find_nearest(self, kind: str, dlvl: int, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """Find nearest known node of 'kind' across all levels (prioritizing current)."""
        # Search current level first
        level = self.get_level(dlvl)
        best_dist = float('inf')
        best_node = None
        
        for (nx, ny), node in level.grid.items():
            if node.kind == kind and node.explored:
                dist = abs(nx - x) + abs(ny - y)
                if dist < best_dist:
                    best_dist = dist
                    best_node = (dlvl, nx, ny)
        
        if best_node:
            return best_node
            
        # Search other levels
        for l_idx, l in self.levels.items():
            if l_idx == dlvl: continue
            for (nx, ny), node in l.grid.items():
                if node.kind == kind and node.explored:
                    return (l_idx, nx, ny) # Return first found for simplicity in other levels
                    
        return None

    def get_local_topology(self, dlvl: int, x: int, y: int, radius: int = 5) -> List[MapNode]:
        """Get nodes within a radius of the current position."""
        level = self.get_level(dlvl)
        nodes = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (nx, ny) in level.grid:
                    nodes.append(level.grid[(nx, ny)])
        return nodes

_INSTANCE: Optional[SpatialAtlas] = None

def get_spatial_atlas() -> SpatialAtlas:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SpatialAtlas()
    return _INSTANCE
