"""core/world/entity_graph.py
Maintains structured tracking of files, apps, and external entities in the workspace.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class EntityNode:
    id: str
    type: str  # "file", "app", "user", etc.
    attributes: Dict[str, Any] = field(default_factory=dict)


class EntityGraph:
    """Graph database modeling active system and environment objects."""

    def __init__(self):
        self._nodes: Dict[str, EntityNode] = {}
        self._edges: Dict[str, List[str]] = {}

    def upsert_entity(self, id: str, type: str, attributes: Dict[str, Any]) -> None:
        self._nodes[id] = EntityNode(id=id, type=type, attributes=attributes)
        if id not in self._edges:
            self._edges[id] = []

    def add_relation(self, from_id: str, to_id: str) -> None:
        if from_id in self._edges and to_id in self._nodes:
            if to_id not in self._edges[from_id]:
                self._edges[from_id].append(to_id)

    def get_entity(self, id: str) -> Optional[EntityNode]:
        return self._nodes.get(id)

    def list_entities_by_type(self, type: str) -> List[Dict[str, Any]]:
        return [
            {"id": n.id, "attributes": n.attributes}
            for n in self._nodes.values() if n.type == type
        ]
