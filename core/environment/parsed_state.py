"""Typed state compiled from raw observations."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .modal import ModalState
from .observation import _json_safe
from .ontology import Affordance, EntityState, HazardState, ObjectState, ResourceState, SemanticEvent


@dataclass
class ParsedState:
    environment_id: str
    context_id: str | None = None
    self_state: dict[str, Any] = field(default_factory=dict)
    entities: list[EntityState] = field(default_factory=list)
    objects: list[ObjectState] = field(default_factory=list)
    resources: dict[str, ResourceState] = field(default_factory=dict)
    hazards: list[HazardState] = field(default_factory=list)
    affordances: list[Affordance] = field(default_factory=list)
    modal_state: ModalState | None = None
    semantic_events: list[SemanticEvent] = field(default_factory=list)
    uncertainty: dict[str, float] = field(default_factory=dict)
    raw_observation_ref: str = ""
    observed_ids: set[str] = field(default_factory=set)
    inferred_ids: set[str] = field(default_factory=set)
    sequence_id: int = 0

    def stable_hash(self) -> str:
        payload = self.to_json_safe()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_json_safe(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_ids"] = sorted(self.observed_ids)
        data["inferred_ids"] = sorted(self.inferred_ids)
        return _json_safe(data)


__all__ = ["ParsedState"]
