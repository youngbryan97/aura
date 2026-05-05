"""Generic ontology for Aura's environment operating system.

The ontology is deliberately small and typed. NetHack, browser work,
codebase repair, email drafting, desktop automation, and simulations all
map their local evidence into these primitives before planning or action.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EntityKind = Literal[
    "self",
    "agent",
    "hostile",
    "neutral",
    "ally",
    "process",
    "user",
    "service",
    "file",
    "unknown",
]

ObjectKind = Literal[
    "item",
    "tool",
    "document",
    "file",
    "button",
    "link",
    "form",
    "menu_entry",
    "resource",
    "container",
    "transition",
    "unknown",
]

HazardKind = Literal[
    "damage",
    "deletion",
    "resource_depletion",
    "privacy_leak",
    "irreversible_submit",
    "hostile_entity",
    "corruption",
    "policy_violation",
    "unknown",
]


@dataclass
class OntologyRecord:
    """Base record shape shared by ontology objects."""

    kind: str = "unknown"
    label: str = ""
    context_id: str = "default"
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    last_seen_seq: int = 0
    evidence_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityState(OntologyRecord):
    entity_id: str = ""
    kind: EntityKind = "unknown"
    position: tuple[int, int] | None = None
    threat_score: float = 0.0
    helpfulness_score: float = 0.0

    @property
    def id(self) -> str:
        return self.entity_id


@dataclass
class ObjectState(OntologyRecord):
    object_id: str = ""
    kind: ObjectKind = "unknown"
    position: tuple[int, int] | None = None
    affordances: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.object_id


@dataclass
class ResourceState:
    name: str
    value: float
    max_value: float | None = None
    normalized: float | None = None
    critical_below: float | None = None
    critical_above: float | None = None
    trend: float = 0.0
    confidence: float = 1.0
    evidence_ref: str = ""
    last_seen_seq: int = 0
    kind: str = "unknown"
    label: str = ""
    context_id: str = "default"
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name

    def __post_init__(self) -> None:
        if self.normalized is None and self.max_value not in (None, 0):
            self.normalized = max(0.0, min(1.0, float(self.value) / float(self.max_value)))
        if not self.label:
            self.label = self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HazardState(OntologyRecord):
    hazard_id: str = ""
    kind: HazardKind = "unknown"
    severity: float = 0.0
    source_id: str | None = None
    mitigations: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.hazard_id


@dataclass
class Affordance:
    affordance_id: str
    name: str
    object_id: str | None = None
    context_id: str = "default"
    preconditions: list[str] = field(default_factory=list)
    expected_effect: str = ""
    risk_score: float = 0.0
    confidence: float = 1.0
    evidence_ref: str = ""
    last_seen_seq: int = 0
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.affordance_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticEvent:
    event_id: str
    kind: str
    label: str
    context_id: str
    evidence_ref: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    properties: dict[str, Any] = field(default_factory=dict)
    related_ids: list[str] = field(default_factory=list)
    last_seen_seq: int = 0

    @property
    def id(self) -> str:
        return self.event_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "EntityKind",
    "ObjectKind",
    "HazardKind",
    "EntityState",
    "ObjectState",
    "ResourceState",
    "HazardState",
    "Affordance",
    "SemanticEvent",
]
