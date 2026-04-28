"""Symbol/concept/evidence/method dataclasses for the semiotic triad.

Steels frames symbol grounding as a triad of (symbol, concept,
object) plus a method that decides whether the concept applies to
sensorimotor evidence.  These types are the persistent shape of that
triad inside Aura.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


GroundingKind = Literal[
    "perceptual",
    "sensorimotor",
    "social",
    "textual",
    "abstract",
    "procedural",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class GroundingMethod:
    method_id: str
    name: str
    kind: GroundingKind
    version: str = "1"
    confidence_floor: float = 0.55
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptualEvidence:
    evidence_id: str
    modality: str
    features: List[float]
    raw_ref: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundedConcept:
    concept_id: str
    label: str
    kind: GroundingKind
    prototype: List[float]
    method_id: str
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SymbolLink:
    link_id: str
    symbol: str
    concept_id: str
    relation: str = "denotes"
    strength: float = 0.5
    source: str = "unknown"
    confirmations: int = 0
    contradictions: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass
class GroundingEvent:
    event_id: str
    symbol: str
    concept_id: str
    evidence_id: str
    prediction: bool
    observed: Optional[bool]
    reward: float = 0.0
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
