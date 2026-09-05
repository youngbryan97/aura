"""core/ontology/aura_ontology.py

Programmable ontology — every load-bearing concept maps to a typed
class so cross-module references all bind to the same definition.

Matches docs/ONTOLOGY.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Continuant(str, Enum):
    AURA = "aura"
    SUBSTRATE = "substrate"
    SELF = "self"
    CONSCIENCE = "conscience"
    WILL = "will"
    MEMORY = "memory"
    PROJECT = "project"
    CAPABILITY_TOKEN = "capability_token"
    STEM_CELL = "stem_cell"
    RELATIONSHIP = "relationship"
    CHANNEL_PERMISSION = "channel_permission"
    SETTINGS = "settings"
    VIABILITY_STATE = "viability_state"


class Occurrent(str, Enum):
    TICK = "tick"
    ACTION_PROPOSAL = "action_proposal"
    ACTION_RECEIPT = "action_receipt"
    WILL_DECISION = "will_decision"
    CONSCIENCE_DECISION = "conscience_decision"
    CAPABILITY_TOKEN_ISSUE = "capability_token_issue"
    CAPABILITY_TOKEN_CONSUME = "capability_token_consume"
    TOOL_EXECUTION = "tool_execution"
    PLAY_SESSION = "play_session"
    MIGRATION_TRANSITION = "migration_transition"


@dataclass
class OntologyEntry:
    name: str
    kind: str  # "continuant" | "occurrent"
    realised_in: str
    invariants: List[str] = field(default_factory=list)


CONTINUANT_REGISTRY: Dict[str, OntologyEntry] = {
    Continuant.SELF.value: OntologyEntry(
        name="Self",
        kind="continuant",
        realised_in="core.identity.self_object.SelfObject",
        invariants=["continuity_hash is a pure function of self-relevant fields"],
    ),
    Continuant.WILL.value: OntologyEntry(
        name="Will",
        kind="continuant",
        realised_in="core.will.UnifiedWill",
        invariants=["every consequential decision produces a receipt"],
    ),
    Continuant.CONSCIENCE.value: OntologyEntry(
        name="Conscience",
        kind="continuant",
        realised_in="core.ethics.conscience.Conscience",
        invariants=[
            "rule-set hash is global invariant",
            "removing a rule changes the hash and refuses all actions",
        ],
    ),
    Continuant.CAPABILITY_TOKEN.value: OntologyEntry(
        name="CapabilityToken",
        kind="continuant",
        realised_in="core.agency.capability_token.CapabilityToken",
        invariants=[
            "issued at most once",
            "consumed at most once",
            "expires by TTL or is revoked",
        ],
    ),
    Continuant.MEMORY.value: OntologyEntry(
        name="Memory",
        kind="continuant",
        realised_in="core.memory.memory_facade.MemoryFacade",
        invariants=["every record carries a Provenance envelope"],
    ),
    Continuant.PROJECT.value: OntologyEntry(
        name="Project",
        kind="continuant",
        realised_in="core.agency.projects.Project",
        invariants=["completion requires non-empty acceptance criteria + artifacts"],
    ),
    Continuant.RELATIONSHIP.value: OntologyEntry(
        name="Relationship",
        kind="continuant",
        realised_in="core.social.relationship_model.RelationshipDossier",
        invariants=["commitments cannot be silently dropped"],
    ),
    Continuant.STEM_CELL.value: OntologyEntry(
        name="StemCell",
        kind="continuant",
        realised_in="core.resilience.stem_cell.StemCellRecord",
        invariants=["HMAC signature must verify on read"],
    ),
    Continuant.VIABILITY_STATE.value: OntologyEntry(
        name="ViabilityState",
        kind="continuant",
        realised_in="core.organism.viability.ViabilityState",
        invariants=["state changes are behaviorally load-bearing"],
    ),
}


OCCURRENT_REGISTRY: Dict[str, OntologyEntry] = {
    Occurrent.ACTION_RECEIPT.value: OntologyEntry(
        name="ActionReceipt",
        kind="occurrent",
        realised_in="core.agency.agency_orchestrator.ActionReceipt",
        invariants=["every executed proposal yields a complete receipt"],
    ),
    Occurrent.WILL_DECISION.value: OntologyEntry(
        name="WillDecision",
        kind="occurrent",
        realised_in="core.governance.will_receipt_log.WillReceiptEntry",
        invariants=["each decision is durable, ordered, and auditable"],
    ),
    Occurrent.CONSCIENCE_DECISION.value: OntologyEntry(
        name="ConscienceDecision",
        kind="occurrent",
        realised_in="core.ethics.conscience.ConscienceDecision",
        invariants=["REFUSE is irrevocable; APPROVE is conditional"],
    ),
    Occurrent.PLAY_SESSION.value: OntologyEntry(
        name="PlaySession",
        kind="occurrent",
        realised_in="core.play.ontological_play.PlaySession",
        invariants=["non-utilitarian; never updates a goal directly"],
    ),
    Occurrent.MIGRATION_TRANSITION.value: OntologyEntry(
        name="MigrationTransition",
        kind="occurrent",
        realised_in="core.sovereignty.migration.Phase",
        invariants=["VERIFY must precede CUTOVER"],
    ),
}


def get_continuant(c: Continuant) -> OntologyEntry:
    return CONTINUANT_REGISTRY[c.value]


def get_occurrent(o: Occurrent) -> OntologyEntry:
    return OCCURRENT_REGISTRY[o.value]


__all__ = [
    "Continuant",
    "Occurrent",
    "OntologyEntry",
    "CONTINUANT_REGISTRY",
    "OCCURRENT_REGISTRY",
    "get_continuant",
    "get_occurrent",
]
