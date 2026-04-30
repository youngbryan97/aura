"""Data models for Aura's Autonomous Architecture Governor."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any


class SemanticSurface(StrEnum):
    AUTHORITY_GOVERNANCE = "authority/governance"
    CAPABILITY_TOOL_EXECUTION = "capability/tool_execution"
    MEMORY_WRITE_READ = "memory_write/read"
    STATE_MUTATION = "state_mutation"
    BOOT_RUNTIME_KERNEL = "boot/runtime/kernel"
    CONSCIOUSNESS_SUBSTRATE = "consciousness/substrate"
    LLM_MODEL_ROUTING = "llm/model_routing"
    IDENTITY_PERSONA = "identity/persona/heartstone"
    SELF_MODIFICATION = "self_modification"
    PROOF_TEST_EVALUATION = "proof/test/evaluation"
    UI_API = "ui/api"
    TRAINING_FINETUNE = "training/finetune"
    UTILITY_PERIPHERAL = "utility/peripheral"


class SmellSeverity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MutationTier(IntEnum):
    T0_SYNTAX_STYLE = 0
    T1_CLEANUP = 1
    T2_REFACTOR = 2
    T3_BEHAVIORAL_IMPROVEMENT = 3
    T4_GOVERNANCE_SENSITIVE = 4
    T5_SEALED = 5

    @classmethod
    def parse(cls, value: str | int | "MutationTier") -> "MutationTier":
        if isinstance(value, MutationTier):
            return value
        if isinstance(value, int):
            return MutationTier(value)
        normalized = value.strip().upper()
        if normalized in cls.__members__:
            return cls[normalized]
        if normalized.startswith("T") and normalized[1:].isdigit():
            return MutationTier(int(normalized[1:]))
        raise ValueError(f"unknown mutation tier: {value!r}")

    @property
    def autonomous_allowed(self) -> bool:
        return self <= MutationTier.T3_BEHAVIORAL_IMPROVEMENT

    @property
    def proposal_only(self) -> bool:
        return self >= MutationTier.T4_GOVERNANCE_SENSITIVE


class PromotionStatus(StrEnum):
    REJECTED = "rejected"
    PROPOSAL_ONLY = "proposal_only"
    SHADOW_PASSED = "shadow_passed"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    MONITORING = "monitoring"


@dataclass(frozen=True)
class ArchitectureNode:
    id: str
    kind: str
    name: str
    path: str
    line_start: int = 0
    line_end: int = 0
    qualified_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureEdge:
    source: str
    target: str
    kind: str
    path: str = ""
    line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeReceipt:
    source: str
    path: str
    timestamp: float
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OwnershipDomain:
    concern: str
    owner: str
    role: str = ""
    file: str = ""
    source: str = ""


@dataclass(frozen=True)
class ArchitecturalSmell:
    id: str
    kind: str
    severity: SmellSeverity
    path: str
    symbol: str = ""
    evidence: tuple[str, ...] = ()
    graph_refs: tuple[str, ...] = ()
    suggested_tier: MutationTier = MutationTier.T1_CLEANUP
    proof_obligations: tuple[str, ...] = ()
    auto_fixable: bool = False


@dataclass(frozen=True)
class MutationProposal:
    id: str
    objective: str
    tier: MutationTier
    affected_files: tuple[str, ...]
    affected_symbols: tuple[str, ...] = ()
    semantic_surfaces: tuple[SemanticSurface, ...] = ()
    expected_behavior_delta: str = "equivalent"
    smell_ids: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RefactorStep:
    id: str
    description: str
    operation: str
    target_path: str
    new_content: str | None = None
    invariants: tuple[str, ...] = ()
    rollback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefactorPlan:
    id: str
    objective: str
    risk_tier: MutationTier
    affected_files: tuple[str, ...]
    affected_symbols: tuple[str, ...]
    semantic_surfaces: tuple[SemanticSurface, ...]
    steps: tuple[RefactorStep, ...]
    proof_obligations: tuple[str, ...]
    expected_smell_reduction: tuple[str, ...]
    expected_behavior_delta: str
    promotion_eligible: bool
    proposal: MutationProposal | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def changed_files(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(step.target_path for step in self.steps))


@dataclass(frozen=True)
class ProofObligation:
    id: str
    description: str
    required_for: MutationTier


@dataclass(frozen=True)
class ProofResult:
    obligation_id: str
    passed: bool
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorFingerprint:
    id: str
    root: str
    graph_metrics: dict[str, Any]
    smell_counts: dict[str, int]
    import_cycle_count: int
    god_file_count: int
    broad_exception_count: int
    protected_bypass_count: int
    tests: dict[str, Any]
    compile_status: dict[str, Any]
    import_status: dict[str, Any]
    boot_status: dict[str, Any]
    changed_public_apis: tuple[str, ...]
    service_registrations: tuple[str, ...]
    authority_path_checks: dict[str, Any]
    memory_state_write_checks: dict[str, Any]
    latency_resource: dict[str, Any]
    optional_runtime_metrics: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BehaviorDelta:
    equivalent: bool
    improved: bool
    regressions: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RollbackPacket:
    run_id: str
    timestamp: float
    repo_root_hash: str
    changed_files: tuple[str, ...]
    original_hashes: dict[str, str]
    candidate_hashes: dict[str, str]
    packet_path: str
    receipt_hash: str = ""
    dry_run_passed: bool = False
    post_restore_verified: bool = False


@dataclass(frozen=True)
class ProofReceipt:
    run_id: str
    plan_id: str
    tier: MutationTier
    results: tuple[ProofResult, ...]
    behavior_delta: BehaviorDelta
    rollback_packet_hash: str
    shadow_artifact_path: str
    decision_hash: str = ""
    generated_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return all(
            result.passed
            or (
                self.tier <= MutationTier.T1_CLEANUP
                and result.status == "BOOT_HARNESS_UNAVAILABLE"
            )
            for result in self.results
        )

    def stable_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("generated_at", None)
        payload.pop("decision_hash", None)
        return payload

    def stable_hash(self) -> str:
        data = json.dumps(self.stable_payload(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def signed(self) -> "ProofReceipt":
        return ProofReceipt(
            run_id=self.run_id,
            plan_id=self.plan_id,
            tier=self.tier,
            results=self.results,
            behavior_delta=self.behavior_delta,
            rollback_packet_hash=self.rollback_packet_hash,
            shadow_artifact_path=self.shadow_artifact_path,
            decision_hash=self.stable_hash(),
            generated_at=self.generated_at,
        )


@dataclass(frozen=True)
class PromotionDecision:
    run_id: str
    plan_id: str
    status: PromotionStatus
    reason: str
    receipt_hash: str
    promoted_files: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PostPromotionObservation:
    run_id: str
    timestamp: float
    status: PromotionStatus
    metrics: dict[str, Any]
    regression_detected: bool = False
    rollback_triggered: bool = False
    reason: str = ""


@dataclass
class ArchitectureGraph:
    root: str
    nodes: dict[str, ArchitectureNode] = field(default_factory=dict)
    edges: list[ArchitectureEdge] = field(default_factory=list)
    semantic_surfaces: dict[str, tuple[SemanticSurface, ...]] = field(default_factory=dict)
    ownership: dict[str, OwnershipDomain] = field(default_factory=dict)
    runtime_receipts: list[RuntimeReceipt] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_node(self, node: ArchitectureNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: ArchitectureEdge) -> None:
        self.edges.append(edge)

    def nodes_for_path(self, path: str) -> list[ArchitectureNode]:
        return [node for node in self.nodes.values() if node.path == path]

    def effects_for_path(self, path: str) -> set[str]:
        effects: set[str] = set()
        for node in self.nodes_for_path(path):
            effects.update(node.metadata.get("effects", ()))
        return effects

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "nodes": {key: asdict(value) for key, value in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
            "semantic_surfaces": {
                key: [surface.value for surface in value]
                for key, value in self.semantic_surfaces.items()
            },
            "ownership": {key: asdict(value) for key, value in self.ownership.items()},
            "runtime_receipts": [asdict(receipt) for receipt in self.runtime_receipts],
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchitectureGraph":
        graph = cls(root=str(payload["root"]), created_at=float(payload.get("created_at", time.time())))
        graph.nodes = {
            key: ArchitectureNode(**value)
            for key, value in payload.get("nodes", {}).items()
        }
        graph.edges = [ArchitectureEdge(**edge) for edge in payload.get("edges", ())]
        graph.semantic_surfaces = {
            key: tuple(SemanticSurface(item) for item in value)
            for key, value in payload.get("semantic_surfaces", {}).items()
        }
        graph.ownership = {
            key: OwnershipDomain(**value)
            for key, value in payload.get("ownership", {}).items()
        }
        graph.runtime_receipts = [
            RuntimeReceipt(**receipt)
            for receipt in payload.get("runtime_receipts", ())
        ]
        graph.metrics = dict(payload.get("metrics", {}))
        return graph

    def persist_json(self, path: str | Path) -> None:
        from core.runtime.atomic_writer import atomic_write_text

        target = Path(path)
        atomic_write_text(target, json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str))

    @classmethod
    def load_json(cls, path: str | Path) -> "ArchitectureGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
