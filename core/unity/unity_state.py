from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
import uuid
from typing import Any, Dict, Literal, Optional

UnityLevel = Literal["coherent", "strained", "fragmented", "dissociated", "unknown"]
Ownership = Literal["self", "world", "other", "ambiguous"]
CommitMode = Literal["clean", "qualified", "conflicted", "defer", "repair_only"]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class TemporalWindow:
    now_id: str = field(default_factory=lambda: _new_id("now"))
    tick_id: str | None = None
    opened_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    subjective_center_t: float = 0.0
    duration_s: float = 0.0
    continuity_from_previous: float = 0.0
    drift_from_previous: float = 0.0
    phase_lag: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TemporalWindow":
        return cls(**dict(data or {}))


@dataclass(frozen=True)
class BoundContent:
    content_id: str
    modality: str
    source: str
    summary: str
    salience: float
    confidence: float
    timestamp: float
    ownership: Ownership
    action_relevance: float
    affective_charge: float
    evidence_ref: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundContent":
        return cls(**dict(data))


@dataclass(frozen=True)
class DraftBinding:
    draft_id: str
    claim: str
    support: float
    conflict: float
    chosen: bool
    suppressed_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DraftBinding":
        return cls(**dict(data))


@dataclass(frozen=True)
class ReconciledDraftSet:
    set_id: str = field(default_factory=lambda: _new_id("draftset"))
    chosen: DraftBinding = field(
        default_factory=lambda: DraftBinding(
            draft_id=_new_id("draft"),
            claim="current interpretation",
            support=1.0,
            conflict=0.0,
            chosen=True,
        )
    )
    alternatives: list[DraftBinding] = field(default_factory=list)
    consensus_score: float = 1.0
    contradiction_score: float = 0.0
    unresolved_residue: list[str] = field(default_factory=list)
    memory_commit_mode: CommitMode = "clean"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "set_id": self.set_id,
            "chosen": self.chosen.to_dict(),
            "alternatives": [item.to_dict() for item in self.alternatives],
            "consensus_score": self.consensus_score,
            "contradiction_score": self.contradiction_score,
            "unresolved_residue": list(self.unresolved_residue),
            "memory_commit_mode": self.memory_commit_mode,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReconciledDraftSet":
        payload = dict(data or {})
        chosen = DraftBinding.from_dict(payload.get("chosen") or {})
        alternatives = [
            DraftBinding.from_dict(item)
            for item in list(payload.get("alternatives") or [])
            if isinstance(item, dict)
        ]
        return cls(
            set_id=str(payload.get("set_id") or _new_id("draftset")),
            chosen=chosen,
            alternatives=alternatives,
            consensus_score=float(payload.get("consensus_score", 1.0) or 1.0),
            contradiction_score=float(payload.get("contradiction_score", 0.0) or 0.0),
            unresolved_residue=[str(item) for item in list(payload.get("unresolved_residue") or [])],
            memory_commit_mode=str(payload.get("memory_commit_mode", "clean") or "clean"),
        )


@dataclass(frozen=True)
class SelfWorldBinding:
    binding_id: str = field(default_factory=lambda: _new_id("selfworld"))
    self_state_refs: list[str] = field(default_factory=list)
    world_state_refs: list[str] = field(default_factory=list)
    authored_action_refs: list[str] = field(default_factory=list)
    external_event_refs: list[str] = field(default_factory=list)
    ownership_confidence: float = 1.0
    agency_score: float = 1.0
    responsibility_score: float = 1.0
    boundary_integrity: float = 1.0
    contamination_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SelfWorldBinding":
        return cls(**dict(data or {}))


@dataclass(frozen=True)
class FragmentationReport:
    report_id: str = field(default_factory=lambda: _new_id("frag"))
    unity_id: str = ""
    fragmentation_score: float = 0.0
    level: UnityLevel = "unknown"
    top_causes: list[tuple[str, float, str]] = field(default_factory=list)
    repair_recommendations: list[str] = field(default_factory=list)
    user_visible_summary: str = ""
    safe_to_act: bool = True
    safe_to_self_report: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FragmentationReport":
        payload = dict(data or {})
        payload["top_causes"] = [
            (str(name), float(weight), str(text))
            for name, weight, text in list(payload.get("top_causes") or [])
        ]
        payload["repair_recommendations"] = [
            str(item) for item in list(payload.get("repair_recommendations") or [])
        ]
        return cls(**payload)


@dataclass(frozen=True)
class UnityRepairPlan:
    plan_id: str = field(default_factory=lambda: _new_id("repair"))
    unity_id: str = ""
    causes: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    requires_will: bool = True
    expected_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UnityRepairPlan":
        return cls(**dict(data or {}))


@dataclass(frozen=True)
class WorkspaceBroadcastFrame:
    frame_id: str = field(default_factory=lambda: _new_id("workspace"))
    focus: BoundContent | None = None
    periphery: list[BoundContent] = field(default_factory=list)
    suppressed: list[DraftBinding] = field(default_factory=list)
    co_presence_cluster_id: str = ""
    unity_score: float = 0.0
    fragmentation_score: float = 0.0
    reentry_targets: list[str] = field(default_factory=list)
    will_receipt_id: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "focus": self.focus.to_dict() if self.focus else None,
            "periphery": [item.to_dict() for item in self.periphery],
            "suppressed": [item.to_dict() for item in self.suppressed],
            "co_presence_cluster_id": self.co_presence_cluster_id,
            "unity_score": self.unity_score,
            "fragmentation_score": self.fragmentation_score,
            "reentry_targets": list(self.reentry_targets),
            "will_receipt_id": self.will_receipt_id,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WorkspaceBroadcastFrame":
        payload = dict(data or {})
        focus = payload.get("focus")
        periphery = [
            BoundContent.from_dict(item)
            for item in list(payload.get("periphery") or [])
            if isinstance(item, dict)
        ]
        suppressed = [
            DraftBinding.from_dict(item)
            for item in list(payload.get("suppressed") or [])
            if isinstance(item, dict)
        ]
        return cls(
            frame_id=str(payload.get("frame_id") or _new_id("workspace")),
            focus=BoundContent.from_dict(focus) if isinstance(focus, dict) else None,
            periphery=periphery,
            suppressed=suppressed,
            co_presence_cluster_id=str(payload.get("co_presence_cluster_id") or ""),
            unity_score=float(payload.get("unity_score", 0.0) or 0.0),
            fragmentation_score=float(payload.get("fragmentation_score", 0.0) or 0.0),
            reentry_targets=[str(item) for item in list(payload.get("reentry_targets") or [])],
            will_receipt_id=payload.get("will_receipt_id"),
        )


@dataclass(frozen=True)
class UnityState:
    unity_id: str = field(default_factory=lambda: _new_id("unity"))
    created_at: float = field(default_factory=time.time)
    temporal: TemporalWindow = field(default_factory=TemporalWindow)
    contents: list[BoundContent] = field(default_factory=list)
    draft_bindings: list[DraftBinding] = field(default_factory=list)
    global_focus_id: str | None = None
    peripheral_content_ids: list[str] = field(default_factory=list)
    self_world_boundary_score: float = 1.0
    temporal_continuity_score: float = 1.0
    cross_modal_coherence_score: float = 1.0
    draft_consensus_score: float = 1.0
    affect_alignment_score: float = 1.0
    agency_ownership_score: float = 1.0
    memory_continuity_score: float = 1.0
    action_readiness_score: float = 1.0
    fragmentation_score: float = 0.0
    unity_score: float = 1.0
    level: UnityLevel = "unknown"
    repair_needed: bool = False
    repair_reasons: list[str] = field(default_factory=list)
    will_receipt_id: str | None = None
    state_version: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unity_id": self.unity_id,
            "created_at": self.created_at,
            "temporal": self.temporal.to_dict(),
            "contents": [item.to_dict() for item in self.contents],
            "draft_bindings": [item.to_dict() for item in self.draft_bindings],
            "global_focus_id": self.global_focus_id,
            "peripheral_content_ids": list(self.peripheral_content_ids),
            "self_world_boundary_score": self.self_world_boundary_score,
            "temporal_continuity_score": self.temporal_continuity_score,
            "cross_modal_coherence_score": self.cross_modal_coherence_score,
            "draft_consensus_score": self.draft_consensus_score,
            "affect_alignment_score": self.affect_alignment_score,
            "agency_ownership_score": self.agency_ownership_score,
            "memory_continuity_score": self.memory_continuity_score,
            "action_readiness_score": self.action_readiness_score,
            "fragmentation_score": self.fragmentation_score,
            "unity_score": self.unity_score,
            "level": self.level,
            "repair_needed": self.repair_needed,
            "repair_reasons": list(self.repair_reasons),
            "will_receipt_id": self.will_receipt_id,
            "state_version": self.state_version,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UnityState":
        payload = dict(data or {})
        temporal = TemporalWindow.from_dict(payload.get("temporal") or {})
        contents = [
            BoundContent.from_dict(item)
            for item in list(payload.get("contents") or [])
            if isinstance(item, dict)
        ]
        drafts = [
            DraftBinding.from_dict(item)
            for item in list(payload.get("draft_bindings") or [])
            if isinstance(item, dict)
        ]
        return cls(
            unity_id=str(payload.get("unity_id") or _new_id("unity")),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            temporal=temporal,
            contents=contents,
            draft_bindings=drafts,
            global_focus_id=payload.get("global_focus_id"),
            peripheral_content_ids=[str(item) for item in list(payload.get("peripheral_content_ids") or [])],
            self_world_boundary_score=float(payload.get("self_world_boundary_score", 1.0) or 1.0),
            temporal_continuity_score=float(payload.get("temporal_continuity_score", 1.0) or 1.0),
            cross_modal_coherence_score=float(payload.get("cross_modal_coherence_score", 1.0) or 1.0),
            draft_consensus_score=float(payload.get("draft_consensus_score", 1.0) or 1.0),
            affect_alignment_score=float(payload.get("affect_alignment_score", 1.0) or 1.0),
            agency_ownership_score=float(payload.get("agency_ownership_score", 1.0) or 1.0),
            memory_continuity_score=float(payload.get("memory_continuity_score", 1.0) or 1.0),
            action_readiness_score=float(payload.get("action_readiness_score", 1.0) or 1.0),
            fragmentation_score=float(payload.get("fragmentation_score", 0.0) or 0.0),
            unity_score=float(payload.get("unity_score", 1.0) or 1.0),
            level=str(payload.get("level", "unknown") or "unknown"),
            repair_needed=bool(payload.get("repair_needed", False)),
            repair_reasons=[str(item) for item in list(payload.get("repair_reasons") or [])],
            will_receipt_id=payload.get("will_receipt_id"),
            state_version=payload.get("state_version"),
            metadata=dict(payload.get("metadata") or {}),
        )
