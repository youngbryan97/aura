"""Adaptive immunity for Aura.

This module adds a true population-based immune ecology on top of Aura's
existing innate defenses. The current immune stack already detects anomalies,
recognizes known signatures, and performs bounded repair. What it lacks is an
adaptive layer that can:

1. Learn reusable receptors over a shared antigen space.
2. Proliferate successful lineages while pruning weak or harmful ones.
3. Preserve immune memory across sessions and dream consolidation cycles.
4. Suppress autoimmune actions against protected identity / sovereignty tissue.
5. Emit bounded repair artifacts instead of free-form self-modification.

The design here deliberately keeps the adaptive layer *advisory and bounded*.
It can execute only a narrow subset of repair actions through the existing
autopoiesis engine. Everything sensitive remains governance-gated.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from core.adaptation.spatial_receptor_code import annotate_antigen_like
from core.cognitive.anomaly_detector import FeatureExtractor
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.lockdep import LockRank, checked_lock

logger = logging.getLogger("Aura.AdaptiveImmunity")

__all__ = [
    "AdaptiveImmuneSystem",
    "AdaptiveImmuneConfig",
    "Antigen",
    "CellKind",
    "EffectorArtifact",
    "EffectorKind",
    "ImmuneCell",
    "ImmuneResponse",
    "OfflineCoevolutionLab",
    "annotate_antigen_like",
    "TissueField",
    "get_adaptive_immune_system",
]

_ANTIGEN_DIM = 16
_EPSILON = 1e-8


def _record_adaptive_immunity_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "adaptive_immunity",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        # This subsystem mutates live state and can trigger repairs — its
        # failures need forensic receipts, not waived ones.
        receipt_required=True,
        extra=extra,
    )


def _maintenance_background_deferral_reason() -> str:
    """Return the runtime background-policy reason for deferring heavy immune work."""
    try:
        from core.container import ServiceContainer
        from core.runtime.background_policy import (
            MAINTENANCE_BACKGROUND_POLICY,
            background_activity_reason,
        )

        orchestrator = ServiceContainer.get("orchestrator", default=None)
        return str(
            background_activity_reason(
                orchestrator,
                profile=MAINTENANCE_BACKGROUND_POLICY,
                allow_no_user_anchor=True,
            )
            or ""
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Adaptive immune background policy unavailable: %s", exc)
        # Fail CLOSED: an unreadable maintenance policy means runtime pressure
        # is UNKNOWN, and heavy immune work (dream consolidation, coevolution)
        # must defer rather than assume the runtime is idle.
        _record_adaptive_immunity_degradation(
            exc,
            action="deferred heavy immune work after background policy probe failed",
        )
        return "background_policy_unavailable"


#: One event is a sighting; history needs more than one (CP126 7c08abf3).
_MIN_TEMPORAL_HISTORY_EVENTS = 2
#: Bumped when the persisted immune-state layout changes incompatibly. State
#: from another version is quarantined to a reseed rather than parsed
#: field-by-field into a live repair-capable population (CP126 5c214831).
IMMUNE_STATE_SCHEMA_VERSION = 1
#: A state file larger than this is not immune state; refuse to parse it.
MAX_IMMUNE_STATE_BYTES = 32 * 1024 * 1024
#: A snapshot older than this cannot describe the event it is attached to.
_MAX_SNAPSHOT_AGE_S = 300.0


def _anomaly_score_is_substantive(anomaly_score: Any) -> bool:
    """Whether an anomaly object actually carries a reading.

    `is not None` credited any object at all, including one whose scoring
    failed and returned an empty shell (CP126 064f40ec).
    """
    if anomaly_score is None:
        return False
    for attribute in ("threat_probability", "score", "anomaly_score"):
        value = getattr(anomaly_score, attribute, None)
        if value is None and isinstance(anomaly_score, dict):
            value = anomaly_score.get(attribute)
        if _optional_unit(value) is not None:
            return True
    return False


def _snapshot_is_usable(snapshot: Any) -> bool:
    """Whether a state snapshot has content and is recent enough to count."""
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    stamp = snapshot.get("timestamp") or snapshot.get("at") or snapshot.get("captured_at")
    if stamp is None:
        # No declared age. Content is all we can check, and we do not invent
        # freshness we cannot see.
        return True
    try:
        age = time.time() - float(stamp)
    except (TypeError, ValueError):
        return True
    return age <= _MAX_SNAPSHOT_AGE_S


def _immune_state_digest(payload: dict[str, Any]) -> str:
    """Digest over the state body, excluding the integrity block itself."""
    body = {key: value for key, value in payload.items() if key != "integrity"}
    try:
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = repr(sorted(body))
    return hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()


def _artifact_payload_digest(artifact: Any) -> str:
    """Digest of the concrete effect an artifact would produce.

    Binds a Will approval to THIS payload rather than to the category of
    action it belongs to (CP126 81f0c6a0).
    """
    try:
        body = json.dumps(
            {
                "kind": getattr(getattr(artifact, "kind", None), "value", ""),
                "component": str(getattr(artifact, "component", "")),
                "payload": _json_safe(getattr(artifact, "bounded_payload", {})),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        body = repr(getattr(artifact, "bounded_payload", ""))
    return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


class CellKind(StrEnum):
    DENDRITIC = "dendritic"
    B = "b_cell"
    CYTOTOXIC = "cytotoxic_t"
    REGULATORY = "regulatory_t"
    MEMORY = "memory"


class EffectorKind(StrEnum):
    CLEAR_CACHE = "clear_cache"
    REDUCE_LOAD = "reduce_load"
    RESTART_COMPONENT = "restart_component"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    QUARANTINE = "quarantine"
    HALT_RUNAWAY = "halt_runaway"
    REVOKE_TOOL = "revoke_tool"
    SCHEMA_MIGRATION = "schema_migration"
    PATCH_PROPOSAL = "patch_proposal"


@dataclass(frozen=True)
class AdaptiveImmuneConfig:
    population_size: int = 24
    max_population: int = 56
    receptor_dim: int = _ANTIGEN_DIM
    initial_receptor_dim: int = 16
    max_receptor_dim: int = 128
    expansion_check_interval: int = 64
    expansion_eigenvalue_threshold: float = 0.15
    contraction_score_floor: float = 0.05
    contraction_min_observations: int = 500
    tau: float = 0.22
    activation_threshold: float = 0.18
    clone_activation_threshold: float = 0.42
    mutation_sigma: float = 0.06
    basal_decay: float = 0.015
    memory_decay: float = 0.003
    persistence_boost: float = 0.18
    lineage_memory_successes: int = 2
    lineage_memory_fitness: float = 0.55
    dream_every_observations: int = 12
    replay_buffer_size: int = 128
    recent_response_buffer: int = 64
    max_artifacts_per_antigen: int = 3
    max_execution_attempts_per_event: int = 2
    execution_confidence_floor: float = 0.45
    low_coverage_floor: float = 0.42
    # CP126 ea9e677e: these were 2 checks 10ms apart, so "verified repair"
    # meant two readings taken within a hundredth of a second — and the
    # stability-window guard below requires >= 3 samples, so at these defaults
    # it could never fire. A guard that cannot run is the shape this whole
    # campaign is about. The window is now long enough to observe whether a
    # recovery HELD rather than whether it happened.
    verification_checks: int = 4
    verification_interval_s: float = 0.25
    min_verified_health_delta: float = 0.02
    #: Below these, an observation is too short to judge and verification
    #: fails closed as unverified rather than passing on a transient.
    min_verification_samples: int = 3
    min_verification_window_s: float = 0.4
    recurrence_window_s: float = 900.0
    species_min_k: int = 2
    species_max_k: int = 4
    species_silhouette_floor: float = 0.22
    tissue_diffusion: float = 0.16
    tissue_decay: float = 0.06

#: Domains an antigen may legitimately claim. CP126 e2f39609: an arbitrary
#: string was accepted, and source_domain gates whether substrate repair is
#: allowed to act on a failure.
_ANTIGEN_SOURCE_DOMAINS = frozenset({"substrate", "environment"})
_MAX_ANTIGEN_TEXT = 4096
_MAX_ANTIGEN_CONTEXT_KEYS = 64


def _on_event_loop() -> bool:
    """Whether the caller is running on an asyncio event loop thread."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _optional_unit(value: Any) -> float | None:
    """A finite 0..1 reading, or None when the field is absent/unusable.

    Distinct from :func:`_unit_scalar` because "not supplied" and "supplied
    as garbage" must both fall through to the computed default rather than
    silently becoming 0.0, which would read as no pressure at all.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


def _first_unit(*values: Any, default: float) -> float:
    """The first usable 0..1 reading among ``values``, else ``default``."""
    for value in values:
        unit = _optional_unit(value)
        if unit is not None:
            return unit
    return _unit_scalar(default)


def _finite_timestamp(value: Any) -> float:
    """A finite timestamp, or now. A NaN here corrupts every age computation."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return time.time()
    return number if math.isfinite(number) and number > 0 else time.time()


def _unit_scalar(value: Any, *, default: float = 0.0) -> float:
    """A finite 0..1 pressure, or the default.

    CP126 e2f39609: persisted antigens were rebuilt with bare ``float(...)``,
    so a NaN or out-of-range score entered live immune state. NaN then
    propagates silently through every comparison that decides whether to act
    — ``nan > threshold`` is False, so a poisoned antigen reads as calm rather
    than as unreadable.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))



@dataclass
class Antigen:
    antigen_id: str
    subsystem: str
    vector: np.ndarray
    danger: float
    subsystem_need: float
    threat_probability: float
    resource_pressure: float
    error_load: float
    health_pressure: float
    temporal_pressure: float
    recurrence_pressure: float
    protected: bool = False
    source_domain: str = "substrate"  # "substrate" or "environment"
    source: str = "unknown"
    error_signature: str = ""
    stack_trace: str = ""
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "antigen_id": self.antigen_id,
            "subsystem": self.subsystem,
            "source_domain": self.source_domain,
            "danger": round(float(self.danger), 4),
            "subsystem_need": round(float(self.subsystem_need), 4),
            "threat_probability": round(float(self.threat_probability), 4),
            "resource_pressure": round(float(self.resource_pressure), 4),
            "error_load": round(float(self.error_load), 4),
            "health_pressure": round(float(self.health_pressure), 4),
            "temporal_pressure": round(float(self.temporal_pressure), 4),
            "recurrence_pressure": round(float(self.recurrence_pressure), 4),
            "protected": bool(self.protected),
            "source": self.source,
            "error_signature": self.error_signature,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp,
            "vector": self.vector.astype(float).tolist(),
            "context": _json_safe(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Antigen:
        vector = np.asarray(data.get("vector", []), dtype=np.float32)
        target_dim = len(vector)
        global _adaptive_immune_singleton
        if _adaptive_immune_singleton is not None:
            target_dim = _adaptive_immune_singleton.expansion_engine.current_dim

        if len(vector) != target_dim:
            resized = np.zeros(target_dim, dtype=np.float32)
            copy_len = min(len(vector), target_dim)
            resized[:copy_len] = vector[:copy_len]
            vector = resized

        # CP126 e2f39609: a NaN in the vector survives np.clip, and NaN reads
        # as calm in every comparison that decides whether to act.
        vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)

        raw_domain = str(data.get("source_domain", "substrate"))
        # Pre-migration runtime-degradation records used ``runtime`` for
        # internal Aura failures. It is a deterministic predecessor of
        # ``substrate``, not an unknown external origin.
        if raw_domain == "runtime":
            raw_domain = "substrate"
        if raw_domain not in _ANTIGEN_SOURCE_DOMAINS:
            logger.warning(
                "Persisted antigen declared unknown source_domain %r; "
                "treating it as environment (the domain substrate repair may "
                "NOT act on)",
                raw_domain[:64],
            )
            # Fail toward the more restrictive domain: an antigen whose origin
            # cannot be trusted must not unlock substrate repair.
            raw_domain = "environment"

        raw_context = data.get("context", {})
        context = dict(raw_context) if isinstance(raw_context, dict) else {}
        if len(context) > _MAX_ANTIGEN_CONTEXT_KEYS:
            context = dict(list(context.items())[:_MAX_ANTIGEN_CONTEXT_KEYS])

        return cls(
            antigen_id=str(data.get("antigen_id", ""))[:_MAX_ANTIGEN_TEXT],
            subsystem=str(data.get("subsystem", "unknown"))[:_MAX_ANTIGEN_TEXT],
            vector=np.clip(vector, 0.0, 1.0),
            danger=_unit_scalar(data.get("danger")),
            subsystem_need=_unit_scalar(data.get("subsystem_need")),
            threat_probability=_unit_scalar(data.get("threat_probability")),
            resource_pressure=_unit_scalar(data.get("resource_pressure")),
            error_load=_unit_scalar(data.get("error_load")),
            health_pressure=_unit_scalar(data.get("health_pressure")),
            temporal_pressure=_unit_scalar(data.get("temporal_pressure")),
            recurrence_pressure=_unit_scalar(data.get("recurrence_pressure")),
            protected=bool(data.get("protected", False)),
            # Restore the origin domain — dropping it reclassified every
            # persisted environmental antigen as substrate, letting
            # environment-caused failures qualify for substrate repair.
            source_domain=raw_domain,
            source=str(data.get("source", "unknown"))[:_MAX_ANTIGEN_TEXT],
            error_signature=str(data.get("error_signature", ""))[:_MAX_ANTIGEN_TEXT],
            stack_trace=str(data.get("stack_trace", ""))[:_MAX_ANTIGEN_TEXT],
            timestamp=_finite_timestamp(data.get("timestamp")),
            context=context,
        )


@dataclass
class EffectorArtifact:
    artifact_id: str
    kind: EffectorKind
    component: str
    confidence: float
    source_cell_id: str
    lineage_id: str
    bounded_payload: dict[str, Any]
    governance_required: bool = True
    suppressed: bool = False
    governance_denied: bool = False
    executed: bool = False
    success: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "component": self.component,
            "confidence": round(float(self.confidence), 4),
            "source_cell_id": self.source_cell_id,
            "lineage_id": self.lineage_id,
            "governance_required": self.governance_required,
            "suppressed": self.suppressed,
            "governance_denied": self.governance_denied,
            "executed": self.executed,
            "success": self.success,
            "notes": self.notes,
            "bounded_payload": self.bounded_payload,
        }


@dataclass
class ImmuneCell:
    cell_id: str
    lineage_id: str
    kind: CellKind
    receptor: np.ndarray
    subsystem_scope: str = "generic"
    age: int = 0
    persistence: float = 0.55
    fitness: float = 0.0
    last_activation: float = 0.0
    successes: int = 0
    failures: int = 0
    species_id: int = 0
    clone_generation: int = 0
    regulatory_strength: float = 1.0
    best_effector: EffectorKind | None = None
    last_antigen_id: str = ""
    born_at: float = field(default_factory=time.time)
    behavioral_rule: dict[str, Any] | None = None

    def resize_receptor(self, new_dim: int, rng: np.random.Generator | None = None) -> None:
        old_dim = len(self.receptor)
        if old_dim == new_dim:
            return
        resized = np.zeros(new_dim, dtype=np.float32)
        copy_len = min(old_dim, new_dim)
        resized[:copy_len] = self.receptor[:copy_len]
        if new_dim > old_dim and rng is not None:
            noise = rng.normal(0.0, 0.05, size=(new_dim - old_dim))
            resized[old_dim:] = np.clip(noise, 0.0, 1.0)
        self.receptor = np.clip(resized, 0.0, 1.0)

    def clone(
        self,
        *,
        rng: np.random.Generator,
        cell_id: str,
        mutation_sigma: float,
        target_dim: int | None = None,
    ) -> ImmuneCell:
        child = copy.deepcopy(self)
        child.cell_id = cell_id
        if target_dim is not None:
            child.resize_receptor(target_dim, rng)
        child.receptor = np.clip(
            child.receptor + rng.normal(0.0, mutation_sigma, size=child.receptor.shape),
            0.0,
            1.0,
        ).astype(np.float32)
        child.age = 0
        child.last_activation = 0.0
        child.clone_generation += 1
        child.persistence = max(0.18, min(1.0, child.persistence * 0.94))
        child.successes = 0
        child.failures = 0
        child.last_antigen_id = ""
        child.born_at = time.time()

        # Mutate behavioral rule
        if child.kind in {CellKind.B, CellKind.MEMORY}:
            child.behavioral_rule = _mutate_behavioral_rule(child.behavioral_rule, rng)

        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "lineage_id": self.lineage_id,
            "kind": self.kind.value,
            "subsystem_scope": self.subsystem_scope,
            "age": self.age,
            "persistence": round(float(self.persistence), 4),
            "fitness": round(float(self.fitness), 4),
            "last_activation": round(float(self.last_activation), 4),
            "successes": self.successes,
            "failures": self.failures,
            "species_id": self.species_id,
            "clone_generation": self.clone_generation,
            "regulatory_strength": round(float(self.regulatory_strength), 4),
            "best_effector": self.best_effector.value if self.best_effector else None,
            "last_antigen_id": self.last_antigen_id,
            "born_at": self.born_at,
            "receptor": self.receptor.astype(float).tolist(),
            "behavioral_rule": self.behavioral_rule,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImmuneCell:
        return cls(
            cell_id=str(data["cell_id"]),
            lineage_id=str(data["lineage_id"]),
            kind=CellKind(data["kind"]),
            receptor=np.asarray(data["receptor"], dtype=np.float32),
            subsystem_scope=str(data.get("subsystem_scope", "generic")),
            age=int(data.get("age", 0)),
            persistence=float(data.get("persistence", 0.55)),
            fitness=float(data.get("fitness", 0.0)),
            last_activation=float(data.get("last_activation", 0.0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
            species_id=int(data.get("species_id", 0)),
            clone_generation=int(data.get("clone_generation", 0)),
            regulatory_strength=float(data.get("regulatory_strength", 1.0)),
            best_effector=(
                EffectorKind(data["best_effector"]) if data.get("best_effector") else None
            ),
            last_antigen_id=str(data.get("last_antigen_id", "")),
            born_at=float(data.get("born_at", time.time())),
            behavioral_rule=data.get("behavioral_rule"),
        )


@dataclass
class ImmuneResponse:
    antigen: Antigen
    activated_cells: list[dict[str, Any]]
    artifacts: list[EffectorArtifact]
    selected_artifact: EffectorArtifact | None
    suppression_applied: float
    metabolic_scale: float
    entropy_pressure: float
    proliferation_count: int
    species_count: int
    tissue_snapshot: dict[str, Any]
    dream_consolidated: bool = False
    coverage_report: dict[str, Any] = field(default_factory=dict)
    verification_report: dict[str, Any] = field(default_factory=dict)
    diagnostic_verdict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "antigen": self.antigen.to_dict(),
            "activated_cells": self.activated_cells,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "selected_artifact": (
                self.selected_artifact.to_dict() if self.selected_artifact else None
            ),
            "suppression_applied": round(float(self.suppression_applied), 4),
            "metabolic_scale": round(float(self.metabolic_scale), 4),
            "entropy_pressure": round(float(self.entropy_pressure), 4),
            "proliferation_count": self.proliferation_count,
            "species_count": self.species_count,
            "tissue_snapshot": self.tissue_snapshot,
            "dream_consolidated": self.dream_consolidated,
            "coverage_report": self.coverage_report,
            "verification_report": self.verification_report,
            "diagnostic_verdict": self.diagnostic_verdict,
        }


class TissueField:
    """Diffusive tissue model over subsystem topology.

    The field tracks four slowly varying values per subsystem:
    - danger
    - inflammation
    - damage
    - repair

    Rather than using brittle one-shot thresholds, antigens perturb one node
    and then those perturbations diffuse through the subsystem graph.
    """

    def __init__(self, *, diffusion: float = 0.16, decay: float = 0.06):
        self._diffusion = diffusion
        self._decay = decay
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)
        self._danger: dict[str, float] = defaultdict(float)
        self._inflammation: dict[str, float] = defaultdict(float)
        self._damage: dict[str, float] = defaultdict(float)
        self._repair: dict[str, float] = defaultdict(float)

    def ensure_node(self, name: str) -> str:
        node = str(name or "unknown")
        self._edges.setdefault(node, {})
        _ = self._danger[node], self._inflammation[node], self._damage[node], self._repair[node]
        return node

    def register_edge(self, a: str, b: str, weight: float = 0.35) -> None:
        a = self.ensure_node(a)
        b = self.ensure_node(b)
        w = max(0.0, min(1.0, float(weight)))
        if a == b:
            return
        self._edges[a][b] = w
        self._edges[b][a] = w

    def ingest_antigen(self, antigen: Antigen) -> None:
        node = self.ensure_node(antigen.subsystem)
        self._danger[node] = self._clip(self._danger[node] + 0.45 * antigen.danger)
        self._inflammation[node] = self._clip(
            self._inflammation[node] + 0.35 * antigen.danger + 0.20 * antigen.subsystem_need
        )
        self._damage[node] = self._clip(
            self._damage[node]
            + 0.25 * max(antigen.resource_pressure, antigen.error_load, antigen.health_pressure)
        )
        self._repair[node] = self._clip(max(0.0, self._repair[node] - 0.08))
        self.diffuse()

    def mark_repair(self, subsystem: str, strength: float = 0.35) -> None:
        node = self.ensure_node(subsystem)
        s = self._clip(strength)
        self._repair[node] = self._clip(self._repair[node] + s)
        self._danger[node] = self._clip(self._danger[node] - 0.5 * s)
        self._damage[node] = self._clip(self._damage[node] - 0.45 * s)
        self._inflammation[node] = self._clip(self._inflammation[node] - 0.35 * s)
        self.diffuse()

    def mark_quarantine(self, subsystem: str, strength: float = 0.4) -> None:
        node = self.ensure_node(subsystem)
        s = self._clip(strength)
        self._danger[node] = self._clip(self._danger[node] - 0.2 * s)
        self._inflammation[node] = self._clip(self._inflammation[node] + 0.15 * s)
        self._repair[node] = self._clip(self._repair[node] + 0.10 * s)
        self.diffuse()

    def diffuse(self, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            self._danger = self._diffuse_scalar(self._danger)
            self._inflammation = self._diffuse_scalar(self._inflammation)
            self._damage = self._diffuse_scalar(self._damage)
            self._repair = self._diffuse_scalar(self._repair)

    def get_need(self, subsystem: str) -> float:
        node = self.ensure_node(subsystem)
        return self._clip(
            0.45 * self._danger[node]
            + 0.35 * self._damage[node]
            + 0.20 * self._inflammation[node]
            - 0.35 * self._repair[node]
        )

    def snapshot(self, top_k: int = 8) -> dict[str, Any]:
        nodes = list(self._edges.keys())
        hot = sorted(
            nodes,
            key=lambda node: self.get_need(node),
            reverse=True,
        )[:top_k]
        return {
            "danger": {node: round(self._danger[node], 4) for node in hot},
            "inflammation": {node: round(self._inflammation[node], 4) for node in hot},
            "damage": {node: round(self._damage[node], 4) for node in hot},
            "repair": {node: round(self._repair[node], 4) for node in hot},
            "hotspots": [
                {"subsystem": node, "need": round(self.get_need(node), 4)} for node in hot
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": {node: dict(neighbors) for node, neighbors in self._edges.items()},
            "danger": dict(self._danger),
            "inflammation": dict(self._inflammation),
            "damage": dict(self._damage),
            "repair": dict(self._repair),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        diffusion: float,
        decay: float,
    ) -> TissueField:
        field_obj = cls(diffusion=diffusion, decay=decay)
        field_obj._edges = defaultdict(
            dict,
            {
                str(node): {str(neighbor): float(weight) for neighbor, weight in neighbors.items()}
                for node, neighbors in data.get("edges", {}).items()
            },
        )
        field_obj._danger = defaultdict(
            float, {str(node): float(value) for node, value in data.get("danger", {}).items()}
        )
        field_obj._inflammation = defaultdict(
            float, {str(node): float(value) for node, value in data.get("inflammation", {}).items()}
        )
        field_obj._damage = defaultdict(
            float, {str(node): float(value) for node, value in data.get("damage", {}).items()}
        )
        field_obj._repair = defaultdict(
            float, {str(node): float(value) for node, value in data.get("repair", {}).items()}
        )
        return field_obj

    def _diffuse_scalar(self, values: dict[str, float]) -> defaultdict[str, float]:
        new_vals: defaultdict[str, float] = defaultdict(float)
        for node in self._edges:
            current = float(values[node])
            neighbors = self._edges.get(node, {})
            if neighbors:
                total_w = sum(max(weight, 0.0) for weight in neighbors.values()) + _EPSILON
                neighbor_mean = (
                    sum(values[neighbor] * weight for neighbor, weight in neighbors.items())
                    / total_w
                )
                diffused = current + self._diffusion * (neighbor_mean - current)
            else:
                diffused = current
            new_vals[node] = self._clip(diffused * (1.0 - self._decay))
        return new_vals

    @staticmethod
    def _clip(value: float) -> float:
        return float(max(0.0, min(1.0, value)))
#: The coevolution lab swaps process-global singletons to run a rule against a
#: copy of the world. CP126 3da4c199: it did so with no lock, so concurrent
#: live code could observe or mutate the SIMULATION while believing it held
#: the real world model. One lab run at a time, and a flag live code can read.
_SIMULATION_ISOLATION_LOCK = checked_lock(
    "adaptive_immunity.simulation_isolation", rank=LockRank.REGISTRY, reentrant=True
)
_simulation_active = threading.Event()


def simulation_isolation_active() -> bool:
    """True while the coevolution lab holds the global world-model swap."""
    return _simulation_active.is_set()


#: Distinguishes "caller did not supply a vocabulary, look one up" from
#: "there IS no vocabulary". Conflating them meant a caller could not express
#: the second, and the live lookup silently supplied the toy sensors instead.
_VOCABULARY_UNSET: Any = object()


def _live_rule_vocabulary() -> dict[str, Any] | None:
    """Sensors and actuators that ACTUALLY exist in this runtime.

    CP126 956ba926: rule generation drew from a hardcoded maritime vocabulary
    — port_east_load, vessel_alpha_speed, reallocate_flow(Port_East,
    Port_West). The immune system exists to repair Aura's subsystems, so a
    learning lane that can only express opinions about a logistics toy was
    optimizing something unrelated to its purpose and reporting the result as
    repair fitness.

    Returns None when neither registry can be read, which is the honest answer
    and makes the caller refuse to author a rule rather than fall back to the
    toy.
    """
    sensors: list[str] = []
    sensor_values: dict[str, float] = {}
    actuators: list[str] = []
    action_templates: dict[str, dict[str, Any]] = {}
    try:
        from core.sensors.sensor_registry import get_sensor_registry

        readings = get_sensor_registry().read_all()
        sensors = sorted(str(name) for name in readings)
        for name, value in readings.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                sensor_values[str(name)] = number
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Immune rule vocabulary: sensors unavailable: %s", exc)
    try:
        from core.actuators.actuator_registry import get_actuator_registry

        registry = get_actuator_registry()
        for name, actuator in registry.actuators.items():
            if not bool(getattr(actuator, "immune_rule_compatible", False)):
                continue
            if bool(getattr(actuator, "requires_authority", True)):
                continue
            params = actuator.immune_rule_seed_params()
            if not isinstance(params, dict) or not actuator.validate_params(params):
                continue
            normalized_name = str(name)
            actuators.append(normalized_name)
            action_templates[normalized_name] = copy.deepcopy(params)
        actuators.sort()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Immune rule vocabulary: actuators unavailable: %s", exc)
    if not sensors or not actuators:
        return None
    return {
        "sensors": sensors,
        "sensor_values": sensor_values,
        "actuators": actuators,
        "action_templates": action_templates,
    }


def _system_pressure(model: Any) -> float | None:
    """Total strain across EVERY entity, whatever they are.

    CP126 691b21ed: fitness was the Port_East/Port_West load imbalance plus
    total latency. On any runtime without those two entities the metric raised
    KeyError, the broad except returned 0.0, and every rule scored identically
    — so the evolution was noise that looked like learning.

    Latency plus over-capacity overflow is defined for any entity set and
    reduces to the old intent on a logistics world. Returns None when there is
    nothing to measure, because a pressure of zero over zero entities is not a
    healthy system.
    """
    try:
        entities = list(getattr(model, "entities", {}).values())
    except (AttributeError, TypeError):
        return None
    if not entities:
        return None
    total = 0.0
    for entity in entities:
        latency = _unit_free_float(getattr(entity, "latency", 0.0))
        load = _unit_free_float(getattr(entity, "load", 0.0))
        capacity = _unit_free_float(getattr(entity, "capacity", 0.0))
        total += latency
        if capacity > 0.0 and load > capacity:
            total += (load - capacity) / capacity
    return total


def _unit_free_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0




def _mutate_behavioral_rule(
    rule: dict[str, Any] | None,
    rng: np.random.Generator,
    *,
    vocabulary: Any = _VOCABULARY_UNSET,
) -> dict[str, Any] | None:
    """Seed or mutate one condition-action rule over the LIVE vocabulary.

    CP126 956ba926: seeding and sensor mutation drew from a hardcoded maritime
    list, so every B-cell Aura ever evolved was an opinion about Port East and
    Port West. Returns None when no live vocabulary can be read — refusing to
    author a rule is correct, because a rule over sensors that do not exist can
    never fire and would occupy a population slot pretending otherwise.
    """
    import re

    vocab = (
        _live_rule_vocabulary() if vocabulary is _VOCABULARY_UNSET else vocabulary
    )

    if rule is None:
        if not vocab:
            logger.debug(
                "Immune rule generation skipped: no live sensor/actuator vocabulary"
            )
            return None
        sensor = str(rng.choice(vocab["sensors"]))
        actuator = str(rng.choice(vocab["actuators"]))
        templates = dict(vocab.get("action_templates") or {})
        params = copy.deepcopy(templates.get(actuator, {}))
        sensor_values = dict(vocab.get("sensor_values") or {})
        baseline = sensor_values.get(sensor)
        condition = {
            "sensor": sensor,
            "operator": ">",
            "value": float(rng.uniform(0.80, 1.10)),
        }
        if isinstance(baseline, (int, float)) and math.isfinite(float(baseline)):
            condition.update(
                {
                    "baseline": float(baseline),
                    "relative_to_baseline": True,
                }
            )
        return {
            "conditions": [condition],
            "actions": [
                {
                    "actuator": actuator,
                    "params": params,
                }
            ],
        }

    allowed_actuators = {
        str(item) for item in list((vocab or {}).get("actuators") or [])
    }
    current_actions = list(rule.get("actions", []) or [])
    if (
        vocab
        and (
            not current_actions
            or any(
                str(action.get("actuator") or "") not in allowed_actuators
                for action in current_actions
            )
        )
    ):
        return _mutate_behavioral_rule(None, rng, vocabulary=vocab)

    new_rule = copy.deepcopy(rule)
    choice = rng.choice(["value", "operator", "multiplier", "sensor"])

    conditions = new_rule.setdefault("conditions", [])
    actions = new_rule.setdefault("actions", [])

    if choice == "value" and conditions:
        cond = rng.choice(conditions)
        if isinstance(cond.get("value"), (int, float)) and not isinstance(
            cond.get("value"), bool
        ):
            cond["value"] = float(
                np.clip(float(cond["value"]) + rng.normal(0.0, 0.08), 0.05, 2.0)
            )
    elif choice == "operator" and conditions:
        cond = rng.choice(conditions)
        cond["operator"] = rng.choice([">", "<", ">=", "<=", "==", "!="])
    elif choice == "multiplier" and actions:
        act = rng.choice(actions)
        params = act.setdefault("params", {})
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$"):

                def repl(match):
                    val = float(match.group(0))
                    new_val = max(0.01, val + rng.normal(0.0, 0.05))
                    return f"{new_val:.2f}"

                params[k] = re.sub(r"\d+\.\d+", repl, v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                params[k] = float(v + rng.normal(0.0, 5.0))
    elif choice == "sensor" and conditions:
        cond = rng.choice(conditions)
        if vocab and vocab["sensors"]:
            cond["sensor"] = str(rng.choice(vocab["sensors"]))
        # With no vocabulary the existing sensor is kept: replacing it with a
        # hardcoded maritime name is what this finding is about, and inventing
        # one would be worse than leaving the rule as it is.

    templates = dict((vocab or {}).get("action_templates") or {})
    if templates:
        try:
            from core.actuators.actuator_registry import get_actuator_registry

            registry = get_actuator_registry()
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            registry = None
        for action in actions:
            name = str(action.get("actuator") or "")
            template = templates.get(name)
            if not isinstance(template, dict):
                continue
            params = action.get("params")
            actuator = registry.get_actuator(name) if registry is not None else None
            structurally_valid = isinstance(params, dict) and all(
                key in params for key in template
            )
            if structurally_valid and actuator is not None:
                structurally_valid = bool(actuator.validate_params(params))
            if not structurally_valid:
                action["params"] = copy.deepcopy(template)

    return new_rule


def _normalize_behavioral_rule(
    rule: dict[str, Any] | None,
    rng: np.random.Generator,
    *,
    vocabulary: Any = _VOCABULARY_UNSET,
) -> tuple[dict[str, Any] | None, bool]:
    """Validate a persisted rule against the current executable grammar."""
    vocab = (
        _live_rule_vocabulary() if vocabulary is _VOCABULARY_UNSET else vocabulary
    )
    if rule is None:
        return None, False
    if not vocab:
        return None, True
    actions = list(rule.get("actions", []) or []) if isinstance(rule, dict) else []
    conditions = list(rule.get("conditions", []) or []) if isinstance(rule, dict) else []
    allowed = {str(item) for item in list(vocab.get("actuators") or [])}
    templates = dict(vocab.get("action_templates") or {})
    valid = bool(actions and conditions) and all(
        str(action.get("actuator") or "") in allowed for action in actions
    )
    if valid:
        for condition in conditions:
            if condition.get("relative_to_reading"):
                valid = False
                break
            if condition.get("relative_to_baseline") and not isinstance(
                condition.get("baseline"), (int, float)
            ):
                valid = False
                break
    if valid:
        for action in actions:
            name = str(action.get("actuator") or "")
            params = action.get("params")
            template = templates.get(name)
            if not isinstance(params, dict):
                valid = False
                break
            if isinstance(template, dict):
                if any(key not in params for key in template):
                    valid = False
                    break
                if any(
                    isinstance(expected, bool)
                    and not isinstance(params.get(key), bool)
                    for key, expected in template.items()
                ):
                    valid = False
                    break
    if valid:
        return copy.deepcopy(rule), False
    return _mutate_behavioral_rule(None, rng, vocabulary=vocab), True


def _clone_world(model: Any) -> Any:
    """An independent copy of the world, for running a rule against."""
    from core.world.world_model import PhysicsWorldModel, WorldEntity

    clone = PhysicsWorldModel()
    clone.entities = {}
    for entity in model.entities.values():
        clone.add_entity(
            WorldEntity(
                entity_id=entity.entity_id,
                kind=entity.kind,
                capacity=entity.capacity,
                load=entity.load,
                flow_rate=entity.flow_rate,
                max_flow_rate=entity.max_flow_rate,
                latency=entity.latency,
                coordinates=entity.coordinates,
                attributes=entity.attributes.copy(),
            )
        )
    return clone


def _evaluate_causal_fitness(rule: dict[str, Any] | None) -> float | None:
    """How much this rule actually relieves system pressure, against a control.

    Returns None when fitness cannot be MEASURED, which is distinct from 0.0
    meaning "measured, changed nothing". CP126 691b21ed: the old version
    scored Port_East/Port_West load imbalance, so on any runtime without those
    entities it raised KeyError, the broad except returned 0.0, and every rule
    scored identically — evolution over pure noise, reported as causal repair
    fitness. A metric that silently returns the same number for every input is
    worse than no metric, because the lane still looks like it is learning.

    Two changes make the score mean something:

    * pressure is :func:`_system_pressure`, defined over whatever entities
      exist, so it measures the world Aura actually has;
    * a SHAM arm runs the identical simulation WITHOUT executing the rule, and
      its drift is subtracted. Without that control a rule is credited for
      improvement the world would have produced on its own, which is the
      difference between "this rule helps" and "things got better".
    """
    if not rule:
        return None

    try:
        import core.sensors.sensor_registry as sr
        import core.world.world_model as wm
        from core.adaptation.immune_executor import ImmuneHeuristicExecutor
        from core.world.world_model import get_physics_world_model

        main_model = get_physics_world_model()
        treatment_model = _clone_world(main_model)
        control_model = _clone_world(main_model)

        before = _system_pressure(treatment_model)
        if before is None:
            # Nothing to measure. Saying so is the point.
            logger.debug("Causal fitness unmeasurable: the world model has no entities")
            return None

        # CP126 3da4c199: the swap below replaces PROCESS-GLOBAL singletons.
        # One lab at a time, and live code can ask whether a simulation holds
        # them via simulation_isolation_active().
        with _SIMULATION_ISOLATION_LOCK:
            original_model = wm._instance
            original_registry = sr._instance
            _simulation_active.set()
            try:
                wm._instance = treatment_model
                sim_registry = sr.SensorRegistry()
                sim_registry.sync_from_world_model()
                sr._instance = sim_registry

                executor = ImmuneHeuristicExecutor()
                exec_res = executor.execute_rule(
                    rule,
                    context={
                        "source": "causal_fitness_lab",
                        "isolated_simulation": True,
                        "world_model_isolated": True,
                        "priority": 0.2,
                    },
                )
                fired = bool(
                    exec_res.get("conditions_met") and exec_res.get("success")
                )
                if fired:
                    treatment_model.simulate(10.0)
                    sim_registry.sync_from_world_model()
            finally:
                wm._instance = original_model
                sr._instance = original_registry
                _simulation_active.clear()

        if not fired:
            # The rule was measured and did not apply. That is a real 0, not
            # an absence of measurement.
            return 0.0

        # Sham arm: the same elapsed simulation with no rule executed.
        control_model.simulate(10.0)

        after = _system_pressure(treatment_model)
        control_after = _system_pressure(control_model)
        if after is None or control_after is None:
            return None

        treatment_relief = before - after
        control_relief = before - control_after
        # Credit only the relief the CONTROL did not also produce.
        return float(treatment_relief - control_relief)
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed evaluating causal fitness in coevolution lab: %s", exc)
        return None


class OfflineCoevolutionLab:
    """Small sandbox for evolving defender receptors against replayed threats."""

    def __init__(self, *, rng: np.random.Generator):
        self._rng = rng

    @staticmethod
    def _objective(
        cell: ImmuneCell,
        antigens: Sequence[Antigen],
        *,
        tau: float,
        weights: Any,
    ) -> float:
        """The single fitness this lab selects on AND ranks by.

        Kept as one function because CP126 714a9713 was two: generations were
        selected with causal fitness included and the survivors were then
        re-sorted without it.
        """
        score = 0.0
        for antigen in antigens:
            affinity = AdaptiveImmuneSystem.compute_affinity_static(
                cell.receptor, antigen.vector, tau=tau, weights=weights
            )
            if antigen.protected:
                if cell.kind == CellKind.REGULATORY:
                    score += 1.1 * affinity
                else:
                    score -= 0.7 * affinity
            else:
                if cell.kind in {CellKind.B, CellKind.CYTOTOXIC, CellKind.MEMORY}:
                    score += affinity * antigen.danger * (0.5 + 0.5 * antigen.subsystem_need)
                elif cell.kind == CellKind.REGULATORY:
                    score -= 0.2 * affinity * antigen.danger

        if cell.kind in {CellKind.B, CellKind.MEMORY}:
            causal_fit = _evaluate_causal_fitness(cell.behavioral_rule)
            # None means fitness could not be MEASURED (no entities, no rule,
            # lab unavailable). Scoring it as 0.0 would rank an unmeasured
            # cell against measured ones as though it had been tested and
            # found useless, making selection pressure an artefact of what
            # happened to be measurable.
            if causal_fit is not None:
                score += causal_fit * 2.5
        return score

    def evolve(
        self,
        cells: Iterable[ImmuneCell],
        antigens: Iterable[Antigen],
        *,
        generations: int = 3,
        population_size: int = 12,
        tau: float = 0.22,
        mutation_sigma: float = 0.05,
        target_dim: int = 16,
        weights: np.ndarray | None = None,
    ) -> list[ImmuneCell]:
        seeds = [
            copy.deepcopy(cell)
            for cell in cells
            if cell.kind in {CellKind.B, CellKind.CYTOTOXIC, CellKind.REGULATORY, CellKind.MEMORY}
        ]
        if not seeds:
            return []
        antigens = list(antigens)
        if not antigens:
            return []

        for cell in seeds:
            cell.resize_receptor(target_dim, self._rng)

        population = seeds[:population_size]
        next_id = 0
        while len(population) < population_size:
            source = self._rng.choice(seeds)
            clone = source.clone(
                rng=self._rng,
                cell_id=f"offline_lab_{next_id}",
                mutation_sigma=mutation_sigma,
                target_dim=target_dim,
            )
            next_id += 1
            population.append(clone)

        for _ in range(max(1, generations)):
            scored: list[tuple[float, ImmuneCell]] = []
            for cell in population:
                score = self._objective(cell, antigens, tau=tau, weights=weights)
                scored.append((score, cell))

            scored.sort(key=lambda item: item[0], reverse=True)
            survivors = [copy.deepcopy(cell) for _, cell in scored[: max(2, population_size // 2)]]
            population = survivors[:]
            while len(population) < population_size:
                parent = copy.deepcopy(self._rng.choice(survivors))
                population.append(
                    parent.clone(
                        rng=self._rng,
                        cell_id=f"offline_lab_{next_id}",
                        mutation_sigma=mutation_sigma,
                        target_dim=target_dim,
                    )
                )
                next_id += 1

        # CP126 714a9713: this used to re-rank by receptor affinity ALONE,
        # discarding the causal-fitness term that every generation of
        # selection had just applied. The champions returned to the caller
        # were therefore the best BINDERS, not the best repairers — the lane
        # optimized one objective and shipped the winner of another. Ranking
        # on the same objective selection used is the whole point of having
        # one.
        population.sort(
            key=lambda cell: self._objective(cell, antigens, tau=tau, weights=weights),
            reverse=True,
        )
        return population[:4]


class AdaptiveImmuneSystem:
    """Adaptive immune ecology for Aura."""

    _PROTECTED_SUBSYSTEM_HINTS = (
        "identity",
        "canonical_self",
        "self_model",
        "soul",
        "will",
        "sovereignty",
        "constitution",
        "executive",
        "continuity",
        "memory_guard",
    )

    _FEATURE_WEIGHTS = np.asarray(
        [
            0.70,
            0.35,
            0.35,
            0.30,
            0.80,
            0.80,
            0.25,
            0.60,
            1.00,
            0.90,
            0.85,
            1.15,
            0.70,
            0.65,
            0.65,
            0.45,
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *,
        config: AdaptiveImmuneConfig | None = None,
        state_dir: Path | None = None,
        rng_seed: int | None = None,
    ):
        self.cfg = config or AdaptiveImmuneConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._extractor = FeatureExtractor()
        self._lock = threading.RLock()
        self._cells: list[ImmuneCell] = []
        self._tissue = TissueField(
            diffusion=self.cfg.tissue_diffusion,
            decay=self.cfg.tissue_decay,
        )
        self._recent_antigens: deque[Antigen] = deque(maxlen=self.cfg.replay_buffer_size)
        self._recent_responses: deque[dict[str, Any]] = deque(
            maxlen=self.cfg.recent_response_buffer
        )
        # Time-windowed event timestamps per subsystem. A cumulative counter
        # never decayed, so temporal pressure permanently saturated after six
        # lifetime events and stopped representing recurrence RECENCY.
        self._recent_subsystem_events: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=64)
        )
        self._subsystem_event_window_s: float = 900.0
        # CP126 b694c436: suppression used to be credited the instant it
        # happened. Whether suppressing was RIGHT is only knowable afterwards,
        # so each one is parked here and settled against what the subsystem
        # actually did next.
        self._pending_suppressions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Guards the expansion engine alone, so a PCA does not serialize every
        # other immune observation (CP126 ad18eba6).
        # Coalescing state for _save_state (CP126 df9f2a05).
        self._state_dirty = False
        self._last_save_at = 0.0
        self._save_min_interval_s = 2.0
        self._deferred_saves = 0
        self._expansion_lock = checked_lock(
            "adaptive_immunity.expansion", rank=LockRank.LANE, reentrant=True
        )
        self._suppression_verdict_window_s: float = 600.0
        self._recurrence_tracker: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "occurrences": 0,
                "last_seen": 0.0,
                "interval_ewma": 0.0,
                "last_interval": None,
                "streak": 0,
                "peak_streak": 0,
                "verified_repairs": 0,
                "failed_repairs": 0,
                "last_verified_at": 0.0,
            }
        )
        self._lineage_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "successes": 0,
                "failures": 0,
                "best_effector": None,
                "best_fitness": 0.0,
            }
        )
        self._observation_count = 0
        self._species_count = 1
        self._last_dream_at = 0
        self._last_dream_defer_log_at = 0.0
        self._last_dream_defer_reason = ""
        self._state_dir = self._resolve_state_dir(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "adaptive_immune_state.json"
        self._migrated_behavioral_rules = 0
        self._lab = OfflineCoevolutionLab(rng=self._rng)

        from core.adaptation.dimensional_expansion import DimensionalExpansionEngine

        self.expansion_engine = DimensionalExpansionEngine(
            initial_dim=self.cfg.initial_receptor_dim,
            max_dim=self.cfg.max_receptor_dim,
            expansion_check_interval=self.cfg.expansion_check_interval,
            expansion_eigenvalue_threshold=self.cfg.expansion_eigenvalue_threshold,
            contraction_score_floor=self.cfg.contraction_score_floor,
            contraction_min_observations=self.cfg.contraction_min_observations,
            base_weights=AdaptiveImmuneSystem._FEATURE_WEIGHTS,
        )

        global _adaptive_immune_singleton
        _adaptive_immune_singleton = self

        if not self._load_state():
            self._cells = self._seed_population()
            self._assign_species()
            self._save_state(force=True)
        elif self._migrated_behavioral_rules:
            self._save_state(force=True)

        logger.info(
            "AdaptiveImmuneSystem online (population=%d, state=%s)",
            len(self._cells),
            self._state_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def observe_event(
        self,
        event: dict[str, Any],
        *,
        anomaly_score: Any | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> ImmuneResponse:
        # OFF-LOOP: present_antigen runs numpy projections per event plus a
        # PCA eigendecomposition on the periodic expansion check — on the
        # loop it froze the runtime for 5.3s during a degradation storm
        # (2026-07-12 23:53 stall receipt: dimensional_expansion
        # _update_contribution_scores on the stamped loop thread). The
        # expansion engine takes its own lock; thread-safe by design.
        antigen = await asyncio.to_thread(
            self.present_antigen,
            event,
            anomaly_score=anomaly_score,
            state_snapshot=state_snapshot,
        )
        coverage_report = await asyncio.to_thread(
            self._assess_coverage,
            event,
            antigen,
            anomaly_score=anomaly_score,
            state_snapshot=state_snapshot,
        )
        # Core evolution snapshots and persists the full immune ecology. The
        # durable snapshot is intentionally synchronous for crash consistency,
        # so the complete mutation transaction belongs on a worker thread.
        response, _top_cell = await asyncio.to_thread(self._observe_core, antigen)
        response.coverage_report = coverage_report
        self._apply_coverage_constraints(response, antigen, coverage_report)

        selected_artifact = None
        verification_report = self._default_verification_report(
            status="not_executed",
            coverage_ratio=coverage_report["coverage_ratio"],
        )
        executed_candidates = 0

        for artifact in self._execution_candidates(response):
            if executed_candidates >= self.cfg.max_execution_attempts_per_event:
                break
            selected_artifact = artifact
            verification_report = await self._maybe_execute_artifact(
                artifact,
                antigen,
                coverage_report=coverage_report,
            )
            executed_candidates += 1
            if artifact.success:
                break

        if selected_artifact is None:
            selected_artifact = self._best_visible_artifact(response)

        response.selected_artifact = selected_artifact
        response.verification_report = verification_report
        response.diagnostic_verdict = self._build_diagnostic_verdict(
            antigen,
            response,
            coverage_report=coverage_report,
            verification_report=verification_report,
        )

        if response.selected_artifact and (
            response.selected_artifact.executed or response.selected_artifact.governance_denied
        ):
            acting_cell = self._find_cell(response.selected_artifact.source_cell_id)
            response.proliferation_count = await asyncio.to_thread(
                self._reinforce_after_execution,
                antigen=antigen,
                response=response,
                acting_cell=acting_cell,
                verification_report=verification_report,
            )
        else:
            await asyncio.to_thread(self._reinforce_without_execution, antigen, response)

        await asyncio.to_thread(self._record_response_summary, response)

        return response

    def _observe_event(
        self, event: dict[str, Any], context: dict[str, Any]
    ) -> ImmuneResponse:
        """The heavy observation path: PCA, clustering, and durable writes.

        CP126 93ff2643: observe_error and observe_signature call this
        synchronously, and asynchronous runtime code calls THEM — so
        dimensional expansion (which can eigendecompose) and the durable
        response write ran on the owner loop. This is the same class as the
        five-second embedding load that hung the app on launch.

        It cannot simply decline on the loop the way a retrieval can: the
        caller needs the response, and dropping an immune observation to
        protect latency would trade a visible stall for an invisible blind
        spot. So the work still happens, and an on-loop call is RECORDED —
        which makes the remaining offenders findable instead of silent.
        Async callers should use :meth:`observe_error_async` /
        :meth:`observe_signature_async`, which do the same work off-loop.
        """
        if _on_event_loop():
            _record_adaptive_immunity_degradation(
                RuntimeError("immune observation ran on the event loop"),
                action=(
                    "completed a heavy immune observation inline; use "
                    "observe_error_async/observe_signature_async from async code"
                ),
                severity="warning",
                extra={"subsystem": str(event.get("subsystem", "unknown"))},
            )
        antigen = self.present_antigen(event, anomaly_score=None, state_snapshot=context)
        response, _top_cell = self._observe_core(antigen)
        response.coverage_report = self._assess_coverage(
            event, antigen, anomaly_score=None, state_snapshot=context
        )
        response.verification_report = self._default_verification_report(
            status="not_executed",
            coverage_ratio=response.coverage_report["coverage_ratio"],
        )
        response.diagnostic_verdict = self._build_diagnostic_verdict(
            antigen,
            response,
            coverage_report=response.coverage_report,
            verification_report=response.verification_report,
        )
        self._reinforce_without_execution(antigen, response)
        self._record_response_summary(response)
        return response

    async def observe_error_async(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> ImmuneResponse:
        """observe_error without occupying the event loop (CP126 93ff2643)."""
        return await asyncio.to_thread(self.observe_error, error, context)

    async def observe_signature_async(
        self,
        component: str,
        exception_type: str,
        *,
        error_count: int = 1,
        context: dict[str, Any] | None = None,
    ) -> ImmuneResponse:
        """observe_signature without occupying the event loop."""
        return await asyncio.to_thread(
            self.observe_signature,
            component,
            exception_type,
            error_count=error_count,
            context=context,
        )

    def observe_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> ImmuneResponse:
        context = context or {}
        event = {
            "type": "exception",
            "text": f"{type(error).__name__}: {error}",
            "source": "exception",
            "subsystem": context.get("component") or context.get("stage") or "unknown",
            "resource_pressure": float(context.get("resource_pressure", 0.0)),
            "error_count": 1,
            "timestamp": time.time(),
            "stack_trace": context.get("stack_trace", ""),
            "exception_type": type(error).__name__,
        }
        return self._observe_event(event, context)

    def observe_signature(
        self,
        component: str,
        exception_type: str,
        *,
        error_count: int = 1,
        context: dict[str, Any] | None = None,
    ) -> ImmuneResponse:
        context = context or {}
        event = {
            "type": "error_signature",
            "text": f"{exception_type} in {component}",
            "source": "signature",
            "subsystem": component,
            "resource_pressure": float(context.get("resource_pressure", 0.0)),
            "error_count": int(error_count),
            "timestamp": time.time(),
            "exception_type": exception_type,
        }
        return self._observe_event(event, context)

    def _recent_subsystem_count(self, subsystem: str) -> int:
        """Count subsystem events inside the recency window, pruning old ones."""
        events = self._recent_subsystem_events.get(subsystem)
        if not events:
            return 0
        cutoff = time.time() - self._subsystem_event_window_s
        while events and events[0] < cutoff:
            events.popleft()
        return len(events)

    def _settle_suppressions(self, subsystem: str, danger: float) -> None:
        """Decide whether earlier suppressions of this subsystem were right.

        Called when a new antigen arrives. A subsystem that returns MORE
        dangerous than when it was silenced is evidence the suppression was
        wrong; one that stays quiet past the window is evidence it was right.
        Either way the regulatory cell is scored on the outcome rather than on
        the act (CP126 b694c436).
        """
        now = time.time()
        for cell_id, pending in list(self._pending_suppressions.items()):
            if not pending:
                self._pending_suppressions.pop(cell_id, None)
                continue
            cell = self._find_cell(cell_id)
            remaining: list[dict[str, Any]] = []
            for record in pending:
                aged_out = (now - record["at"]) > self._suppression_verdict_window_s
                recurred = (
                    record["subsystem"] == subsystem
                    and danger > record["danger"] + 0.05
                )
                if not recurred and not aged_out:
                    remaining.append(record)
                    continue
                if cell is None:
                    continue
                if recurred:
                    # Suppression was wrong: the danger it silenced came back
                    # worse. No success is counted and fitness drops.
                    cell.fitness = 0.85 * cell.fitness - 0.15 * min(1.0, danger)
                else:
                    # Quiet through the whole window — the suppression held.
                    cell.successes += 1
                    cell.fitness = 0.85 * cell.fitness + 0.15 * max(
                        cell.fitness, record["danger"] * max(record["suppression"], 0.25)
                    )
            if remaining:
                self._pending_suppressions[cell_id] = remaining
            else:
                self._pending_suppressions.pop(cell_id, None)

    def present_antigen(
        self,
        event: dict[str, Any],
        *,
        anomaly_score: Any | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> Antigen:
        with self._lock:
            subsystem = self._canonical_subsystem(
                event.get("subsystem")
                or event.get("component")
                or event.get("source")
                or event.get("type")
                or "unknown"
            )
            self._ensure_graph_links(subsystem)

            base_vec = self._extractor.extract(event)
            # CP126 8bd58283: these were `float(event.get(...)) or ...` with
            # min/max clipping. NaN is TRUTHY, so `or` passed it straight
            # through, and min/max propagate NaN rather than clipping it — a
            # single malformed telemetry field then flowed into danger and
            # activation, where every threshold comparison against NaN is
            # False and the antigen reads as calm. Negative counts and huge
            # values were likewise accepted.
            declared_pressure = _optional_unit(event.get("resource_pressure"))
            if declared_pressure is None:
                resource_pressure = max(
                    0.0,
                    _unit_scalar(_unit_free_float(event.get("cpu", 0.0)) / 100.0),
                    _unit_scalar(_unit_free_float(event.get("ram", 0.0)) / 100.0),
                )
            else:
                resource_pressure = declared_pressure
            error_load = _unit_scalar(
                max(0.0, _unit_free_float(event.get("error_rate", 0.0)))
                + max(0.0, _unit_free_float(event.get("error_count", 0))) / 10.0
            )
            error_signature = str(
                event.get("exception_type")
                or event.get("error_signature")
                or event.get("type")
                or ""
            )
            threat_probability = _first_unit(
                getattr(anomaly_score, "threat_probability", None),
                event.get("threat_probability"),
                event.get("danger"),
                default=max(resource_pressure, error_load * 0.7),
            )
            stack_trace = str(event.get("stack_trace", "") or "")
            stack_complexity = min(1.0, len(stack_trace) / 1200.0)
            protected = bool(
                event.get("protected", False) or self._is_protected_subsystem(subsystem)
            )
            health_pressure = self._component_health_pressure(subsystem, state_snapshot)
            temporal_pressure = min(
                1.0,
                float(self._recent_subsystem_count(subsystem)) / 6.0,
            )
            recurrence_pressure = self._estimate_recurrence_pressure(subsystem, error_signature)
            tissue_need_prior = self._tissue.get_need(subsystem)

            danger = max(
                0.0,
                min(
                    1.0,
                    0.48 * threat_probability
                    + 0.20 * error_load
                    + 0.15 * resource_pressure
                    + 0.09 * stack_complexity
                    + 0.08 * recurrence_pressure,
                ),
            )
            subsystem_need = max(
                tissue_need_prior,
                0.48 * danger
                + 0.32 * max(health_pressure, resource_pressure, error_load)
                + 0.20 * recurrence_pressure,
            )

            # Build canonical 16-dim vector
            base_vector = np.zeros(16, dtype=np.float32)
            base_vector[:8] = base_vec[:8]
            base_vector[8] = danger
            base_vector[9] = resource_pressure
            base_vector[10] = error_load
            base_vector[11] = 1.0 if protected else 0.0
            base_vector[12] = health_pressure
            base_vector[13] = tissue_need_prior
            base_vector[14] = max(temporal_pressure, recurrence_pressure)
            base_vector[15] = stack_complexity

        # CP126 ad18eba6: dimensional expansion can EIGENDECOMPOSE, and it ran
        # while holding the process-wide immune RLock — so every other
        # observation queued behind one PCA. The heavy call is done outside
        # that lock, under its own, so expansions still serialize with each
        # other but no longer serialize the whole immune system.
        with self._expansion_lock:
            vector, new_events = self.expansion_engine.evaluate_expansion(event, base_vector)
            target_dim = self.expansion_engine.current_dim

        with self._lock:
            # Resize receptors of all cells dynamically if new dimensions are born
            if new_events:
                for cell in self._cells:
                    cell.resize_receptor(target_dim, self._rng)

            antigen_id = (
                f"ag_{hashlib.sha1(f'{subsystem}:{time.time()}'.encode()).hexdigest()[:12]}"
            )

            antigen = Antigen(
                antigen_id=antigen_id,
                subsystem=subsystem,
                vector=vector,
                danger=danger,
                subsystem_need=max(0.0, min(1.0, subsystem_need)),
                threat_probability=max(0.0, min(1.0, threat_probability)),
                resource_pressure=max(0.0, min(1.0, resource_pressure)),
                error_load=max(0.0, min(1.0, error_load)),
                health_pressure=max(0.0, min(1.0, health_pressure)),
                temporal_pressure=max(0.0, min(1.0, temporal_pressure)),
                recurrence_pressure=max(0.0, min(1.0, recurrence_pressure)),
                protected=protected,
                # Carry the event's origin domain — omitting it classified
                # every live event as substrate, defeating the environmental
                # repair guard.
                source_domain=str(event.get("source_domain") or "substrate"),
                source=str(event.get("source") or event.get("type") or "unknown"),
                error_signature=error_signature,
                stack_trace=stack_trace,
                context=dict(state_snapshot or {}),
            )
            antigen.context.setdefault("spatial_receptor_code", annotate_antigen_like(antigen))
            self._settle_suppressions(subsystem, danger)
            return antigen

    def dream_consolidate(self) -> dict[str, Any]:
        with self._lock:
            promotions = 0
            removed = 0

            for cell in self._cells:
                cell.age += 1
                decay = (
                    self.cfg.memory_decay if cell.kind == CellKind.MEMORY else self.cfg.basal_decay
                )
                cell.persistence = max(0.0, cell.persistence - decay * (1.0 + 0.15 * cell.failures))
                cell.fitness *= 0.98

            for cell in list(self._cells):
                lineage = self._lineage_stats[cell.lineage_id]
                if (
                    cell.kind != CellKind.MEMORY
                    and (
                        cell.successes >= self.cfg.lineage_memory_successes
                        or lineage["successes"] >= self.cfg.lineage_memory_successes
                    )
                    and max(cell.fitness, float(lineage["best_fitness"]))
                    >= self.cfg.lineage_memory_fitness
                ):
                    if not any(
                        existing.lineage_id == cell.lineage_id and existing.kind == CellKind.MEMORY
                        for existing in self._cells
                    ):
                        memory = copy.deepcopy(cell)
                        memory.cell_id = f"mem_{hashlib.sha1((cell.cell_id + str(time.time())).encode()).hexdigest()[:10]}"
                        memory.kind = CellKind.MEMORY
                        memory.persistence = min(
                            1.0, memory.persistence + self.cfg.persistence_boost + 0.15
                        )
                        memory.regulatory_strength = max(memory.regulatory_strength, 1.0)
                        self._cells.append(memory)
                        promotions += 1

            champions = self._lab.evolve(
                self._cells,
                list(self._recent_antigens)[-24:],
                generations=2,
                population_size=10,
                tau=self.cfg.tau,
                mutation_sigma=self.cfg.mutation_sigma,
                target_dim=self.expansion_engine.current_dim,
                weights=self.expansion_engine.feature_weights.get(),
            )
            for champion in champions[:2]:
                if len(self._cells) < self.cfg.max_population:
                    champion.cell_id = f"lab_{hashlib.sha1((champion.cell_id + str(time.time())).encode()).hexdigest()[:10]}"
                    champion.persistence = min(1.0, champion.persistence + 0.08)
                    self._cells.append(champion)

            # Evaluate dimensional contraction to retire under-used dimensions
            retired = self.expansion_engine.evaluate_contraction()
            if retired:
                target_dim = self.expansion_engine.current_dim
                for cell in self._cells:
                    cell.resize_receptor(target_dim, self._rng)

            self._assign_species()
            self._prune_population()

            for cell in list(self._cells):
                if cell.persistence <= 0.03 or (
                    cell.fitness < -0.55 and cell.kind != CellKind.REGULATORY
                ):
                    self._cells.remove(cell)
                    removed += 1

            self._save_state(force=True)
            self._last_dream_at = self._observation_count
            return {
                "promotions": promotions,
                "removed": removed,
                "population": len(self._cells),
                "species_count": self._species_count,
            }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            by_kind = Counter(cell.kind.value for cell in self._cells)
            hot_tissue = self._tissue.snapshot()
            top_lineages = sorted(
                self._lineage_stats.items(),
                key=lambda item: float(item[1]["best_fitness"]),
                reverse=True,
            )[:5]
            return {
                "population": len(self._cells),
                "species_count": self._species_count,
                "observation_count": self._observation_count,
                "last_dream_at": self._last_dream_at,
                "cells_by_kind": dict(by_kind),
                "coverage": self._system_coverage_summary(),
                "recurrence_hotspots": self._recurrence_hotspots(),
                "top_lineages": [
                    {
                        "lineage_id": lineage_id,
                        "successes": stats["successes"],
                        "failures": stats["failures"],
                        "best_effector": (
                            stats["best_effector"].value
                            if isinstance(stats["best_effector"], EffectorKind)
                            else None
                        ),
                        "best_fitness": round(float(stats["best_fitness"]), 4),
                    }
                    for lineage_id, stats in top_lineages
                ],
                "tissue": hot_tissue,
                "recent_responses": list(self._recent_responses)[-6:],
            }

    @staticmethod
    def compute_affinity_static(
        receptor: np.ndarray,
        antigen_vector: np.ndarray,
        *,
        tau: float,
        weights: np.ndarray | None = None,
    ) -> float:
        r_len = receptor.shape[0]
        a_len = antigen_vector.shape[0]
        max_len = max(r_len, a_len)

        r_pad = np.zeros(max_len, dtype=np.float32)
        r_pad[:r_len] = receptor

        a_pad = np.zeros(max_len, dtype=np.float32)
        a_pad[:a_len] = antigen_vector

        if weights is not None:
            w = weights
            if len(w) < max_len:
                w_pad = np.ones(max_len, dtype=np.float32) * 0.5
                w_pad[: len(w)] = w
                w = w_pad
            else:
                w = w[:max_len]
        else:
            w = np.ones(max_len, dtype=np.float32) * 0.5

        diff = (r_pad - a_pad) * w
        distance = float(np.linalg.norm(diff) / math.sqrt(max(1, max_len)))
        return float(math.exp(-((distance * distance) / max(tau, 1e-6))))

    # ------------------------------------------------------------------
    # Core observation and evolution
    # ------------------------------------------------------------------

    def _observe_core(self, antigen: Antigen) -> tuple[ImmuneResponse, ImmuneCell | None]:
        with self._lock:
            self._observation_count += 1
            self._recent_subsystem_events[antigen.subsystem].append(time.time())
            self._recent_antigens.append(antigen)
            self._record_recurrence_observation(antigen)
            self._tissue.ingest_antigen(antigen)

            metabolic_scale, entropy_pressure = self._metabolic_context()
            activated: list[tuple[ImmuneCell, float, float]] = []
            regulatory_suppression = 0.0
            dominant_regulatory: ImmuneCell | None = None

            for cell in self._cells:
                affinity = self._affinity(cell, antigen)
                activation = self._activation(cell, antigen, affinity, metabolic_scale)
                cell.last_activation = activation
                cell.last_antigen_id = antigen.antigen_id
                if cell.kind == CellKind.REGULATORY:
                    if antigen.protected or activation >= self.cfg.activation_threshold * 0.75:
                        suppression = min(0.97, activation * cell.regulatory_strength)
                        if suppression > regulatory_suppression:
                            regulatory_suppression = suppression
                            dominant_regulatory = cell
                    continue
                if activation >= self.cfg.activation_threshold:
                    activated.append((cell, affinity, activation))

            activated.sort(key=lambda item: item[2], reverse=True)
            artifacts: list[EffectorArtifact] = []
            top_cell: ImmuneCell | None = None

            for cell, affinity, activation in activated[: self.cfg.max_artifacts_per_antigen]:
                if top_cell is None:
                    top_cell = cell
                artifact = self._emit_artifact(cell, antigen, affinity, activation)
                if artifact is None:
                    continue
                if artifact.governance_required and antigen.protected:
                    artifact.suppressed = regulatory_suppression > 0.18
                    if artifact.suppressed and not artifact.notes:
                        artifact.notes = "regulatory suppression on protected tissue"
                artifacts.append(artifact)

            if dominant_regulatory and antigen.protected:
                # CP126 b694c436: this incremented `successes` and raised
                # fitness the moment suppression happened — for suppressing at
                # all, not for suppressing correctly. Silencing a GENUINE
                # threat that happens to sit on protected tissue scored
                # exactly like preventing an autoimmune response, so the
                # regulatory lineage was selected for quietness rather than
                # for judgement.
                #
                # The credit is deferred. What settles it is what the
                # subsystem does next: staying quiet means the suppression was
                # right, coming back louder means it was wrong.
                self._pending_suppressions[dominant_regulatory.cell_id].append(
                    {
                        "subsystem": antigen.subsystem,
                        "danger": float(antigen.danger),
                        "suppression": float(regulatory_suppression),
                        "at": time.time(),
                    }
                )

            selected_artifact = max(
                (artifact for artifact in artifacts if not artifact.suppressed),
                key=lambda artifact: artifact.confidence,
                default=None,
            )

            dream_consolidated = False
            if self._observation_count - self._last_dream_at >= self.cfg.dream_every_observations:
                defer_reason = _maintenance_background_deferral_reason()
                if defer_reason:
                    self._log_dream_deferred(defer_reason)
                else:
                    self.dream_consolidate()
                    dream_consolidated = True

            activated_cells = [
                {
                    "cell_id": cell.cell_id,
                    "lineage_id": cell.lineage_id,
                    "kind": cell.kind.value,
                    "subsystem_scope": cell.subsystem_scope,
                    "affinity": round(float(affinity), 4),
                    "activation": round(float(activation), 4),
                    "fitness": round(float(cell.fitness), 4),
                    "species_id": cell.species_id,
                }
                for cell, affinity, activation in activated[:6]
            ]

            response = ImmuneResponse(
                antigen=antigen,
                activated_cells=activated_cells,
                artifacts=artifacts,
                selected_artifact=selected_artifact,
                suppression_applied=regulatory_suppression,
                metabolic_scale=metabolic_scale,
                entropy_pressure=entropy_pressure,
                proliferation_count=0,
                species_count=self._species_count,
                tissue_snapshot=self._tissue.snapshot(),
                dream_consolidated=dream_consolidated,
            )
            self._recent_responses.append(
                {
                    "subsystem": antigen.subsystem,
                    "danger": round(float(antigen.danger), 4),
                    "recurrence_pressure": round(float(antigen.recurrence_pressure), 4),
                    "selected_artifact": (
                        selected_artifact.kind.value if selected_artifact else None
                    ),
                    "suppression": round(float(regulatory_suppression), 4),
                }
            )
            self._save_state()
        return response, top_cell

    def _log_dream_deferred(self, defer_reason: str) -> None:
        now = time.monotonic()
        if defer_reason.startswith("boot_grace_"):
            reason_key = "boot_grace"
        elif defer_reason.startswith("recent_user_"):
            reason_key = "recent_user"
        elif defer_reason.startswith("failure_lockdown_"):
            reason_key = "failure_lockdown"
        elif defer_reason.startswith("memory_pressure_"):
            reason_key = "memory_pressure"
        else:
            reason_key = defer_reason
        if (
            reason_key == self._last_dream_defer_reason
            and now - self._last_dream_defer_log_at < 30.0
        ):
            logger.debug(
                "AdaptiveImmuneSystem: dream consolidation still deferred — %s.",
                defer_reason,
            )
            return
        self._last_dream_defer_reason = reason_key
        self._last_dream_defer_log_at = now
        logger.info(
            "AdaptiveImmuneSystem: dream consolidation deferred — %s.",
            defer_reason,
        )

    def _reinforce_without_execution(self, antigen: Antigen, response: ImmuneResponse) -> None:
        with self._lock:
            false_positive_cost = 0.25 if antigen.danger < 0.22 else 0.0
            entropy_cost = 0.12 * response.entropy_pressure
            regulatory_reward = (
                0.25 if antigen.protected and response.suppression_applied > 0.18 else 0.0
            )
            for cell_summary in response.activated_cells[:3]:
                cell = self._find_cell(cell_summary["cell_id"])
                if cell is None:
                    continue
                reward = 0.15 * cell_summary["activation"] - false_positive_cost - entropy_cost
                if cell.kind == CellKind.REGULATORY:
                    reward += regulatory_reward
                cell.fitness = 0.82 * cell.fitness + 0.18 * reward
                cell.failures += int(reward < 0.0)

    def _reinforce_after_execution(
        self,
        *,
        antigen: Antigen,
        response: ImmuneResponse,
        acting_cell: ImmuneCell | None,
        verification_report: dict[str, Any] | None = None,
    ) -> int:
        if not response.selected_artifact or acting_cell is None:
            return 0

        artifact = response.selected_artifact
        verification_report = verification_report or {}
        verified_success = bool(verification_report.get("verified_success", artifact.success))
        raw_success = bool(verification_report.get("raw_success", artifact.success))
        health_delta = max(0.0, float(verification_report.get("health_delta", 0.0) or 0.0))
        repair_gain = 1.0 if verified_success else 0.22 if raw_success else 0.0
        recurrence_reduction = min(0.45, 0.15 + 0.45 * health_delta) if verified_success else 0.0
        recovery_speed = (
            min(1.0, artifact.confidence * (0.35 + 0.90 * health_delta))
            if artifact.executed
            else 0.0
        )
        false_positive_cost = 0.35 if antigen.danger < 0.25 else 0.0
        entropy_cost = 0.18 * response.entropy_pressure + 0.05
        governance_penalty = 0.70 if artifact.governance_denied else 0.0
        verification_penalty = 0.20 if artifact.executed and not verified_success else 0.0
        fitness = (
            repair_gain
            + recurrence_reduction
            + recovery_speed
            - false_positive_cost
            - entropy_cost
            - governance_penalty
            - verification_penalty
        )
        proliferation_count = 0

        with self._lock:
            acting_cell.fitness = 0.68 * acting_cell.fitness + 0.32 * fitness
            self._record_repair_outcome(artifact, antigen, verified_success=verified_success)
            if verified_success:
                acting_cell.successes += 1
                acting_cell.best_effector = artifact.kind
                self._lineage_stats[acting_cell.lineage_id]["successes"] += 1
                self._lineage_stats[acting_cell.lineage_id]["best_effector"] = artifact.kind
                self._lineage_stats[acting_cell.lineage_id]["best_fitness"] = max(
                    float(self._lineage_stats[acting_cell.lineage_id]["best_fitness"]),
                    float(acting_cell.fitness),
                )
                proliferation_count = self._clone_successful_lineages(
                    top_cell=acting_cell,
                    antigen=antigen,
                )
            else:
                acting_cell.failures += 1
                self._lineage_stats[acting_cell.lineage_id]["failures"] += 1
                if artifact.governance_denied:
                    acting_cell.fitness -= 0.15
            self._save_state()
        return proliferation_count

    def _clone_successful_lineages(
        self,
        *,
        top_cell: ImmuneCell,
        antigen: Antigen,
    ) -> int:
        with self._lock:
            if top_cell.kind not in {CellKind.B, CellKind.CYTOTOXIC, CellKind.MEMORY}:
                return 0
            if top_cell.last_activation < self.cfg.activation_threshold:
                return 0

            clones = 0
            target_clones = 1 + int(top_cell.last_activation > 0.75 and antigen.danger > 0.7)
            for _ in range(target_clones):
                if len(self._cells) >= self.cfg.max_population:
                    break
                child = top_cell.clone(
                    rng=self._rng,
                    cell_id=self._new_cell_id(top_cell.kind),
                    mutation_sigma=self.cfg.mutation_sigma,
                )
                child.persistence = min(1.0, child.persistence + self.cfg.persistence_boost)
                child.fitness = max(top_cell.fitness * 0.75, 0.05)
                self._cells.append(child)
                clones += 1
            if clones:
                self._assign_species()
                self._prune_population()
            return clones

    # ------------------------------------------------------------------
    # Artifact generation and execution
    # ------------------------------------------------------------------

    def _emit_artifact(
        self,
        cell: ImmuneCell,
        antigen: Antigen,
        affinity: float,
        activation: float,
    ) -> EffectorArtifact | None:
        kind = None
        notes = ""
        if cell.kind == CellKind.DENDRITIC:
            return None

        if cell.kind in {CellKind.B, CellKind.MEMORY}:
            sig = antigen.error_signature.lower()
            text = f"{antigen.source} {sig}".lower()
            if (
                any(
                    token in text
                    for token in (
                        "zerodivision",
                        "typeerror",
                        "attributeerror",
                        "nameerror",
                        "importerror",
                        "keyerror",
                        "indexerror",
                        "schema drift",
                        "null",
                        "none",
                    )
                )
                and antigen.stack_trace
            ):
                kind = EffectorKind.PATCH_PROPOSAL
            elif "lock" in text or "cache" in text:
                kind = EffectorKind.CLEAR_CACHE
            elif "schema" in text or "migration" in text:
                kind = EffectorKind.SCHEMA_MIGRATION
            elif antigen.resource_pressure > 0.78:
                kind = EffectorKind.REDUCE_LOAD
            elif antigen.error_load > 0.45 or antigen.health_pressure > 0.4:
                kind = EffectorKind.RESTART_COMPONENT
            elif antigen.danger > 0.72:
                kind = EffectorKind.RESTORE_CHECKPOINT
            else:
                kind = cell.best_effector or EffectorKind.PATCH_PROPOSAL

        elif cell.kind == CellKind.CYTOTOXIC:
            if antigen.resource_pressure > 0.86:
                kind = EffectorKind.HALT_RUNAWAY
            elif "tool" in antigen.subsystem or "skill" in antigen.subsystem:
                kind = EffectorKind.REVOKE_TOOL
            else:
                kind = EffectorKind.QUARANTINE

        elif cell.kind == CellKind.REGULATORY:
            return None

        if kind is None:
            return None

        if antigen.protected and kind in {
            EffectorKind.CLEAR_CACHE,
            EffectorKind.REDUCE_LOAD,
            EffectorKind.RESTART_COMPONENT,
            EffectorKind.RESTORE_CHECKPOINT,
            EffectorKind.QUARANTINE,
            EffectorKind.HALT_RUNAWAY,
            EffectorKind.REVOKE_TOOL,
            EffectorKind.SCHEMA_MIGRATION,
        }:
            notes = "protected tissue requires regulatory pass + will approval"

        confidence = max(
            0.0,
            min(
                0.99,
                0.25 + 0.45 * activation + 0.20 * affinity + 0.10 * max(cell.fitness, 0.0),
            ),
        )
        artifact_id = f"eff_{hashlib.sha1(f'{cell.cell_id}:{antigen.antigen_id}:{kind.value}'.encode()).hexdigest()[:12]}"
        return EffectorArtifact(
            artifact_id=artifact_id,
            kind=kind,
            component=antigen.subsystem,
            confidence=confidence,
            source_cell_id=cell.cell_id,
            lineage_id=cell.lineage_id,
            governance_required=True,
            notes=notes,
            bounded_payload={
                "reason": antigen.error_signature or antigen.source,
                "danger": round(float(antigen.danger), 4),
                "subsystem_need": round(float(antigen.subsystem_need), 4),
                "protected": antigen.protected,
                "activation": round(float(activation), 4),
            },
        )

    async def _maybe_execute_artifact(
        self,
        artifact: EffectorArtifact,
        antigen: Antigen,
        *,
        coverage_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        coverage_report = coverage_report or {"coverage_ratio": 0.0}
        coverage_ratio = float(coverage_report.get("coverage_ratio", 0.0) or 0.0)
        if artifact.suppressed:
            return self._default_verification_report(
                status="suppressed",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

        # [HARDENING] Prevent substrate repair for environmental antigens
        if antigen.source_domain == "environment" and artifact.kind in {
            EffectorKind.RESTART_COMPONENT,
            EffectorKind.CLEAR_CACHE,
            EffectorKind.RESTORE_CHECKPOINT,
            EffectorKind.QUARANTINE,
            EffectorKind.HALT_RUNAWAY,
            EffectorKind.SCHEMA_MIGRATION,
            EffectorKind.PATCH_PROPOSAL,
        }:
            artifact.suppressed = True
            artifact.notes = "environmental antigen forbidden from substrate repair"
            return self._default_verification_report(
                status="suppressed",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )
        if artifact.confidence < self.cfg.execution_confidence_floor:
            artifact.notes = artifact.notes or "execution confidence below floor"
            return self._default_verification_report(
                status="low_confidence",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )
        if not self._is_executable_artifact(artifact):
            return self._default_verification_report(
                status="advisory_only",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

        # EVERY executable effect passes one artifact-bound Will decision, not
        # only protected-tissue ones. Restart, restore, quarantine, halt,
        # revoke, and patch effects on ordinary subsystems are consequential
        # state mutations and must be governed too — previously they relied on
        # downstream conventions with no authority decision at this boundary.
        if not self._authorize_protected_action(artifact, antigen):
            artifact.governance_denied = True
            artifact.suppressed = True
            artifact.notes = artifact.notes or "adaptive immune effector denied by Unified Will"
            return self._default_verification_report(
                status="governance_denied",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

        # Behavioral rules are learned in cloned, effect-free world models.
        # They influence lineage fitness and therefore which cell proposes this
        # artifact; they are not a second actuator lane. The former design let
        # speculative persisted rules replace the concrete governed repair.
        # Real repair continues through the bounded patch or autopoiesis path
        # below, where the resulting health change can be measured.

        if artifact.kind == EffectorKind.PATCH_PROPOSAL:
            resilience_mesh = self._get_service("autonomous_resilience_mesh")
            if resilience_mesh is None or not hasattr(resilience_mesh, "attempt_patch_for_antigen"):
                try:
                    from core.resilience.autonomous_repair_executor import (
                        get_autonomous_repair_executor,
                    )

                    patch_result = await get_autonomous_repair_executor().attempt_patch_for_antigen(
                        artifact,
                        antigen,
                    )
                    attempted = bool(patch_result.get("attempted", False))
                    artifact.executed = attempted
                    artifact.success = False
                    if patch_result.get("notes"):
                        artifact.notes = str(patch_result["notes"])
                    return {
                        "status": str(patch_result.get("status", "patch_scheduled")),
                        "raw_success": False,
                        "verified_success": False,
                        "health_before": None,
                        "health_after": None,
                        "health_delta": 0.0,
                        "health_samples": [],
                        "coverage_ratio": round(coverage_ratio, 4),
                        "recurrence_risk": round(
                            max(0.0, min(1.0, antigen.recurrence_pressure)),
                            4,
                        ),
                        "notes": artifact.notes or "patch scheduled through autonomous repair executor",
                    }
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    artifact.notes = artifact.notes or f"patch pipeline unavailable: {exc}"
                    return self._default_verification_report(
                        status="unavailable",
                        coverage_ratio=coverage_ratio,
                        notes=artifact.notes,
                    )
            try:
                patch_result = await resilience_mesh.attempt_patch_for_antigen(artifact, antigen)
                attempted = bool(patch_result.get("attempted", False))
                applied = bool(patch_result.get("applied", False))
                artifact.executed = attempted
                artifact.success = applied
                if patch_result.get("notes"):
                    artifact.notes = str(patch_result["notes"])
                if applied:
                    self._tissue.mark_repair(artifact.component, 0.32)
                # "applied" means the patch pipeline ran — this layer has no
                # post-apply health, regression, or recurrence evidence, so
                # an applied patch is NOT a verified recovery.
                return {
                    "status": str(patch_result.get("status", "patch_attempted")),
                    "raw_success": applied,
                    "verified_success": False,
                    "health_before": None,
                    "health_after": None,
                    "health_delta": 0.0,
                    "health_samples": [],
                    "coverage_ratio": round(coverage_ratio, 4),
                    "recurrence_risk": round(
                        max(
                            0.0, min(1.0, antigen.recurrence_pressure * (0.80 if applied else 1.0))
                        ),
                        4,
                    ),
                    "notes": artifact.notes or "",
                }
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                _record_adaptive_immunity_degradation(
                    exc,
                    action="Marked patch artifact execution failed and returned an execution_error verification report",
                    severity="degraded",
                    extra={
                        "artifact_id": artifact.artifact_id,
                        "artifact_kind": artifact.kind.value,
                        "component": artifact.component,
                        "antigen_id": antigen.antigen_id,
                    },
                )
                artifact.executed = True
                artifact.success = False
                artifact.notes = artifact.notes or f"patch execution failed: {exc}"
                return self._default_verification_report(
                    status="execution_error",
                    coverage_ratio=coverage_ratio,
                    notes=artifact.notes,
                )

        autopoiesis = self._get_service("autopoiesis")
        if not autopoiesis or not hasattr(autopoiesis, "request_repair"):
            artifact.notes = artifact.notes or "autopoiesis repair path unavailable"
            return self._default_verification_report(
                status="unavailable",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

        try:
            from core.cognitive.autopoiesis import RepairStrategy

            strategy = getattr(RepairStrategy, self._artifact_strategy_name(artifact.kind), None)
            if strategy is None:
                return self._default_verification_report(
                    status="unsupported",
                    coverage_ratio=coverage_ratio,
                    notes=f"unsupported strategy for {artifact.kind.value}",
                )
            result = await autopoiesis.request_repair(artifact.component, strategy)
            artifact.executed = True
            raw_success = bool(getattr(result, "success", False))
            health_before = self._coerce_optional_float(getattr(result, "health_before", None))
            health_after = self._coerce_optional_float(getattr(result, "health_after", None))
            health_samples = await self._sample_component_health(artifact.component)
            if health_after is None and health_samples:
                health_after = health_samples[-1]
            verified_success = self._verify_repair_success(
                raw_success=raw_success,
                health_before=health_before,
                health_after=health_after,
                health_samples=health_samples,
            )
            artifact.success = verified_success
            if verified_success:
                self._tissue.mark_repair(artifact.component, 0.40)
            elif artifact.kind in {
                EffectorKind.QUARANTINE,
                EffectorKind.HALT_RUNAWAY,
                EffectorKind.REVOKE_TOOL,
            }:
                self._tissue.mark_quarantine(artifact.component, 0.28)
            verification_report = {
                "status": (
                    "verified_success"
                    if verified_success
                    else "attempted_unverified"
                    if raw_success
                    else "failed"
                ),
                "raw_success": raw_success,
                "verified_success": verified_success,
                "health_before": self._round_optional(health_before),
                "health_after": self._round_optional(health_after),
                "health_delta": round(
                    float((health_after or 0.0) - (health_before or 0.0)),
                    4,
                )
                if health_before is not None and health_after is not None
                else 0.0,
                "health_samples": [round(float(sample), 4) for sample in health_samples],
                "coverage_ratio": round(coverage_ratio, 4),
                "recurrence_risk": round(
                    max(
                        0.0,
                        min(1.0, antigen.recurrence_pressure * (0.55 if verified_success else 1.0)),
                    ),
                    4,
                ),
                "notes": artifact.notes or "",
            }
            if raw_success and not verified_success and not artifact.notes:
                artifact.notes = "repair executed but could not be verified as durable"
                verification_report["notes"] = artifact.notes
            return verification_report
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_adaptive_immunity_degradation(
                exc,
                action="Marked autopoiesis artifact execution failed and returned an execution_error verification report",
                severity="degraded",
                extra={
                    "artifact_id": artifact.artifact_id,
                    "artifact_kind": artifact.kind.value,
                    "component": artifact.component,
                    "antigen_id": antigen.antigen_id,
                },
            )
            artifact.executed = True
            artifact.success = False
            artifact.notes = artifact.notes or f"execution failed: {exc}"
            return self._default_verification_report(
                status="execution_error",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

    async def _sample_component_health(self, component: str) -> list[float]:
        samples: list[float] = []
        checks = max(0, int(self.cfg.verification_checks))
        for idx in range(checks):
            if idx > 0 and self.cfg.verification_interval_s > 0.0:
                await asyncio.sleep(self.cfg.verification_interval_s)
            reading = self._read_component_health(component)
            if reading is not None:
                samples.append(reading)
        return samples

    def _verify_repair_success(
        self,
        *,
        raw_success: bool,
        health_before: float | None,
        health_after: float | None,
        health_samples: list[float],
    ) -> bool:
        if not raw_success:
            return False
        threshold = float(self.cfg.min_verified_health_delta)
        if health_before is None or health_after is None:
            return False
        # A single transient uptick is not verification. Require the FINAL
        # observation to clear the threshold, and when multiple samples
        # exist, require the recovery to have held across the tail of the
        # window (min of the last half must also clear it) — a spike that
        # decays back is a failed repair, not a verified one.
        if (health_after - health_before) < threshold:
            return False
        # Too few samples to distinguish a recovery from a blip. Unverified is
        # the honest answer; the caller already reports actuated_unverified.
        if len(health_samples) < max(2, int(self.cfg.min_verification_samples)):
            return False
        # The observation must also have SPANNED enough time. Samples taken
        # microseconds apart are one measurement repeated.
        observed_window_s = max(0, len(health_samples) - 1) * float(
            self.cfg.verification_interval_s
        )
        if observed_window_s < float(self.cfg.min_verification_window_s):
            return False
        # A spike that decays back is a failed repair, not a verified one:
        # the tail of the window must hold the gain too.
        tail = health_samples[len(health_samples) // 2 :]
        if (min(tail) - health_before) < threshold:
            return False
        return True

    @staticmethod
    def _artifact_strategy_name(kind: EffectorKind) -> str:
        mapping = {
            EffectorKind.CLEAR_CACHE: "CLEAR_CACHE",
            EffectorKind.REDUCE_LOAD: "REDUCE_LOAD",
            EffectorKind.RESTART_COMPONENT: "RESTART_COMPONENT",
            EffectorKind.RESTORE_CHECKPOINT: "RESTORE_CHECKPOINT",
            EffectorKind.QUARANTINE: "ISOLATE",
            EffectorKind.HALT_RUNAWAY: "ISOLATE",
            EffectorKind.REVOKE_TOOL: "ISOLATE",
        }
        return mapping.get(kind, "")

    @staticmethod
    def _is_executable_artifact(artifact: EffectorArtifact) -> bool:
        return artifact.kind in {
            EffectorKind.CLEAR_CACHE,
            EffectorKind.REDUCE_LOAD,
            EffectorKind.RESTART_COMPONENT,
            EffectorKind.RESTORE_CHECKPOINT,
            EffectorKind.QUARANTINE,
            EffectorKind.HALT_RUNAWAY,
            EffectorKind.REVOKE_TOOL,
            EffectorKind.PATCH_PROPOSAL,
        }

    def _default_verification_report(
        self,
        *,
        status: str,
        coverage_ratio: float,
        notes: str = "",
    ) -> dict[str, Any]:
        return {
            "status": status,
            "raw_success": False,
            "verified_success": False,
            "health_before": None,
            "health_after": None,
            "health_delta": 0.0,
            "health_samples": [],
            "coverage_ratio": round(float(coverage_ratio), 4),
            # Nothing ran, so the underlying risk is UNRESOLVED — reporting
            # zero here understated recurrence risk for suppressed,
            # unavailable, and advisory outcomes.
            "recurrence_risk": 0.5,
            "notes": notes,
        }

    def _cell_generation(self, cell_id: str) -> int:
        """Generation of the cell that produced an artifact, or -1 if gone."""
        cell = self._find_cell(cell_id)
        return int(getattr(cell, "clone_generation", -1)) if cell is not None else -1

    def _authorize_protected_action(
        self,
        artifact: EffectorArtifact,
        antigen: Antigen,
    ) -> bool:
        try:
            from core.governance.recovery_authority import (
                build_internal_recovery_context,
            )
            from core.runtime.action_executor import ActionExecutor
            from core.will import ActionDomain

            # CP126 81f0c6a0: Will saw only component, kind, danger and
            # lineage — it authorized a CATEGORY of action, not the action.
            # The exact parameters, the behavioural rule that produced them,
            # the cell generation, and the strategy were all outside the
            # decision, so an approval for "restart component X" covered any
            # payload that later arrived under that description. The payload
            # digest binds the approval to the concrete effect, and the
            # decision is refused if the payload changes afterwards.
            payload_digest = _artifact_payload_digest(artifact)
            recovery_context = build_internal_recovery_context(
                "adaptive_immune_system",
                artifact.kind.value,
                evidence={
                    "component": artifact.component,
                    "artifact_id": artifact.artifact_id,
                    "artifact_kind": artifact.kind.value,
                    "antigen_id": antigen.antigen_id,
                    "danger": antigen.danger,
                    "protected": antigen.protected,
                    "lineage_id": artifact.lineage_id,
                    # What is actually going to be done, not just to what.
                    "payload_digest": payload_digest,
                    "bounded_payload": _json_safe(artifact.bounded_payload),
                    "repair_strategy": self._artifact_strategy_name(artifact.kind),
                    "source_cell_id": artifact.source_cell_id,
                    "source_cell_generation": self._cell_generation(artifact.source_cell_id),
                    "confidence": round(float(artifact.confidence), 4),
                    "subsystem": antigen.subsystem,
                    "source_domain": antigen.source_domain,
                    "error_signature": antigen.error_signature[:200],
                },
            )
            admission = ActionExecutor.authorize_action(
                action_name=f"adaptive_immune.{artifact.kind.value}",
                params={
                    "artifact_id": artifact.artifact_id,
                    "component": artifact.component,
                    "payload_digest": payload_digest,
                },
                source="adaptive_immune_system",
                domain=ActionDomain.STATE_MUTATION,
                priority=min(0.95, 0.45 + 0.35 * antigen.danger),
                context=recovery_context,
            )
            if not admission.approved:
                return False
            # The approval covered THIS payload. If anything mutated the
            # artifact between the decision and here, the approval no longer
            # describes what would run.
            if _artifact_payload_digest(artifact) != payload_digest:
                _record_adaptive_immunity_degradation(
                    RuntimeError("artifact payload changed after authorization"),
                    action="refused an immune effector whose payload changed after the Will decision",
                    severity="degraded",
                    extra={"artifact_id": artifact.artifact_id},
                )
                return False
            return True
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_adaptive_immunity_degradation(
                exc,
                action="Denied protected adaptive immune action because authorization was unavailable",
                severity="degraded",
                extra={
                    "artifact_id": artifact.artifact_id,
                    "artifact_kind": artifact.kind.value,
                    "component": artifact.component,
                    "antigen_id": antigen.antigen_id,
                },
            )
            logger.debug("Protected-action authorization unavailable: %s", exc)
            return False

    def _assess_coverage(
        self,
        event: dict[str, Any],
        antigen: Antigen,
        *,
        anomaly_score: Any | None,
        state_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # CP126 064f40ec: every channel counted mere PRESENCE — any non-None
        # anomaly object, any truthy snapshot, a fuzzy monitor-name match.
        # Coverage is the number this subsystem reports as "how well was this
        # observed", and presence is not observation: a channel that exists
        # but is empty, stale, or carries no reading tells you nothing while
        # raising the score that says it did.
        component_matches = self._component_monitor_matches(antigen.subsystem)
        channels = {
            # An anomaly object only corroborates if it actually carries a
            # reading.
            "anomaly_model": _anomaly_score_is_substantive(anomaly_score),
            "subsystem_identity": antigen.subsystem not in {"", "unknown"},
            "error_telemetry": antigen.error_load > 0.0
            or bool(event.get("error_count"))
            or bool(antigen.error_signature),
            "resource_telemetry": any(key in event for key in ("resource_pressure", "cpu", "ram")),
            "health_probe": bool(component_matches),
            "causal_trace": bool(
                event.get("stack_trace") or event.get("causal_trace") or antigen.stack_trace
            ),
            # A snapshot must have content and, when it declares its age, be
            # recent enough to describe the event it is attached to.
            "state_snapshot": _snapshot_is_usable(state_snapshot),
            # CP126 7c08abf3: a SINGLE event was credited as temporal history.
            # One observation is not longitudinal, and calling it that made
            # first-sightings look as well-understood as recurring failures.
            "temporal_history": antigen.recurrence_pressure > 0.0
            or self._recent_subsystem_count(antigen.subsystem) >= _MIN_TEMPORAL_HISTORY_EVENTS,
        }
        coverage_ratio = sum(1.0 for present in channels.values() if present) / max(
            len(channels), 1
        )
        missing_channels = [name for name, present in channels.items() if not present]
        blind_spots: list[str] = []
        if "health_probe" in missing_channels:
            blind_spots.append("no direct health probe for this subsystem")
        if "causal_trace" in missing_channels:
            blind_spots.append("no stack or causal trace")
        if "anomaly_model" in missing_channels:
            blind_spots.append("no anomaly-model corroboration")
        if "state_snapshot" in missing_channels:
            blind_spots.append("no rich state snapshot for this observation")
        if "temporal_history" in missing_channels:
            blind_spots.append("little longitudinal history for this subsystem")

        if coverage_ratio >= 0.8:
            coverage_label = "strong"
        elif coverage_ratio >= 0.55:
            coverage_label = "moderate"
        else:
            coverage_label = "thin"

        return {
            "coverage_ratio": round(float(coverage_ratio), 4),
            "coverage_label": coverage_label,
            "observed_channels": [name for name, present in channels.items() if present],
            "missing_channels": missing_channels,
            "known_blind_spots": blind_spots,
            "monitored_components": component_matches,
            "system_coverage": self._system_coverage_summary(),
        }

    def _apply_coverage_constraints(
        self,
        response: ImmuneResponse,
        antigen: Antigen,
        coverage_report: dict[str, Any],
    ) -> None:
        coverage_ratio = float(coverage_report.get("coverage_ratio", 0.0) or 0.0)
        risky_kinds = {
            EffectorKind.RESTART_COMPONENT,
            EffectorKind.RESTORE_CHECKPOINT,
            EffectorKind.REVOKE_TOOL,
            EffectorKind.SCHEMA_MIGRATION,
        }
        for artifact in response.artifacts:
            artifact.confidence = max(
                0.0, min(0.99, artifact.confidence * (0.55 + 0.45 * coverage_ratio))
            )
            if (
                coverage_ratio < self.cfg.low_coverage_floor
                and artifact.kind in risky_kinds
            ):
                # Danger alone must not buy a bypass: with health and causal
                # visibility missing, a high danger score is itself
                # low-confidence evidence. Risky actions under low
                # observability require at least SOME coverage even at
                # extreme danger.
                # A ratio floor alone can be met by any two channels — the
                # two that MATTER for a risky action are a direct health
                # probe and a causal trace, and the finding names exactly
                # those. Without at least one of them, "danger 0.88" is a
                # number produced from telemetry nobody corroborated, and
                # restart/restore/revoke/migrate are irreversible enough that
                # it must not authorize itself.
                observed = set(coverage_report.get("observed_channels") or [])
                has_grounding = bool(observed & {"health_probe", "causal_trace"})
                extreme_danger_with_minimum_visibility = (
                    antigen.danger >= 0.88 and coverage_ratio >= 0.25 and has_grounding
                )
                if not extreme_danger_with_minimum_visibility:
                    artifact.suppressed = True
                    artifact.notes = artifact.notes or "suppressed under low observability"

    def _execution_candidates(self, response: ImmuneResponse) -> list[EffectorArtifact]:
        candidates = [
            artifact
            for artifact in response.artifacts
            if not artifact.suppressed and self._is_executable_artifact(artifact)
        ]
        candidates.sort(key=lambda artifact: artifact.confidence, reverse=True)
        return candidates

    def _best_visible_artifact(self, response: ImmuneResponse) -> EffectorArtifact | None:
        visible = [artifact for artifact in response.artifacts if not artifact.suppressed]
        if not visible:
            return None
        return max(visible, key=lambda artifact: artifact.confidence)

    def _build_diagnostic_verdict(
        self,
        antigen: Antigen,
        response: ImmuneResponse,
        *,
        coverage_report: dict[str, Any],
        verification_report: dict[str, Any],
    ) -> dict[str, Any]:
        coverage_ratio = float(coverage_report.get("coverage_ratio", 0.0) or 0.0)
        verification_status = str(verification_report.get("status", "not_executed"))
        verified_success = bool(verification_report.get("verified_success", False))
        # Only INPUT signals count as evidence. The immune model's own cell
        # activation is an output of this same model — counting it alongside
        # the inputs double-counted one inference as corroboration.
        evidence_count = sum(
            1
            for present in (
                antigen.error_load > 0.08,
                antigen.health_pressure > 0.12,
                antigen.resource_pressure > 0.45,
                antigen.recurrence_pressure > 0.3,
                bool(antigen.stack_trace),
            )
            if present
        )
        issue_confirmed = evidence_count >= 2 and antigen.danger >= 0.28
        escalation_recommended = False

        observed_channels = set(coverage_report.get("observed_channels") or [])
        if verified_success:
            status = "verified_recovery"
            # A verified recovery already required health samples, so the
            # probe is implied — asserting it explicitly keeps all_clear
            # keyed on a DIRECT observation rather than on that implication
            # surviving future edits (CP126 37f929c1).
            all_clear = (
                coverage_ratio >= 0.7
                and antigen.recurrence_pressure < 0.45
                and "health_probe" in observed_channels
            )
        elif verification_status in {"failed", "execution_error"}:
            status = "persistent_issue"
            all_clear = False
            escalation_recommended = antigen.danger >= 0.45 or antigen.recurrence_pressure >= 0.35
        elif verification_status == "attempted_unverified":
            status = "repair_attempted_unverified"
            all_clear = False
            escalation_recommended = True
        elif issue_confirmed:
            status = "confirmed_issue"
            all_clear = False
        elif antigen.danger >= 0.48 or antigen.recurrence_pressure >= 0.4:
            status = "suspected_issue"
            all_clear = False
        elif coverage_ratio >= 0.75 and "health_probe" in observed_channels:
            # All-clear requires a DIRECT component probe — channel presence
            # alone (telemetry existing) is not evidence the component is well.
            status = "no_confirmed_issue_under_current_visibility"
            all_clear = True
        else:
            status = "no_confirmed_issue_under_limited_visibility"
            all_clear = False

        confidence = max(
            0.05,
            min(
                0.98,
                0.30
                + 0.45 * coverage_ratio
                + 0.20 * min(1.0, evidence_count / 4.0)
                + 0.05 * float(verified_success),
            ),
        )
        summary = self._summarize_verdict(
            status=status,
            antigen=antigen,
            coverage_report=coverage_report,
            verification_report=verification_report,
        )
        return {
            "status": status,
            "all_clear": all_clear,
            "confidence": round(float(confidence), 4),
            "issue_confirmed": issue_confirmed,
            "repair_verified": verified_success,
            "coverage_ratio": round(coverage_ratio, 4),
            "coverage_label": coverage_report.get("coverage_label", "thin"),
            "evidence_count": evidence_count,
            "escalation_recommended": escalation_recommended,
            "known_blind_spots": list(coverage_report.get("known_blind_spots", [])),
            "summary": summary,
        }

    def _summarize_verdict(
        self,
        *,
        status: str,
        antigen: Antigen,
        coverage_report: dict[str, Any],
        verification_report: dict[str, Any],
    ) -> str:
        coverage_label = coverage_report.get("coverage_label", "thin")
        if status == "verified_recovery":
            return f"issue was detected in {antigen.subsystem} and a bounded repair verified successfully under {coverage_label} coverage"
        if status == "persistent_issue":
            return f"repair did not hold for {antigen.subsystem}; recurrence or low health still indicates a persistent issue"
        if status == "repair_attempted_unverified":
            return f"repair was attempted in {antigen.subsystem}, but success could not be verified under current visibility"
        if status == "confirmed_issue":
            return f"multiple signals confirm an active issue in {antigen.subsystem}"
        if status == "suspected_issue":
            return (
                f"signals suggest risk in {antigen.subsystem}, but confirmation remains incomplete"
            )
        if status == "no_confirmed_issue_under_current_visibility":
            return (
                f"no confirmed issue was observed in {antigen.subsystem} under current visibility"
            )
        return f"no confirmed issue was observed in {antigen.subsystem}, but visibility is limited and blind spots remain"

    def _record_response_summary(self, response: ImmuneResponse) -> None:
        with self._lock:
            if self._recent_responses:
                self._recent_responses[-1].update(
                    {
                        "coverage_ratio": response.coverage_report.get("coverage_ratio", 0.0),
                        "verdict": response.diagnostic_verdict.get("status"),
                        "verification": response.verification_report.get("status"),
                        "all_clear": response.diagnostic_verdict.get("all_clear", False),
                    }
                )
            # The last write of an observation is the durable one: the
            # intermediate writes above coalesce into it, so one event
            # costs one snapshot instead of three (CP126 df9f2a05)
            # without giving up recurrence memory across a reload.
            self._save_state(force=True)

    def _component_monitor_matches(self, subsystem: str) -> list[str]:
        """Match a subsystem to registered health monitors — exact-first.

        Loose substring / shared-prefix matching bound the WRONG health probe
        (e.g. "memory" matched "memory_guard"), producing false coverage and
        verification against another subsystem. Exact identity wins outright;
        containment is accepted only for full underscore-token prefixes
        ("llm_router" ↔ "llm_router_gate"), never bare first-token overlap.
        """
        if not subsystem:
            return []
        autopoiesis = self._get_service("autopoiesis")
        health_fns = getattr(autopoiesis, "_health_fns", {}) if autopoiesis is not None else {}
        lowered = subsystem.lower()
        exact: list[str] = []
        token_prefix: list[str] = []
        for component in health_fns:
            candidate = str(component).lower()
            if candidate == lowered:
                exact.append(str(component))
                continue
            if candidate.startswith(lowered + "_") or lowered.startswith(candidate + "_"):
                token_prefix.append(str(component))
        if exact:
            return sorted(set(exact))
        return sorted(set(token_prefix))

    def _system_coverage_summary(self) -> dict[str, Any]:
        active_subsystems = {
            subsystem
            for subsystem in set(self._recent_subsystem_events) | set(self._tissue._edges)
            if subsystem and subsystem != "unknown"
        }
        monitored = set()
        autopoiesis = self._get_service("autopoiesis")
        if autopoiesis is not None:
            monitored = {str(name) for name in getattr(autopoiesis, "_health_fns", {}).keys()}
        covered = {
            subsystem
            for subsystem in active_subsystems
            if self._component_monitor_matches(subsystem)
        }
        coverage_ratio = len(covered) / max(len(active_subsystems), 1)
        uncovered = sorted(active_subsystems - covered)[:8]
        return {
            "active_components": len(active_subsystems),
            "monitored_components": len(monitored),
            "covered_active_components": len(covered),
            "coverage_ratio": round(float(coverage_ratio), 4),
            "uncovered_hotspots": uncovered,
        }

    def _recurrence_hotspots(self, limit: int = 6) -> list[dict[str, Any]]:
        hotspots: list[tuple[float, str, dict[str, Any]]] = []
        for key, stats in self._recurrence_tracker.items():
            if not key.startswith("subsystem::"):
                continue
            subsystem = key.split("::", 1)[1]
            pressure = self._estimate_recurrence_pressure(subsystem, "")
            if pressure <= 0.0:
                continue
            hotspots.append((pressure, subsystem, stats))
        hotspots.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "subsystem": subsystem,
                "pressure": round(float(pressure), 4),
                "occurrences": int(stats.get("occurrences", 0)),
                "streak": int(stats.get("streak", 0)),
                "verified_repairs": int(stats.get("verified_repairs", 0)),
                "failed_repairs": int(stats.get("failed_repairs", 0)),
            }
            for pressure, subsystem, stats in hotspots[:limit]
        ]

    def _read_component_health(self, subsystem: str) -> float | None:
        autopoiesis = self._get_service("autopoiesis")
        if autopoiesis is None or not hasattr(autopoiesis, "get_component_health"):
            return None
        matches = self._component_monitor_matches(subsystem)
        for component in matches:
            try:
                raw = float(autopoiesis.get_component_health(component))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue
            # CP126 adbedaea. min/max propagate NaN silently — max(0.0, nan)
            # is nan — so a non-finite health reading passed straight through
            # into comparisons and persisted fitness, where every comparison
            # against it is False and the value looks benign forever.
            # Unknown health is None, which callers already handle; it is not
            # a number.
            if not math.isfinite(raw):
                record_degradation(
                    "adaptive_immunity",
                    ValueError(
                        f"non-finite health for component {component!r}: {raw!r}"
                    ),
                    severity="warning",
                    action="treated a non-finite component health reading as unknown",
                )
                continue
            return max(0.0, min(1.0, raw))
        return None

    def _estimate_recurrence_pressure(self, subsystem: str, error_signature: str) -> float:
        keys = self._recurrence_keys(subsystem, error_signature)
        if not keys:
            return 0.0
        pressures: list[float] = []
        for key in keys:
            stats = self._recurrence_tracker.get(key)
            if not stats:
                continue
            occurrences = float(stats.get("occurrences", 0))
            streak = float(stats.get("streak", 0))
            interval_ewma = float(stats.get("interval_ewma", 0.0) or 0.0)
            verified = float(stats.get("verified_repairs", 0))
            failed = float(stats.get("failed_repairs", 0))
            # CP126 61cc8648: occurrences and streak were LIFETIME totals, so
            # after six events the count term pinned at 1.0 and stayed there
            # for the life of the process — recurrence pressure stopped
            # meaning "this is happening again" and started meaning "this
            # happened six times once". A subsystem that failed a lot last
            # year looked identical to one failing right now.
            last_seen = float(stats.get("last_seen", 0.0) or 0.0)
            window = max(float(self.cfg.recurrence_window_s), 1.0)
            age = max(0.0, time.time() - last_seen) if last_seen > 0.0 else window
            # Linear decay to zero across one window: history informs, but
            # only recent history is pressure.
            recency = max(0.0, 1.0 - (age / window))

            count_term = min(1.0, occurrences / 6.0) * recency
            streak_term = min(1.0, streak / 4.0) * recency
            interval_term = 0.0
            if interval_ewma > 0.0:
                interval_term = 1.0 - min(1.0, interval_ewma / window)
            repair_term = failed / max(verified + failed + 1.0, 1.0)
            pressure = (
                0.35 * count_term + 0.25 * streak_term + 0.20 * interval_term + 0.20 * repair_term
            )
            # A verified repair MORE RECENT than the last occurrence opens a
            # healthy epoch. Decrementing the streak by one left the old
            # history dominating, so a fixed subsystem kept reporting the
            # pressure of the problem that was fixed.
            last_verified_at = float(stats.get("last_verified_at", 0.0) or 0.0)
            if last_verified_at > last_seen > 0.0:
                pressure *= 0.35
            pressures.append(pressure)
        return float(max(pressures, default=0.0))

    def _record_recurrence_observation(self, antigen: Antigen) -> None:
        if (
            antigen.danger < 0.18
            and antigen.error_load <= 0.0
            and antigen.health_pressure <= 0.0
            and antigen.resource_pressure < 0.45
        ):
            return
        now = antigen.timestamp
        for key in self._recurrence_keys(antigen.subsystem, antigen.error_signature):
            stats = self._recurrence_tracker[key]
            last_seen = float(stats.get("last_seen", 0.0) or 0.0)
            interval = now - last_seen if last_seen > 0.0 else None
            stats["occurrences"] = int(stats.get("occurrences", 0)) + 1
            stats["last_seen"] = now
            if interval is not None and interval >= 0.0:
                prev_ewma = float(stats.get("interval_ewma", 0.0) or 0.0)
                stats["last_interval"] = interval
                stats["interval_ewma"] = (
                    interval if prev_ewma <= 0.0 else 0.7 * prev_ewma + 0.3 * interval
                )
                if interval <= self.cfg.recurrence_window_s:
                    stats["streak"] = int(stats.get("streak", 0)) + 1
                else:
                    stats["streak"] = 1
            else:
                stats["streak"] = max(1, int(stats.get("streak", 0)))
            stats["peak_streak"] = max(
                int(stats.get("peak_streak", 0)), int(stats.get("streak", 0))
            )

    def _record_repair_outcome(
        self,
        artifact: EffectorArtifact,
        antigen: Antigen,
        *,
        verified_success: bool,
    ) -> None:
        now = time.time()
        for key in self._recurrence_keys(antigen.subsystem, antigen.error_signature):
            stats = self._recurrence_tracker[key]
            if verified_success:
                stats["verified_repairs"] = int(stats.get("verified_repairs", 0)) + 1
                stats["last_verified_at"] = now
                stats["streak"] = max(0, int(stats.get("streak", 0)) - 1)
            else:
                stats["failed_repairs"] = int(stats.get("failed_repairs", 0)) + 1

    @staticmethod
    def _recurrence_keys(subsystem: str, error_signature: str) -> list[str]:
        keys = [f"subsystem::{subsystem}"]
        if error_signature:
            keys.append(f"signature::{subsystem}::{error_signature.lower()}")
        return keys

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _round_optional(value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 4)

    # ------------------------------------------------------------------
    # Species, metabolism, persistence
    # ------------------------------------------------------------------

    def _assign_species(self) -> None:
        if len(self._cells) < 4:
            self._species_count = 1
            for cell in self._cells:
                cell.species_id = 0
            return

        vectors = np.asarray([cell.receptor for cell in self._cells], dtype=np.float32)
        max_k = min(self.cfg.species_max_k, len(self._cells) - 1)
        best_labels = np.zeros(len(self._cells), dtype=np.int32)
        best_score = -1.0
        best_k = 1

        for k in range(self.cfg.species_min_k, max_k + 1):
            labels = self._kmeans(vectors, k)
            score = self._silhouette_score(vectors, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        if best_score < self.cfg.species_silhouette_floor:
            best_labels = np.zeros(len(self._cells), dtype=np.int32)
            best_k = 1

        for idx, cell in enumerate(self._cells):
            cell.species_id = int(best_labels[idx])
        self._species_count = best_k

    def _prune_population(self) -> None:
        if len(self._cells) <= self.cfg.max_population:
            return

        keep: list[ImmuneCell] = []
        by_species: dict[int, list[ImmuneCell]] = defaultdict(list)
        for cell in self._cells:
            by_species[cell.species_id].append(cell)

        for species_cells in by_species.values():
            best = max(
                species_cells,
                key=lambda cell: (cell.fitness, cell.persistence, cell.successes - cell.failures),
            )
            keep.append(best)

        remaining = [cell for cell in self._cells if cell not in keep]
        remaining.sort(
            key=lambda cell: (
                cell.kind == CellKind.MEMORY,
                cell.kind == CellKind.REGULATORY,
                cell.fitness,
                cell.persistence,
                cell.successes - cell.failures,
            ),
            reverse=True,
        )
        slots = max(0, self.cfg.max_population - len(keep))
        self._cells = keep + remaining[:slots]

    def _metabolic_context(self) -> tuple[float, float]:
        vitality = 0.72
        metabolism = 0.65
        entropy_pressure = 0.0

        homeostasis = self._get_service("homeostasis") or self._get_service("homeostatic_coupling")
        if homeostasis is not None:
            try:
                if hasattr(homeostasis, "compute_vitality"):
                    vitality = float(homeostasis.compute_vitality())
                metabolism = float(getattr(homeostasis, "metabolism", metabolism))
            except (RuntimeError, AttributeError, TypeError):
                pass  # no-op: intentional

        alife_dynamics = self._get_service("alife_dynamics")
        if alife_dynamics is not None:
            try:
                status = (
                    alife_dynamics.get_status() if hasattr(alife_dynamics, "get_status") else {}
                )
                entropy_pressure = float(
                    status.get("entropy_pressure")
                    or status.get("pressure")
                    or status.get("entropy", 0.0) / max(status.get("max_entropy", 100.0), 1.0)
                )
            except (OSError, ConnectionError, TimeoutError):
                entropy_pressure = 0.0

        scale = max(
            0.10,
            min(
                1.20,
                0.25
                + 0.75 * vitality * (0.55 + 0.45 * metabolism) * (1.0 - 0.45 * entropy_pressure),
            ),
        )
        return float(scale), float(max(0.0, min(1.0, entropy_pressure)))

    def _save_state(self, *, force: bool = False) -> None:
        """Persist the immune ecology, coalescing bursts.

        CP126 df9f2a05: core observation writes state, reinforcement can write
        again, and the response summary writes again — so ONE event
        serialized the whole ecology several times. During a failure storm,
        which is exactly when many events arrive at once, that multiplied I/O
        and lock time in the subsystem meant to be responding to the storm.

        Writes inside the coalescing interval are deferred, not dropped: the
        dirty flag survives and the next call past the interval writes the
        latest state. The honest cost is that a crash can lose up to
        ``_save_min_interval_s`` of fitness updates — which is a far better
        trade than amplifying the storm that causes the crash, and callers
        that need a durable point (boot seeding, consolidation) pass
        force=True.
        """
        now = time.time()
        self._state_dirty = True
        if not force and (now - self._last_save_at) < self._save_min_interval_s:
            self._deferred_saves += 1
            return
        self._last_save_at = now
        self._state_dirty = False
        payload = {
            "cells": [cell.to_dict() for cell in self._cells],
            "tissue": self._tissue.to_dict(),
            "lineage_stats": {
                lineage_id: {
                    "successes": int(stats["successes"]),
                    "failures": int(stats["failures"]),
                    "best_effector": (
                        stats["best_effector"].value
                        if isinstance(stats["best_effector"], EffectorKind)
                        else None
                    ),
                    "best_fitness": float(stats["best_fitness"]),
                }
                for lineage_id, stats in self._lineage_stats.items()
            },
            "observation_count": self._observation_count,
            "last_dream_at": self._last_dream_at,
            "recent_antigens": [antigen.to_dict() for antigen in list(self._recent_antigens)[-24:]],
            "recent_responses": list(self._recent_responses)[-24:],
            "recurrence_tracker": {
                key: {
                    "occurrences": int(stats.get("occurrences", 0)),
                    "last_seen": float(stats.get("last_seen", 0.0)),
                    "interval_ewma": float(stats.get("interval_ewma", 0.0)),
                    "last_interval": (
                        float(stats["last_interval"])
                        if stats.get("last_interval") is not None
                        else None
                    ),
                    "streak": int(stats.get("streak", 0)),
                    "peak_streak": int(stats.get("peak_streak", 0)),
                    "verified_repairs": int(stats.get("verified_repairs", 0)),
                    "failed_repairs": int(stats.get("failed_repairs", 0)),
                    "last_verified_at": float(stats.get("last_verified_at", 0.0)),
                }
                for key, stats in self._recurrence_tracker.items()
            },
            "expansion_engine": self.expansion_engine.to_dict(),
        }
        payload["schema_version"] = IMMUNE_STATE_SCHEMA_VERSION
        # An unkeyed digest over the body: it detects CORRUPTION and truncation,
        # not tampering by anyone who can write the file. Labelled as what it
        # is rather than as a trust root — a signed state file needs a key this
        # subsystem does not hold (CP126 5c214831).
        payload["integrity"] = {
            "algorithm": "sha256-unkeyed",
            "digest": _immune_state_digest(payload),
        }
        try:
            # Route through the governed file-write gateway: a repair-capable,
            # behavior-evolving state file is a consequential write and must
            # be authorized and receipt-bound like every other one.
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope(
                "adaptation.adaptive_immunity.state",
                domain="file_write",
                receipt_prefix="adaptive-immunity-state",
            ):
                get_file_write_gateway().write_text(
                    self._state_path,
                    json.dumps(payload, indent=2),
                    source="adaptation.adaptive_immunity.state",
                )
        except (ImportError, OSError, RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _record_adaptive_immunity_degradation(
                exc,
                action="Skipped adaptive immune persistence write and kept in-memory immune state active",
                extra={"state_path": str(self._state_path), "cells": len(self._cells)},
            )
            logger.debug("Adaptive immune state save skipped: %s", exc)

    def _load_state(self) -> bool:
        if not self._state_path.exists():
            return False
        try:
            size = self._state_path.stat().st_size
            if size > MAX_IMMUNE_STATE_BYTES:
                raise ValueError(
                    f"immune state file is {size} bytes, over the "
                    f"{MAX_IMMUNE_STATE_BYTES} bound"
                )
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("immune state must be a JSON object")
            # A file written by a different layout is quarantined to a reseed
            # rather than parsed field-by-field into a live repair-capable
            # population (CP126 5c214831).
            found_version = int(payload.get("schema_version", 0) or 0)
            if found_version != IMMUNE_STATE_SCHEMA_VERSION:
                raise ValueError(
                    f"immune state schema {found_version} != "
                    f"{IMMUNE_STATE_SCHEMA_VERSION}"
                )
            integrity = payload.get("integrity")
            if isinstance(integrity, dict) and integrity.get("digest"):
                if _immune_state_digest(payload) != str(integrity["digest"]):
                    raise ValueError("immune state digest does not match its contents")
            if "expansion_engine" in payload:
                from core.adaptation.dimensional_expansion import DimensionalExpansionEngine

                self.expansion_engine = DimensionalExpansionEngine.from_dict(
                    payload["expansion_engine"]
                )

            self._cells = [ImmuneCell.from_dict(item) for item in payload.get("cells", [])]
            vocabulary = _live_rule_vocabulary()
            migrated_rules = 0
            for cell in self._cells:
                if cell.kind not in {CellKind.B, CellKind.MEMORY}:
                    if cell.behavioral_rule is not None:
                        cell.behavioral_rule = None
                        migrated_rules += 1
                    continue
                normalized, migrated = _normalize_behavioral_rule(
                    cell.behavioral_rule,
                    self._rng,
                    vocabulary=vocabulary,
                )
                cell.behavioral_rule = normalized
                migrated_rules += int(migrated)
            self._migrated_behavioral_rules = migrated_rules
            if migrated_rules:
                logger.info(
                    "Migrated %d persisted immune behavioral rule(s) to the bounded grammar",
                    migrated_rules,
                )

            # Reconcile receptor vectors of loaded cells with system current_dim
            target_dim = self.expansion_engine.current_dim
            for cell in self._cells:
                cell.resize_receptor(target_dim, self._rng)

            self._tissue = TissueField.from_dict(
                payload.get("tissue", {}),
                diffusion=self.cfg.tissue_diffusion,
                decay=self.cfg.tissue_decay,
            )
            self._lineage_stats = defaultdict(
                lambda: {
                    "successes": 0,
                    "failures": 0,
                    "best_effector": None,
                    "best_fitness": 0.0,
                }
            )
            for lineage_id, stats in payload.get("lineage_stats", {}).items():
                self._lineage_stats[lineage_id] = {
                    "successes": int(stats.get("successes", 0)),
                    "failures": int(stats.get("failures", 0)),
                    "best_effector": (
                        EffectorKind(stats["best_effector"]) if stats.get("best_effector") else None
                    ),
                    "best_fitness": float(stats.get("best_fitness", 0.0)),
                }
            self._observation_count = int(payload.get("observation_count", 0))
            self._last_dream_at = int(payload.get("last_dream_at", 0))
            self._recent_antigens = deque(
                [Antigen.from_dict(item) for item in payload.get("recent_antigens", [])],
                maxlen=self.cfg.replay_buffer_size,
            )
            self._recent_responses = deque(
                [dict(item) for item in payload.get("recent_responses", [])],
                maxlen=self.cfg.recent_response_buffer,
            )
            self._recurrence_tracker = defaultdict(
                lambda: {
                    "occurrences": 0,
                    "last_seen": 0.0,
                    "interval_ewma": 0.0,
                    "last_interval": None,
                    "streak": 0,
                    "peak_streak": 0,
                    "verified_repairs": 0,
                    "failed_repairs": 0,
                    "last_verified_at": 0.0,
                }
            )
            for key, stats in payload.get("recurrence_tracker", {}).items():
                self._recurrence_tracker[str(key)] = {
                    "occurrences": int(stats.get("occurrences", 0)),
                    "last_seen": float(stats.get("last_seen", 0.0)),
                    "interval_ewma": float(stats.get("interval_ewma", 0.0)),
                    "last_interval": self._coerce_optional_float(stats.get("last_interval")),
                    "streak": int(stats.get("streak", 0)),
                    "peak_streak": int(stats.get("peak_streak", 0)),
                    "verified_repairs": int(stats.get("verified_repairs", 0)),
                    "failed_repairs": int(stats.get("failed_repairs", 0)),
                    "last_verified_at": float(stats.get("last_verified_at", 0.0)),
                }
            self._assign_species()
            return bool(self._cells)
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            # Corrupt or hostile persisted state must quarantine to a reseed,
            # never abort immune construction: JSON, schema, enum, and
            # numeric failures were previously uncaught here.
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            _record_adaptive_immunity_degradation(
                exc,
                action="Rejected persisted adaptive immune state and reseeded immune population",
                severity="degraded",
                extra={"state_path": str(self._state_path)},
            )
            logger.warning("Adaptive immune state load failed; reseeding: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _seed_population(self) -> list[ImmuneCell]:
        seeds: list[ImmuneCell] = []
        scopes = [
            "llm_router",
            "state_repository",
            "continuity",
            "memory_guard",
            "voice_engine",
            "identity",
            "prompt_boundary",
            "tool_boundary",
            "memory_boundary",
            "generic",
        ]
        kind_sequence = (
            [CellKind.DENDRITIC] * 5
            + [CellKind.B] * 7
            + [CellKind.CYTOTOXIC] * 7
            + [CellKind.REGULATORY] * 5
        )
        for idx, kind in enumerate(kind_sequence[: self.cfg.population_size]):
            receptor = self._seed_receptor(kind)
            rule = None
            if kind in {CellKind.B, CellKind.MEMORY}:
                rule = _mutate_behavioral_rule(None, self._rng)
            seeds.append(
                ImmuneCell(
                    cell_id=self._new_cell_id(kind),
                    lineage_id=f"{kind.value}_{idx}",
                    kind=kind,
                    receptor=receptor,
                    subsystem_scope=scopes[idx % len(scopes)],
                    persistence=0.58 if kind != CellKind.REGULATORY else 0.72,
                    regulatory_strength=1.25 if kind == CellKind.REGULATORY else 1.0,
                    behavioral_rule=rule,
                )
            )
        return seeds

    def _seed_receptor(self, kind: CellKind) -> np.ndarray:
        current_dim = self.expansion_engine.current_dim
        vec = self._rng.uniform(0.05, 0.55, size=current_dim).astype(np.float32)
        if kind == CellKind.DENDRITIC:
            vec[8] = 0.65
            vec[10] = 0.55
        elif kind == CellKind.B:
            vec[8] = 0.72
            vec[9] = 0.42
            vec[10] = 0.70
        elif kind == CellKind.CYTOTOXIC:
            vec[8] = 0.92
            vec[9] = 0.75
            vec[10] = 0.84
            vec[12] = 0.70
        elif kind == CellKind.REGULATORY:
            vec[8] = 0.55
            vec[11] = 0.98
            vec[12] = 0.60
            vec[13] = 0.52
        elif kind == CellKind.MEMORY:
            vec[8] = 0.80
            vec[10] = 0.65
        return np.clip(vec, 0.0, 1.0)

    def _affinity(self, cell: ImmuneCell, antigen: Antigen) -> float:
        affinity = self.compute_affinity_static(
            cell.receptor,
            antigen.vector,
            tau=self.cfg.tau,
            weights=self.expansion_engine.feature_weights.get(),
        )
        if cell.subsystem_scope == antigen.subsystem:
            affinity *= 1.15
        elif cell.subsystem_scope != "generic":
            cell_hint = cell.subsystem_scope.split("_", 1)[0]
            antigen_hint = antigen.subsystem.split("_", 1)[0]
            if cell_hint == antigen_hint:
                affinity *= 1.06
        if cell.kind == CellKind.REGULATORY and antigen.protected:
            affinity *= 1.25
        if cell.kind in {CellKind.B, CellKind.CYTOTOXIC} and antigen.protected:
            affinity *= 0.82
        return float(max(0.0, min(1.25, affinity)))

    def _activation(
        self,
        cell: ImmuneCell,
        antigen: Antigen,
        affinity: float,
        metabolic_scale: float,
    ) -> float:
        activation = affinity * antigen.danger * max(0.12, antigen.subsystem_need) * metabolic_scale
        if cell.kind == CellKind.DENDRITIC:
            activation *= 1.08
        elif cell.kind == CellKind.REGULATORY and antigen.protected:
            activation *= 1.18
        elif cell.kind == CellKind.MEMORY:
            activation *= 1.12
        activation *= self._spatial_receptor_activation_prior(cell, antigen)
        return float(max(0.0, min(1.20, activation)))

    def _spatial_receptor_activation_prior(self, cell: ImmuneCell, antigen: Antigen) -> float:
        code = antigen.context.get("spatial_receptor_code") if isinstance(antigen.context, dict) else None
        if not isinstance(code, dict):
            return 1.0
        top = code.get("top_receptor")
        if not isinstance(top, dict):
            return 1.0
        preferred = top.get("preferred_cell_kinds") or []
        try:
            confidence = float(top.get("probability", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if cell.kind.value in set(map(str, preferred)):
            return 1.0 + min(0.18, max(0.0, confidence) * 0.18)
        return 1.0

    def _component_health_pressure(
        self,
        subsystem: str,
        state_snapshot: dict[str, Any] | None,
    ) -> float:
        if state_snapshot and "health_pressure" in state_snapshot:
            return float(max(0.0, min(1.0, state_snapshot["health_pressure"])))
        autopoiesis = self._get_service("autopoiesis")
        if autopoiesis and hasattr(autopoiesis, "get_component_health"):
            try:
                health = float(autopoiesis.get_component_health(subsystem))
                if health > 0.0:
                    return float(max(0.0, min(1.0, 1.0 - health)))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass  # no-op: intentional
        return 0.0

    def _ensure_graph_links(self, subsystem: str) -> None:
        node = self._tissue.ensure_node(subsystem)
        known = list(self._tissue._edges.keys())
        for other in known:
            if other == node:
                continue
            if other.split("_", 1)[0] == node.split("_", 1)[0]:
                self._tissue.register_edge(node, other, 0.45)
            elif other in {
                "llm_router",
                "inference_gate",
                "state_repository",
                "continuity",
                "identity",
                "memory_guard",
            }:
                self._tissue.register_edge(node, other, 0.18)

    def _find_cell(self, cell_id: str) -> ImmuneCell | None:
        for cell in self._cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def _canonical_subsystem(self, value: str) -> str:
        return str(value or "unknown").strip().lower().replace(" ", "_")

    def _is_protected_subsystem(self, subsystem: str) -> bool:
        """Whether this subsystem is protected tissue.

        CP126 5b472fda: this was bare substring containment, which fails in
        BOTH directions. "will" matched "goodwill_tracker" and "self_model"
        matched "self_model_debug_dump", suppressing repair on things that
        were never protected; meanwhile a genuinely protected component with
        a novel name matched nothing and got no protection at all.

        Matching is now on underscore/dot TOKENS, so a hint matches a whole
        name component rather than any letters inside one. This still is not
        a canonical registry with signed ownership labels — that needs an
        owner declaration this module does not have — so hint matching remains
        the mechanism, made precise rather than replaced.
        """
        lowered = str(subsystem or "").lower()
        if not lowered:
            return False
        tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
        for hint in self._PROTECTED_SUBSYSTEM_HINTS:
            hint_tokens = [token for token in re.split(r"[^a-z0-9]+", hint) if token]
            if not hint_tokens:
                continue
            if len(hint_tokens) == 1:
                if hint_tokens[0] in tokens:
                    return True
            # A multi-token hint ("memory_guard", "canonical_self") must appear
            # as a contiguous token run.
            elif all(token in tokens for token in hint_tokens):
                return True
        return False

    def _new_cell_id(self, kind: CellKind) -> str:
        digest = hashlib.sha1(
            f"{kind.value}:{time.time()}:{self._rng.random()}".encode()
        ).hexdigest()
        return f"{kind.value[:3]}_{digest[:10]}"

    def _resolve_state_dir(self, state_dir: Path | None) -> Path:
        if state_dir is not None:
            return Path(state_dir)
        try:
            from core.config import config

            return config.paths.data_dir / "adaptive_immunity"
        except (ImportError, AttributeError, RuntimeError):
            return Path.home().expanduser() / ".aura" / "data" / "adaptive_immunity"

    def _get_service(self, name: str) -> Any:
        try:
            from core.container import ServiceContainer

            return ServiceContainer.get(name, default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            # A broken lookup is a repair-infrastructure fault, not merely
            # absent evidence — record it so the immune system's own
            # dependencies show up in forensics.
            _record_adaptive_immunity_degradation(
                exc,
                action=f"treated service '{name}' as unavailable after lookup failure",
            )
            return None

    @staticmethod
    def _kmeans(x: np.ndarray, k: int, max_iter: int = 32) -> np.ndarray:
        rng = np.random.default_rng(0)
        n = x.shape[0]
        if n <= k:
            return np.arange(n, dtype=np.int32)
        centroids = np.empty((k, x.shape[1]), dtype=np.float32)
        centroids[0] = x[rng.integers(0, n)]
        for idx in range(1, k):
            dist_sq = np.min(
                np.sum((x[:, None, :] - centroids[None, :idx, :]) ** 2, axis=2), axis=1
            )
            total = float(dist_sq.sum())
            if total <= _EPSILON:
                centroids[idx] = x[rng.integers(0, n)]
            else:
                probs = dist_sq / total
                probs = probs / max(float(probs.sum()), _EPSILON)
                centroids[idx] = x[rng.choice(n, p=probs)]

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            dists = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centroids[idx] = x[mask].mean(axis=0)
        return labels

    @staticmethod
    def _silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
        unique = np.unique(labels)
        if len(unique) < 2 or len(x) < 3:
            return 0.0
        norms = np.sum(x**2, axis=1)
        dist_sq = norms[:, None] + norms[None, :] - 2.0 * (x @ x.T)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        sil = np.zeros(len(x), dtype=np.float32)
        for idx in range(len(x)):
            own = labels[idx]
            own_mask = labels == own
            own_count = int(np.sum(own_mask)) - 1
            if own_count <= 0:
                continue
            a_i = np.sum(dist[idx, own_mask]) / max(1, own_count)
            b_i = np.inf
            for other in unique:
                if other == own:
                    continue
                other_mask = labels == other
                if not np.any(other_mask):
                    continue
                b_i = min(b_i, np.mean(dist[idx, other_mask]))
            if not np.isfinite(b_i):
                continue
            sil[idx] = (b_i - a_i) / max(a_i, b_i, _EPSILON)
        return float(np.mean(sil))


_adaptive_immune_singleton: AdaptiveImmuneSystem | None = None


def get_adaptive_immune_system() -> AdaptiveImmuneSystem:
    global _adaptive_immune_singleton
    if _adaptive_immune_singleton is None:
        _adaptive_immune_singleton = AdaptiveImmuneSystem()
    return _adaptive_immune_singleton
