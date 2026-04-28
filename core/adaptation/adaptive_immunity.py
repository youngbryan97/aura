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
from core.runtime.errors import record_degradation


from core.runtime.atomic_writer import atomic_write_text

import asyncio
import copy
import hashlib
import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.cognitive.anomaly_detector import FeatureExtractor

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
    "TissueField",
    "get_adaptive_immune_system",
]

_ANTIGEN_DIM = 16
_EPSILON = 1e-8


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


class CellKind(str, Enum):
    DENDRITIC = "dendritic"
    B = "b_cell"
    CYTOTOXIC = "cytotoxic_t"
    REGULATORY = "regulatory_t"
    MEMORY = "memory"


class EffectorKind(str, Enum):
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
    verification_checks: int = 2
    verification_interval_s: float = 0.01
    min_verified_health_delta: float = 0.02
    recurrence_window_s: float = 900.0
    species_min_k: int = 2
    species_max_k: int = 4
    species_silhouette_floor: float = 0.22
    tissue_diffusion: float = 0.16
    tissue_decay: float = 0.06


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
    protected: bool
    source: str = "unknown"
    error_signature: str = ""
    stack_trace: str = ""
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antigen_id": self.antigen_id,
            "subsystem": self.subsystem,
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
    def from_dict(cls, data: Dict[str, Any]) -> "Antigen":
        vector = np.asarray(data.get("vector", [0.0] * _ANTIGEN_DIM), dtype=np.float32)
        if vector.shape[0] != _ANTIGEN_DIM:
            vector = np.resize(vector, (_ANTIGEN_DIM,)).astype(np.float32)
        return cls(
            antigen_id=str(data.get("antigen_id", "")),
            subsystem=str(data.get("subsystem", "unknown")),
            vector=np.clip(vector, 0.0, 1.0),
            danger=float(data.get("danger", 0.0)),
            subsystem_need=float(data.get("subsystem_need", 0.0)),
            threat_probability=float(data.get("threat_probability", 0.0)),
            resource_pressure=float(data.get("resource_pressure", 0.0)),
            error_load=float(data.get("error_load", 0.0)),
            health_pressure=float(data.get("health_pressure", 0.0)),
            temporal_pressure=float(data.get("temporal_pressure", 0.0)),
            recurrence_pressure=float(data.get("recurrence_pressure", 0.0)),
            protected=bool(data.get("protected", False)),
            source=str(data.get("source", "unknown")),
            error_signature=str(data.get("error_signature", "")),
            stack_trace=str(data.get("stack_trace", "")),
            timestamp=float(data.get("timestamp", time.time())),
            context=dict(data.get("context", {})),
        )


@dataclass
class EffectorArtifact:
    artifact_id: str
    kind: EffectorKind
    component: str
    confidence: float
    source_cell_id: str
    lineage_id: str
    bounded_payload: Dict[str, Any]
    governance_required: bool = True
    suppressed: bool = False
    governance_denied: bool = False
    executed: bool = False
    success: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
    best_effector: Optional[EffectorKind] = None
    last_antigen_id: str = ""
    born_at: float = field(default_factory=time.time)

    def clone(
        self,
        *,
        rng: np.random.Generator,
        cell_id: str,
        mutation_sigma: float,
    ) -> "ImmuneCell":
        child = copy.deepcopy(self)
        child.cell_id = cell_id
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
        return child

    def to_dict(self) -> Dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImmuneCell":
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
                EffectorKind(data["best_effector"])
                if data.get("best_effector")
                else None
            ),
            last_antigen_id=str(data.get("last_antigen_id", "")),
            born_at=float(data.get("born_at", time.time())),
        )


@dataclass
class ImmuneResponse:
    antigen: Antigen
    activated_cells: List[Dict[str, Any]]
    artifacts: List[EffectorArtifact]
    selected_artifact: Optional[EffectorArtifact]
    suppression_applied: float
    metabolic_scale: float
    entropy_pressure: float
    proliferation_count: int
    species_count: int
    tissue_snapshot: Dict[str, Any]
    dream_consolidated: bool = False
    coverage_report: Dict[str, Any] = field(default_factory=dict)
    verification_report: Dict[str, Any] = field(default_factory=dict)
    diagnostic_verdict: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
        self._edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._danger: Dict[str, float] = defaultdict(float)
        self._inflammation: Dict[str, float] = defaultdict(float)
        self._damage: Dict[str, float] = defaultdict(float)
        self._repair: Dict[str, float] = defaultdict(float)

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
            self._damage[node] + 0.25 * max(antigen.resource_pressure, antigen.error_load, antigen.health_pressure)
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

    def snapshot(self, top_k: int = 8) -> Dict[str, Any]:
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
                {"subsystem": node, "need": round(self.get_need(node), 4)}
                for node in hot
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
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
        data: Dict[str, Any],
        *,
        diffusion: float,
        decay: float,
    ) -> "TissueField":
        field_obj = cls(diffusion=diffusion, decay=decay)
        field_obj._edges = defaultdict(dict, {
            str(node): {str(neighbor): float(weight) for neighbor, weight in neighbors.items()}
            for node, neighbors in data.get("edges", {}).items()
        })
        field_obj._danger = defaultdict(float, {
            str(node): float(value) for node, value in data.get("danger", {}).items()
        })
        field_obj._inflammation = defaultdict(float, {
            str(node): float(value) for node, value in data.get("inflammation", {}).items()
        })
        field_obj._damage = defaultdict(float, {
            str(node): float(value) for node, value in data.get("damage", {}).items()
        })
        field_obj._repair = defaultdict(float, {
            str(node): float(value) for node, value in data.get("repair", {}).items()
        })
        return field_obj

    def _diffuse_scalar(self, values: Dict[str, float]) -> defaultdict[str, float]:
        new_vals: defaultdict[str, float] = defaultdict(float)
        for node in self._edges:
            current = float(values[node])
            neighbors = self._edges.get(node, {})
            if neighbors:
                total_w = sum(max(weight, 0.0) for weight in neighbors.values()) + _EPSILON
                neighbor_mean = sum(values[neighbor] * weight for neighbor, weight in neighbors.items()) / total_w
                diffused = current + self._diffusion * (neighbor_mean - current)
            else:
                diffused = current
            new_vals[node] = self._clip(diffused * (1.0 - self._decay))
        return new_vals

    @staticmethod
    def _clip(value: float) -> float:
        return float(max(0.0, min(1.0, value)))


class OfflineCoevolutionLab:
    """Small sandbox for evolving defender receptors against replayed threats."""

    def __init__(self, *, rng: np.random.Generator):
        self._rng = rng

    def evolve(
        self,
        cells: Iterable[ImmuneCell],
        antigens: Iterable[Antigen],
        *,
        generations: int = 3,
        population_size: int = 12,
        tau: float = 0.22,
        mutation_sigma: float = 0.05,
    ) -> List[ImmuneCell]:
        seeds = [copy.deepcopy(cell) for cell in cells if cell.kind in {CellKind.B, CellKind.CYTOTOXIC, CellKind.REGULATORY, CellKind.MEMORY}]
        if not seeds:
            return []
        antigens = list(antigens)
        if not antigens:
            return []

        population = seeds[:population_size]
        next_id = 0
        while len(population) < population_size:
            source = self._rng.choice(seeds)
            clone = source.clone(
                rng=self._rng,
                cell_id=f"offline_lab_{next_id}",
                mutation_sigma=mutation_sigma,
            )
            next_id += 1
            population.append(clone)

        for _ in range(max(1, generations)):
            scored: List[Tuple[float, ImmuneCell]] = []
            for cell in population:
                score = 0.0
                for antigen in antigens:
                    affinity = AdaptiveImmuneSystem.compute_affinity_static(
                        cell.receptor,
                        antigen.vector,
                        tau=tau,
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
                    )
                )
                next_id += 1

        population.sort(
            key=lambda cell: sum(
                AdaptiveImmuneSystem.compute_affinity_static(cell.receptor, antigen.vector, tau=tau)
                for antigen in antigens
            ),
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
            0.70, 0.35, 0.35, 0.30,
            0.80, 0.80, 0.25, 0.60,
            1.00, 0.90, 0.85, 1.15,
            0.70, 0.65, 0.65, 0.45,
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *,
        config: AdaptiveImmuneConfig | None = None,
        state_dir: Optional[Path] = None,
        rng_seed: Optional[int] = None,
    ):
        self.cfg = config or AdaptiveImmuneConfig()
        self._rng = np.random.default_rng(rng_seed)
        self._extractor = FeatureExtractor()
        self._lock = threading.RLock()
        self._cells: List[ImmuneCell] = []
        self._tissue = TissueField(
            diffusion=self.cfg.tissue_diffusion,
            decay=self.cfg.tissue_decay,
        )
        self._recent_antigens: Deque[Antigen] = deque(maxlen=self.cfg.replay_buffer_size)
        self._recent_responses: Deque[Dict[str, Any]] = deque(
            maxlen=self.cfg.recent_response_buffer
        )
        self._recent_subsystem_counts: Counter[str] = Counter()
        self._recurrence_tracker: Dict[str, Dict[str, Any]] = defaultdict(
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
        self._lineage_stats: Dict[str, Dict[str, Any]] = defaultdict(
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
        self._state_dir = self._resolve_state_dir(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "adaptive_immune_state.json"
        self._lab = OfflineCoevolutionLab(rng=self._rng)

        if not self._load_state():
            self._cells = self._seed_population()
            self._assign_species()
            self._save_state()

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
        event: Dict[str, Any],
        *,
        anomaly_score: Any | None = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> ImmuneResponse:
        antigen = self.present_antigen(
            event,
            anomaly_score=anomaly_score,
            state_snapshot=state_snapshot,
        )
        coverage_report = self._assess_coverage(
            event,
            antigen,
            anomaly_score=anomaly_score,
            state_snapshot=state_snapshot,
        )
        response, _top_cell = self._observe_core(antigen)
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
            response.proliferation_count = self._reinforce_after_execution(
                antigen=antigen,
                response=response,
                acting_cell=acting_cell,
                verification_report=verification_report,
            )
        else:
            self._reinforce_without_execution(antigen, response)

        self._record_response_summary(response)

        return response

    def observe_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
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
        antigen = self.present_antigen(event, anomaly_score=None, state_snapshot=context)
        response, _top_cell = self._observe_core(antigen)
        response.coverage_report = self._assess_coverage(event, antigen, anomaly_score=None, state_snapshot=context)
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

    def observe_signature(
        self,
        component: str,
        exception_type: str,
        *,
        error_count: int = 1,
        context: Optional[Dict[str, Any]] = None,
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
        antigen = self.present_antigen(event, anomaly_score=None, state_snapshot=context)
        response, _top_cell = self._observe_core(antigen)
        response.coverage_report = self._assess_coverage(event, antigen, anomaly_score=None, state_snapshot=context)
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

    def present_antigen(
        self,
        event: Dict[str, Any],
        *,
        anomaly_score: Any | None = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
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
            resource_pressure = float(
                event.get("resource_pressure")
                or max(
                    0.0,
                    min(1.0, float(event.get("cpu", 0.0)) / 100.0),
                    min(1.0, float(event.get("ram", 0.0)) / 100.0),
                )
            )
            error_load = min(1.0, float(event.get("error_rate", 0.0)) + float(event.get("error_count", 0) or 0) / 10.0)
            error_signature = str(
                event.get("exception_type")
                or event.get("error_signature")
                or event.get("type")
                or ""
            )
            threat_probability = float(
                getattr(anomaly_score, "threat_probability", None)
                or event.get("threat_probability")
                or event.get("danger")
                or max(resource_pressure, error_load * 0.7)
            )
            stack_trace = str(event.get("stack_trace", "") or "")
            stack_complexity = min(1.0, len(stack_trace) / 1200.0)
            protected = bool(event.get("protected", False) or self._is_protected_subsystem(subsystem))
            health_pressure = self._component_health_pressure(subsystem, state_snapshot)
            temporal_pressure = min(
                1.0,
                float(self._recent_subsystem_counts.get(subsystem, 0)) / 6.0,
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
                0.48 * danger + 0.32 * max(health_pressure, resource_pressure, error_load) + 0.20 * recurrence_pressure,
            )

            vector = np.zeros(self.cfg.receptor_dim, dtype=np.float32)
            vector[:8] = base_vec[:8]
            vector[8] = danger
            vector[9] = resource_pressure
            vector[10] = error_load
            vector[11] = 1.0 if protected else 0.0
            vector[12] = health_pressure
            vector[13] = tissue_need_prior
            vector[14] = max(temporal_pressure, recurrence_pressure)
            vector[15] = stack_complexity

            antigen_id = f"ag_{hashlib.sha1(f'{subsystem}:{time.time()}'.encode()).hexdigest()[:12]}"

            antigen = Antigen(
                antigen_id=antigen_id,
                subsystem=subsystem,
                vector=np.clip(vector, 0.0, 1.0),
                danger=danger,
                subsystem_need=max(0.0, min(1.0, subsystem_need)),
                threat_probability=max(0.0, min(1.0, threat_probability)),
                resource_pressure=max(0.0, min(1.0, resource_pressure)),
                error_load=max(0.0, min(1.0, error_load)),
                health_pressure=max(0.0, min(1.0, health_pressure)),
                temporal_pressure=max(0.0, min(1.0, temporal_pressure)),
                recurrence_pressure=max(0.0, min(1.0, recurrence_pressure)),
                protected=protected,
                source=str(event.get("source") or event.get("type") or "unknown"),
                error_signature=error_signature,
                stack_trace=stack_trace,
                context=dict(state_snapshot or {}),
            )
            return antigen

    def dream_consolidate(self) -> Dict[str, Any]:
        with self._lock:
            promotions = 0
            removed = 0

            for cell in self._cells:
                cell.age += 1
                decay = self.cfg.memory_decay if cell.kind == CellKind.MEMORY else self.cfg.basal_decay
                cell.persistence = max(0.0, cell.persistence - decay * (1.0 + 0.15 * cell.failures))
                cell.fitness *= 0.98

            for cell in list(self._cells):
                lineage = self._lineage_stats[cell.lineage_id]
                if (
                    cell.kind != CellKind.MEMORY
                    and (cell.successes >= self.cfg.lineage_memory_successes or lineage["successes"] >= self.cfg.lineage_memory_successes)
                    and max(cell.fitness, float(lineage["best_fitness"])) >= self.cfg.lineage_memory_fitness
                ):
                    if not any(existing.lineage_id == cell.lineage_id and existing.kind == CellKind.MEMORY for existing in self._cells):
                        memory = copy.deepcopy(cell)
                        memory.cell_id = f"mem_{hashlib.sha1((cell.cell_id + str(time.time())).encode()).hexdigest()[:10]}"
                        memory.kind = CellKind.MEMORY
                        memory.persistence = min(1.0, memory.persistence + self.cfg.persistence_boost + 0.15)
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
            )
            for champion in champions[:2]:
                if len(self._cells) < self.cfg.max_population:
                    champion.cell_id = f"lab_{hashlib.sha1((champion.cell_id + str(time.time())).encode()).hexdigest()[:10]}"
                    champion.persistence = min(1.0, champion.persistence + 0.08)
                    self._cells.append(champion)

            self._assign_species()
            self._prune_population()

            for cell in list(self._cells):
                if cell.persistence <= 0.03 or (cell.fitness < -0.55 and cell.kind != CellKind.REGULATORY):
                    self._cells.remove(cell)
                    removed += 1

            self._save_state()
            self._last_dream_at = self._observation_count
            return {
                "promotions": promotions,
                "removed": removed,
                "population": len(self._cells),
                "species_count": self._species_count,
            }

    def get_status(self) -> Dict[str, Any]:
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
    ) -> float:
        diff = (receptor - antigen_vector) * AdaptiveImmuneSystem._FEATURE_WEIGHTS
        distance = float(np.linalg.norm(diff) / math.sqrt(max(1, receptor.shape[0])))
        return float(math.exp(-((distance * distance) / max(tau, 1e-6))))

    # ------------------------------------------------------------------
    # Core observation and evolution
    # ------------------------------------------------------------------

    def _observe_core(self, antigen: Antigen) -> Tuple[ImmuneResponse, Optional[ImmuneCell]]:
        with self._lock:
            self._observation_count += 1
            self._recent_subsystem_counts[antigen.subsystem] += 1
            self._recent_antigens.append(antigen)
            self._record_recurrence_observation(antigen)
            self._tissue.ingest_antigen(antigen)

            metabolic_scale, entropy_pressure = self._metabolic_context()
            activated: List[Tuple[ImmuneCell, float, float]] = []
            regulatory_suppression = 0.0
            dominant_regulatory: Optional[ImmuneCell] = None

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
            artifacts: List[EffectorArtifact] = []
            top_cell: Optional[ImmuneCell] = None

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
                dominant_regulatory.successes += 1
                dominant_regulatory.fitness = 0.85 * dominant_regulatory.fitness + 0.15 * max(
                    dominant_regulatory.fitness, antigen.danger * max(regulatory_suppression, 0.25)
                )

            selected_artifact = max(
                (
                    artifact
                    for artifact in artifacts
                    if not artifact.suppressed
                ),
                key=lambda artifact: artifact.confidence,
                default=None,
            )

            dream_consolidated = False
            if self._observation_count - self._last_dream_at >= self.cfg.dream_every_observations:
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

    def _reinforce_without_execution(self, antigen: Antigen, response: ImmuneResponse) -> None:
        with self._lock:
            false_positive_cost = 0.25 if antigen.danger < 0.22 else 0.0
            entropy_cost = 0.12 * response.entropy_pressure
            regulatory_reward = 0.25 if antigen.protected and response.suppression_applied > 0.18 else 0.0
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
        acting_cell: Optional[ImmuneCell],
        verification_report: Optional[Dict[str, Any]] = None,
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
        recovery_speed = min(1.0, artifact.confidence * (0.35 + 0.90 * health_delta)) if artifact.executed else 0.0
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
    ) -> Optional[EffectorArtifact]:
        kind = None
        notes = ""
        if cell.kind == CellKind.DENDRITIC:
            return None

        if cell.kind in {CellKind.B, CellKind.MEMORY}:
            sig = antigen.error_signature.lower()
            text = f"{antigen.source} {sig}".lower()
            if any(
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
            ) and antigen.stack_trace:
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
                0.25
                + 0.45 * activation
                + 0.20 * affinity
                + 0.10 * max(cell.fitness, 0.0),
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
        coverage_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        coverage_report = coverage_report or {"coverage_ratio": 0.0}
        coverage_ratio = float(coverage_report.get("coverage_ratio", 0.0) or 0.0)
        if artifact.suppressed:
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

        if antigen.protected and not self._authorize_protected_action(artifact, antigen):
            artifact.governance_denied = True
            artifact.suppressed = True
            artifact.notes = artifact.notes or "protected tissue denied by Unified Will"
            return self._default_verification_report(
                status="governance_denied",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

        if artifact.kind == EffectorKind.PATCH_PROPOSAL:
            resilience_mesh = self._get_service("autonomous_resilience_mesh")
            if resilience_mesh is None or not hasattr(resilience_mesh, "attempt_patch_for_antigen"):
                artifact.notes = artifact.notes or "patch pipeline unavailable"
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
                return {
                    "status": str(patch_result.get("status", "patch_attempted")),
                    "raw_success": applied,
                    "verified_success": applied,
                    "health_before": None,
                    "health_after": None,
                    "health_delta": 0.0,
                    "health_samples": [],
                    "coverage_ratio": round(coverage_ratio, 4),
                    "recurrence_risk": round(
                        max(0.0, min(1.0, antigen.recurrence_pressure * (0.45 if applied else 1.0))),
                        4,
                    ),
                    "notes": artifact.notes or "",
                }
            except Exception as exc:
                record_degradation('adaptive_immunity', exc)
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
            elif artifact.kind in {EffectorKind.QUARANTINE, EffectorKind.HALT_RUNAWAY, EffectorKind.REVOKE_TOOL}:
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
                    max(0.0, min(1.0, antigen.recurrence_pressure * (0.55 if verified_success else 1.0))),
                    4,
                ),
                "notes": artifact.notes or "",
            }
            if raw_success and not verified_success and not artifact.notes:
                artifact.notes = "repair executed but could not be verified as durable"
                verification_report["notes"] = artifact.notes
            return verification_report
        except Exception as exc:
            record_degradation('adaptive_immunity', exc)
            artifact.executed = True
            artifact.success = False
            artifact.notes = artifact.notes or f"execution failed: {exc}"
            return self._default_verification_report(
                status="execution_error",
                coverage_ratio=coverage_ratio,
                notes=artifact.notes,
            )

    async def _sample_component_health(self, component: str) -> List[float]:
        samples: List[float] = []
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
        health_before: Optional[float],
        health_after: Optional[float],
        health_samples: List[float],
    ) -> bool:
        if not raw_success:
            return False
        threshold = float(self.cfg.min_verified_health_delta)
        if health_before is not None and health_after is not None:
            if (health_after - health_before) >= threshold:
                return True
            if health_samples and (max(health_samples) - health_before) >= threshold:
                return True
        return False

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
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "raw_success": False,
            "verified_success": False,
            "health_before": None,
            "health_after": None,
            "health_delta": 0.0,
            "health_samples": [],
            "coverage_ratio": round(float(coverage_ratio), 4),
            "recurrence_risk": 0.0,
            "notes": notes,
        }

    def _authorize_protected_action(
        self,
        artifact: EffectorArtifact,
        antigen: Antigen,
    ) -> bool:
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"Adaptive immune effector: {artifact.kind.value} on {artifact.component}",
                source="adaptive_immune_system",
                domain=ActionDomain.STATE_MUTATION,
                priority=min(0.95, 0.45 + 0.35 * antigen.danger),
                context={
                    "component": artifact.component,
                    "artifact_kind": artifact.kind.value,
                    "danger": antigen.danger,
                    "protected": antigen.protected,
                    "lineage_id": artifact.lineage_id,
                },
            )
            return bool(decision.is_approved())
        except Exception as exc:
            record_degradation('adaptive_immunity', exc)
            logger.debug("Protected-action authorization unavailable: %s", exc)
            return False

    def _assess_coverage(
        self,
        event: Dict[str, Any],
        antigen: Antigen,
        *,
        anomaly_score: Any | None,
        state_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        component_matches = self._component_monitor_matches(antigen.subsystem)
        channels = {
            "anomaly_model": anomaly_score is not None,
            "subsystem_identity": antigen.subsystem not in {"", "unknown"},
            "error_telemetry": antigen.error_load > 0.0 or bool(event.get("error_count")) or bool(antigen.error_signature),
            "resource_telemetry": any(key in event for key in ("resource_pressure", "cpu", "ram")),
            "health_probe": bool(component_matches),
            "causal_trace": bool(event.get("stack_trace") or event.get("causal_trace") or antigen.stack_trace),
            "state_snapshot": bool(state_snapshot),
            "temporal_history": antigen.recurrence_pressure > 0.0 or self._recent_subsystem_counts.get(antigen.subsystem, 0) > 0,
        }
        coverage_ratio = sum(1.0 for present in channels.values() if present) / max(len(channels), 1)
        missing_channels = [name for name, present in channels.items() if not present]
        blind_spots: List[str] = []
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
        coverage_report: Dict[str, Any],
    ) -> None:
        coverage_ratio = float(coverage_report.get("coverage_ratio", 0.0) or 0.0)
        risky_kinds = {
            EffectorKind.RESTART_COMPONENT,
            EffectorKind.RESTORE_CHECKPOINT,
            EffectorKind.REVOKE_TOOL,
            EffectorKind.SCHEMA_MIGRATION,
        }
        for artifact in response.artifacts:
            artifact.confidence = max(0.0, min(0.99, artifact.confidence * (0.55 + 0.45 * coverage_ratio)))
            if (
                coverage_ratio < self.cfg.low_coverage_floor
                and artifact.kind in risky_kinds
                and antigen.danger < 0.88
            ):
                artifact.suppressed = True
                artifact.notes = artifact.notes or "suppressed under low observability"

    def _execution_candidates(self, response: ImmuneResponse) -> List[EffectorArtifact]:
        candidates = [
            artifact
            for artifact in response.artifacts
            if not artifact.suppressed and self._is_executable_artifact(artifact)
        ]
        candidates.sort(key=lambda artifact: artifact.confidence, reverse=True)
        return candidates

    def _best_visible_artifact(self, response: ImmuneResponse) -> Optional[EffectorArtifact]:
        visible = [artifact for artifact in response.artifacts if not artifact.suppressed]
        if not visible:
            return None
        return max(visible, key=lambda artifact: artifact.confidence)

    def _build_diagnostic_verdict(
        self,
        antigen: Antigen,
        response: ImmuneResponse,
        *,
        coverage_report: Dict[str, Any],
        verification_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        coverage_ratio = float(coverage_report.get("coverage_ratio", 0.0) or 0.0)
        verification_status = str(verification_report.get("status", "not_executed"))
        verified_success = bool(verification_report.get("verified_success", False))
        evidence_count = sum(
            1
            for present in (
                antigen.error_load > 0.08,
                antigen.health_pressure > 0.12,
                antigen.resource_pressure > 0.45,
                antigen.recurrence_pressure > 0.3,
                bool(antigen.stack_trace),
                bool(response.activated_cells),
            )
            if present
        )
        issue_confirmed = evidence_count >= 2 and antigen.danger >= 0.28
        escalation_recommended = False

        if verified_success:
            status = "verified_recovery"
            all_clear = coverage_ratio >= 0.7 and antigen.recurrence_pressure < 0.45
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
        elif coverage_ratio >= 0.75:
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
        coverage_report: Dict[str, Any],
        verification_report: Dict[str, Any],
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
            return f"signals suggest risk in {antigen.subsystem}, but confirmation remains incomplete"
        if status == "no_confirmed_issue_under_current_visibility":
            return f"no confirmed issue was observed in {antigen.subsystem} under current visibility"
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
            self._save_state()

    def _component_monitor_matches(self, subsystem: str) -> List[str]:
        if not subsystem:
            return []
        autopoiesis = self._get_service("autopoiesis")
        health_fns = getattr(autopoiesis, "_health_fns", {}) if autopoiesis is not None else {}
        lowered = subsystem.lower()
        matches: List[str] = []
        for component in health_fns:
            candidate = str(component).lower()
            if candidate == lowered or candidate in lowered or lowered in candidate:
                matches.append(str(component))
                continue
            if candidate.split("_", 1)[0] == lowered.split("_", 1)[0]:
                matches.append(str(component))
        return sorted(set(matches))

    def _system_coverage_summary(self) -> Dict[str, Any]:
        active_subsystems = {
            subsystem
            for subsystem in set(self._recent_subsystem_counts) | set(self._tissue._edges)
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

    def _recurrence_hotspots(self, limit: int = 6) -> List[Dict[str, Any]]:
        hotspots: List[Tuple[float, str, Dict[str, Any]]] = []
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

    def _read_component_health(self, subsystem: str) -> Optional[float]:
        autopoiesis = self._get_service("autopoiesis")
        if autopoiesis is None or not hasattr(autopoiesis, "get_component_health"):
            return None
        matches = self._component_monitor_matches(subsystem)
        for component in matches:
            try:
                return float(max(0.0, min(1.0, autopoiesis.get_component_health(component))))
            except Exception:
                continue
        return None

    def _estimate_recurrence_pressure(self, subsystem: str, error_signature: str) -> float:
        keys = self._recurrence_keys(subsystem, error_signature)
        if not keys:
            return 0.0
        pressures: List[float] = []
        for key in keys:
            stats = self._recurrence_tracker.get(key)
            if not stats:
                continue
            occurrences = float(stats.get("occurrences", 0))
            streak = float(stats.get("streak", 0))
            interval_ewma = float(stats.get("interval_ewma", 0.0) or 0.0)
            verified = float(stats.get("verified_repairs", 0))
            failed = float(stats.get("failed_repairs", 0))
            count_term = min(1.0, occurrences / 6.0)
            streak_term = min(1.0, streak / 4.0)
            interval_term = 0.0
            if interval_ewma > 0.0:
                interval_term = 1.0 - min(1.0, interval_ewma / max(self.cfg.recurrence_window_s, 1.0))
            repair_term = failed / max(verified + failed + 1.0, 1.0)
            pressures.append(0.35 * count_term + 0.25 * streak_term + 0.20 * interval_term + 0.20 * repair_term)
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
                stats["interval_ewma"] = interval if prev_ewma <= 0.0 else 0.7 * prev_ewma + 0.3 * interval
                if interval <= self.cfg.recurrence_window_s:
                    stats["streak"] = int(stats.get("streak", 0)) + 1
                else:
                    stats["streak"] = 1
            else:
                stats["streak"] = max(1, int(stats.get("streak", 0)))
            stats["peak_streak"] = max(int(stats.get("peak_streak", 0)), int(stats.get("streak", 0)))

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
    def _recurrence_keys(subsystem: str, error_signature: str) -> List[str]:
        keys = [f"subsystem::{subsystem}"]
        if error_signature:
            keys.append(f"signature::{subsystem}::{error_signature.lower()}")
        return keys

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _round_optional(value: Optional[float]) -> Optional[float]:
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

        keep: List[ImmuneCell] = []
        by_species: Dict[int, List[ImmuneCell]] = defaultdict(list)
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

    def _metabolic_context(self) -> Tuple[float, float]:
        vitality = 0.72
        metabolism = 0.65
        entropy_pressure = 0.0

        homeostasis = self._get_service("homeostasis") or self._get_service("homeostatic_coupling")
        if homeostasis is not None:
            try:
                if hasattr(homeostasis, "compute_vitality"):
                    vitality = float(homeostasis.compute_vitality())
                metabolism = float(getattr(homeostasis, "metabolism", metabolism))
            except Exception:
                pass  # no-op: intentional

        alife_dynamics = self._get_service("alife_dynamics")
        if alife_dynamics is not None:
            try:
                status = alife_dynamics.get_status() if hasattr(alife_dynamics, "get_status") else {}
                entropy_pressure = float(
                    status.get("entropy_pressure")
                    or status.get("pressure")
                    or status.get("entropy", 0.0) / max(status.get("max_entropy", 100.0), 1.0)
                )
            except Exception:
                entropy_pressure = 0.0

        scale = max(
            0.10,
            min(
                1.20,
                0.25 + 0.75 * vitality * (0.55 + 0.45 * metabolism) * (1.0 - 0.45 * entropy_pressure),
            ),
        )
        return float(scale), float(max(0.0, min(1.0, entropy_pressure)))

    def _save_state(self) -> None:
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
        }
        try:
            atomic_write_text(self._state_path, json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            record_degradation('adaptive_immunity', exc)
            logger.debug("Adaptive immune state save skipped: %s", exc)

    def _load_state(self) -> bool:
        if not self._state_path.exists():
            return False
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._cells = [ImmuneCell.from_dict(item) for item in payload.get("cells", [])]
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
                        EffectorKind(stats["best_effector"])
                        if stats.get("best_effector")
                        else None
                    ),
                    "best_fitness": float(stats.get("best_fitness", 0.0)),
                }
            self._observation_count = int(payload.get("observation_count", 0))
            self._last_dream_at = int(payload.get("last_dream_at", 0))
            self._recent_antigens = deque(
                [
                    Antigen.from_dict(item)
                    for item in payload.get("recent_antigens", [])
                ],
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
        except Exception as exc:
            record_degradation('adaptive_immunity', exc)
            logger.warning("Adaptive immune state load failed; reseeding: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _seed_population(self) -> List[ImmuneCell]:
        seeds: List[ImmuneCell] = []
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
            seeds.append(
                ImmuneCell(
                    cell_id=self._new_cell_id(kind),
                    lineage_id=f"{kind.value}_{idx}",
                    kind=kind,
                    receptor=receptor,
                    subsystem_scope=scopes[idx % len(scopes)],
                    persistence=0.58 if kind != CellKind.REGULATORY else 0.72,
                    regulatory_strength=1.25 if kind == CellKind.REGULATORY else 1.0,
                )
            )
        return seeds

    def _seed_receptor(self, kind: CellKind) -> np.ndarray:
        vec = self._rng.uniform(0.05, 0.55, size=self.cfg.receptor_dim).astype(np.float32)
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
        affinity = self.compute_affinity_static(cell.receptor, antigen.vector, tau=self.cfg.tau)
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
        return float(max(0.0, min(1.20, activation)))

    def _component_health_pressure(
        self,
        subsystem: str,
        state_snapshot: Optional[Dict[str, Any]],
    ) -> float:
        if state_snapshot and "health_pressure" in state_snapshot:
            return float(max(0.0, min(1.0, state_snapshot["health_pressure"])))
        autopoiesis = self._get_service("autopoiesis")
        if autopoiesis and hasattr(autopoiesis, "get_component_health"):
            try:
                health = float(autopoiesis.get_component_health(subsystem))
                if health > 0.0:
                    return float(max(0.0, min(1.0, 1.0 - health)))
            except Exception:
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
            elif other in {"llm_router", "inference_gate", "state_repository", "continuity", "identity", "memory_guard"}:
                self._tissue.register_edge(node, other, 0.18)

    def _find_cell(self, cell_id: str) -> Optional[ImmuneCell]:
        for cell in self._cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def _canonical_subsystem(self, value: str) -> str:
        return str(value or "unknown").strip().lower().replace(" ", "_")

    def _is_protected_subsystem(self, subsystem: str) -> bool:
        lowered = subsystem.lower()
        return any(hint in lowered for hint in self._PROTECTED_SUBSYSTEM_HINTS)

    def _new_cell_id(self, kind: CellKind) -> str:
        digest = hashlib.sha1(f"{kind.value}:{time.time()}:{self._rng.random()}".encode()).hexdigest()
        return f"{kind.value[:3]}_{digest[:10]}"

    def _resolve_state_dir(self, state_dir: Optional[Path]) -> Path:
        if state_dir is not None:
            return Path(state_dir)
        try:
            from core.config import config

            return config.paths.data_dir / "adaptive_immunity"
        except Exception:
            return Path.home().expanduser() / ".aura" / "data" / "adaptive_immunity"

    def _get_service(self, name: str) -> Any:
        try:
            from core.container import ServiceContainer

            return ServiceContainer.get(name, default=None)
        except Exception:
            return None

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iter: int = 32) -> np.ndarray:
        rng = np.random.default_rng(0)
        n = X.shape[0]
        if n <= k:
            return np.arange(n, dtype=np.int32)
        centroids = np.empty((k, X.shape[1]), dtype=np.float32)
        centroids[0] = X[rng.integers(0, n)]
        for idx in range(1, k):
            dist_sq = np.min(np.sum((X[:, None, :] - centroids[None, :idx, :]) ** 2, axis=2), axis=1)
            total = float(dist_sq.sum())
            if total <= _EPSILON:
                centroids[idx] = X[rng.integers(0, n)]
            else:
                probs = dist_sq / total
                probs = probs / max(float(probs.sum()), _EPSILON)
                centroids[idx] = X[rng.choice(n, p=probs)]

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centroids[idx] = X[mask].mean(axis=0)
        return labels

    @staticmethod
    def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        unique = np.unique(labels)
        if len(unique) < 2 or len(X) < 3:
            return 0.0
        norms = np.sum(X ** 2, axis=1)
        dist_sq = norms[:, None] + norms[None, :] - 2.0 * (X @ X.T)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        sil = np.zeros(len(X), dtype=np.float32)
        for idx in range(len(X)):
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


_adaptive_immune_singleton: Optional[AdaptiveImmuneSystem] = None


def get_adaptive_immune_system() -> AdaptiveImmuneSystem:
    global _adaptive_immune_singleton
    if _adaptive_immune_singleton is None:
        _adaptive_immune_singleton = AdaptiveImmuneSystem()
    return _adaptive_immune_singleton
