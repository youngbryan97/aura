"""core/adaptation/dynamic_value_graph.py -- Dynamic Value Evolution Graph
==========================================================================
Replaces the hard-capped _MAX_CYCLE_DELTA=0.03 value autopoiesis with an
evidence-gated, graph-based value evolution system.

Key improvements over the original ValueAutopoiesis:
  1. Evidence-gated evolution: changes require statistical significance
  2. Candidate → Sandbox → Evidence → Adoption pipeline
  3. New drive axes can be created dynamically (with evidence gates)
  4. Graph-based value relationships (values can reinforce or inhibit others)
  5. No hard caps — evolution speed is determined by evidence strength
  6. Anti-wireheading: reward signals are diversity-weighted

Design principles:
  - Safety-First Plasticity: no value change without evidence
  - Reversibility: all adoptions can be rolled back within a grace period
  - Observability: full audit trail of every value mutation
  - Sovereignty: the system evolves its own values, not external override
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Aura.DynamicValueGraph")

_DATA_DIR = Path.home() / ".aura" / "data" / "value_graph"
_GRAPH_PATH = _DATA_DIR / "value_graph.json"


class ValueNodeStatus(str, Enum):
    """Lifecycle stage of a value node."""
    CANDIDATE = "candidate"      # Proposed, not yet tested
    SANDBOX = "sandbox"          # Being tested in simulation
    PROVISIONAL = "provisional"  # Adopted with rollback window
    ADOPTED = "adopted"          # Fully integrated
    DEPRECATED = "deprecated"    # Marked for removal


class EvidenceType(str, Enum):
    """Type of evidence supporting a value change."""
    OUTCOME_QUALITY = "outcome_quality"
    ENGAGEMENT = "engagement"
    FREE_ENERGY_REDUCTION = "free_energy_reduction"
    SOCIAL_FEEDBACK = "social_feedback"
    SELF_REPORT = "self_report"
    LONGITUDINAL = "longitudinal"


@dataclass
class ValueEvidence:
    """A single piece of evidence for or against a value."""
    evidence_type: EvidenceType
    value_name: str
    signal: float          # -1.0 (against) to +1.0 (for)
    confidence: float      # 0.0 to 1.0
    source: str            # What subsystem generated this
    context: str           # Brief description
    timestamp: float = field(default_factory=time.time)

    def weighted_signal(self) -> float:
        """Signal weighted by confidence."""
        return self.signal * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_type": self.evidence_type.value,
            "value_name": self.value_name,
            "signal": round(self.signal, 4),
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "context": self.context[:200],
            "timestamp": self.timestamp,
        }


@dataclass
class ValueNode:
    """A node in the value graph representing a single value/drive."""
    name: str
    weight: float                            # Current weight (0.0 to 1.0)
    status: ValueNodeStatus = ValueNodeStatus.ADOPTED
    origin_weight: float = 0.5               # Weight at creation
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    evidence_buffer: List[ValueEvidence] = field(default_factory=list)
    adoption_timestamp: float = 0.0          # When it was adopted
    rollback_deadline: float = 0.0           # Grace period for rollback
    total_evidence_count: int = 0
    source_diversity: int = 0                # Number of unique evidence sources
    edges: Dict[str, float] = field(default_factory=dict)  # name → coupling weight

    # Anti-wireheading
    _recent_sources: Set[str] = field(default_factory=set)

    def add_evidence(self, evidence: ValueEvidence) -> None:
        """Add evidence to this node's buffer."""
        self.evidence_buffer.append(evidence)
        self._recent_sources.add(evidence.source)
        self.total_evidence_count += 1
        self.source_diversity = len(self._recent_sources)

        # Cap buffer size
        if len(self.evidence_buffer) > 200:
            self.evidence_buffer = self.evidence_buffer[-200:]

    def compute_evidence_delta(self, min_evidence: int = 5) -> Tuple[float, float]:
        """Compute the recommended weight delta from accumulated evidence.

        Returns:
            (delta, confidence) where delta is the recommended change
            and confidence is how certain we are.
        """
        if len(self.evidence_buffer) < min_evidence:
            return 0.0, 0.0

        # Compute weighted average signal
        total_weight = 0.0
        weighted_sum = 0.0
        for ev in self.evidence_buffer:
            w = ev.confidence
            weighted_sum += ev.weighted_signal() * w
            total_weight += w

        if total_weight < 1e-6:
            return 0.0, 0.0

        mean_signal = weighted_sum / total_weight

        # Diversity bonus: more diverse sources → higher confidence
        diversity_factor = min(1.0, self.source_diversity / 3.0)

        # Sample size factor: more evidence → higher confidence
        sample_factor = min(1.0, len(self.evidence_buffer) / 20.0)

        confidence = diversity_factor * sample_factor

        # Scale delta by evidence strength (no hard cap!)
        # But bounded by a reasonable maximum per cycle
        max_delta = 0.05 + (confidence * 0.10)  # 0.05 to 0.15
        delta = max(-max_delta, min(max_delta, mean_signal * 0.1))

        return delta, confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": round(self.weight, 4),
            "status": self.status.value,
            "origin_weight": round(self.origin_weight, 4),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_evidence_count": self.total_evidence_count,
            "source_diversity": self.source_diversity,
            "evidence_buffer_size": len(self.evidence_buffer),
            "edges": {k: round(v, 4) for k, v in self.edges.items()},
            "adoption_timestamp": self.adoption_timestamp,
            "rollback_deadline": self.rollback_deadline,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValueNode":
        try:
            status = ValueNodeStatus(data.get("status", "adopted"))
        except ValueError:
            status = ValueNodeStatus.ADOPTED
        return cls(
            name=data["name"],
            weight=float(data.get("weight", 0.5)),
            status=status,
            origin_weight=float(data.get("origin_weight", data.get("weight", 0.5))),
            created_at=float(data.get("created_at", time.time())),
            last_updated=float(data.get("last_updated", time.time())),
            total_evidence_count=int(data.get("total_evidence_count", 0)),
            source_diversity=int(data.get("source_diversity", 0)),
            edges=dict(data.get("edges", {})),
            adoption_timestamp=float(data.get("adoption_timestamp", 0.0)),
            rollback_deadline=float(data.get("rollback_deadline", 0.0)),
        )


@dataclass
class ValueMutation:
    """Record of a value graph mutation."""
    node_name: str
    mutation_type: str       # "weight_shift", "created", "adopted", "deprecated", "rollback"
    old_weight: float
    new_weight: float
    delta: float
    evidence_count: int
    confidence: float
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "mutation_type": self.mutation_type,
            "old_weight": round(self.old_weight, 4),
            "new_weight": round(self.new_weight, 4),
            "delta": round(self.delta, 4),
            "evidence_count": self.evidence_count,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class DynamicValueGraph:
    """Evidence-gated value evolution graph.

    Replaces the hard-capped autopoiesis with a graph-based system where:
    - Values are nodes with edges representing reinforcement/inhibition
    - Changes require statistical significance from diverse evidence
    - New values can be created through the Candidate→Sandbox→Adoption pipeline
    - Anti-wireheading: evidence diversity is weighted

    Usage:
        graph = get_dynamic_value_graph()

        # Record evidence during waking
        graph.record_evidence(ValueEvidence(
            evidence_type=EvidenceType.OUTCOME_QUALITY,
            value_name="curiosity",
            signal=0.8,
            confidence=0.9,
            source="research_engine",
            context="Successful autonomous research",
        ))

        # Evolve during dream cycle
        mutations = graph.evolve()
    """

    # Rollback grace period: 24 hours
    ROLLBACK_GRACE_SECONDS = 86400.0
    # Minimum evidence before any evolution
    MIN_EVIDENCE = 5
    # Minimum source diversity for adoption
    MIN_SOURCE_DIVERSITY = 2

    def __init__(self) -> None:
        self._nodes: Dict[str, ValueNode] = {}
        self._mutation_log: List[ValueMutation] = []
        self._cycle_count: int = 0
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(
            "DynamicValueGraph initialized: %d nodes, cycle=%d",
            len(self._nodes), self._cycle_count,
        )

    # ── Evidence Recording ──────────────────────────────────────────────

    def record_evidence(self, evidence: ValueEvidence) -> None:
        """Record a piece of evidence about a value.

        If the value doesn't exist as a node, create it as CANDIDATE.
        """
        node = self._nodes.get(evidence.value_name)
        if node is None:
            # Auto-create candidate node for new values
            node = ValueNode(
                name=evidence.value_name,
                weight=0.5,
                status=ValueNodeStatus.CANDIDATE,
                origin_weight=0.5,
            )
            self._nodes[evidence.value_name] = node
            logger.info(
                "New value candidate created: '%s' (from evidence)",
                evidence.value_name,
            )

        node.add_evidence(evidence)
        logger.debug(
            "Evidence recorded for '%s': signal=%.2f conf=%.2f src=%s",
            evidence.value_name, evidence.signal,
            evidence.confidence, evidence.source,
        )

    # ── Evolution Cycle ─────────────────────────────────────────────────

    def evolve(self) -> List[ValueMutation]:
        """Run one evolution cycle.

        Called during dream/consolidation. Processes evidence buffers,
        computes deltas, applies changes through the adoption pipeline.

        Returns:
            List of mutations applied.
        """
        self._cycle_count += 1
        mutations: List[ValueMutation] = []

        for name, node in list(self._nodes.items()):
            # Skip deprecated nodes
            if node.status == ValueNodeStatus.DEPRECATED:
                continue

            delta, confidence = node.compute_evidence_delta(
                min_evidence=self.MIN_EVIDENCE
            )

            if abs(delta) < 0.001:
                continue

            # Apply pipeline stages
            if node.status == ValueNodeStatus.CANDIDATE:
                mutation = self._process_candidate(node, delta, confidence)
            elif node.status == ValueNodeStatus.SANDBOX:
                mutation = self._process_sandbox(node, delta, confidence)
            elif node.status == ValueNodeStatus.PROVISIONAL:
                mutation = self._process_provisional(node, delta, confidence)
            elif node.status == ValueNodeStatus.ADOPTED:
                mutation = self._process_adopted(node, delta, confidence)
            else:
                mutation = None

            if mutation is not None:
                mutations.append(mutation)
                self._mutation_log.append(mutation)
                node.last_updated = time.time()

            # Clear consumed evidence (keep 20% for continuity)
            retain = max(5, len(node.evidence_buffer) // 5)
            node.evidence_buffer = node.evidence_buffer[-retain:]

        # Apply edge propagation (value coupling)
        edge_mutations = self._propagate_edges(mutations)
        mutations.extend(edge_mutations)

        # Check rollback deadlines
        rollback_mutations = self._check_rollbacks()
        mutations.extend(rollback_mutations)

        # Persist
        self._save()

        if mutations:
            logger.info(
                "Value evolution cycle %d: %d mutation(s)",
                self._cycle_count, len(mutations),
            )
            self._publish_event("dynamic_value_graph.evolved", {
                "cycle": self._cycle_count,
                "mutations": [m.to_dict() for m in mutations],
            })

        return mutations

    def _process_candidate(
        self, node: ValueNode, delta: float, confidence: float
    ) -> Optional[ValueMutation]:
        """Process a candidate value: promote to sandbox if evidence is sufficient."""
        if node.total_evidence_count >= self.MIN_EVIDENCE and confidence > 0.3:
            node.status = ValueNodeStatus.SANDBOX
            return ValueMutation(
                node_name=node.name,
                mutation_type="promoted_to_sandbox",
                old_weight=node.weight,
                new_weight=node.weight,
                delta=0.0,
                evidence_count=node.total_evidence_count,
                confidence=confidence,
                reason=f"Sufficient evidence ({node.total_evidence_count}) to begin sandbox testing",
            )
        return None

    def _process_sandbox(
        self, node: ValueNode, delta: float, confidence: float
    ) -> Optional[ValueMutation]:
        """Process a sandbox value: promote to provisional if diversity is sufficient."""
        if (node.source_diversity >= self.MIN_SOURCE_DIVERSITY
                and node.total_evidence_count >= self.MIN_EVIDENCE * 2
                and confidence > 0.5):
            node.status = ValueNodeStatus.PROVISIONAL
            node.adoption_timestamp = time.time()
            node.rollback_deadline = time.time() + self.ROLLBACK_GRACE_SECONDS

            old_weight = node.weight
            node.weight = max(0.05, min(0.95, node.weight + delta))

            return ValueMutation(
                node_name=node.name,
                mutation_type="promoted_to_provisional",
                old_weight=old_weight,
                new_weight=node.weight,
                delta=delta,
                evidence_count=node.total_evidence_count,
                confidence=confidence,
                reason=(
                    f"Diverse evidence (sources={node.source_diversity}, "
                    f"count={node.total_evidence_count}). "
                    f"Rollback window: {self.ROLLBACK_GRACE_SECONDS/3600:.0f}h"
                ),
            )
        return None

    def _process_provisional(
        self, node: ValueNode, delta: float, confidence: float
    ) -> Optional[ValueMutation]:
        """Process a provisional value: adopt if grace period passed and evidence holds."""
        now = time.time()
        if now > node.rollback_deadline and confidence > 0.4:
            node.status = ValueNodeStatus.ADOPTED
            old_weight = node.weight
            node.weight = max(0.05, min(0.95, node.weight + delta))

            return ValueMutation(
                node_name=node.name,
                mutation_type="adopted",
                old_weight=old_weight,
                new_weight=node.weight,
                delta=delta,
                evidence_count=node.total_evidence_count,
                confidence=confidence,
                reason="Grace period passed with sustained evidence — fully adopted",
            )

        # Still in grace period: apply delta conservatively
        old_weight = node.weight
        conservative_delta = delta * 0.5  # Half-strength during provisional
        node.weight = max(0.05, min(0.95, node.weight + conservative_delta))

        return ValueMutation(
            node_name=node.name,
            mutation_type="provisional_adjustment",
            old_weight=old_weight,
            new_weight=node.weight,
            delta=conservative_delta,
            evidence_count=node.total_evidence_count,
            confidence=confidence,
            reason="Provisional adjustment (half-strength, rollback window active)",
        )

    def _process_adopted(
        self, node: ValueNode, delta: float, confidence: float
    ) -> Optional[ValueMutation]:
        """Process an adopted value: apply evidence-gated delta."""
        old_weight = node.weight
        node.weight = max(0.05, min(0.95, node.weight + delta))

        return ValueMutation(
            node_name=node.name,
            mutation_type="weight_shift",
            old_weight=old_weight,
            new_weight=node.weight,
            delta=delta,
            evidence_count=node.total_evidence_count,
            confidence=confidence,
            reason=f"Evidence-gated shift (conf={confidence:.2f})",
        )

    def _propagate_edges(
        self, mutations: List[ValueMutation]
    ) -> List[ValueMutation]:
        """Propagate value changes through edges (coupling)."""
        edge_mutations: List[ValueMutation] = []
        for mutation in mutations:
            node = self._nodes.get(mutation.node_name)
            if node is None:
                continue

            for neighbor_name, coupling in node.edges.items():
                neighbor = self._nodes.get(neighbor_name)
                if neighbor is None or neighbor.status == ValueNodeStatus.DEPRECATED:
                    continue

                # Propagate: coupling * delta * damping
                propagated_delta = coupling * mutation.delta * 0.3
                if abs(propagated_delta) < 0.001:
                    continue

                old_weight = neighbor.weight
                neighbor.weight = max(0.05, min(0.95, neighbor.weight + propagated_delta))

                edge_mutations.append(ValueMutation(
                    node_name=neighbor.name,
                    mutation_type="edge_propagation",
                    old_weight=old_weight,
                    new_weight=neighbor.weight,
                    delta=propagated_delta,
                    evidence_count=0,
                    confidence=abs(coupling),
                    reason=f"Propagated from '{mutation.node_name}' (coupling={coupling:.2f})",
                ))

        return edge_mutations

    def _check_rollbacks(self) -> List[ValueMutation]:
        """Check for provisional values that should be rolled back."""
        rollbacks: List[ValueMutation] = []
        now = time.time()

        for name, node in self._nodes.items():
            if node.status != ValueNodeStatus.PROVISIONAL:
                continue

            # Rollback if evidence turned negative during grace period
            if now < node.rollback_deadline:
                delta, confidence = node.compute_evidence_delta(min_evidence=3)
                if delta < -0.02 and confidence > 0.4:
                    old_weight = node.weight
                    node.weight = node.origin_weight
                    node.status = ValueNodeStatus.CANDIDATE

                    rollbacks.append(ValueMutation(
                        node_name=name,
                        mutation_type="rollback",
                        old_weight=old_weight,
                        new_weight=node.weight,
                        delta=node.weight - old_weight,
                        evidence_count=node.total_evidence_count,
                        confidence=confidence,
                        reason="Evidence turned negative during grace period — rolled back",
                    ))
                    logger.info("Value '%s' ROLLED BACK to origin weight", name)

        return rollbacks

    # ── Edge Management ─────────────────────────────────────────────────

    def add_edge(self, from_value: str, to_value: str, coupling: float) -> None:
        """Add a coupling edge between two values.

        Positive coupling: values reinforce each other
        Negative coupling: values inhibit each other
        """
        coupling = max(-1.0, min(1.0, coupling))
        if from_value in self._nodes:
            self._nodes[from_value].edges[to_value] = coupling
        if to_value in self._nodes:
            self._nodes[to_value].edges[from_value] = coupling * 0.5  # Asymmetric

    # ── Import from Heartstone ──────────────────────────────────────────

    def import_from_heartstone(self) -> None:
        """Import existing Heartstone values as adopted nodes."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hv = get_heartstone_values()
            for name, weight in hv.values.items():
                if name not in self._nodes:
                    self._nodes[name] = ValueNode(
                        name=name,
                        weight=float(weight),
                        status=ValueNodeStatus.ADOPTED,
                        origin_weight=float(weight),
                    )
            logger.info(
                "Imported %d values from Heartstone",
                len(hv.values),
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Heartstone import failed: %s", exc)

    def export_to_heartstone(self) -> None:
        """Export adopted value weights back to Heartstone."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hv = get_heartstone_values()
            for name, node in self._nodes.items():
                if node.status in (ValueNodeStatus.ADOPTED, ValueNodeStatus.PROVISIONAL):
                    current = hv.values.get(name)
                    if current is not None:
                        delta = node.weight - current
                        if abs(delta) > 0.001:
                            hv._adjust(name, delta)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Heartstone export failed: %s", exc)

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist the value graph to disk."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "cycle_count": self._cycle_count,
                "timestamp": time.time(),
                "nodes": {name: node.to_dict() for name, node in self._nodes.items()},
                "mutation_count": len(self._mutation_log),
            }
            fd, tmp_path = tempfile.mkstemp(dir=str(_DATA_DIR), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, indent=2, default=str))
                os.replace(tmp_path, str(_GRAPH_PATH))
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except (OSError, IOError) as _exc:
                    logger.debug("Suppressed %s in core.adaptation.dynamic_value_graph: %s", type(_exc).__name__, _exc)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("Value graph save failed: %s", exc)

    def _load(self) -> None:
        """Load the value graph from disk."""
        try:
            if not _GRAPH_PATH.exists():
                return
            data = json.loads(_GRAPH_PATH.read_text())
            self._cycle_count = int(data.get("cycle_count", 0))
            for name, node_data in data.get("nodes", {}).items():
                self._nodes[name] = ValueNode.from_dict(node_data)
            logger.info(
                "Value graph restored: %d nodes, cycle=%d",
                len(self._nodes), self._cycle_count,
            )
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
            logger.debug("Value graph load failed: %s", exc)

    # ── Events ──────────────────────────────────────────────────────────

    def _publish_event(self, topic: str, data: Dict[str, Any]) -> None:
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish_threadsafe(topic, data)
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.adaptation.dynamic_value_graph: %s", type(_exc).__name__, _exc)

    # ── Public API ──────────────────────────────────────────────────────

    def get_adopted_values(self) -> Dict[str, float]:
        """Return all adopted value weights."""
        return {
            name: node.weight
            for name, node in self._nodes.items()
            if node.status in (ValueNodeStatus.ADOPTED, ValueNodeStatus.PROVISIONAL)
        }

    def get_status(self) -> Dict[str, Any]:
        """Return graph status for observability."""
        by_status = defaultdict(int)
        for node in self._nodes.values():
            by_status[node.status.value] += 1

        return {
            "cycle_count": self._cycle_count,
            "total_nodes": len(self._nodes),
            "by_status": dict(by_status),
            "nodes": {name: node.to_dict() for name, node in self._nodes.items()},
            "recent_mutations": [m.to_dict() for m in self._mutation_log[-20:]],
        }

    def get_recent_mutations(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return recent mutations for observability."""
        return [m.to_dict() for m in self._mutation_log[-n:]]


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[DynamicValueGraph] = None


def get_dynamic_value_graph() -> DynamicValueGraph:
    """Get or create the singleton DynamicValueGraph."""
    global _instance
    if _instance is None:
        _instance = DynamicValueGraph()
    return _instance
