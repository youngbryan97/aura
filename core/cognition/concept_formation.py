"""Concept formation from prediction error — inventing primitives, not just distilling patterns.

The critique: "she distills patterns, she doesn't invent physics… she doesn't form new conceptual
primitives the way a human does from experience," and the move it names is "concept formation from
repeated prediction errors." Aura's existing AbstractionEngine distills principles from *successes*
— that is the pattern-distillation the doc already credits her with. This is the missing other
half: a concept is born from *being repeatedly wrong in the same way*.

Mechanism: surprising events (high prediction error) are clustered by the similarity of their
feature signatures. When the same kind of surprise recurs — enough high-error events sharing a
signature — that is evidence of a regularity the current model has no name for, so the engine
abstracts a new **concept primitive**: it names it from the recurring features, records its
defining signature, and registers it. Once formed, the concept *recognizes* future occurrences of
that signature (closing the loop — a learned primitive now explains what used to surprise) and raises
a scientific-engine hypothesis to test it. Independent held and transfer evidence is required
before consolidation; repetition alone does not publish a belief.

This is deliberately bounded: it abstracts regularities from experience (real, testable), it does
not claim to derive novel physical law. But it is the honest first step from "good notes" toward
forming concepts the designer never hand-coded.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Cognition.ConceptFormation")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class _ErrorCluster:
    """An accumulating group of similar surprising events — a candidate concept."""

    features: Set[str]
    support: int = 1
    error_sum: float = 0.0
    last_seen: float = field(default_factory=lambda: time.time())

    @property
    def mean_error(self) -> float:
        return self.error_sum / max(1, self.support)


@dataclass
class Concept:
    """A formed conceptual primitive: a named, recurring regularity abstracted from surprise."""

    concept_id: str
    name: str
    defining_features: List[str]
    support: int
    mean_error_at_formation: float
    confidence: float
    status: str = "provisional"      # provisional | consolidated
    explained: int = 0               # times it has since recognized a matching event
    created_at: float = field(default_factory=lambda: time.time())
    validation_provenance: str = ""
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    validation_revoked_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "defining_features": self.defining_features,
            "support": self.support,
            "mean_error_at_formation": round(self.mean_error_at_formation, 4),
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "explained": self.explained,
            "created_at": self.created_at,
            "validation_provenance": self.validation_provenance,
            "validation_evidence": self.validation_evidence,
            "validation_revoked_by": self.validation_revoked_by,
        }


@dataclass
class FormationResult:
    recognized: Optional[str]        # concept_id if an existing concept already explains it
    formed: Optional[Concept]        # a newly-formed concept, if this event triggered one
    cluster_support: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recognized": self.recognized,
            "formed": self.formed.to_dict() if self.formed else None,
            "cluster_support": self.cluster_support,
            "reason": self.reason,
        }


class ConceptFormationEngine:
    """Forms new concept primitives from repeated, similar prediction errors."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        *,
        similarity_threshold: float = 0.5,
        error_threshold: float = 0.5,
        min_support: int = 3,
        max_clusters: int = 64,
        autosave: bool = True,
        min_save_interval_s: float = 5.0,
    ) -> None:
        if storage_path is None:
            try:
                from core.config import config
                storage_path = config.paths.memory_dir / "concepts.json"
            except (ImportError, AttributeError, RuntimeError):
                storage_path = state_root() / "data" / "memory" / "concepts.json"
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sim_t = similarity_threshold
        self._err_t = error_threshold
        self._min_support = min_support
        self._max_clusters = max_clusters
        self._autosave = autosave
        self._min_save_interval = min_save_interval_s
        self._lock = threading.RLock()
        self._clusters: List[_ErrorCluster] = []
        self._concepts: Dict[str, Concept] = {}
        self._last_save = 0.0
        self._counter = 0
        self._load()
        logger.info("ConceptFormationEngine initialized (%d concepts).", len(self._concepts))

    # ── the formation loop ────────────────────────────────────────────────

    def observe_prediction_error(
        self,
        features: Sequence[str],
        magnitude: float,
        *,
        context: str = "",
        now: Optional[float] = None,
    ) -> FormationResult:
        """Feed one surprising event. May recognize an existing concept or form a new one."""
        now = time.time() if now is None else now
        sig = {str(f).strip().lower() for f in features if str(f).strip()}
        magnitude = _clamp(magnitude)
        if not sig:
            return FormationResult(recognized=None, formed=None, reason="empty_signature")

        with self._lock:
            # Already have a concept for this? Then it's explained, not novel.
            existing = self._recognize_locked(sig)
            if existing is not None:
                existing.explained += 1
                self._maybe_save()
                return FormationResult(recognized=existing.concept_id, formed=None,
                                       reason="recognized_known_concept")

            # Only sustained, genuinely surprising events feed concept formation.
            if magnitude < self._err_t:
                return FormationResult(recognized=None, formed=None, reason="below_error_threshold")

            cluster = self._assign_cluster(sig, magnitude, now)
            if cluster.support >= self._min_support and cluster.mean_error >= self._err_t:
                concept = self._form_concept(cluster, now)
                self._clusters.remove(cluster)
                self._maybe_save()
                return FormationResult(recognized=None, formed=concept,
                                       cluster_support=concept.support, reason="formed_new_concept")
            self._maybe_save()
            return FormationResult(recognized=None, formed=None,
                                   cluster_support=cluster.support, reason="accumulating")

    def _assign_cluster(self, sig: Set[str], magnitude: float, now: float) -> _ErrorCluster:
        best, best_sim = None, 0.0
        for c in self._clusters:
            sim = _jaccard(sig, c.features)
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= self._sim_t:
            # Reinforce: keep the features the events agree on (intersection), grow support.
            best.features = (best.features & sig) or sig
            best.support += 1
            best.error_sum += magnitude
            best.last_seen = now
            return best
        cluster = _ErrorCluster(features=set(sig), support=1, error_sum=magnitude, last_seen=now)
        self._clusters.append(cluster)
        self._prune_clusters(now)
        return cluster

    def _form_concept(self, cluster: _ErrorCluster, now: float) -> Concept:
        self._counter += 1
        features = sorted(cluster.features)
        name = "concept:" + "+".join(features[:3]) if features else f"concept_{self._counter}"
        concept = Concept(
            concept_id=f"con-{self._counter}-{int(now)}",
            name=name,
            defining_features=features,
            support=cluster.support,
            mean_error_at_formation=cluster.mean_error,
            confidence=_clamp(0.4 + 0.1 * cluster.support),
        )
        self._concepts[concept.concept_id] = concept
        self._on_formation(concept)
        return concept

    def _on_formation(self, concept: Concept) -> None:
        """Submit a proposal to the experiment lane without publishing a belief."""
        try:
            from core.cognition.scientific_engine import get_scientific_engine
            get_scientific_engine().form_hypothesis(
                f"newly-formed concept holds: {concept.name}",
                predicted_observable="recognizes_future_instances",
                expected=concept.confidence, prior_confidence=concept.confidence,
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("concept_formation", exc, severity="debug")

    def certify(self, concept_id: str, discovered: Any) -> Concept:
        """Promote only a matching law with held and transfer evidence."""
        from core.brain.ontology_discovery import (
            MAX_P_VALUE,
            MIN_HELDOUT_LIFT,
            DiscoveredLaw,
        )

        if not isinstance(discovered, DiscoveredLaw):
            raise TypeError("concept certification requires a discovered law")
        evidence = discovered.evidence
        if (evidence.p_value > MAX_P_VALUE or evidence.heldout_lift < MIN_HELDOUT_LIFT
                or evidence.transfer_lift <= 1.0
                or evidence.transfer_p_value > MAX_P_VALUE or not evidence.ablation):
            raise ValueError("concept law lacks independent validation")
        with self._lock:
            concept = self._concepts[concept_id]
            cues = {"cue:" + feature for feature in concept.defining_features}
            if not any(predicate.feature in cues for predicate in discovered.law.predicates):
                raise ValueError("discovered law does not test this concept")
            concept.status = "consolidated"
            concept.validation_provenance = discovered.provenance()
            concept.validation_evidence = discovered.to_dict()
            concept.validation_revoked_by = ""
            concept.confidence = min(evidence.heldout.precision, evidence.transfer.precision)
            self._maybe_save()
            return concept

    def revoke_certification(
        self, concept_id: str, validation_provenance: str, *, counterevidence: str
    ) -> bool:
        """Reopen a concept when fresh evidence defeats its validating law."""
        with self._lock:
            concept = self._concepts.get(concept_id)
            if (concept is None or concept.status != "consolidated"
                    or concept.validation_provenance != validation_provenance):
                return False
            concept.status = "provisional"
            concept.validation_revoked_by = counterevidence
            concept.confidence = min(concept.confidence, 0.5)
        self.save()
        return True

    # ── recognition (the closed loop) ─────────────────────────────────────

    def recognize(self, features: Sequence[str]) -> Optional[Concept]:
        """Does a formed concept explain this signature? (a learned primitive recognizing reality)"""
        sig = {str(f).strip().lower() for f in features if str(f).strip()}
        with self._lock:
            return self._recognize_locked(sig)

    def _recognize_locked(self, sig: Set[str]) -> Optional[Concept]:
        best, best_sim = None, 0.0
        for c in self._concepts.values():
            sim = _jaccard(sig, set(c.defining_features))
            if sim > best_sim:
                best, best_sim = c, sim
        return best if best is not None and best_sim >= self._sim_t else None

    # ── readout ───────────────────────────────────────────────────────────

    def concepts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._concepts.values()]

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Back a concept memory store: concepts whose features overlap the query."""
        toks = {t for t in str(query or "").lower().split() if len(t) > 2}
        out: List[Dict[str, Any]] = []
        with self._lock:
            for c in self._concepts.values():
                overlap = len(toks & set(" ".join(c.defining_features).split()))
                score = 0.4 + 0.15 * overlap + 0.2 * c.confidence
                out.append({"content": f"Concept '{c.name}' ({c.status}, support {c.support})",
                            "score": score, "source": "concept_formation"})
        out.sort(key=lambda d: d["score"], reverse=True)
        return out[:limit]

    def get_health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "module": "ConceptFormationEngine",
                "concepts": len(self._concepts),
                "open_clusters": len(self._clusters),
                "consolidated": sum(1 for c in self._concepts.values() if c.status == "consolidated"),
                "status": "online",
            }

    # ── housekeeping / persistence ────────────────────────────────────────

    def _prune_clusters(self, now: float) -> None:
        if len(self._clusters) <= self._max_clusters:
            return
        # Drop the weakest/stalest candidates (low support, long unseen).
        self._clusters.sort(key=lambda c: (c.support, -(now - c.last_seen)), reverse=True)
        self._clusters = self._clusters[: self._max_clusters]

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for c in raw.get("concepts", []):
                if isinstance(c, dict) and c.get("concept_id"):
                    self._concepts[c["concept_id"]] = Concept(
                        concept_id=c["concept_id"], name=c.get("name", ""),
                        defining_features=list(c.get("defining_features", [])),
                        support=int(c.get("support", 0)),
                        mean_error_at_formation=float(c.get("mean_error_at_formation", 0.0)),
                        confidence=float(c.get("confidence", 0.4)),
                        status=c.get("status", "provisional"),
                        explained=int(c.get("explained", 0)),
                        created_at=float(c.get("created_at", time.time())),
                        validation_provenance=str(c.get("validation_provenance", "")),
                        validation_evidence=dict(c.get("validation_evidence") or {}),
                        validation_revoked_by=str(c.get("validation_revoked_by", "")),
                    )
                    concept = self._concepts[c["concept_id"]]
                    if concept.status == "consolidated" and (
                        not concept.validation_provenance or concept.validation_revoked_by
                    ):
                        concept.status = "provisional"
            self._counter = int(raw.get("counter", len(self._concepts)))
        except (OSError, ValueError) as exc:
            record_degradation("concept_formation", exc)

    def save(self) -> None:
        try:
            from core.runtime.atomic_writer import atomic_write_text
            with self._lock:
                payload = {"concepts": [c.to_dict() for c in self._concepts.values()],
                           "counter": self._counter}
            atomic_write_text(self._path, json.dumps(payload, indent=2))
            self._last_save = time.time()
        except (OSError, TypeError, ValueError) as exc:
            record_degradation("concept_formation", exc)

    def _maybe_save(self) -> None:
        if self._autosave and (time.time() - self._last_save) >= self._min_save_interval:
            self.save()


_instance: Optional[ConceptFormationEngine] = None
_instance_lock = threading.Lock()


def get_concept_formation_engine() -> ConceptFormationEngine:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ConceptFormationEngine()
    return _instance
