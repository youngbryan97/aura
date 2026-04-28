"""SemioticNetwork — the persistent symbol↔concept↔evidence↔method graph.

Concepts are clusters of feature vectors; evidence is the raw signal
that supports a concept; symbols are the linguistic tokens that
denote concepts; methods are the classifiers that decide whether a
concept applies.  This is the durable form of Steels' triad.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.grounding.types import (
    GroundedConcept,
    GroundingEvent,
    GroundingMethod,
    PerceptualEvidence,
    SymbolLink,
    new_id,
)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(av, bv) / denom)


class SemioticNetwork:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.methods: Dict[str, GroundingMethod] = {}
        self.evidence: Dict[str, PerceptualEvidence] = {}
        self.concepts: Dict[str, GroundedConcept] = {}
        self.links: Dict[str, SymbolLink] = {}
        self.events: List[GroundingEvent] = []
        self.load()

    # ------------------------------------------------------------------
    # write paths
    # ------------------------------------------------------------------
    def register_method(self, method: GroundingMethod) -> None:
        self.methods[method.method_id] = method
        self.save()

    def add_evidence(self, evidence: PerceptualEvidence) -> None:
        self.evidence[evidence.evidence_id] = evidence
        self.save()

    def create_or_update_concept(
        self,
        *,
        label: str,
        kind: str,
        features: List[float],
        method_id: str,
        positive: bool = True,
    ) -> GroundedConcept:
        existing = self.find_concept(label)
        now = time.time()
        if existing is None:
            concept = GroundedConcept(
                concept_id=new_id("concept"),
                label=label,
                kind=kind,
                prototype=list(features),
                method_id=method_id,
                confidence=0.55 if positive else 0.25,
                positive_count=1 if positive else 0,
                negative_count=0 if positive else 1,
            )
            self.concepts[concept.concept_id] = concept
            self.save()
            return concept

        proto = np.asarray(existing.prototype, dtype=np.float32)
        feat = np.asarray(features, dtype=np.float32)
        n = max(1, existing.positive_count)
        if positive:
            proto = (proto * n + feat) / (n + 1)
            existing.positive_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.05)
        else:
            existing.negative_count += 1
            existing.confidence = max(0.0, existing.confidence - 0.07)
        existing.prototype = proto.tolist()
        existing.updated_at = now
        self.save()
        return existing

    def link_symbol(
        self,
        *,
        symbol: str,
        concept_id: str,
        relation: str = "denotes",
        source: str = "unknown",
        delta: float = 0.10,
    ) -> SymbolLink:
        symbol_norm = symbol.strip().lower()
        for link in self.links.values():
            if (
                link.symbol == symbol_norm
                and link.concept_id == concept_id
                and link.relation == relation
            ):
                link.strength = float(max(0.0, min(1.0, link.strength + delta)))
                if delta >= 0:
                    link.confirmations += 1
                else:
                    link.contradictions += 1
                link.updated_at = time.time()
                self.save()
                return link
        link = SymbolLink(
            link_id=new_id("link"),
            symbol=symbol_norm,
            concept_id=concept_id,
            relation=relation,
            strength=0.55 if delta >= 0 else 0.40,
            source=source,
            confirmations=1 if delta >= 0 else 0,
            contradictions=0 if delta >= 0 else 1,
        )
        self.links[link.link_id] = link
        self.save()
        return link

    def record_event(self, event: GroundingEvent) -> None:
        self.events.append(event)
        self.events = self.events[-10_000:]
        self.save()

    # ------------------------------------------------------------------
    # read paths
    # ------------------------------------------------------------------
    def find_concept(self, label: str) -> Optional[GroundedConcept]:
        target = label.strip().lower()
        for c in self.concepts.values():
            if c.label.strip().lower() == target:
                return c
        return None

    def concepts_for_symbol(self, symbol: str) -> List[Tuple[GroundedConcept, SymbolLink]]:
        target = symbol.strip().lower()
        out: List[Tuple[GroundedConcept, SymbolLink]] = []
        for link in self.links.values():
            if link.symbol == target and link.concept_id in self.concepts:
                out.append((self.concepts[link.concept_id], link))
        out.sort(key=lambda pair: pair[1].strength * pair[0].confidence, reverse=True)
        return out

    def score_evidence_for_concept(
        self, evidence_id: str, concept_id: str
    ) -> float:
        ev = self.evidence[evidence_id]
        concept = self.concepts[concept_id]
        return cosine_similarity(ev.features, concept.prototype)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        payload = {
            "methods": {k: v.__dict__ for k, v in self.methods.items()},
            "evidence": {k: v.__dict__ for k, v in self.evidence.items()},
            "concepts": {k: v.__dict__ for k, v in self.concepts.items()},
            "links": {k: v.__dict__ for k, v in self.links.items()},
            "events": [e.__dict__ for e in self.events],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.methods = {k: GroundingMethod(**v) for k, v in payload.get("methods", {}).items()}
        self.evidence = {k: PerceptualEvidence(**v) for k, v in payload.get("evidence", {}).items()}
        self.concepts = {k: GroundedConcept(**v) for k, v in payload.get("concepts", {}).items()}
        self.links = {k: SymbolLink(**v) for k, v in payload.get("links", {}).items()}
        self.events = [GroundingEvent(**v) for v in payload.get("events", [])]
