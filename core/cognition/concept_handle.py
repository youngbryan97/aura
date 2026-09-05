"""core/cognition/concept_handle.py — the same concept, wherever it is stored.

"Kettle" exists in this repository at least five times over. It is a
``Concept`` in concept_formation.py with defining features and a support
count. It is a ``GroundedConcept`` in core/grounding with a prototype vector.
It is a ``Node`` in the AtomSpace with a truth value. It is a row in a vector
store with an embedding. It is a slot in a world-model latent. Each is a
correct representation for what its own organ does, and none of them knows
about the others, so a concept learned in one place has to be re-recognised in
every other place it is needed — usually by string match on its label, which
fails the moment two organs spell it differently.

A :class:`ConceptHandle` is the identity those five share. It stores no
content: the representations stay where they are, owned by their organs. What
it stores is *where the representations are* — a projection per substrate,
with the confidence of the binding and how it was made. Retrieving a handle
gets you every correlate a concept has; adding a substrate later does not
require touching the others.

Why the binding carries confidence
----------------------------------
Most cross-substrate identity in this repository is asserted by a label
match, and a label match between an embedding cluster and an AtomSpace node
is a guess. Recording it as a guess — ``method="label"``, low confidence —
lets a consumer that needs certainty ask for it, and lets a later
verification upgrade the binding rather than silently overwrite it. A
projection made by measurement (a prototype vector that actually retrieves
the atom, a probe that decodes the latent) carries ``method="measured"`` and
the evidence that established it.

What this is not
----------------
It is not a merge. Two representations bound to one handle stay distinct and
can disagree; :meth:`ConceptHandle.disagreement` reports when they do, which
is card 056's substrate conflict made visible rather than blended away in a
final score. Binding is also not transitive by default: A-bound-to-B and
B-bound-to-C does not create A-to-C, because two label matches in a row is
how a kettle becomes a kestrel.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from core.evidence.packet import EvidencePacket
from core.runtime.lockdep import checked_lock

__all__ = [
    "Substrate",
    "BindingMethod",
    "Projection",
    "ConceptHandle",
    "ConceptRegistry",
    "get_concept_registry",
    "reset_concept_registry_for_test",
]


class Substrate(StrEnum):
    """Where a concept can be represented. One value per real store."""

    #: core/cognition/concept_formation.py — features and support.
    FORMED = "formed"
    #: core/grounding — a prototype vector tied to perceptual evidence.
    GROUNDED = "grounded"
    #: core/knowledge/atomspace.py — a typed node with a truth value.
    ATOMSPACE = "atomspace"
    #: A vector store row.
    EMBEDDING = "embedding"
    #: A slot or direction in a world-model latent.
    WORLD_LATENT = "world_latent"
    #: An RLC / learned-tissue state channel.
    NEURAL = "neural"
    #: An episodic memory record.
    EPISODIC = "episodic"
    #: A procedure or operator in the procedure registry.
    PROCEDURAL = "procedural"
    #: A word or phrase as the cortex uses it.
    LEXICAL = "lexical"


class BindingMethod(StrEnum):
    """How a projection was established, worst to best."""

    #: The labels matched. A guess, and recorded as one.
    LABEL = "label"
    #: A human or an organ asserted the binding directly.
    DECLARED = "declared"
    #: Vector similarity above a threshold.
    SIMILARITY = "similarity"
    #: The binding was checked: the projection retrieves or decodes the other.
    MEASURED = "measured"


#: Confidence ceiling per method. A label match cannot report high confidence
#: however sure the caller feels, because the evidence is a string comparison.
_METHOD_CEILING = {
    BindingMethod.LABEL: 0.4,
    BindingMethod.DECLARED: 0.6,
    BindingMethod.SIMILARITY: 0.8,
    BindingMethod.MEASURED: 1.0,
}


@dataclass(frozen=True, slots=True)
class Projection:
    """One concept as one substrate holds it."""

    substrate: Substrate
    ref: str
    method: BindingMethod = BindingMethod.LABEL
    confidence: float = 0.0
    evidence: EvidencePacket | None = None
    at: float = field(default_factory=time.time)
    #: Whatever the substrate needs to find it again — a store name, a slot
    #: index, a model version. Never identity.
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ceiling = _METHOD_CEILING[self.method]
        object.__setattr__(self, "confidence", max(0.0, min(float(self.confidence), ceiling)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate.value,
            "ref": self.ref,
            "method": self.method.value,
            "confidence": self.confidence,
            "at": self.at,
            "detail": dict(self.detail),
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass(frozen=True, slots=True)
class ConceptHandle:
    """A concept's identity, and every substrate that holds a version of it."""

    handle_id: str
    label: str
    projections: tuple[Projection, ...] = ()
    #: Handles this one was split from or merged with, so a rename does not
    #: orphan the history.
    lineage: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)

    def projection(self, substrate: Substrate) -> Projection | None:
        best = [p for p in self.projections if p.substrate is substrate]
        return max(best, key=lambda p: p.confidence) if best else None

    @property
    def substrates(self) -> frozenset[Substrate]:
        return frozenset(p.substrate for p in self.projections)

    @property
    def reach(self) -> int:
        """How many substrates hold this concept. The number card 173 asks for."""
        return len(self.substrates)

    def bound_confidence(self) -> float:
        """Weakest link. A chain of bindings is as good as its worst hop."""
        return min((p.confidence for p in self.projections), default=0.0)

    def with_projection(self, projection: Projection) -> ConceptHandle:
        """Add or upgrade a projection.

        A better-evidenced binding replaces a weaker one for the same
        substrate and ref; a weaker one never overwrites a stronger, so a
        later label match cannot undo a measurement.
        """
        kept = []
        replaced = False
        for existing in self.projections:
            same = existing.substrate is projection.substrate and existing.ref == projection.ref
            if not same:
                kept.append(existing)
                continue
            replaced = True
            kept.append(projection if projection.confidence >= existing.confidence else existing)
        if not replaced:
            kept.append(projection)
        return replace(self, projections=tuple(kept))

    def disagreement(self, readings: dict[Substrate, float]) -> dict[str, Any]:
        """How far apart the substrates are about this concept.

        ``readings`` is whatever number each substrate reports for the same
        question — a truth strength, a probe probability, a match score. The
        spread is the signal: a wide one means two representations of one
        concept have diverged, which is an occasion for an experiment rather
        than for averaging.
        """
        present = {s: v for s, v in readings.items() if s in self.substrates}
        if len(present) < 2:
            return {"comparable": False, "spread": 0.0, "readings": {}}
        values = list(present.values())
        return {
            "comparable": True,
            "spread": max(values) - min(values),
            "readings": {s.value: v for s, v in present.items()},
            "highest": max(present, key=present.get).value,
            "lowest": min(present, key=present.get).value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle_id": self.handle_id,
            "label": self.label,
            "reach": self.reach,
            "bound_confidence": self.bound_confidence(),
            "projections": [p.to_dict() for p in self.projections],
            "lineage": list(self.lineage),
            "created_at": self.created_at,
        }


def _handle_id(label: str) -> str:
    normalised = " ".join(str(label or "").lower().split())
    return "c_" + hashlib.blake2s(normalised.encode("utf-8"), digest_size=8).hexdigest()


class ConceptRegistry:
    """The handles, and the index from any substrate reference back to one."""

    def __init__(self, *, max_handles: int = 100_000) -> None:
        self._lock = checked_lock("core.cognition.concept_handle.ConceptRegistry", reentrant=True)
        self._handles: dict[str, ConceptHandle] = {}
        self._index: dict[tuple[Substrate, str], str] = {}
        self._max_handles = int(max_handles)

    def handle_for(self, label: str) -> ConceptHandle:
        """Get or create the handle for a label."""
        hid = _handle_id(label)
        with self._lock:
            existing = self._handles.get(hid)
            if existing is not None:
                return existing
            if len(self._handles) >= self._max_handles:
                oldest = min(self._handles.values(), key=lambda h: h.created_at)
                self._forget_locked(oldest.handle_id)
            handle = ConceptHandle(handle_id=hid, label=label)
            self._handles[hid] = handle
            return handle

    def bind(
        self,
        label: str,
        substrate: Substrate,
        ref: str,
        *,
        method: BindingMethod = BindingMethod.LABEL,
        confidence: float = 0.0,
        evidence: EvidencePacket | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ConceptHandle:
        """Record that ``substrate`` holds this concept as ``ref``."""
        projection = Projection(
            substrate=substrate,
            ref=ref,
            method=method,
            confidence=confidence if confidence else _METHOD_CEILING[method],
            evidence=evidence,
            detail=dict(detail or {}),
        )
        with self._lock:
            handle = self.handle_for(label).with_projection(projection)
            self._handles[handle.handle_id] = handle
            self._index[(substrate, ref)] = handle.handle_id
            return handle

    def resolve(self, substrate: Substrate, ref: str) -> ConceptHandle | None:
        """Which concept is this substrate reference part of."""
        with self._lock:
            hid = self._index.get((substrate, ref))
            return self._handles.get(hid) if hid else None

    def get(self, label_or_id: str) -> ConceptHandle | None:
        with self._lock:
            return self._handles.get(label_or_id) or self._handles.get(_handle_id(label_or_id))

    def merge(self, keep: str, absorb: str) -> ConceptHandle:
        """Fold one handle into another, keeping the lineage.

        Used when two labels turn out to name one concept. The absorbed
        handle's id stays resolvable through ``lineage`` so a stored reference
        to it does not dangle.
        """
        with self._lock:
            a = self.get(keep)
            b = self.get(absorb)
            if a is None or b is None:
                raise KeyError(f"cannot merge {keep!r} into {absorb!r}: one does not exist")
            if a.handle_id == b.handle_id:
                return a
            merged = a
            for projection in b.projections:
                merged = merged.with_projection(projection)
            merged = replace(merged, lineage=(*merged.lineage, b.handle_id))
            self._handles[merged.handle_id] = merged
            del self._handles[b.handle_id]
            for key, hid in list(self._index.items()):
                if hid == b.handle_id:
                    self._index[key] = merged.handle_id
            return merged

    def _forget_locked(self, handle_id: str) -> None:
        self._handles.pop(handle_id, None)
        for key, hid in list(self._index.items()):
            if hid == handle_id:
                del self._index[key]

    def report(self) -> dict[str, Any]:
        """Reach distribution — the evidence that identity actually crosses.

        ``multi_substrate`` is the number card 173 is about: a concept present
        in one store is a row, and a concept present in four is an identity.
        """
        with self._lock:
            handles = list(self._handles.values())
            by_reach: dict[int, int] = {}
            for handle in handles:
                by_reach[handle.reach] = by_reach.get(handle.reach, 0) + 1
            substrate_counts: dict[str, int] = {}
            for handle in handles:
                for substrate in handle.substrates:
                    substrate_counts[substrate.value] = substrate_counts.get(substrate.value, 0) + 1
            return {
                "handles": len(handles),
                "multi_substrate": sum(1 for h in handles if h.reach >= 2),
                "reach_four_or_more": sum(1 for h in handles if h.reach >= 4),
                "by_reach": dict(sorted(by_reach.items())),
                "by_substrate": dict(sorted(substrate_counts.items())),
                "measured_bindings": sum(
                    1 for h in handles for p in h.projections if p.method is BindingMethod.MEASURED
                ),
            }

    def handles(self) -> Iterable[ConceptHandle]:
        with self._lock:
            return list(self._handles.values())


_registry_lock = checked_lock("core.cognition.concept_handle.singleton")
_registry: ConceptRegistry | None = None


def get_concept_registry() -> ConceptRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ConceptRegistry()
        return _registry


def reset_concept_registry_for_test() -> ConceptRegistry:
    global _registry
    with _registry_lock:
        _registry = ConceptRegistry()
        return _registry
