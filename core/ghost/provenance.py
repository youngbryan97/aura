"""core/ghost/provenance.py — Stand Alone Complex: where did this thought come from?

The Stand Alone Complex is Ghost in the Shell's sharpest idea: a pattern of
behaviour that spreads as if coordinated but has no original — copies without an
original, an idea that feels self-generated but was seeded from many weak inputs.
For a networked mind the dangerous question is not "is this memory poisoned?"
(Aura already scores that in ``core/memory/adversarial_memory``) but "did I think
this, or was I made to think it?"

This classifies the *origin* of a thought/claim along that axis:

  self_generated      — arose from Aura's own internal process (a reflection).
  memory_supported    — grounded in a few trusted autobiographical memories.
  internalized_pattern — many trusted memories converge across clusters into a
                         self-reinforcing pattern with no single source: the
                         Stand Alone Complex proper.
  external_input      — fresh input with little internal support.
  possibly_implanted  — support is dominated by quarantined memories, or the
                        ghost-hack guard flags it: treat as suspect, not self.
  self_maintenance    — no content (an idle tick).

It reuses ``AdversarialMemoryScanner`` for the poisoning signal rather than
re-deriving trust, and takes the guard's risk as an input, so it composes with
the rest of the Ghost substrate instead of forking a parallel judgement. The
output feeds two real decisions in the facade: whether a thought may update the
self-model, and how it is bound into the mind-moment (a possibly-implanted
thought enters as low-confidence, high-salience — held at arm's length).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# Origin labels
SELF_GENERATED = "self_generated"
MEMORY_SUPPORTED = "memory_supported"
INTERNALIZED_PATTERN = "internalized_pattern"
EXTERNAL_INPUT = "external_input"
POSSIBLY_IMPLANTED = "possibly_implanted"
SELF_MAINTENANCE = "self_maintenance"

# Origins whose content must NOT be allowed to silently update the self-model.
SUSPECT_ORIGINS = frozenset({POSSIBLY_IMPLANTED})


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ProvenanceSignals:
    """The evidence a provenance judgement is made from."""

    has_text: bool = True
    internally_originated: bool = False
    trusted_support: int = 0
    quarantined_support: int = 0
    convergent_clusters: int = 0
    repetition: float = 0.0
    guard_risk: float = 0.0


@dataclass(frozen=True)
class ProvenanceVerdict:
    origin: str
    complex_score: float          # how much this is an internalised complex (SAC-ness)
    confidence: float             # how much evidence backs the classification
    rationale: str
    signals: ProvenanceSignals = field(default_factory=ProvenanceSignals)

    @property
    def is_suspect(self) -> bool:
        return self.origin in SUSPECT_ORIGINS

    @property
    def may_update_self(self) -> bool:
        """Only genuinely self-owned, well-supported thought reshapes the self."""
        return self.origin in {SELF_GENERATED, MEMORY_SUPPORTED, INTERNALIZED_PATTERN}

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "complex_score": round(self.complex_score, 4),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
        }


def classify(signals: ProvenanceSignals) -> ProvenanceVerdict:
    """Classify a thought's origin from its evidence."""
    total_support = signals.trusted_support + signals.quarantined_support

    if not signals.has_text:
        origin = SELF_MAINTENANCE
        rationale = "no content — homeostatic tick"
    elif signals.guard_risk >= 0.4 or (
        signals.quarantined_support > 0 and signals.trusted_support == 0
    ):
        origin = POSSIBLY_IMPLANTED
        rationale = (
            "support is quarantined and/or the ghost-hack guard flagged it — "
            "held as suspect external input, not a self-fact"
        )
    elif signals.internally_originated:
        origin = SELF_GENERATED
        rationale = "arose from Aura's own internal process"
    elif signals.trusted_support >= 4 and signals.convergent_clusters >= 2:
        origin = INTERNALIZED_PATTERN
        rationale = (
            "many trusted memories converge across clusters — a self-reinforcing "
            "pattern with no single source (Stand Alone Complex)"
        )
    elif signals.trusted_support >= 2:
        origin = MEMORY_SUPPORTED
        rationale = "grounded in trusted autobiographical memory"
    else:
        origin = EXTERNAL_INPUT
        rationale = "fresh external input with little internal support"

    complex_score = _clamp(
        0.12 * signals.trusted_support
        + 0.08 * signals.convergent_clusters
        + 0.30 * signals.repetition
        - 0.15 * signals.quarantined_support
    )
    # Confidence rises with evidence and with a clear guard signal.
    confidence = _clamp(
        0.35
        + 0.10 * min(6, total_support)
        + (0.25 if signals.guard_risk >= 0.4 else 0.0)
        + (0.15 if not signals.has_text else 0.0)
    )
    return ProvenanceVerdict(
        origin=origin,
        complex_score=complex_score,
        confidence=confidence,
        rationale=rationale,
        signals=signals,
    )


def signals_from_recall(
    text: str,
    hits: Optional[Sequence[dict[str, Any]]] = None,
    *,
    guard_risk: float = 0.0,
    internally_originated: bool = False,
    use_scanner: bool = True,
) -> ProvenanceSignals:
    """Derive provenance signals from memory-recall hits.

    Each hit is a dict that may carry ``layer``, ``trust``, ``tags``, and
    ``content``. Quarantine is decided by the existing AdversarialMemoryScanner
    when content is present (so the poisoning judgement is shared, not forked),
    falling back to layer/trust otherwise.
    """
    hits = list(hits or [])
    trusted = 0
    quarantined = 0
    clusters: set[str] = set()

    scanner = None
    if use_scanner and any("content" in h for h in hits):
        try:
            from core.memory.adversarial_memory import AdversarialMemoryScanner
            scanner = AdversarialMemoryScanner()
        except (ImportError, AttributeError):
            scanner = None

    for hit in hits:
        layer = str(hit.get("layer", "")).lower()
        trust = float(hit.get("trust", 0.5) or 0.5)
        is_quarantined = layer == "quarantine" or trust < 0.25
        if scanner is not None and "content" in hit:
            try:
                is_quarantined = bool(scanner.score(str(hit.get("content", "")))["quarantine"])
            except (KeyError, TypeError, ValueError):
                pass
        if is_quarantined:
            quarantined += 1
        else:
            trusted += 1
            for tag in hit.get("tags", []) or []:
                clusters.add(str(tag))

    repetition = _clamp(trusted / 6.0)
    return ProvenanceSignals(
        has_text=bool((text or "").strip()),
        internally_originated=internally_originated,
        trusted_support=trusted,
        quarantined_support=quarantined,
        convergent_clusters=len(clusters),
        repetition=repetition,
        guard_risk=guard_risk,
    )


def classify_thought(
    text: str,
    hits: Optional[Sequence[dict[str, Any]]] = None,
    *,
    guard_risk: float = 0.0,
    internally_originated: bool = False,
) -> ProvenanceVerdict:
    """Convenience: build signals from recall + classify in one call."""
    return classify(
        signals_from_recall(
            text,
            hits,
            guard_risk=guard_risk,
            internally_originated=internally_originated,
        )
    )


__all__ = [
    "ProvenanceSignals",
    "ProvenanceVerdict",
    "classify",
    "classify_thought",
    "signals_from_recall",
    "SELF_GENERATED",
    "MEMORY_SUPPORTED",
    "INTERNALIZED_PATTERN",
    "EXTERNAL_INPUT",
    "POSSIBLY_IMPLANTED",
    "SELF_MAINTENANCE",
    "SUSPECT_ORIGINS",
]
