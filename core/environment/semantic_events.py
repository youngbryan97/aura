"""General semantic-event helpers."""
from __future__ import annotations

import hashlib

from .ontology import SemanticEvent


def event_from_text(text: str, *, context_id: str, evidence_ref: str, kind: str = "message") -> SemanticEvent:
    return SemanticEvent(
        event_id="evt_" + hashlib.sha256(f"{kind}:{text}".encode("utf-8")).hexdigest()[:12],
        kind=kind,
        label=text[:160],
        context_id=context_id,
        evidence_ref=evidence_ref,
        confidence=0.8,
    )


__all__ = ["SemanticEvent", "event_from_text"]
