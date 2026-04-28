"""core/memory/provenance.py

Universal provenance tags for every memory write.

Every memory record carries a provenance envelope that answers:

    when_created       — wall-clock timestamp
    source             — "user_provided" | "self_inferred" | "observed"
                          | "generated" | "consolidated" | "imported"
    confidence         — 0..1, calibrated against the substrate's
                          uncertainty estimate at the moment of write
    contested          — bool; true iff a contradicting record already
                          exists in the same scope
    identity_relevant  — bool; true iff this record contributes to the
                          identity continuity hash inputs
    recalled_in_actions — list of action receipt IDs that have used this
                          memory; appended every time the record is
                          retrieved into a working-memory context
    reviewed_at        — last review timestamp (for stale detection)

The envelope is wrapped around the payload at write time and unwrapped
at read time. ``wrap()`` and ``unwrap()`` are the only legal touchpoints.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.MemoryProvenance")


@dataclass
class Provenance:
    record_id: str = field(default_factory=lambda: f"M-{uuid.uuid4().hex[:14]}")
    when_created: float = field(default_factory=time.time)
    source: str = "self_inferred"
    confidence: float = 0.7
    contested: bool = False
    identity_relevant: bool = False
    recalled_in_actions: List[str] = field(default_factory=list)
    reviewed_at: Optional[float] = None
    schema_version: int = 1


@dataclass
class StampedMemory:
    payload: Any
    provenance: Provenance

    def to_dict(self) -> Dict[str, Any]:
        return {"payload": self.payload, "provenance": asdict(self.provenance)}


def wrap(
    payload: Any,
    *,
    source: str = "self_inferred",
    confidence: Optional[float] = None,
    identity_relevant: bool = False,
    contested: bool = False,
) -> StampedMemory:
    if confidence is None:
        confidence = _read_confidence_from_substrate()
    prov = Provenance(
        source=source,
        confidence=float(confidence),
        contested=contested,
        identity_relevant=identity_relevant,
    )
    return StampedMemory(payload=payload, provenance=prov)


def unwrap(record: Any) -> StampedMemory:
    if isinstance(record, StampedMemory):
        return record
    if isinstance(record, dict) and "provenance" in record and "payload" in record:
        prov_data = record.get("provenance") or {}
        prov = Provenance(**{k: v for k, v in prov_data.items() if k in Provenance.__dataclass_fields__})
        return StampedMemory(payload=record.get("payload"), provenance=prov)
    # Pre-provenance record: fabricate a conservative envelope so the
    # rest of the system can use the uniform API. The fact that this
    # record lacks a real envelope is itself recorded by the source tag.
    return StampedMemory(payload=record, provenance=Provenance(source="legacy_unstamped", confidence=0.4))


def annotate_recall(rec: StampedMemory, *, action_receipt_id: str) -> None:
    rec.provenance.recalled_in_actions.append(action_receipt_id)


def mark_reviewed(rec: StampedMemory) -> None:
    rec.provenance.reviewed_at = time.time()


def _read_confidence_from_substrate() -> float:
    try:
        from core.container import ServiceContainer
        fe = ServiceContainer.get("free_energy_engine", default=None)
        if fe is not None and getattr(fe, "current", None) is not None:
            cur = fe.current
            # Lower free energy → higher subjective confidence
            free_energy = float(getattr(cur, "free_energy", 0.5) or 0.5)
            return max(0.05, min(0.99, 1.0 - free_energy))
    except Exception:
        pass  # no-op: intentional
    return 0.7


__all__ = [
    "Provenance",
    "StampedMemory",
    "wrap",
    "unwrap",
    "annotate_recall",
    "mark_reviewed",
]
