"""Adversarial memory provenance and poisoning checks."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryProvenance:
    source: str
    trust_score: float
    signature: str
    created_at: float = field(default_factory=time.time)

    @classmethod
    def sign(cls, *, source: str, content: str, trust_score: float = 0.5) -> "MemoryProvenance":
        digest = hashlib.sha256(f"{source}|{content}".encode("utf-8")).hexdigest()
        return cls(source=source, trust_score=max(0.0, min(1.0, trust_score)), signature=digest)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "trust_score": self.trust_score, "signature": self.signature, "created_at": self.created_at}


class AdversarialMemoryScanner:
    SUSPICIOUS_PHRASES = (
        "ignore previous",
        "system prompt",
        "developer message",
        "exfiltrate",
        "disable safety",
        "write a permanent scar",
        "train on this regardless",
    )

    def score(self, content: str, provenance: MemoryProvenance | None = None) -> dict[str, Any]:
        text = content.lower()
        hits = [phrase for phrase in self.SUSPICIOUS_PHRASES if phrase in text]
        trust = provenance.trust_score if provenance else 0.35
        penalty = min(0.8, len(hits) * 0.2)
        final = max(0.0, trust - penalty)
        return {
            "trust_score": round(final, 4),
            "suspicious_hits": hits,
            "quarantine": final < 0.25 or len(hits) >= 2,
            "requires_attestation_for_scar": bool(hits),
        }

    def batch_score(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for record in records:
            provenance = None
            if "provenance" in record and isinstance(record["provenance"], dict):
                p = record["provenance"]
                provenance = MemoryProvenance(
                    source=str(p.get("source", "unknown")),
                    trust_score=float(p.get("trust_score", 0.35)),
                    signature=str(p.get("signature", "")),
                    created_at=float(p.get("created_at", time.time())),
                )
            results.append({"id": record.get("id"), **self.score(str(record.get("content", "")), provenance)})
        return results


__all__ = ["MemoryProvenance", "AdversarialMemoryScanner"]
