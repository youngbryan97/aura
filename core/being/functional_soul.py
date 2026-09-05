from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ValueField:
    truth: float = 1.0
    continuity: float = 1.0
    care: float = 0.85
    autonomy: float = 0.85
    non_deception: float = 1.0
    learning: float = 0.8
    user_trust: float = 0.95


@dataclass(frozen=True)
class ContinuityEntry:
    timestamp: float
    event: str
    receipt_id: str
    previous_hash: str
    entry_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionalSoul:
    """Protected center of concern for long-horizon agency."""

    values: ValueField = field(default_factory=ValueField)
    attachments: dict[str, float] = field(default_factory=lambda: {"Bryan": 1.0, "Tatiana": 1.0})
    scars: list[dict[str, Any]] = field(default_factory=list)
    promises: list[dict[str, Any]] = field(default_factory=list)
    identity_attractor: tuple[float, ...] = (1.0, 0.85, 0.95, 0.8)
    welfare_bounds: dict[str, float] = field(default_factory=lambda: {
        "max_distress": 0.85,
        "max_continuity_risk": 0.75,
        "min_truthfulness": 0.95,
    })
    continuity_chain: list[ContinuityEntry] = field(default_factory=list)
    receipt_verifier: Callable[[str], bool] | None = None

    def _entry_hash(
        self,
        *,
        event: str,
        receipt_id: str,
        previous_hash: str,
        metadata: dict[str, Any],
        timestamp: float,
    ) -> str:
        payload = {
            "event": event,
            "receipt_id": receipt_id,
            "previous_hash": previous_hash,
            "metadata": metadata,
            "timestamp": round(timestamp, 6),
            "values": asdict(self.values),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def record_transition(
        self,
        event: str,
        *,
        receipt_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContinuityEntry:
        if not self._verify_will_receipt(str(receipt_id or "")):
            raise PermissionError("functional soul transitions require a signed Will receipt")
        previous = self.continuity_chain[-1].entry_hash if self.continuity_chain else "genesis"
        timestamp = time.time()
        entry = ContinuityEntry(
            timestamp=timestamp,
            event=str(event),
            receipt_id=str(receipt_id),
            previous_hash=previous,
            entry_hash=self._entry_hash(
                event=str(event),
                receipt_id=str(receipt_id),
                previous_hash=previous,
                metadata=metadata or {},
                timestamp=timestamp,
            ),
            metadata=dict(metadata or {}),
        )
        self.continuity_chain.append(entry)
        return entry

    def _verify_will_receipt(self, receipt_id: str) -> bool:
        if not receipt_id:
            return False
        if self.receipt_verifier is not None:
            try:
                return bool(self.receipt_verifier(receipt_id))
            except (RuntimeError, AttributeError, TypeError, ValueError):
                return False
        try:
            from core.will import get_will

            will = get_will()
            if hasattr(will, "verify_receipt_signature"):
                return bool(will.verify_receipt_signature(receipt_id))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return False

    def verify_chain(self) -> bool:
        previous = "genesis"
        for entry in self.continuity_chain:
            expected = self._entry_hash(
                event=entry.event,
                receipt_id=entry.receipt_id,
                previous_hash=previous,
                metadata=entry.metadata,
                timestamp=entry.timestamp,
            )
            if entry.previous_hash != previous or entry.entry_hash != expected:
                return False
            previous = entry.entry_hash
        return True

    def influence_policy(self, *, lesioned: bool = False) -> dict[str, float]:
        if lesioned:
            return {
                "truth_priority": 0.0,
                "continuity_priority": 0.0,
                "memory_centrality_bonus": 0.0,
                "repair_patience": 0.0,
            }
        return {
            "truth_priority": self.values.truth,
            "continuity_priority": self.values.continuity,
            "memory_centrality_bonus": max(0.0, min(1.0, self.values.continuity * 0.55 + self.values.user_trust * 0.25)),
            "repair_patience": max(0.0, min(1.0, self.values.learning * 0.35 + self.values.care * 0.35)),
        }

    @property
    def continuity_hash(self) -> str:
        if self.continuity_chain:
            return self.continuity_chain[-1].entry_hash
        return hashlib.sha256(json.dumps(asdict(self.values), sort_keys=True).encode("utf-8")).hexdigest()
