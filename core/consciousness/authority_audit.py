"""core/consciousness/authority_audit.py — Runtime Causal Receipt System

Every SubstrateAuthority.authorize() call produces a receipt with a unique ID,
the original content hash, source, category, priority, and decision.

Every outward effect MUST carry the receipt_id it was authorized under.
Matching is exact on receipt_id — no fuzzy source/category guessing.

An effect without a valid receipt_id is flagged as UNMATCHED.

Queryable at any time via get_audit().verify() or the /audit endpoint.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

logger = logging.getLogger("Consciousness.AuthorityAudit")

_MAX_ENTRIES = 5000


@dataclass
class AuthorityReceipt:
    """Immutable record of one SubstrateAuthority.authorize() call."""
    receipt_id: str           # unique ID (hash of timestamp + source + content)
    timestamp: float
    content_hash: str         # first 80 chars of authorized content
    source: str               # original source passed to authorize()
    category: str             # ActionCategory.name
    priority: float
    decision: str             # ALLOW | CONSTRAIN | BLOCK | CRITICAL_PASS
    reason: str


@dataclass
class EffectRecord:
    """Record of an outward effect and its authority receipt."""
    timestamp: float
    effect_type: str          # response | memory_write | tool_execution | belief_update | expression | state_mutation
    source: str               # who produced this effect
    content_hash: str         # first 80 chars
    receipt_id: Optional[str] # must match a receipt — None = unmatched
    matched: bool = False


def _make_receipt_id(timestamp: float, source: str, content: str) -> str:
    """Deterministic receipt ID from the authorization parameters."""
    raw = f"{timestamp:.6f}:{source}:{content[:80]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class AuthorityAudit:
    """Live causal receipt system.

    Thread-safe. Every authorize() produces a receipt.
    Every outward effect must carry that receipt's ID.
    verify() reports exact match coverage.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._receipts: Deque[AuthorityReceipt] = deque(maxlen=_MAX_ENTRIES)
        self._effects: Deque[EffectRecord] = deque(maxlen=_MAX_ENTRIES)
        self._receipt_ids: Set[str] = set()  # fast lookup for matching
        self._allow_ids: Set[str] = set()    # only ALLOW/CONSTRAIN/CRITICAL_PASS
        self._total_receipts: int = 0
        self._total_effects: int = 0
        self._matched_count: int = 0
        self._unmatched_count: int = 0

    # ── Receipt recording (called by SubstrateAuthority) ─────────────

    def record_receipt(
        self,
        receipt_id: str,
        content: str,
        source: str,
        category: str,
        priority: float,
        decision: str,
        reason: str = "",
    ) -> None:
        with self._lock:
            receipt = AuthorityReceipt(
                receipt_id=receipt_id,
                timestamp=time.time(),
                content_hash=content[:80] if content else "",
                source=source,
                category=category,
                priority=round(priority, 4),
                decision=decision,
                reason=reason,
            )
            self._receipts.append(receipt)
            self._receipt_ids.add(receipt_id)
            if decision in ("ALLOW", "CONSTRAIN", "CRITICAL_PASS"):
                self._allow_ids.add(receipt_id)
            self._total_receipts += 1

            # Evict old IDs if set grows too large
            if len(self._receipt_ids) > _MAX_ENTRIES * 2:
                keep = {r.receipt_id for r in self._receipts}
                self._receipt_ids = keep
                self._allow_ids = {rid for rid in self._allow_ids if rid in keep}

    # ── Effect recording (called at output points) ───────────────────

    def record_effect(
        self,
        effect_type: str,
        source: str,
        content: str = "",
        receipt_id: Optional[str] = None,
    ) -> None:
        """Record an outward effect.

        receipt_id: the ID returned by SubstrateAuthority.authorize().
                    If None or not found in allow_ids, the effect is UNMATCHED.
        """
        with self._lock:
            matched = receipt_id is not None and receipt_id in self._allow_ids

            record = EffectRecord(
                timestamp=time.time(),
                effect_type=effect_type,
                source=source,
                content_hash=content[:80] if content else "",
                receipt_id=receipt_id,
                matched=matched,
            )
            self._effects.append(record)
            self._total_effects += 1

            if matched:
                self._matched_count += 1
            else:
                self._unmatched_count += 1
                logger.warning(
                    "⚠️ UNMATCHED EFFECT: type=%s source=%s receipt_id=%s",
                    effect_type, source, receipt_id,
                )

    # ── Verification ─────────────────────────────────────────────────

    def verify(self) -> Dict:
        """Return full audit report with exact provenance matching."""
        with self._lock:
            unmatched = [
                {
                    "timestamp": round(e.timestamp, 3),
                    "type": e.effect_type,
                    "source": e.source,
                    "content": e.content_hash,
                    "receipt_id": e.receipt_id,
                }
                for e in self._effects if not e.matched
            ]

            total = self._total_effects
            coverage = self._matched_count / max(1, total)

            return {
                "total_receipts": self._total_receipts,
                "total_effects": self._total_effects,
                "matched_effects": self._matched_count,
                "unmatched_effects": self._unmatched_count,
                "coverage_ratio": round(coverage, 4),
                "recent_unmatched": unmatched[-20:],
                "verdict": "CLEAN" if self._unmatched_count == 0 else "UNMATCHED_EFFECTS_FOUND",
            }

    def get_recent_receipts(self, n: int = 20) -> List[Dict]:
        with self._lock:
            return [
                {
                    "receipt_id": r.receipt_id,
                    "source": r.source,
                    "category": r.category,
                    "decision": r.decision,
                    "content": r.content_hash[:40],
                    "timestamp": round(r.timestamp, 3),
                }
                for r in list(self._receipts)[-n:]
            ]

    def get_recent_effects(self, n: int = 20) -> List[Dict]:
        with self._lock:
            return [
                {
                    "timestamp": round(e.timestamp, 3),
                    "effect_type": e.effect_type,
                    "source": e.source,
                    "content": e.content_hash,
                    "receipt_id": e.receipt_id,
                    "matched": e.matched,
                }
                for e in list(self._effects)[-n:]
            ]

    def get_status(self) -> Dict:
        return self.verify()


# ── Singleton ────────────────────────────────────────────────────────

_instance: Optional[AuthorityAudit] = None
_instance_lock = threading.Lock()


def get_audit() -> AuthorityAudit:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AuthorityAudit()
    return _instance
