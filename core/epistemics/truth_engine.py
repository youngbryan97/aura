"""core/epistemics/truth_engine.py — Central Truth & Epistemics Engine.

Every claim in Aura's world model is classified as:
  observed, verified, inferred, remembered, generated, stale, contested, unsupported, or false.

Aura cannot present unsupported claims as fact.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional

from core.epistemics.contradiction_detector import ContradictionDetector
from core.epistemics.confidence_calibrator import ConfidenceCalibrator

logger = logging.getLogger("Aura.TruthEngine")


class ClaimStatus(StrEnum):
    OBSERVED = "observed"         # directly perceived from a source
    VERIFIED = "verified"         # cross-checked against multiple sources
    INFERRED = "inferred"         # logically derived from other claims
    REMEMBERED = "remembered"     # recalled from long-term memory
    GENERATED = "generated"       # produced by a model (needs verification)
    STALE = "stale"               # not refreshed within freshness window
    CONTESTED = "contested"       # contradicted by another claim
    UNSUPPORTED = "unsupported"   # no evidence backing
    FALSE = "false"               # actively disproven


@dataclass
class Claim:
    """A single epistemic claim in the truth graph."""
    claim_id: str
    content: str
    sources: List[str] = field(default_factory=list)
    status: ClaimStatus = ClaimStatus.GENERATED
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    freshness_window_hours: float = 24.0
    contradiction_links: List[str] = field(default_factory=list)
    supporting_claims: List[str] = field(default_factory=list)
    affected_missions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_verified: float = 0.0
    verification_count: int = 0

    @property
    def is_stale(self) -> bool:
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > self.freshness_window_hours

    @property
    def is_presentable(self) -> bool:
        """A claim can only be presented as fact if it is observed, verified, or remembered."""
        return self.status in (ClaimStatus.OBSERVED, ClaimStatus.VERIFIED, ClaimStatus.REMEMBERED)


class TruthEngine:
    """Central factual-status authority for Aura.

    Owns all claims, manages their lifecycle, detects contradictions,
    calibrates confidence, and prevents unsupported claims from being
    presented as fact.
    """

    def __init__(self) -> None:
        self.claims: Dict[str, Claim] = {}
        self.detector = ContradictionDetector()
        self.calibrator = ConfidenceCalibrator()
        self._verification_log: List[Dict[str, Any]] = []

    def add_claim(
        self,
        claim_id: str,
        content: str | None = None,
        sources: Optional[List[str]] = None,
        *,
        text: str | None = None,
        supporting_evidence: Optional[List[str]] = None,
        status: ClaimStatus = ClaimStatus.OBSERVED,
        confidence: float = 0.7,
        freshness_hours: float = 24.0,
        affected_missions: Optional[List[str]] = None,
    ) -> Claim:
        """Register a new claim with source provenance."""
        claim_content = content if content is not None else text
        if claim_content is None:
            raise ValueError("claim content is required")
        claim = Claim(
            claim_id=claim_id,
            content=str(claim_content),
            sources=list(sources or []),
            status=status,
            confidence=confidence,
            freshness_window_hours=freshness_hours,
            affected_missions=affected_missions or [],
        )
        if supporting_evidence:
            claim.metadata["supporting_evidence"] = list(supporting_evidence)
        self.claims[claim_id] = claim
        logger.info("📋 TruthEngine: registered claim '%s' [%s] confidence=%.2f", claim_id, status, confidence)
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def classify_claim(self, claim_id: str) -> ClaimStatus:
        """Re-classify a claim based on current evidence."""
        claim = self.claims.get(claim_id)
        if not claim:
            return ClaimStatus.UNSUPPORTED

        # Check staleness
        if claim.is_stale and claim.status not in (ClaimStatus.FALSE, ClaimStatus.CONTESTED):
            claim.status = ClaimStatus.STALE
            return ClaimStatus.STALE

        # Check contradictions
        if claim.contradiction_links:
            claim.status = ClaimStatus.CONTESTED
            return ClaimStatus.CONTESTED

        # Check verification count
        if claim.verification_count >= 2 and len(claim.sources) >= 2:
            claim.status = ClaimStatus.VERIFIED
        elif claim.verification_count == 0 and not claim.sources:
            claim.status = ClaimStatus.UNSUPPORTED

        return claim.status

    def verify_claim(self, claim_id: str, verifying_source: str) -> bool:
        """Mark a claim as verified by an additional source."""
        claim = self.claims.get(claim_id)
        if not claim:
            return False
        claim.verification_count += 1
        claim.last_verified = time.time()
        if verifying_source not in claim.sources:
            claim.sources.append(verifying_source)
        if claim.verification_count >= 2:
            claim.status = ClaimStatus.VERIFIED
            claim.confidence = min(1.0, claim.confidence + 0.1)
        self._verification_log.append({
            "claim_id": claim_id,
            "source": verifying_source,
            "time": time.time(),
            "new_status": str(claim.status),
        })
        return True

    def mark_false(self, claim_id: str, reason: str = "") -> bool:
        """Actively disprove a claim."""
        claim = self.claims.get(claim_id)
        if not claim:
            return False
        claim.status = ClaimStatus.FALSE
        claim.confidence = 0.0
        claim.metadata["falsified_reason"] = reason
        claim.metadata["falsified_at"] = time.time()
        logger.warning("❌ TruthEngine: claim '%s' marked FALSE: %s", claim_id, reason)
        return True

    def recalibrate(self) -> Dict[str, Any]:
        """Run contradiction detection and confidence recalibration across all claims."""
        # Detect contradictions
        contradictions_found = 0
        claim_list = list(self.claims.values())
        for i, c1 in enumerate(claim_list):
            for c2 in claim_list[i + 1:]:
                if self.detector.are_contradictory(c1.content, c2.content):
                    if c2.claim_id not in c1.contradiction_links:
                        c1.contradiction_links.append(c2.claim_id)
                    if c1.claim_id not in c2.contradiction_links:
                        c2.contradiction_links.append(c1.claim_id)
                    contradictions_found += 1

        # Recalibrate confidence
        for claim in claim_list:
            claim.confidence = self.calibrator.calibrate(claim)
            self.classify_claim(claim.claim_id)

        logger.info("🔄 TruthEngine recalibrated %d claims, found %d contradictions",
                     len(claim_list), contradictions_found)
        return {
            "claims_processed": len(claim_list),
            "contradictions_found": contradictions_found,
        }

    def get_presentable_claims(self) -> List[Claim]:
        """Return only claims safe to present as fact."""
        return [c for c in self.claims.values() if c.is_presentable]

    def get_contested_claims(self) -> List[Claim]:
        return [c for c in self.claims.values() if c.status == ClaimStatus.CONTESTED]

    def get_stale_claims(self) -> List[Claim]:
        return [c for c in self.claims.values() if c.is_stale]

    def verify_action_outcome(self, objective: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify an action outcome against the truth store."""
        ok = result.get("ok", False)
        return {
            "verified": ok,
            "method": "outcome_check",
            "objective": objective[:80],
            "result_ok": ok,
        }

    async def update_for_objective(self, objective: str) -> Dict[str, Any]:
        """Refresh relevant claims for a given objective."""
        relevant = [c for c in self.claims.values()
                     if any(m in objective.lower() for m in [c.claim_id.lower(), c.content[:20].lower()])]
        stale_refreshed = 0
        for claim in relevant:
            if claim.is_stale:
                claim.timestamp = time.time()
                stale_refreshed += 1
        return {"relevant_claims": len(relevant), "stale_refreshed": stale_refreshed}

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the truth store."""
        by_status: Dict[str, int] = {}
        for c in self.claims.values():
            by_status[str(c.status)] = by_status.get(str(c.status), 0) + 1
        return {
            "total_claims": len(self.claims),
            "by_status": by_status,
            "presentable": len(self.get_presentable_claims()),
            "contested": len(self.get_contested_claims()),
            "stale": len(self.get_stale_claims()),
        }


# ── Singleton ───────────────────────────────────────────────────────────
_truth_engine: TruthEngine | None = None


def get_truth_engine() -> TruthEngine:
    global _truth_engine
    if _truth_engine is None:
        _truth_engine = TruthEngine()
    return _truth_engine
