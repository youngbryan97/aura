"""core/world/claim_store.py — Living Claim Store & Graph.

Stores factual claims about the external world with:
  source, timestamp, confidence, contradiction links, freshness,
  uncertainty, affected missions, and possible actions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.ClaimStore")


@dataclass
class WorldClaim:
    claim_id: str
    content: str
    source: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    freshness: float = 1.0  # 1.0 is fresh, decaying to 0.0 over time
    uncertainty: float = 0.5
    contradiction_links: List[str] = field(default_factory=list)
    affected_missions: List[str] = field(default_factory=list)
    possible_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaimStore:
    """Stores and indexes all claims ingested from external sources."""

    def __init__(self) -> None:
        self.claims: Dict[str, WorldClaim] = {}

    def add_claim(
        self,
        content: str,
        source: str,
        *,
        confidence: float = 0.5,
        uncertainty: float = 0.5,
        affected_missions: Optional[List[str]] = None,
        possible_actions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorldClaim:
        claim_id = f"claim_{int(time.time())}_{hash(content) % 100000}"
        claim = WorldClaim(
            claim_id=claim_id,
            content=content,
            source=source,
            timestamp=time.time(),
            confidence=confidence,
            uncertainty=uncertainty,
            affected_missions=affected_missions or [],
            possible_actions=possible_actions or [],
            metadata=metadata or {},
        )
        self.claims[claim_id] = claim
        logger.info("📋 Claim Ingested: '%s' from %s", content[:60], source)
        return claim

    def get_claim(self, claim_id: str) -> Optional[WorldClaim]:
        return self.claims.get(claim_id)

    def link_contradiction(self, cid1: str, cid2: str) -> None:
        c1 = self.claims.get(cid1)
        c2 = self.claims.get(cid2)
        if c1 and c2:
            if cid2 not in c1.contradiction_links:
                c1.contradiction_links.append(cid2)
            if cid1 not in c2.contradiction_links:
                c2.contradiction_links.append(cid1)
            c1.uncertainty = min(1.0, c1.uncertainty + 0.2)
            c2.uncertainty = min(1.0, c2.uncertainty + 0.2)

    def decay_freshness(self, decay_rate_per_hour: float = 0.01) -> None:
        now = time.time()
        for claim in self.claims.values():
            age_hours = (now - claim.timestamp) / 3600.0
            claim.freshness = max(0.0, 1.0 - (age_hours * decay_rate_per_hour))

    def get_fresh_claims(self, threshold: float = 0.5) -> List[WorldClaim]:
        self.decay_freshness()
        return [c for c in self.claims.values() if c.freshness >= threshold]

    def query_by_mission(self, mission_id: str) -> List[WorldClaim]:
        return [c for c in self.claims.values() if mission_id in c.affected_missions]
