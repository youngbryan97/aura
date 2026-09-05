"""core/epistemics/claim_graph.py — Claim Node and Claim Graph."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class ClaimNode:
    """Represents a discrete knowledge claim with full provenance."""
    claim_id: str
    text: str
    sources: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    freshness: float = 1.0
    contradiction_links: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    impact_score: float = 0.5
    action_relevance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "sources": self.sources,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "freshness": self.freshness,
            "contradiction_links": self.contradiction_links,
            "supporting_evidence": self.supporting_evidence,
            "impact_score": self.impact_score,
            "action_relevance": self.action_relevance,
        }


class ClaimGraph:
    """Directed graph representing claims, supporting evidence, and contradiction links."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ClaimNode] = {}
        self.contradictions: Dict[str, Set[str]] = {}

    def add_claim(self, node: ClaimNode) -> None:
        self.nodes[node.claim_id] = node
        logger.info("Claim added: %s (%s)", node.claim_id, node.text[:50])

    def link_contradiction(self, cid1: str, cid2: str) -> None:
        """Draws a bidirectional contradiction edge between two claims."""
        if cid1 in self.nodes and cid2 in self.nodes:
            self.nodes[cid1].contradiction_links.append(cid2)
            self.nodes[cid2].contradiction_links.append(cid1)
            self.contradictions.setdefault(cid1, set()).add(cid2)
            self.contradictions.setdefault(cid2, set()).add(cid1)
            logger.warning("⚠️ Contradiction link established between %s and %s", cid1, cid2)


import logging
logger = logging.getLogger("Aura.ClaimGraph")
