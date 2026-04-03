import logging
import asyncio
from typing import Any, Dict, List

logger = logging.getLogger("Cybernetics.TheMachine")

class TheMachine:
    """
    [ZENITH] The Machine: Social graph threat propagation (PageRank variant).
    Scores external event sources based on interaction history.
    """
    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self._event_bus = None
        # source_id -> threat_score
        self._trust_graph: Dict[str, float] = {}
        # adjacency list for propagation
        self._influence_map: Dict[str, List[str]] = {}

    async def load(self):
        try:
            from core.event_bus import get_event_bus
            self._event_bus = get_event_bus()
        except ImportError as _exc:
            logger.debug("Suppressed ImportError: %s", _exc)
        logger.info("👁️ [THE MACHINE] Social Graph Prophet ACTIVE. Sourcing PoIs.")

    async def register_interaction(self, source_id: str, success: bool, related_sources: List[str] = ()):
        """Update threat scores using PageRank flow logic."""
        # 1. Update base score
        current = self._trust_graph.get(source_id, 0.5)
        adjustment = -0.1 if success else 0.2
        self._trust_graph[source_id] = max(0.0, min(1.0, current + adjustment))
        
        # 2. Update influence map
        for related in related_sources:
            if related not in self._influence_map.setdefault(source_id, []):
                self._influence_map[source_id].append(related)
        
        # 3. Propagate threat (simplified PageRank iteration)
        await self._propagate_threat()

    async def _propagate_threat(self):
        """Dampened PageRank-style propagation: Threat flows through edges."""
        new_scores = self._trust_graph.copy()
        for source, relatives in self._influence_map.items():
            if not relatives: continue
            threat_flow = self._trust_graph.get(source, 0.5) * 0.15
            for rel in relatives:
                cur_rel = new_scores.get(rel, 0.5)
                new_scores[rel] = min(1.0, cur_rel + (threat_flow / len(relatives)))
        
        self._trust_graph = new_scores
        
        if any(t > 0.8 for t in self._trust_graph.values()):
            logger.warning("📍 [THE MACHINE] HIGH-RISK PARTICIPANT DETECTED in social graph.")

    def get_threat(self, source_id: str) -> float:
        return self._trust_graph.get(source_id, 0.5)
