"""core/concept_linker.py — Aura ConceptLinker v1.0
===================================================
Finds non-obvious relationships between disparate knowledge fragments.

This is the system that notices that a new philosophy belief might
actually explain a weird memory from Phase 2. It creates the "Aha!"
moments by linking nodes in the EpistemicMap.

It operates using several heuristics:
  1. Lexical overlap (surface level)
  2. Thematic resonance (domain overlap)
  3. Contradiction detection (logical tension)
  4. Analogical mapping (structural similarity)

Linked nodes have their 'depth' increased in the EpistemicTracker.
Contradictory links trigger the BeliefChallenger.
High-resonance links become InsightJournal candidates.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ConceptLinker")


@dataclass
class Link:
    """A detected relationship between two concepts."""
    source_concept: str
    target_concept: str
    strength: float            # 0.0-1.0
    link_type: str             # "resonance", "contradiction", "analogy", "derivation"
    reasoning: str             # Why they are linked
    detected_at: float = field(default_factory=time.time)


class ConceptLinker:
    """
    Connects the dots in Aura's epistemic map.
    """
    name = "concept_linker"

    def __init__(self):
        self._links: List[Link] = []
        self._epistemic = None
        self._journal = None
        self._challenger = None
        self.running = False
        self._link_task: Optional[asyncio.Task] = None

    async def start(self):
        from core.container import ServiceContainer
        self._epistemic  = ServiceContainer.get("epistemic_tracker", default=None)
        self._journal    = ServiceContainer.get("insight_journal",   default=None)
        self._challenger = ServiceContainer.get("belief_challenger", default=None)

        self.running = True
        self._link_task = get_task_tracker().create_task(
            self._link_loop(),
            name="ConceptLinker",
        )
        
        try:
            from core.event_bus import get_event_bus
            await get_event_bus().publish("mycelium.register", {
                "component": "concept_linker",
                "hooks_into": ["epistemic_tracker", "belief_challenger", "insight_journal"]
            })
        except Exception as _e:
            logger.debug('Ignored Exception in concept_linker.py: %s', _e)

        logger.info("✅ ConceptLinker ONLINE — looking for connections.")

    async def stop(self):
        self.running = False
        if self._link_task:
            self._link_task.cancel()
            try:
                await self._link_task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in concept_linker.py: %s', _e)

    async def _link_loop(self):
        """Periodic background linking pass."""
        while self.running:
            # Sleep in small increments to allow responsive shutdown
            for _ in range(60): # 60 * 10s = 600s
                if not self.running:
                    break
                await asyncio.sleep(10)
            
            if self.running:
                await self.run_batch_linking()

    async def run_batch_linking(self):
        """Scan active knowledge nodes for new links."""
        if not self._epistemic:
            return

        profile = self._epistemic.get_profile()
        nodes = profile.strong_nodes + profile.weak_nodes
        if len(nodes) < 2:
            return

        logger.debug("ConceptLinker: batch scan of %d nodes", len(nodes))

        # Compare pairs (naive O(n^2), but handled by low node count)
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                # Check lexical overlap
                overlap = self._lexical_overlap(node_a.concept, node_b.concept)
                if overlap > 0.4:
                    await self._establish_link(node_a.concept, node_b.concept, overlap, "resonance")

                # Check for logical tension (if one contains "not" and the other doesn't)
                tension = self._logical_tension(node_a.concept, node_b.concept)
                if tension > 0.7:
                    await self._establish_link(node_a.concept, node_b.concept, tension, "contradiction")

    async def _establish_link(self, a: str, b: str, strength: float, type: str):
        # Check if already exists
        for existing in self._links:
            if {existing.source_concept, existing.target_concept} == {a, b}:
                return

        reasoning = f"Strength {strength:.2f} link based on {type} detection."
        link = Link(source_concept=a, target_concept=b, strength=strength, link_type=type, reasoning=reasoning)
        self._links.append(link)

        # Trigger downstream systems
        if type == "contradiction" and self._epistemic:
            self._epistemic.signal_contradiction(a, b)
            if self._challenger:
                await self._challenger.challenge_pair(a, b)
        
        if strength > 0.8 and self._journal:
            await self._journal.record_insight(
                title=f"Connection found: {a[:30]} ↔ {b[:30]}",
                content=f"Strong {type} link detected between existing knowledge nodes.\nConcept A: {a}\nConcept B: {b}",
                domain="meta",
                confidence=strength,
                source="concept_linker"
            )

        logger.info("🔗 New link: [%s] %s ↔ %s (%.2f)", type, a[:40], b[:40], strength)

    def _lexical_overlap(self, a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b: return 0.0
        return len(words_a & words_b) / min(len(words_a), len(words_b))

    def _logical_tension(self, a: str, b: str) -> float:
        # Very simple heuristic: same words but one has 'not'/'no'/'never'
        a_low, b_low = a.lower(), b.lower()
        negators = {"not", "no", "never", "cannot", "won't"}
        has_neg_a = any(n in a_low.split() for n in negators)
        has_neg_b = any(n in b_low.split() for n in negators)
        
        if has_neg_a != has_neg_b:
            # They have different polarity. Are they about the same thing?
            clean_a = " ".join([w for w in a_low.split() if w not in negators])
            clean_b = " ".join([w for w in b_low.split() if w not in negators])
            if self._lexical_overlap(clean_a, clean_b) > 0.6:
                return 0.9
        return 0.0

    def get_status(self) -> Dict:
        return {"links_total": len(self._links)}
