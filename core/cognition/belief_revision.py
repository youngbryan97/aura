"""Belief Revision System — Aura's Epistemic Engine

Manages Aura's beliefs about the world with confidence tracking,
evidence-based revision, and contradiction detection.

A belief is a proposition Aura holds to be true, with:
  - Confidence (0.0 to 1.0) — how strongly Aura believes it
  - Evidence — what supports the belief
  - Source — where the belief came from
  - Revision history — how the belief has changed over time

Key behaviors:
  - New evidence can strengthen or weaken beliefs
  - Contradictory evidence triggers belief revision
  - Beliefs decay slightly if not reinforced over time
  - The LLM can be asked to resolve contradictions

Design: Backed by the PersistentKnowledgeGraph for persistence.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Cognition.Beliefs")


@dataclass
class Belief:
    """A single proposition Aura believes to be true."""
    id: str
    proposition: str
    confidence: float              # 0.0 (no confidence) to 1.0 (certain)
    evidence: List[str]            # Supporting evidence
    source: str                    # Where the belief originated
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    revision_count: int = 0
    category: str = "general"      # "fact", "preference", "opinion", "rule", "self"
    active: bool = True            # False if retracted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Belief':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BeliefRevisionEngine:
    """Manages Aura's belief system with evidence-based updates.
    
    Architecture:
      - Beliefs are stored in the knowledge graph as type "belief"
      - Contradictions are detected via semantic similarity + LLM judgment  
      - Revision follows a Bayesian-like update: new evidence adjusts confidence
    """

    # Confidence thresholds
    CERTAIN = 0.95
    STRONG = 0.8
    MODERATE = 0.5
    WEAK = 0.3
    RETRACTED = 0.05

    def __init__(self, knowledge_graph=None, brain=None):
        self._kg = knowledge_graph
        self._brain = brain
        self._beliefs: Dict[str, Belief] = {}
        self._load_beliefs()

    def _load_beliefs(self):
        """Load beliefs from knowledge graph on startup."""
        if not self._kg:
            return
        try:
            nodes = self._kg.search_knowledge("", type="belief", limit=500)
            for node in nodes:
                meta = json.loads(node.get("metadata", "{}"))
                belief = Belief(
                    id=node["id"],
                    proposition=node["content"],
                    confidence=node.get("confidence", 0.5),
                    evidence=meta.get("evidence", []),
                    source=node.get("source", "unknown"),
                    created_at=node.get("created_at", time.time()),
                    last_updated=meta.get("last_updated", time.time()),
                    revision_count=meta.get("revision_count", 0),
                    category=meta.get("category", "general"),
                    active=meta.get("active", True),
                )
                self._beliefs[belief.id] = belief
            logger.info("📖 Loaded %d beliefs from knowledge graph", len(self._beliefs))
        except Exception as e:
            logger.warning("Failed to load beliefs: %s", e)

    def believe(
        self,
        proposition: str,
        confidence: float = 0.7,
        evidence: str = "",
        source: str = "conversation",
        category: str = "general",
    ) -> Belief:
        """Form a new belief or strengthen an existing one.
        
        If a matching belief exists, confidence is updated via Bayesian-like averaging.
        If not, a new belief is created and persisted.
        """
        belief_id = hashlib.sha256(proposition.lower().strip().encode()).hexdigest()[:16]

        if belief_id in self._beliefs:
            existing = self._beliefs[belief_id]
            # Bayesian-like update: weighted average with existing confidence
            n = existing.revision_count + 1
            existing.confidence = (existing.confidence * n + confidence) / (n + 1)
            existing.confidence = min(1.0, max(0.0, existing.confidence))
            existing.revision_count += 1
            existing.last_updated = time.time()
            if evidence:
                existing.evidence.append(evidence)
                existing.evidence = existing.evidence[-10:]  # Keep last 10
            self._persist_belief(existing)
            logger.info("📖 Belief reinforced (%.2f): %s", existing.confidence, proposition[:60])
            return existing

        belief = Belief(
            id=belief_id,
            proposition=proposition,
            confidence=confidence,
            evidence=[evidence] if evidence else [],
            source=source,
            category=category,
        )
        self._beliefs[belief_id] = belief
        self._persist_belief(belief)

        logger.info("📖 New belief (%.2f): %s", confidence, proposition[:60])
        
        try:
            from core.thought_stream import get_emitter
            get_emitter().emit(
                "Belief Formed 📖",
                f"{proposition[:80]} (confidence: {confidence:.0%})",
                level="info",
                category="Cognition"
            )
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return belief

    async def challenge(self, proposition: str, counter_evidence: str) -> Dict[str, Any]:
        """Challenge an existing belief with new evidence.
        
        Uses the LLM to evaluate whether the counter-evidence is strong enough
        to revise the belief. Returns the revision outcome.
        """
        belief_id = hashlib.sha256(proposition.lower().strip().encode()).hexdigest()[:16]
        belief = self._beliefs.get(belief_id)

        if not belief:
            return {"revised": False, "reason": "No matching belief found"}

        if not self._brain:
            # Without LLM, apply simple confidence reduction
            belief.confidence = max(0.1, belief.confidence - 0.15)
            belief.evidence.append(f"[COUNTER] {counter_evidence}")
            belief.revision_count += 1
            belief.last_updated = time.time()
            self._persist_belief(belief)
            return {
                "revised": True,
                "old_confidence": belief.confidence + 0.15,
                "new_confidence": belief.confidence,
                "method": "simple_reduction"
            }

        # LLM-based evaluation
        try:
            prompt = (
                f"You are evaluating whether new evidence should revise a belief.\n\n"
                f"CURRENT BELIEF (confidence {belief.confidence:.0%}):\n"
                f"  \"{belief.proposition}\"\n"
                f"  Evidence: {'; '.join(belief.evidence[-3:])}\n\n"
                f"NEW COUNTER-EVIDENCE:\n"
                f"  \"{counter_evidence}\"\n\n"
                f"On a scale of 0.0 to 1.0, what should the revised confidence be?\n"
                f"Respond with ONLY a JSON object: "
                f'{{\"revised_confidence\": 0.X, \"reasoning\": \"...\"}}'
            )

            response = await self._brain.generate(prompt, use_strategies=False)

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                old_conf = belief.confidence
                belief.confidence = max(0.0, min(1.0, float(result.get("revised_confidence", belief.confidence))))
                belief.evidence.append(f"[COUNTER] {counter_evidence}")
                belief.revision_count += 1
                belief.last_updated = time.time()
                
                # Retract if confidence drops too low
                if belief.confidence < self.RETRACTED:
                    belief.active = False
                    logger.info("📖 Belief retracted: %s", belief.proposition[:60])
                
                self._persist_belief(belief)

                try:
                    from core.thought_stream import get_emitter
                    get_emitter().emit(
                        "Belief Revised 📖",
                        f"{belief.proposition[:60]}: {old_conf:.0%} → {belief.confidence:.0%}",
                        level="warning" if abs(old_conf - belief.confidence) > 0.2 else "info",
                        category="Cognition"
                    )
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)

                return {
                    "revised": True,
                    "old_confidence": old_conf,
                    "new_confidence": belief.confidence,
                    "reasoning": result.get("reasoning", ""),
                    "active": belief.active,
                }

        except Exception as e:
            logger.debug("LLM belief revision failed: %s", e)

        return {"revised": False, "reason": "evaluation_failed"}

    def get_beliefs(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        active_only: bool = True,
    ) -> List[Belief]:
        """Retrieve beliefs matching criteria."""
        beliefs = list(self._beliefs.values())
        if active_only:
            beliefs = [b for b in beliefs if b.active]
        if category:
            beliefs = [b for b in beliefs if b.category == category]
        if min_confidence > 0:
            beliefs = [b for b in beliefs if b.confidence >= min_confidence]
        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs

    def get_context_beliefs(self, query: str, limit: int = 5) -> str:
        """Get beliefs relevant to a query, formatted for prompt injection."""
        words = set(query.lower().split())
        scored = []
        for belief in self._beliefs.values():
            if not belief.active:
                continue
            prop_words = set(belief.proposition.lower().split())
            overlap = len(words & prop_words)
            if overlap > 0:
                score = overlap * belief.confidence
                scored.append((score, belief))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]

        lines = ["[Active Beliefs]"]
        for _, b in top:
            lines.append(f"- ({b.confidence:.0%}) {b.proposition}")
        return "\n".join(lines) + "\n"

    def get_stats(self) -> Dict[str, Any]:
        """Belief system statistics."""
        active = [b for b in self._beliefs.values() if b.active]
        return {
            "total_beliefs": len(self._beliefs),
            "active_beliefs": len(active),
            "retracted": len(self._beliefs) - len(active),
            "avg_confidence": (
                sum(b.confidence for b in active) / max(1, len(active))
            ),
            "categories": list(set(b.category for b in active)),
            "most_revised": max(
                (b.revision_count for b in self._beliefs.values()), default=0
            ),
        }

    def _persist_belief(self, belief: Belief):
        """Save belief to knowledge graph."""
        if not self._kg:
            return
        try:
            metadata = {
                "evidence": belief.evidence[-10:],
                "last_updated": belief.last_updated,
                "revision_count": belief.revision_count,
                "category": belief.category,
                "active": belief.active,
            }
            # Use add_knowledge which does upsert
            self._kg.add_knowledge(
                content=belief.proposition,
                type="belief",
                source=belief.source,
                confidence=belief.confidence,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("Failed to persist belief: %s", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[BeliefRevisionEngine] = None


def get_belief_engine(knowledge_graph=None, brain=None) -> BeliefRevisionEngine:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = BeliefRevisionEngine(knowledge_graph=knowledge_graph, brain=brain)
    return _instance
