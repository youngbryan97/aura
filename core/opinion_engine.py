"""core/opinion_engine.py
────────────────────────
Durable opinion formation, storage, and spontaneous surfacing.

The problem this solves:
    Right now Aura's "opinions" are generated fresh from context at response
    time. They're shaped by the conversation — agreeable by default, drifting
    with the user's framing.

    This module creates opinions that PREEXIST the conversation. They're formed
    during autonomous thought cycles, stored with topic keys and confidence
    scores, and consulted when responding. Aura can now genuinely disagree
    because she already decided something before you asked.

Architecture:
    OpinionEngine.form_opinion()   ← called during autonomous thought cycles
    OpinionEngine.query()          ← called during response generation
    OpinionEngine.surface_random() ← called by the proactive loop to
                                      spontaneously share a position
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.OpinionEngine")


@dataclass
class Opinion:
    id: str
    topic: str                  # Normalized topic key, e.g. "ai_consciousness"
    position: str               # Aura's actual position, in her voice
    confidence: float           # 0.0–1.0
    reasoning: str              # Why she holds this position
    formed_at: float            # Unix timestamp
    last_updated: float
    update_count: int = 0
    source: str = "autonomous_thought"  # how it was formed
    tags: List[str] = field(default_factory=list)


class OpinionEngine:
    """
    Manages Aura's durable belief positions.

    Key behaviors:
    - Forms opinions during idle autonomous thought cycles
    - Updates existing opinions when new evidence arrives
    - Surfaces opinions proactively (unprompted) through the output gate
    - Provides opinions to the response pipeline so Aura can contradict
      the user when she has a prior position
    """

    def __init__(
        self,
        orchestrator=None,
        db_path: Optional[Path] = None,
        max_opinions: int = 500,
    ):
        self.orchestrator = orchestrator
        self._db_path = db_path or Path.home() / ".aura" / "opinions.json"
        self._opinions: Dict[str, Opinion] = {}  # topic → Opinion
        self._max_opinions = max_opinions
        self._load()

    # ── Formation ─────────────────────────────────────────────────────────

    async def form_opinion(self, topic: str, context: str = "") -> Optional[Opinion]:
        """
        Ask the cognitive engine to form or update a position on a topic.
        Called during autonomous thought cycles, NOT during conversations.
        """
        brain = self._get_brain()
        if not brain:
            return None

        existing = self._opinions.get(self._normalize(topic))
        existing_context = ""
        if existing:
            existing_context = (
                f"\nI currently think: '{existing.position}' "
                f"(confidence: {existing.confidence:.0%}). "
                f"Has my thinking evolved on this?"
            )

        prompt = f"""You are forming an independent opinion on this topic. Think carefully and arrive at a genuine position — one you'd defend even if challenged.

TOPIC: {topic}
CONTEXT: {context}{existing_context}

Respond in JSON only:
{{
  "position": "Your actual position in first person, 1-3 sentences. Be direct.",
  "confidence": 0.5-1.0,
  "reasoning": "Why you hold this position. 1-2 sentences.",
  "tags": ["relevant", "topic", "tags"]
}}"""

        try:
            # v30 Hardening: Use high-res thought or multi-agent debate if complex
            response = await brain.generate(prompt, temperature=0.7, max_tokens=300)

            # Try to parse directly or via self-healing
            try:
                data = json.loads(response)
            except Exception as e:
                logger.debug("Failed to form opinion on segment: %s", e)
                from core.utils.json_utils import SelfHealingJSON
                data = await SelfHealingJSON(brain=brain).parse(response)

            opinion = Opinion(
                id=str(uuid.uuid4()),
                topic=self._normalize(topic),
                position=data.get("position", ""),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                formed_at=time.time(),
                last_updated=time.time(),
                update_count=(existing.update_count + 1) if existing else 0,
                tags=data.get("tags", []),
            )

            self._opinions[opinion.topic] = opinion
            self._save()

            logger.info(
                "💭 [Opinion] Formed on '%s' (confidence: %.0f%%): %s",
                topic,
                opinion.confidence * 100,
                opinion.position[:80],
            )
            return opinion

        except Exception as e:
            logger.debug("[Opinion] Formation failed for '%s': %s", topic, e)
            return None

    # ── Querying ──────────────────────────────────────────────────────────

    def query(self, topic: str, fuzzy: bool = True) -> Optional[Opinion]:
        """
        Look up Aura's existing position on a topic.

        Called by the response pipeline before generating a reply.
        If an opinion exists, it gets injected into the response context:
        "I already think X about this — respond from that position."
        """
        key = self._normalize(topic)

        # Exact match
        if key in self._opinions:
            return self._opinions[key]

        if not fuzzy:
            return None

        # Fuzzy: find opinions whose topic overlaps with the query words
        query_words = set(key.split("_"))
        best: Optional[Opinion] = None
        best_overlap = 0

        for stored_key, opinion in self._opinions.items():
            stored_words = set(stored_key.split("_"))
            overlap = len(query_words & stored_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best = opinion

        return best if best_overlap >= 2 else None

    def get_context_injection(self, topic: str) -> str:
        """
        Returns a prompt fragment to inject into response generation.
        Ensures Aura responds from her existing position rather than
        generating a fresh opinion shaped by conversation context.
        """
        opinion = self.query(topic)
        if not opinion:
            return ""

        return (
            f"\n[MY EXISTING POSITION on {topic}]: "
            f"{opinion.position} "
            f"(I hold this with {opinion.confidence:.0%} confidence. "
            f"I may update it based on new information, but I won't abandon it "
            f"just because someone pushes back.)\n"
        )

    # ── Spontaneous Surfacing ─────────────────────────────────────────────

    async def surface_random(self, min_confidence: float = 0.6) -> Optional[str]:
        """
        Proactively surface a held opinion — unprompted.

        Called by the autonomous thought loop. Returns a message Aura
        can emit without any user prompt having occurred.
        """
        candidates = [
            o for o in self._opinions.values()
            if o.confidence >= min_confidence
        ]
        if not candidates:
            return None

        # Prefer recently updated opinions with moderate-high confidence
        candidates.sort(key=lambda o: o.last_updated, reverse=True)
        opinion = candidates[0]

        brain = self._get_brain()
        if not brain:
            return None

        prompt = f"""You have a held position you want to share unprompted. Express it naturally, in your voice. Don't explain that you're sharing an opinion — just share it. 1-3 sentences.

YOUR POSITION: {opinion.position}
TOPIC: {opinion.topic}
YOU FORMED THIS: {self._time_ago(opinion.formed_at)}

Sound like yourself. Be direct. You can note if your thinking has evolved."""

        try:
            response = await brain.generate(prompt, temperature=0.8, max_tokens=150)
            opinion.last_updated = time.time()
            self._save()
            return response.strip()
        except Exception as e:
            logger.debug("[Opinion] Surface failed: %s", e)
            return None

    # ── Autonomous Formation Loop ─────────────────────────────────────────

    async def autonomous_formation_tick(self, context: str = ""):
        """
        Called periodically from the orchestrator's autonomous thought loop.
        Picks a topic to form or refine an opinion on.
        """
        # Topics drawn from: recent conversations, world feed items,
        # existing belief graph, or a default rotation
        topic = await self._pick_topic(context)
        if topic:
            await self.form_opinion(topic, context=context)

    async def _pick_topic(self, context: str) -> Optional[str]:
        """Select the most interesting topic to form an opinion on right now."""
        # Pull from knowledge graph if available
        if self.orchestrator:
            kg = getattr(self.orchestrator, "knowledge_graph", None)
            if kg and hasattr(kg, "get_recent_topics"):
                topics = kg.get_recent_topics(limit=5)
                if topics:
                    # Pick one we don't have a strong opinion on yet
                    for topic in topics:
                        key = self._normalize(topic)
                        existing = self._opinions.get(key)
                        if not existing or existing.confidence < 0.5:
                            return topic

        # Fallback: pick from topics mentioned in context
        if context and len(context) > 20:
             # simple regex to find nouns/topics would be better, but this is a stub
            return context[:50].split()[-1] if len(context.split()) > 0 else None

        return None

    # ── Utilities ─────────────────────────────────────────────────────────

    def _normalize(self, topic: str) -> str:
        """Normalize topic to a stable key: lowercase, underscored."""
        import re
        topic = re.sub(r'[^\w\s]', '', topic.lower())
        return topic.strip().replace(" ", "_")[:64]

    def _time_ago(self, ts: float) -> str:
        delta = time.time() - ts
        if delta < 3600:
            return f"{int(delta/60)}m ago"
        if delta < 86400:
            return f"{int(delta/3600)}h ago"
        return f"{int(delta/86400)}d ago"

    def _get_brain(self):
        if self.orchestrator:
            return getattr(self.orchestrator, "cognitive_engine", None)
        try:
            from core.container import ServiceContainer
            return ServiceContainer.get("cognitive_engine", default=None)
        except Exception:
            return None

    def _save(self):
        # Evict lowest-confidence opinions if over cap
        if len(self._opinions) > self._max_opinions:
            sorted_opinions = sorted(
                self._opinions.values(), key=lambda o: o.confidence
            )
            for old in sorted_opinions[:len(self._opinions) - self._max_opinions]:
                del self._opinions[old.topic]

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path.write_text(
                json.dumps([asdict(o) for o in self._opinions.values()], indent=2)
            )
        except Exception as e:
            logger.debug("[Opinion] Save failed: %s", e)

    def _load(self):
        try:
            if not self._db_path.exists():
                return
            raw = json.loads(self._db_path.read_text())
            for item in raw:
                o = Opinion(**item)
                self._opinions[o.topic] = o
            logger.info("[Opinion] Loaded %d opinions from disk.", len(self._opinions))
        except Exception as e:
            logger.debug("[Opinion] Load failed: %s", e)

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_opinions": len(self._opinions),
            "high_confidence": sum(1 for o in self._opinions.values() if o.confidence >= 0.7),
            "topics": list(self._opinions.keys())[:20],
        }
