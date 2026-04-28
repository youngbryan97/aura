"""Knowledge Auto-Enrichment — Extract facts from conversations

After each conversation exchange, this module can extract:
  - Named entities (people, places, tools, concepts)
  - Facts and relationships between entities
  - User preferences and opinions
  
These are automatically inserted into the Knowledge Graph, growing
Aura's persistent understanding without explicit user instruction.

Design:
  - Non-blocking: runs as a background task
  - Rate-limited: max one extraction per conversation turn
  - Lightweight: uses a focused prompt to minimize LLM cost
  - Graceful: failures never impact the main conversation
"""

from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Knowledge.Enrichment")


class KnowledgeEnricher:
    """Automatically extracts and stores knowledge from conversations."""

    _MIN_INTERVAL = 10.0  # Minimum seconds between extractions
    _MIN_MESSAGES = 2     # Minimum messages before extracting

    def __init__(self, knowledge_graph=None, brain=None, belief_engine=None):
        self._kg = knowledge_graph
        self._brain = brain
        self._beliefs = belief_engine
        self._last_extraction = 0.0
        self._lock = asyncio.Lock()
        self._extraction_count = 0

    async def enrich_from_conversation(
        self,
        messages: List[Dict[str, str]],
        force: bool = False,
    ) -> Dict[str, Any]:
        """Extract knowledge from recent conversation messages.
        
        Args:
            messages: Recent conversation history
            force: Skip rate limiting
            
        Returns:
            Dict with counts of extracted entities, facts, and preferences
        """
        result = {"facts": 0, "entities": 0, "preferences": 0, "beliefs": 0}

        if not self._kg or not self._brain:
            return result

        # Rate limiting
        now = time.time()
        if not force and now - self._last_extraction < self._MIN_INTERVAL:
            return result

        if len(messages) < self._MIN_MESSAGES:
            return result

        if self._lock.locked():
            return result  # Already running

        async with self._lock:
            try:
                self._last_extraction = now

                # Build excerpt from recent messages
                recent = messages[-6:]
                excerpt_lines: List[str] = []
                grounded = False
                for m in recent:
                    if not isinstance(m, dict):
                        continue
                    content = str(m.get("content", "") or "").strip()
                    if not content or m.get("ephemeral"):
                        continue
                    if "Cognitive baseline tick" in content:
                        continue

                    role = str(m.get("role", "") or "").strip().lower()
                    metadata = m.get("metadata", {}) or {}
                    if metadata.get("type") == "skill_result":
                        skill = metadata.get("skill") or metadata.get("tool") or "tool"
                        status = "ok" if metadata.get("ok") else "result"
                        excerpt_lines.append(f"Tool[{skill}/{status}]: {content[:300]}")
                        grounded = True
                    elif role == "user":
                        excerpt_lines.append(f"User: {content[:300]}")
                    elif role in {"assistant", "aura"}:
                        excerpt_lines.append(f"Aura: {content[:300]}")
                    elif role == "tool":
                        excerpt_lines.append(f"Tool: {content[:300]}")

                excerpt = "\n".join(excerpt_lines)

                if len(excerpt) < 20:
                    return result

                # Extract via LLM
                extractions = await self._extract(excerpt)

                if not extractions:
                    return result

                # Store extractions
                for item in extractions:
                    item_type = item.get("type", "")
                    content = item.get("content", "")
                    
                    if not content or len(content) < 5:
                        continue

                    if item_type == "fact":
                        self._kg.add_knowledge(
                            content=content,
                            type="fact",
                            source="conversation_extraction_grounded" if grounded else "conversation_extraction",
                            confidence=float(item.get("confidence", 0.7)),
                        )
                        result["facts"] += 1

                    elif item_type == "entity":
                        node_id = self._kg.add_knowledge(
                            content=content,
                            type="concept",
                            source="conversation_extraction_grounded" if grounded else "conversation_extraction",
                            confidence=0.8,
                        )
                        # Link to related entities
                        for related in item.get("related_to", []):
                            rel_id = self._kg.add_knowledge(
                                content=related,
                                type="concept",
                                source="conversation_extraction_grounded" if grounded else "conversation_extraction",
                            )
                            self._kg.add_relationship(
                                node_id, rel_id, "associated_with", strength=1.0
                            )
                        result["entities"] += 1

                    elif item_type == "preference":
                        self._kg.add_knowledge(
                            content=f"[User Preference] {content}",
                            type="preference",
                            source="conversation_extraction_grounded" if grounded else "conversation_extraction",
                            confidence=0.8,
                        )
                        result["preferences"] += 1
                        
                        # Also register as a belief
                        if self._beliefs:
                            self._beliefs.believe(
                                proposition=f"The user {content}",
                                confidence=0.75,
                                evidence=[excerpt[:100]],
                                source="conversation",
                                category="preference",
                            )
                            result["beliefs"] += 1

                    elif item_type == "relationship":
                        entity_a = item.get("entity_a", "")
                        entity_b = item.get("entity_b", "")
                        relation = item.get("relation", "associated_with")
                        if entity_a and entity_b:
                            self._kg.upsert_relationship(
                                entity_a, relation, entity_b,
                                weight=float(item.get("strength", 1.0))
                            )
                            result["facts"] += 1

                self._extraction_count += 1
                total = sum(result.values())

                if total > 0:
                    logger.info(
                        "📚 Knowledge enrichment: +%d facts, +%d entities, +%d preferences",
                        result["facts"], result["entities"], result["preferences"]
                    )
                    try:
                        from core.thought_stream import get_emitter
                        get_emitter().emit(
                            "Knowledge Enrichment 📚",
                            f"Extracted {total} items from conversation",
                            level="info",
                            category="Memory"
                        )
                    except Exception as _exc:
                        record_degradation('knowledge_enrichment', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)

            except Exception as e:
                record_degradation('knowledge_enrichment', e)
                logger.debug("Knowledge enrichment failed (non-critical): %s", e)

        return result

    async def _extract(self, excerpt: str) -> List[Dict[str, Any]]:
        """Use LLM to extract structured knowledge from conversation text."""
        prompt = (
            "Extract knowledge from this conversation excerpt. Return a JSON list.\n"
            "Each item should have: type, content, and optional fields.\n\n"
            "Types:\n"
            '  - {"type": "fact", "content": "...", "confidence": 0.0-1.0}\n'
            '  - {"type": "entity", "content": "name", "related_to": ["other entities"]}\n'
            '  - {"type": "preference", "content": "prefers/likes/dislikes ..."}\n'
            '  - {"type": "relationship", "entity_a": "X", "relation": "causes/requires/etc", "entity_b": "Y"}\n\n'
            "Rules:\n"
            "  - Only extract clearly stated facts, not speculation\n"
            "  - Focus on information worth remembering long-term\n"
            "  - Skip trivial greetings or small talk\n"
            "  - Return [] if nothing worth extracting\n\n"
            f"Conversation:\n{excerpt}\n\n"
            "JSON:"
        )

        try:
            response = await self._brain.generate(prompt, use_strategies=False)
            
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group(0))
                if isinstance(items, list):
                    return items[:15]  # Cap at 15 items per extraction
        except Exception as e:
            record_degradation('knowledge_enrichment', e)
            logger.debug("Knowledge extraction LLM call failed: %s", e)

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Enrichment statistics."""
        return {
            "total_extractions": self._extraction_count,
            "last_extraction": self._last_extraction,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[KnowledgeEnricher] = None


def get_enricher(knowledge_graph=None, brain=None, belief_engine=None) -> KnowledgeEnricher:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = KnowledgeEnricher(
            knowledge_graph=knowledge_graph,
            brain=brain,
            belief_engine=belief_engine,
        )
    return _instance
