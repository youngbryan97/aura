"""core/learning/formalizer.py
──────────────────────────────
Knowledge Formalizer: Background distillation of research data into
atomic facts committed to the PersistentKnowledgeGraph.

This runs as a background task triggered after sovereign_browser fetches
page content, extracting structured facts and relationships without
blocking the user-facing response.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.utils.task_tracker import get_task_tracker

import asyncio
import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Formalizer")


class KnowledgeFormalizer:
    """Distills raw page/research content into atomic knowledge graph nodes.

    Design:
        - Runs in a background asyncio task (never blocks foreground)
        - Extracts factual claims, definitions, and key entities
        - Deduplicates against existing graph nodes before insertion
        - Links related concepts via the graph's relationship system
    """

    # Minimum content length worth formalizing
    MIN_CONTENT_LENGTH = 200
    # Maximum facts to extract per page
    MAX_FACTS_PER_PAGE = 15
    # Cooldown between formalization runs (seconds)
    COOLDOWN_SECONDS = 10.0

    def __init__(self) -> None:
        self._last_run_at: float = 0.0
        self._running: bool = False
        self._total_facts_committed: int = 0

    def _get_knowledge_graph(self) -> Any:
        """Resolve the live KnowledgeGraph from ServiceContainer."""
        try:
            from core.container import ServiceContainer
            kg = ServiceContainer.get("knowledge_graph", default=None)
            if kg is not None:
                return kg

            # Fallback: try memory provider
            mem = ServiceContainer.get("memory", default=None)
            if mem and hasattr(mem, "knowledge_graph"):
                return mem.knowledge_graph
        except Exception as exc:
            record_degradation('formalizer', exc)
            logger.debug("KnowledgeGraph resolution failed: %s", exc)
        return None

    def _extract_atomic_facts(self, content: str, source_title: str = "") -> List[Dict[str, str]]:
        """Extract atomic factual claims from raw text content.

        Uses heuristic sentence-level extraction rather than LLM inference
        to keep this fast and non-blocking.
        """
        facts: List[Dict[str, str]] = []
        if not content or len(content) < self.MIN_CONTENT_LENGTH:
            return facts

        # Split into sentences (rough but fast)
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # Fact-bearing sentence patterns
        fact_patterns = [
            # "X is Y" definitions
            re.compile(r'^(.{10,80})\s+(?:is|are|was|were)\s+(.{10,200})\.?$', re.IGNORECASE),
            # "X was founded/created/built in Y"
            re.compile(r'^(.{5,80})\s+(?:was|were)\s+(?:founded|created|built|established|invented|discovered)\s+(.{5,150})\.?$', re.IGNORECASE),
            # "According to X, Y"
            re.compile(r'^(?:according to\s+.{3,60},\s*)(.{15,250})\.?$', re.IGNORECASE),
            # Sentences with numbers/dates (often factual)
            re.compile(r'^(.{15,250}?\b\d{4}\b.{5,200})\.?$', re.IGNORECASE),
            # "X percent/%" patterns
            re.compile(r'^(.{10,250}?\b\d+\.?\d*\s*(?:%|percent)\b.{5,150})\.?$', re.IGNORECASE),
        ]

        seen_hashes = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            if len(facts) >= self.MAX_FACTS_PER_PAGE:
                break

            # Skip navigational/UI noise
            if any(noise in sentence.lower() for noise in (
                "click here", "subscribe", "sign up", "cookie", "privacy policy",
                "terms of service", "advertisement", "loading", "menu",
            )):
                continue

            # Check against fact patterns
            for pattern in fact_patterns:
                if pattern.match(sentence):
                    # Deduplicate by content hash
                    content_hash = hashlib.md5(sentence.lower().encode()).hexdigest()[:12]
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        facts.append({
                            "content": sentence,
                            "type": "fact",
                            "source": source_title or "web_research",
                            "confidence": 0.6,
                        })
                    break

        return facts

    def _extract_entities(self, content: str) -> List[str]:
        """Extract likely named entities from content (fast heuristic)."""
        # Find capitalized multi-word phrases (likely proper nouns/entities)
        entities = set()
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', content):
            entity = match.group(1).strip()
            if len(entity) > 3 and len(entity) < 60:
                entities.add(entity)

        return list(entities)[:10]

    async def formalize(
        self,
        content: str,
        source_title: str = "",
        source_url: str = "",
    ) -> Dict[str, Any]:
        """Formalize raw content into knowledge graph entries.

        This is the main entry point, designed to be called via
        get_task_tracker().create_task() from the response pipeline.

        Returns:
            Summary dict with counts of facts and relationships committed.
        """
        result = {
            "facts_committed": 0,
            "relationships_created": 0,
            "entities_found": 0,
            "skipped": False,
            "error": None,
        }

        # Cooldown check
        if time.monotonic() - self._last_run_at < self.COOLDOWN_SECONDS:
            result["skipped"] = True
            return result

        if self._running:
            result["skipped"] = True
            return result

        self._running = True
        self._last_run_at = time.monotonic()

        try:
            kg = self._get_knowledge_graph()
            if kg is None:
                result["error"] = "KnowledgeGraph not available"
                return result

            # 1. Extract atomic facts
            facts = self._extract_atomic_facts(content, source_title)
            if not facts:
                logger.debug("Formalizer: No facts extracted from '%s'", source_title[:60])
                return result

            # 2. Commit facts to graph
            fact_ids: List[str] = []
            for fact in facts:
                try:
                    node_id = kg.add_knowledge(
                        content=fact["content"],
                        type=fact["type"],
                        source=fact["source"],
                        confidence=float(fact.get("confidence", 0.6)),
                        metadata={
                            "source_url": source_url,
                            "source_title": source_title,
                            "formalized_at": time.time(),
                        },
                    )
                    fact_ids.append(node_id)
                    result["facts_committed"] += 1
                except Exception as exc:
                    record_degradation('formalizer', exc)
                    logger.debug("Formalizer: fact commit failed: %s", exc)

                # Yield to event loop periodically
                await asyncio.sleep(0)

            # 3. Extract entities and create concept nodes + relationships
            entities = self._extract_entities(content)
            result["entities_found"] = len(entities)

            entity_ids: Dict[str, str] = {}
            for entity in entities:
                try:
                    eid = kg.add_knowledge(
                        content=entity,
                        type="concept",
                        source="formalization",
                        confidence=0.5,
                        metadata={"source_url": source_url},
                    )
                    entity_ids[entity] = eid
                except Exception as exc:
                    record_degradation('formalizer', exc)
                    logger.debug("Formalizer: entity commit failed: %s", exc)

            # 4. Link facts to entities they mention
            for fact_id, fact in zip(fact_ids, facts):
                for entity, eid in entity_ids.items():
                    if entity.lower() in fact["content"].lower():
                        try:
                            kg.add_relationship(
                                from_id=fact_id,
                                to_id=eid,
                                relation_type="related_to",
                                strength=1.0,
                            )
                            result["relationships_created"] += 1
                        except Exception:
                            pass
                await asyncio.sleep(0)

            self._total_facts_committed += result["facts_committed"]
            logger.info(
                "📚 Formalizer: Committed %d facts, %d relationships from '%s' (total: %d)",
                result["facts_committed"],
                result["relationships_created"],
                source_title[:60],
                self._total_facts_committed,
            )

        except Exception as exc:
            record_degradation('formalizer', exc)
            result["error"] = str(exc)
            logger.error("Formalizer error: %s", exc, exc_info=True)
        finally:
            self._running = False

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return formalization statistics."""
        return {
            "total_facts_committed": self._total_facts_committed,
            "running": self._running,
            "last_run_at": self._last_run_at,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_formalizer_instance: Optional[KnowledgeFormalizer] = None


def get_formalizer() -> KnowledgeFormalizer:
    """Get or create the singleton KnowledgeFormalizer."""
    global _formalizer_instance
    if _formalizer_instance is None:
        _formalizer_instance = KnowledgeFormalizer()
    return _formalizer_instance


async def formalize_content(
    content: str,
    source_title: str = "",
    source_url: str = "",
) -> Dict[str, Any]:
    """Convenience function to formalize content via the singleton."""
    return await get_formalizer().formalize(
        content=content,
        source_title=source_title,
        source_url=source_url,
    )
