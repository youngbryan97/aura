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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.Formalizer")


@dataclass
class DistilledClaim:
    """A machine-readable claim extracted from source text."""

    content: str
    claim_type: str
    source: str
    confidence: float
    subject: str = ""
    predicate: str = ""
    conditions: list[str] = field(default_factory=list)
    consequences: list[str] = field(default_factory=list)
    evidence_span: str = ""
    source_quality: float = 0.5

    def to_knowledge(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "type": self.claim_type,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": {
                "subject": self.subject,
                "predicate": self.predicate,
                "conditions": list(self.conditions),
                "consequences": list(self.consequences),
                "evidence_span": self.evidence_span,
                "source_quality": self.source_quality,
                "verification_status": "extractive_unverified",
                "distillation_method": "deterministic_claim_parser_v2",
            },
        }


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

    def _extract_atomic_facts(self, content: str, source_title: str = "", source_url: str = "") -> List[Dict[str, Any]]:
        """Extract definitions, causal rules, procedures, and quantitative claims.

        This is not regex scraping dressed up as learning: every retained claim
        carries claim type, subject/predicate, conditions, consequences,
        source quality, and an evidence span. Downstream systems can therefore
        verify, contradict, promote, or suppress the claim instead of treating a
        sentence as unquestioned memory.
        """
        claims: list[DistilledClaim] = []
        if not content or len(content) < self.MIN_CONTENT_LENGTH:
            return []

        sentences = self._sentence_candidates(content)
        source_quality = self._source_quality(source_title, source_url)

        seen_hashes = set()

        for sentence in sentences:
            sentence = self._clean_sentence(sentence)
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            if len(claims) >= self.MAX_FACTS_PER_PAGE:
                break

            if self._is_noise(sentence):
                continue

            claim = self._distill_sentence(sentence, source_title or "web_research", source_quality)
            if claim is None:
                continue
            content_hash = hashlib.md5(claim.content.lower().encode()).hexdigest()[:12]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            claims.append(claim)

        return [claim.to_knowledge() for claim in claims]

    @staticmethod
    def _sentence_candidates(content: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", str(content or "").replace("\r", "\n")).strip()
        chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?:\n+)|(?:;\s+(?=(?:if|when|after|before|to|avoid|use)\b))", normalized)
        return [chunk.strip(" -•\t") for chunk in chunks if chunk.strip()]

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        sentence = re.sub(r"\[[^\]]{1,40}\]", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        return sentence.rstrip(".") + "."

    @staticmethod
    def _is_noise(sentence: str) -> bool:
        lowered = sentence.lower()
        if any(noise in lowered for noise in (
            "click here", "subscribe", "sign up", "cookie", "privacy policy",
            "terms of service", "advertisement", "loading", "menu", "share this",
            "all rights reserved", "javascript", "enable cookies",
        )):
            return True
        alpha_ratio = sum(ch.isalpha() for ch in sentence) / max(1, len(sentence))
        return alpha_ratio < 0.35

    @staticmethod
    def _source_quality(source_title: str, source_url: str) -> float:
        text = f"{source_title} {source_url}".lower()
        quality = 0.55
        if any(token in text for token in (".edu", ".gov", "docs.", "manual", "reference", "specification", "paper", "arxiv")):
            quality += 0.2
        if any(token in text for token in ("forum", "reddit", "comment", "blog")):
            quality -= 0.08
        if source_url.startswith("https://"):
            quality += 0.05
        return max(0.2, min(0.95, quality))

    def _distill_sentence(self, sentence: str, source: str, source_quality: float) -> DistilledClaim | None:
        lowered = sentence.lower()
        patterns: list[tuple[str, re.Pattern[str]]] = [
            ("conditional_rule", re.compile(r"^(?:if|when|whenever|after|before)\s+(.{5,140}?),\s+(.{8,180})\.$", re.IGNORECASE)),
            ("procedure", re.compile(r"^(?:to|in order to)\s+(.{5,90}?),\s+(.{8,190})\.$", re.IGNORECASE)),
            ("requirement", re.compile(r"^(.{5,120}?)\s+(?:requires|must|should|needs|depends on)\s+(.{5,180})\.$", re.IGNORECASE)),
            ("affordance", re.compile(r"^(.{5,120}?)\s+(?:can|may|allows?|enables?)\s+(.{5,180})\.$", re.IGNORECASE)),
            ("definition", re.compile(r"^(.{5,100}?)\s+(?:is|are|means|refers to|was|were)\s+(.{8,190})\.$", re.IGNORECASE)),
            ("causal_rule", re.compile(r"^(.{5,140}?)\s+(?:causes|prevents|reduces|increases|leads to|results in)\s+(.{5,180})\.$", re.IGNORECASE)),
            ("quantitative_fact", re.compile(r"^(.{10,260}?\b(?:\d{4}|\d+\.?\d*\s*(?:%|percent|ms|s|kb|mb|gb|turns?|steps?))\b.{0,160})\.$", re.IGNORECASE)),
        ]

        for claim_type, pattern in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            groups = [g.strip() for g in match.groups()]
            subject = groups[0] if groups else ""
            predicate = groups[1] if len(groups) > 1 else sentence
            conditions: list[str] = []
            consequences: list[str] = []
            if claim_type == "conditional_rule":
                conditions = [subject]
                consequences = [predicate]
            elif claim_type in {"procedure", "requirement", "affordance", "causal_rule"}:
                consequences = [predicate]
            confidence = self._claim_confidence(sentence, claim_type, source_quality)
            return DistilledClaim(
                content=sentence,
                claim_type=claim_type,
                source=source,
                confidence=confidence,
                subject=subject,
                predicate=predicate,
                conditions=conditions,
                consequences=consequences,
                evidence_span=sentence,
                source_quality=source_quality,
            )

        if any(token in lowered for token in ("avoid ", "do not ", "never ", "risk ", "unsafe ", "failure ", "error ")):
            return DistilledClaim(
                content=sentence,
                claim_type="risk_rule",
                source=source,
                confidence=self._claim_confidence(sentence, "risk_rule", source_quality),
                predicate=sentence,
                consequences=[sentence],
                evidence_span=sentence,
                source_quality=source_quality,
            )
        return None

    @staticmethod
    def _claim_confidence(sentence: str, claim_type: str, source_quality: float) -> float:
        confidence = source_quality
        if claim_type in {"conditional_rule", "causal_rule", "requirement", "risk_rule"}:
            confidence += 0.08
        if re.search(r"\b(?:maybe|might|could|often|usually|sometimes|appears)\b", sentence, re.IGNORECASE):
            confidence -= 0.12
        if re.search(r"\b(?:must|always|never|guarantees?)\b", sentence, re.IGNORECASE):
            confidence -= 0.05
        if re.search(r"\b\d", sentence):
            confidence += 0.04
        return round(max(0.2, min(0.92, confidence)), 3)

    def _extract_entities(self, content: str) -> List[str]:
        """Extract likely named entities and durable technical concepts."""
        entities = set()
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', content):
            entity = match.group(1).strip()
            if len(entity) > 3 and len(entity) < 60:
                entities.add(entity)
        for match in re.finditer(r"\b([a-z][a-z0-9_/-]{2,}(?:\s+[a-z][a-z0-9_/-]{2,}){1,3})\b", content):
            phrase = match.group(1).strip()
            if any(token in phrase for token in (" risk", " model", " graph", " policy", " planner", " parser", " state", " action", " memory", " rule")):
                entities.add(phrase)

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
            facts = self._extract_atomic_facts(content, source_title, source_url)
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
                            **dict(fact.get("metadata", {}) or {}),
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
                            record_degradation(
                                "formalizer",
                                RuntimeError("relationship_commit_failed"),
                                severity="debug",
                                action="continued committing independent claims",
                                extra={"entity": entity, "fact_id": fact_id},
                            )
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
