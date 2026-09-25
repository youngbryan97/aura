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

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

from core.epistemics.belief_revision import BeliefDomain
from core.runtime.errors import record_degradation
from core.runtime.runtime_settings import get_runtime_setting

logger = logging.getLogger("Knowledge.Enrichment")


def _automatic_enrichment_enabled() -> bool:
    return bool(
        get_runtime_setting("learning.auto_enrichment_enabled", True)
    )


class KnowledgeEnricher:
    """Automatically extracts and stores knowledge from conversations."""

    _MIN_INTERVAL = 10.0  # Minimum seconds between extractions
    _MIN_MESSAGES = 2     # Minimum messages before extracting

    def __init__(self, knowledge_graph=None, brain=None, belief_engine=None):
        self._kg = knowledge_graph
        self._brain = brain
        self._beliefs = belief_engine
        self._last_attempt = 0.0
        self._last_success = 0.0
        self._lock = asyncio.Lock()
        self._extraction_count = 0
        self._rejection_count = 0
        self._last_outcome = "never_run"

    def _belief_engine(self) -> Any:
        """Her one belief engine, as registered, or None before boot registers it.

        The learning phase passed the container's "belief_engine", a name
        nothing registers, and this enricher is built once, so the preferences
        it heard never reached her beliefs for the life of the process. The call
        it made, `believe()`, is on no belief engine either. Read at use rather
        than held from construction, and never built here: building the engine
        loads its store from disk, which boot does off the loop.
        """
        if self._beliefs is not None:
            return self._beliefs
        from core.container import ServiceContainer

        return ServiceContainer.get("belief_revision_engine", default=None)

    async def enrich_from_conversation(
        self,
        messages: list[dict[str, str]],
        force: bool = False,
    ) -> dict[str, Any]:
        """Extract knowledge from recent conversation messages.
        
        Args:
            messages: Recent conversation history
            force: Skip rate limiting
            
        Returns:
            Dict with counts of extracted entities, facts, and preferences
        """
        result = {"facts": 0, "entities": 0, "preferences": 0, "beliefs": 0}

        if not _automatic_enrichment_enabled():
            self._last_outcome = "disabled_by_runtime_setting"
            return result

        if not self._kg or not self._brain:
            self._last_outcome = "owner_dependency_unavailable"
            return result

        if not isinstance(messages, (list, tuple)):
            self._last_outcome = "invalid_messages_container"
            return result

        # Rate limiting
        now = time.time()
        if not force and now - self._last_attempt < self._MIN_INTERVAL:
            self._last_outcome = "rate_limited"
            return result

        if len(messages) < self._MIN_MESSAGES:
            self._last_outcome = "insufficient_messages"
            return result

        if self._lock.locked():
            self._last_outcome = "already_running"
            return result  # Already running

        async with self._lock:
            try:
                # Build excerpt from recent messages
                recent = messages[-6:]
                excerpt_lines: list[str] = []
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
                    metadata_value = m.get("metadata", {}) or {}
                    metadata = (
                        metadata_value if isinstance(metadata_value, dict) else {}
                    )
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
                    self._last_outcome = "insufficient_grounded_text"
                    return result

                # Extract via LLM
                self._last_attempt = time.time()
                extractions = await self._extract(excerpt)

                if not extractions:
                    return result

                # Store extractions
                storage_errors: list[tuple[int, str, Exception]] = []
                for item_index, item in enumerate(extractions):
                    item_type = item.get("type", "")
                    content = item.get("content", "")
                    try:
                        source = (
                            "conversation_extraction_grounded"
                            if grounded
                            else "conversation_extraction"
                        )
                        if item_type == "fact":
                            self._kg.add_knowledge(
                                content=content,
                                type="fact",
                                source=source,
                                confidence=float(item.get("confidence", 0.7)),
                            )
                            result["facts"] += 1

                        elif item_type == "entity":
                            node_id = self._kg.add_knowledge(
                                content=content,
                                type="concept",
                                source=source,
                                confidence=0.8,
                            )
                            result["entities"] += 1
                            for related in item.get("related_to", []):
                                try:
                                    rel_id = self._kg.add_knowledge(
                                        content=related,
                                        type="concept",
                                        source=source,
                                    )
                                    self._kg.add_relationship(
                                        node_id,
                                        rel_id,
                                        "associated_with",
                                        strength=1.0,
                                    )
                                except Exception as exc:  # noqa: BLE001 - pluggable storage boundary
                                    storage_errors.append(
                                        (item_index, "entity_relationship", exc)
                                    )

                        elif item_type == "preference":
                            self._kg.add_knowledge(
                                content=f"[User Preference] {content}",
                                type="preference",
                                source=source,
                                confidence=0.8,
                            )
                            result["preferences"] += 1

                            beliefs = self._belief_engine()
                            if beliefs is not None:
                                try:
                                    await beliefs.process_new_claim(
                                        f"The user {content}",
                                        domain=BeliefDomain.USER,
                                        source="conversation",
                                        confidence=0.75,
                                    )
                                    result["beliefs"] += 1
                                except Exception as exc:  # noqa: BLE001 - pluggable belief boundary
                                    storage_errors.append(
                                        (item_index, "preference_belief", exc)
                                    )

                        elif item_type == "relationship":
                            self._kg.upsert_relationship(
                                item.get("entity_a", ""),
                                item.get("relation", "associated_with"),
                                item.get("entity_b", ""),
                                weight=float(item.get("strength", 1.0)),
                            )
                            result["facts"] += 1
                    except Exception as exc:  # noqa: BLE001 - isolate independent external writes
                        # This background isolation boundary deliberately does
                        # not catch CancelledError, which derives BaseException.
                        storage_errors.append((item_index, item_type, exc))

                self._extraction_count += 1
                total = sum(result.values())
                if total > 0:
                    self._last_success = time.time()
                if storage_errors:
                    self._last_outcome = "completed_with_storage_errors"
                    first_index, first_stage, first_error = storage_errors[0]
                    record_degradation(
                        "knowledge_enrichment.storage",
                        first_error,
                        severity="warning",
                        action="continued independent enrichment writes after a partial storage failure",
                        receipt_required=True,
                        extra={
                            "error_count": len(storage_errors),
                            "first_item_index": first_index,
                            "first_stage": first_stage,
                            "stored_items": total,
                        },
                        enforce_failure_policy=False,
                    )
                else:
                    self._last_outcome = "completed"

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
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        record_degradation('knowledge_enrichment', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)

            except (ImportError, AttributeError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as e:
                self._last_outcome = f"failed:{type(e).__name__}"
                record_degradation('knowledge_enrichment', e)
                logger.debug("Knowledge enrichment failed (non-critical): %s", e)

        return result

    async def _extract(self, excerpt: str) -> list[dict[str, Any]]:
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
        except (AttributeError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            self._last_outcome = f"model_call_failed:{type(exc).__name__}"
            record_degradation("knowledge_enrichment", exc)
            logger.debug("Knowledge extraction model call failed: %s", exc)
            return []

        items = self._decode_extraction_items(response)
        if not items:
            self._rejection_count += 1
            self._last_outcome = (
                "empty_model_response"
                if response is None
                or isinstance(response, (str, bytes))
                and not response.strip()
                else "invalid_structured_response"
            )
            logger.debug("Knowledge extraction produced no valid structured items")
            return []

        validated = [
            normalized
            for item in items[:30]
            if (normalized := self._validate_extraction_item(item)) is not None
        ][:15]
        if not validated:
            self._rejection_count += 1
            self._last_outcome = "invalid_extraction_items"
            return []
        return validated

    @staticmethod
    def _decode_extraction_items(response: Any) -> list[Any]:
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            for key in ("items", "extractions", "data"):
                candidate = response.get(key)
                if isinstance(candidate, list):
                    return candidate
            for key in ("content", "text", "response", "output"):
                candidate = response.get(key)
                if isinstance(candidate, str):
                    response = candidate
                    break
        elif not isinstance(response, str):
            for attribute in ("content", "text", "response", "output"):
                candidate = getattr(response, attribute, None)
                if isinstance(candidate, str):
                    response = candidate
                    break
        if isinstance(response, bytes):
            response = response.decode("utf-8", errors="replace")
        if not isinstance(response, str) or not response.strip():
            return []

        decoder = json.JSONDecoder()
        for index, character in enumerate(response):
            if character != "[":
                continue
            try:
                candidate, _end = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, list):
                return candidate
        return []

    @staticmethod
    def _validate_extraction_item(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        raw_type = item.get("type")
        if not isinstance(raw_type, str):
            return None
        item_type = raw_type.strip().lower()
        if item_type not in {"entity", "fact", "preference", "relationship"}:
            return None
        if item_type == "relationship":
            raw_entity_a = item.get("entity_a")
            raw_entity_b = item.get("entity_b")
            raw_relation = item.get("relation", "associated_with")
            if not all(
                isinstance(value, str)
                for value in (raw_entity_a, raw_entity_b, raw_relation)
            ):
                return None
            entity_a = raw_entity_a.strip()
            entity_b = raw_entity_b.strip()
            relation = raw_relation.strip()
            if min(len(entity_a), len(entity_b), len(relation)) < 2:
                return None
            strength = item.get("strength", 1.0)
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                return None
            strength = float(strength)
            if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
                return None
            return {
                "type": item_type,
                "entity_a": entity_a[:240],
                "entity_b": entity_b[:240],
                "relation": relation[:120],
                "strength": strength,
            }

        raw_content = item.get("content")
        if not isinstance(raw_content, str):
            return None
        content = raw_content.strip()
        if len(content) < 5:
            return None
        normalized: dict[str, Any] = {
            "type": item_type,
            "content": content[:1000],
        }
        confidence = item.get("confidence", 0.8 if item_type != "fact" else 0.7)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return None
        normalized["confidence"] = confidence
        if item_type == "entity":
            related = item.get("related_to", [])
            if not isinstance(related, list):
                return None
            normalized["related_to"] = [
                value.strip()[:240]
                for value in related[:12]
                if isinstance(value, str) and value.strip()
            ]
        return normalized

    def get_stats(self) -> dict[str, Any]:
        """Enrichment statistics."""
        return {
            "total_extractions": self._extraction_count,
            "rejected_model_outputs": self._rejection_count,
            "last_extraction": self._last_success,
            "last_attempt": self._last_attempt,
            "last_success": self._last_success,
            "last_outcome": self._last_outcome,
            "enabled": _automatic_enrichment_enabled(),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: KnowledgeEnricher | None = None


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
