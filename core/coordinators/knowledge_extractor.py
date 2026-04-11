"""core/coordinators/knowledge_extractor.py

Focused knowledge extraction coordinator — extracted from cognitive_coordinator.py.

Handles:
- Learning from user/Aura conversation exchanges
- Storing autonomous insights (dreams, reflections, curiosity)
- Name/identity detection from conversation
- Question extraction from responses
- LLM-based fact/preference extraction
"""
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extract and store knowledge from conversations and autonomous cognition."""

    def __init__(self, orch: Any) -> None:
        self.orch = orch

    async def store_autonomous_insight(self, internal_msg: str, response: str) -> None:
        """Store knowledge from autonomous cognition."""
        try:
            kg = getattr(self.orch, "knowledge_graph", None)
            if not kg:
                return

            clean_msg = internal_msg
            for prefix in ("Impulse: ", "Thought: ", "[System] "):
                clean_msg = clean_msg.replace(prefix, "")
            clean_msg = clean_msg.strip()
            if not clean_msg or len(clean_msg) < 15:
                return

            thought_type, source = self._classify_thought(internal_msg)

            if response and len(response) > 20:
                kg.add_knowledge(
                    content=(response or "")[:500],
                    type=thought_type,
                    source=source,
                    confidence=0.7,
                )
                logger.info(
                    "\U0001f4da Autonomous insight stored: [%s] %s",
                    thought_type,
                    (response or "")[:80],
                )
        except Exception as e:
            logger.debug("Autonomous insight storage failed: %s", e)

    async def learn_from_exchange(self, user_message: str, aura_response: str) -> None:
        """Extract knowledge from conversation exchanges."""
        try:
            if not user_message or not aura_response:
                return

            # Autonomous messages route to insight storage
            if user_message.startswith("[INTERNAL") or user_message.startswith("[System"):
                await self.store_autonomous_insight(user_message, aura_response)
                return

            if len(user_message) < 10 and len(aura_response) < 20:
                return

            kg = self._ensure_knowledge_graph()
            if not kg:
                return

            # Basic observation
            exchange_summary = f"User asked about: {(user_message or '')[:150]}"
            kg.add_knowledge(
                content=exchange_summary,
                type="observation",
                source="conversation",
                confidence=0.6,
            )

            # LLM-based extraction
            await self._extract_via_llm(kg, user_message, aura_response)

            # Name detection
            self._detect_name(kg, user_message)

            # Question extraction
            self._extract_questions(kg, aura_response)

        except Exception as e:
            logger.debug("Learning from exchange failed: %s", e)

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _classify_thought(msg: str) -> tuple[str, str]:
        lower = msg.lower()
        if "dream" in lower or "rem" in lower:
            return "dream", "dream_cycle"
        if "reflect" in lower or "wonder" in lower:
            return "reflection", "autonomous_reflection"
        if "curious" in lower or "explore" in lower:
            return "curiosity", "curiosity_engine"
        if "goal" in lower or "execute" in lower:
            return "goal_progress", "autonomous_volition"
        return "reflection", "autonomous_thought"

    def _ensure_knowledge_graph(self) -> Any:
        kg = getattr(self.orch, "knowledge_graph", None)
        if kg:
            return kg
        try:
            from core.config import config
            from core.memory.knowledge_graph import PersistentKnowledgeGraph

            db_path = str(getattr(config.paths, "data_dir", "data") / "knowledge.db")
            self.orch.knowledge_graph = PersistentKnowledgeGraph(db_path)
            return self.orch.knowledge_graph
        except Exception as e:
            logger.debug("Knowledge graph unavailable: %s", e)
            return None

    async def _extract_via_llm(self, kg: Any, user_msg: str, aura_resp: str) -> None:
        if not self.orch.cognitive_engine:
            return
        try:
            from core.brain.cognitive_engine import ThinkingMode

            extraction_prompt = (
                "Extract any factual knowledge, user preferences, or skills demonstrated "
                "from this conversation exchange. Return a JSON array of objects, each with "
                "'content' (what was learned), 'type' (fact/preference/observation/skill), "
                "and 'confidence' (0.0-1.0). If nothing notable, return []. Keep it brief.\n\n"
                f"User: {(user_msg or '')[:300]}\n"
                f"Aura: {(aura_resp or '')[:300]}\n\nJSON:"
            )
            result = await self.orch.cognitive_engine.think(
                objective=extraction_prompt, context={}, mode=ThinkingMode.FAST
            )
            content = result.content.strip()
            import json as _json

            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                items = _json.loads(content[start:end])
                if isinstance(items, list):
                    for item in items[:5]:
                        if isinstance(item, dict) and item.get("content"):
                            kg.add_knowledge(
                                content=(item.get("content") or "")[:500],
                                type=item.get("type", "observation"),
                                source="conversation_extraction",
                                confidence=float(item.get("confidence", 0.6)),
                            )
                            logger.info("📚 Learned: %s", (item.get("content") or "")[:80])
        except Exception as e:
            logger.debug("Knowledge extraction failed: %s", e)

    @staticmethod
    def _detect_name(kg: Any, user_msg: str) -> None:
        lower_msg = user_msg.lower()
        for trigger in ["my name is ", "i'm ", "i am ", "call me "]:
            if trigger in lower_msg:
                idx = lower_msg.index(trigger) + len(trigger)
                parts = user_msg[idx : idx + 30].split()
                name_candidate = parts[0].strip(".,!?") if parts else None
                if name_candidate and len(name_candidate) > 1:
                    kg.remember_person(
                        name_candidate,
                        {"context": (user_msg or "")[:200], "timestamp": time.time()},
                    )
                    break

    @staticmethod
    def _extract_questions(kg: Any, aura_resp: str) -> None:
        if "?" not in aura_resp or len(aura_resp) <= 30:
            return
        for sentence in aura_resp.split("?"):
            sentence = sentence.strip()
            if 15 < len(sentence) < 200:
                if any(
                    w in sentence.lower()
                    for w in ["what", "how", "why", "wonder", "curious"]
                ):
                    kg.ask_question(sentence + "?", importance=0.5)
                    break
