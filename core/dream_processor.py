from core.runtime.errors import record_degradation
import asyncio
import inspect
import logging
import os
import time
from typing import List, Optional, Dict, Any
from core.container import ServiceContainer

logger = logging.getLogger("Kernel.Dream")

class DreamProcessor:
    """Offline Memory Consolidation.
    Turning short-term logs into long-term wisdom.
    """

    def __init__(self, memory_nexus, brain):
        self.memory = memory_nexus
        self.brain = brain
        self.fragment_threshold = 10 # Process after 10 new events

    async def dream(self):
        """The Dreaming Cycle.
        1. Fetch recent raw episodes.
        2. Summarize them into a narrative.
        3. Extract 'Lessons Learned'.
        4. Store in Vector DB.
        5. Archive raw episodes.
        """
        logger.info("🌙 Entering Dream State...")

        episodes = await self._load_recent_episodes(limit=self.fragment_threshold)
        if not episodes:
            logger.info("DreamProcessor: no recent episodic memories available.")
            return

        if len(episodes) < self.fragment_threshold:
            logger.info("Not enough experiences to dream about.")
            return

        recent_batch = episodes[-self.fragment_threshold:]

        reflection = self._compose_reflection(recent_batch)
        try:
            if self._llm_dream_enabled():
                summary_prompt = (
                    "Review these recent events and extract 3 key lessons or facts.\n"
                    f"Events: {[self._episode_text(ep) for ep in recent_batch]}\n"
                    "Format: Bullet points."
                )
                reflection = self._brain_text(await self.brain.think(summary_prompt))

            logger.info("Dream Insight: %s", reflection)

            # 3. Consolidate to Vector Memory
            memory_saved = await self._store_reflection(reflection)

            # 4. Phase 10: Graph Contraction (Long-term Wisdom)
            graph_writes_raw = await self._contract_graph(reflection)
            try:
                graph_writes = int(graph_writes_raw or 0)
            except (TypeError, ValueError):
                graph_writes = 0

            if memory_saved or graph_writes:
                logger.info(
                    "✓ Dream cycle complete: committed reflection=%s graph_edges=%d.",
                    bool(memory_saved),
                    graph_writes,
                )
            else:
                logger.info(
                    "Dream cycle complete: no memory writes committed; governance blocked or no stable graph edges."
                )

        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('dream_processor', e)
            logger.error("Nightmare error: %s", e)

    async def _load_recent_episodes(self, limit: int) -> List[Any]:
        """Load recent episodes from the current persistent memory stack."""
        try:
            if hasattr(self.memory, "_refresh_subsystems"):
                self.memory._refresh_subsystems()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("dream_processor", exc)
            logger.debug("DreamProcessor memory refresh skipped: %s", exc)

        candidates = [
            getattr(self.memory, "episodic", None),
            ServiceContainer.get("episodic_memory", default=None),
        ]

        for episodic_store in candidates:
            if episodic_store is None:
                continue
            try:
                if hasattr(episodic_store, "recall_recent_async"):
                    episodes = await episodic_store.recall_recent_async(limit=limit)
                    if episodes:
                        return list(reversed(episodes))
                if hasattr(episodic_store, "recall_recent"):
                    episodes = episodic_store.recall_recent(limit=limit)
                    if episodes:
                        return list(reversed(episodes))
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("dream_processor", exc)
                logger.debug("DreamProcessor episodic recall failed: %s", exc)

        try:
            if hasattr(self.memory, "get_hot_memory"):
                hot = await self.memory.get_hot_memory(limit=limit)
                episodes = hot.get("recent_episodes") or []
                if episodes:
                    return list(reversed(episodes))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("dream_processor", exc)
            logger.debug("DreamProcessor hot memory recall failed: %s", exc)

        # Legacy fallback for old test fixtures and stale in-memory stores.
        episodic_store = getattr(self.memory, "episodic", None)
        data = getattr(episodic_store, "data", None)
        if isinstance(data, dict):
            return list(data.get("episodic", []) or [])[-limit:]
        return []

    @staticmethod
    def _episode_text(episode: Any) -> str:
        if hasattr(episode, "to_retrieval_text"):
            try:
                return str(episode.to_retrieval_text())
            except (AttributeError, RuntimeError, TypeError, ValueError) as _exc:
                logger.debug("Suppressed %s in core.dream_processor: %s", type(_exc).__name__, _exc)
        if isinstance(episode, dict):
            context = episode.get("context") or episode.get("description") or episode.get("content") or ""
            action = episode.get("action") or ""
            outcome = episode.get("outcome") or ""
            return f"Context: {context} | Action: {action} | Outcome: {outcome}".strip()
        return str(episode)

    def _compose_reflection(self, episodes: List[Any]) -> str:
        lines = [self._episode_text(ep) for ep in episodes]
        compact = []
        seen = set()
        for line in lines:
            normalized = " ".join(line.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            compact.append(normalized[:220])
            if len(compact) >= 5:
                break
        if not compact:
            return "Dream consolidation found recent activity, but no stable narrative pattern was extractable yet."
        return "Dream consolidation is integrating: " + " | ".join(compact)

    @staticmethod
    def _llm_dream_enabled() -> bool:
        return os.getenv("AURA_DREAM_PROCESSOR_USE_LLM", "").strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _brain_text(value: Any) -> str:
        """Normalize CognitiveEngine results before string parsing."""
        if value is None:
            return ""
        for attr in ("content", "text", "response"):
            try:
                text = getattr(value, attr, None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                text = None
            if text:
                return str(text)
        return str(value)

    async def _store_reflection(self, reflection: str) -> bool:
        metadata = {"source": "dream_cycle", "timestamp": time.time()}
        try:
            if hasattr(self.memory, "add_memory"):
                return bool(await self._call_memory_writer(self.memory.add_memory, reflection, metadata))

            vector = getattr(self.memory, "vector", None) or ServiceContainer.get("vector_memory", default=None)
            if vector is None:
                return False
            if hasattr(vector, "add"):
                return bool(await self._call_memory_writer(vector.add, text=reflection, metadata=metadata))
            elif hasattr(vector, "add_memory"):
                return bool(await self._call_memory_writer(vector.add_memory, reflection, metadata))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("dream_processor", exc)
            logger.debug("DreamProcessor reflection store failed: %s", exc)
        return False

    @staticmethod
    async def _call_memory_writer(fn, *args, **kwargs):
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _contract_graph(self, reflection: str) -> int:
        """Distill the fuzzy reflection into structured Graph nodes/edges."""
        kg = ServiceContainer.get("knowledge_graph", default=None)
        if not kg:
            logger.warning("Graph Contraction: Knowledge Graph not found in container.")
            return 0

        logger.debug("Graph Contraction: Starting distillation of reflection...")

        try:
            if self._llm_dream_enabled():
                contract_prompt = (
                    "From this reflection, extract the core entities and their relationships.\n"
                    f"Reflection: {reflection}\n"
                    "Format: entity1 | relation | entity2\n"
                    "One per line."
                )
                struct_data = self._brain_text(await self.brain.think(contract_prompt))
            else:
                struct_data = self._deterministic_relationships(reflection)
            logger.debug("Graph Contraction raw output: %s", struct_data)
            committed = 0
            for line in struct_data.split("\n"):
                if "|" in line:
                    logger.debug("Parsing line: %s", line)
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) == 3:
                        e1, rel, e2 = parts
                        logger.info("🕸️ Upserting relationship: %s -[%s]-> %s", e1, rel, e2)
                        if kg.upsert_relationship(e1, rel, e2, weight=1.5):
                            committed += 1
            return committed
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('dream_processor', e)
            logger.debug("Graph contraction failed: %s", e)
            return 0

    @staticmethod
    def _deterministic_relationships(reflection: str) -> str:
        compact = " ".join(str(reflection or "").split())
        if not compact:
            return ""
        relationships = ["recent_experience | consolidates_into | continuity_memory"]
        lowered = compact.lower()
        if "email" in lowered:
            relationships.append("autonomous_email | requires | content_aware_follow_through")
        if "reddit" in lowered:
            relationships.append("autonomous_reddit | requires | login_state_and_read_result_tracking")
        if "live chat" in lowered or "conversation" in lowered:
            relationships.append("live_conversation | depends_on | coherent_response_lane")
        if "error" in lowered or "failed" in lowered or "degraded" in lowered:
            relationships.append("runtime_failure | should_trigger | bounded_repair_plan")
        return "\n".join(relationships)
