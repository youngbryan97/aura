import logging
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
        
        # 1. Fetch recent episodes via safe accessor chain
        episodic_store = getattr(self.memory, 'episodic', None)
        if episodic_store is None:
            logger.warning("Episodic memory subsystem not attached.")
            return
        data = getattr(episodic_store, 'data', None)
        if data is None or not isinstance(data, dict):
            logger.warning("Episodic memory has no accessible data store.")
            return
        episodes = data.get("episodic", [])


        if len(episodes) < self.fragment_threshold:
            logger.info("Not enough experiences to dream about.")
            return

        recent_batch = episodes[-self.fragment_threshold:]
        
        # 2. Cognitive Reflection
        summary_prompt = (
            "Review these recent events and extract 3 key lessons or facts.\n"
            f"Events: {recent_batch}\n"
            "Format: Bullet points."
        )
        try:
            reflection = await self.brain.think(summary_prompt)
            logger.info("Dream Insight: %s", reflection)
            
            # 3. Consolidate to Vector Memory
            self.memory.vector.add(
                text=reflection,
                metadata={"source": "dream_cycle", "timestamp": time.time()}
            )
            
            # 4. Phase 10: Graph Contraction (Long-term Wisdom)
            await self._contract_graph(reflection)
            
            logger.info("✓ Dream cycle complete: Consolidated wisdom into Knowledge Graph.")
            
        except Exception as e:
            logger.error("Nightmare error: %s", e)

    async def _contract_graph(self, reflection: str):
        """Distill the fuzzy reflection into structured Graph nodes/edges."""
        kg = ServiceContainer.get("knowledge_graph", default=None)
        if not kg:
            logger.warning("Graph Contraction: Knowledge Graph not found in container.")
            return

        logger.debug("Graph Contraction: Starting distillation of reflection...")

        contract_prompt = (
            "From this reflection, extract the core entities and their relationships.\n"
            f"Reflection: {reflection}\n"
            "Format: entity1 | relation | entity2\n"
            "One per line."
        )
        
        try:
            struct_data = await self.brain.think(contract_prompt)
            logger.debug("Graph Contraction raw output: %s", struct_data)
            for line in struct_data.split("\n"):
                if "|" in line:
                    logger.debug("Parsing line: %s", line)
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) == 3:
                        e1, rel, e2 = parts
                        logger.info("🕸️ Upserting relationship: %s -[%s]-> %s", e1, rel, e2)
                        kg.upsert_relationship(e1, rel, e2, weight=1.5)
        except Exception as e:
            logger.debug("Graph contraction failed: %s", e)