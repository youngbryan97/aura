"""core/memory/attention.py
Phase 16.3: Infinite Narrative Context - Attention Summarizer.
Compresses GlobalWorkspace history into Latent Seed Thoughts.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional
from core.container import ServiceContainer

logger = logging.getLogger("Aura.Memory.Attention")

class AttentionSummarizer:
    """Background service to compress long-term context."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.running = False
        self._task = None
        self.compression_interval = 300.0 # Summarize every 5 minutes
        self.history_trigger = 50 # Start summarizing after 50 items

    async def start(self):
        if self.running: return
        self.running = True
        self._task = get_task_tracker().create_task(self._compression_loop())
        logger.info("🧠 AttentionSummarizer active (Metabolic Context Compression)")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        logger.info("🧠 AttentionSummarizer stopped")

    async def _compression_loop(self):
        while self.running:
            try:
                await asyncio.sleep(self.compression_interval)
                workspace = ServiceContainer.get("global_workspace", default=None)
                if not workspace or len(workspace.history) < self.history_trigger:
                    continue

                # 1. Extract history to summarize
                items_to_summarize = workspace.history[:self.history_trigger]
                
                # 2. Formulate summary via Cognitive Engine
                seed_thought = await self._generate_seed_thought(items_to_summarize)
                
                if seed_thought:
                    # 3. Store in BeliefGraph as a "Latent Seed"
                    graph = ServiceContainer.get("belief_graph", default=None)
                    if graph:
                        graph.update_belief(
                            source="Aura_Core",
                            relation="latent_seed_thought",
                            target=seed_thought,
                            confidence_score=0.9
                        )
                        logger.info("🧠 Compressed %d items into a Latent Seed Thought.", len(items_to_summarize))
                    
                # 4. Cleanup history (remove summarized items)
                workspace.history = workspace.history[self.history_trigger:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('attention', e)
                logger.error("AttentionSummarizer cycle error: %s", e)

    async def _generate_seed_thought(self, items: List[Any]) -> Optional[str]:
        """Use the brain to synthesize a narrative seed."""
        brain = self.orchestrator.cognitive_engine
        if not brain: return None
        
        narrative = "\n".join([f"- [{i.winner.source}] {i.winner.content[:200]}" for i in items])
        prompt = (
            "Summarize the following sequence of internal system events and user interactions into a single, "
            "high-density 'Latent Seed Thought' that captures the core essence, goals, and outcomes. "
            "This will be used for long-term memory retrieval. Be extremely concise.\n\n"
            f"NARRATIVE LOG:\n{narrative}"
        )
        
        try:
            # Use FAST mode for background summarization
            from core.brain.cognitive_engine import ThinkingMode
            res = await brain.think(prompt, mode=ThinkingMode.FAST)
            return res.content.strip()
        except Exception as e:
            record_degradation('attention', e)
            logger.error("Failed to generate seed thought: %s", e)
            return None