"""core/memory/attention.py
Phase 16.3: Infinite Narrative Context - Attention Summarizer.
Compresses GlobalWorkspace history into Latent Seed Thoughts.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
from typing import List, Any, Optional
from core.runtime.service_registry import get_runtime_service

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
                workspace = get_runtime_service("global_workspace", default=None)
                if not workspace or len(workspace.history) < self.history_trigger:
                    continue

                # 1. Extract history to summarize
                items_to_summarize = workspace.history[:self.history_trigger]
                
                # 2. Formulate summary via Cognitive Engine
                seed_thought = await self._generate_seed_thought(items_to_summarize)
                
                if seed_thought:
                    # 3. Store in BeliefGraph as a "Latent Seed"
                    graph = get_runtime_service("belief_graph", default=None)
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
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation('attention', e)
                logger.error("AttentionSummarizer cycle error: %s", e)

    async def _generate_seed_thought(self, items: List[Any]) -> Optional[str]:
        """Ask the language organ for one sentence that stands for these items.

        Through the router, not the cognitive engine: a full cognitive turn
        assembles the whole conversational context around whatever it is
        handed, and for fifty workspace items that came to 96,998 characters —
        double the brainstem's prefill ceiling, so the middle was dropped
        before the model saw it (live, 2026-09-15). The items are the whole
        input here, fenced as data.
        """
        from core.container import ServiceContainer
        from core.security.prompt_fencing import fence

        router = ServiceContainer.get("llm_router", default=None)
        if router is None:
            return None

        narrative = "\n".join(
            f"- [{i.winner.source}] {str(i.winner.content)[:200]}" for i in items
        )
        prompt = (
            "One sentence that stands for this sequence of events, for finding "
            "it again later.\n\n"
            + fence(narrative, label="events")
        )
        try:
            text = await router.think(
                prompt=prompt,
                prefer_tier="local_fast",
                max_tokens=120,
                temperature=0.3,
                purpose="attention_summary",
                origin="attention_summarizer",
                allow_cloud_fallback=False,
            )
            return str(text or "").strip() or None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_degradation('attention', e)
            logger.error("Failed to generate seed thought: %s", e)
            return None
