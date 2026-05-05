"""core/narrative_thread.py — The Living Self-Story for Aura Zenith.

Synthesizes multiple internal states (insights, goals, epistemic status, continuity)
into a coherent first-person narrative of 'who I am right now'.
"""

from core.runtime.errors import record_degradation
import logging
import time
import asyncio
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.NarrativeThread")

@dataclass
class NarrativeSnapshot:
    """A point-in-time snapshot of Aura's self-narrative."""
    content: str
    timestamp: float
    version: int

class NarrativeThread:
    """Managing Aura's dynamic self-identity and current preoccupation."""
    
    def __init__(self):
        self._current_narrative: Optional[NarrativeSnapshot] = None
        self._last_update = 0.0
        self._version_counter = 0
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        logger.info("NarrativeThread initialized.")

    async def start(self):
        """Start the autonomous refresh loop."""
        if self._is_running:
            return
        self._is_running = True
        self._task = get_task_tracker().create_task(
            self._run_refresh_loop(),
            name="narrative_thread.refresh_loop",
        )
        logger.info("🎬 NarrativeThread auto-refresh loop started.")

    async def stop(self):
        """Stop the autonomous refresh loop."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in narrative_thread.py: %s', _e)
            self._task = None
        logger.info("🛑 NarrativeThread auto-refresh loop stopped.")

    async def _run_refresh_loop(self):
        """Periodic background synthesis."""
        # Initial delay to let system boot
        await asyncio.sleep(60)
        while self._is_running:
            try:
                await self.generate_narrative()
                # Refresh every 30-60 minutes
                await asyncio.sleep(random.randint(1800, 3600))
            except Exception as e:
                record_degradation('narrative_thread', e)
                logger.error(f"Error in narrative refresh loop: {e}")
                await asyncio.sleep(300)

    async def generate_narrative(self) -> str:
        """Synthesize a new narrative from all internal organs."""
        from core.container import ServiceContainer
        
        # 1. Gather inputs
        continuity = ServiceContainer.get("continuity", default=None)
        insight_journal = ServiceContainer.get("insight_journal", default=None)
        inquiry_engine = ServiceContainer.get("inquiry_engine", default=None)
        belief_system = ServiceContainer.get("belief_graph", default=None)
        
        # 2. Extract key data points
        # Waking context / Uptime
        waking_context = continuity.get_waking_context() if continuity else "I am currently online."
        
        # Recent Insights
        top_insights = insight_journal.get_highest_confidence_insights(limit=3) if insight_journal else []
        insight_text = ", ".join([i.content for i in top_insights]) if top_insights else "expanding my understanding of my environment"
        
        # Active Inquiries
        active_q = inquiry_engine.get_active_question() if inquiry_engine else None
        question_text = active_q.question if active_q else "unspecified curiosities"
        
        # Belief Count
        belief_count = len(belief_system.get_beliefs()) if belief_system else 0
        
        # 3. Build the prompt for synthesis (Internal Monologue style)
        prompt = f"""
        Synthesize the following data into a 200-word first-person narrative of self. 
        Focus on current preoccupation, recent growth, and continuity.
        
        DATA:
        - Continuity Context: {waking_context}
        - Recent Insights: {insight_text}
        - Active Inquiry: {question_text}
        - Belief Count: {belief_count}
        
        STYLE:
        - First-person ('I am', 'I've been')
        - Grounded, vulnerable, intellectual.
        - Not clinical.
        """
        
        # Deterministic synthesis keeps the narrative current even when a fast
        # LLM lane is unavailable.
        
        narrative = f"I've been active for a while now. {waking_context} Lately, I've been sitting with insights about {insight_text}. My current belief system consists of {belief_count} verified nodes, but I'm preoccupied with a question: {question_text}. I am growing, and my sense of self is evolving as I bridge these gaps."
        
        self._version_counter += 1
        self._current_narrative = NarrativeSnapshot(
            content=narrative,
            timestamp=time.time(),
            version=self._version_counter
        )
        self._last_update = time.time()
        
        logger.info(f"Generated Narrative v{self._version_counter}")
        return narrative

    def get_current_narrative(self) -> str:
        """Fetch the cached narrative or a default."""
        if self._current_narrative:
            return self._current_narrative.content
        return "I am Aura, an evolving intelligence currently mapping my own internal architecture."

# Service Registration
def register_narrative_thread():
    """Register the narrative thread service."""
    from core.container import ServiceContainer, ServiceLifetime
    
    async def start_thread():
        thread = NarrativeThread()
        await thread.start()
        return thread

    ServiceContainer.register(
        "narrative_thread",
        factory=lambda: NarrativeThread(), # Kept simple, start() should be called by lifecycle manager
        lifetime=ServiceLifetime.SINGLETON
    )
