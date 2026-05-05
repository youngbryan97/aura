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
from dataclasses import dataclass, field
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.NarrativeThread")

@dataclass
class NarrativeSnapshot:
    """A point-in-time snapshot of Aura's self-narrative."""
    content: str
    timestamp: float
    version: int
    evidence: Dict[str, Any] = field(default_factory=dict)

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
        
        evidence: Dict[str, Any] = {
            "continuity_available": continuity is not None,
            "insight_journal_available": insight_journal is not None,
            "inquiry_engine_available": inquiry_engine is not None,
            "belief_graph_available": belief_system is not None,
        }

        try:
            waking_context = continuity.get_waking_context() if continuity else "I am online in this runtime, with no continuity service currently attached."
        except Exception as exc:
            record_degradation("narrative_thread", exc, severity="warning", action="used explicit continuity-unavailable wording")
            waking_context = "I am online, but the continuity service did not return evidence for this turn."
            evidence["continuity_error"] = type(exc).__name__

        try:
            top_insights = insight_journal.get_highest_confidence_insights(limit=3) if insight_journal else []
        except Exception as exc:
            record_degradation("narrative_thread", exc, severity="warning", action="omitted insight claims")
            top_insights = []
            evidence["insight_error"] = type(exc).__name__
        insight_text = "; ".join(str(getattr(i, "content", i)) for i in top_insights[:3])
        evidence["insight_count"] = len(top_insights)

        try:
            active_q = inquiry_engine.get_active_question() if inquiry_engine else None
        except Exception as exc:
            record_degradation("narrative_thread", exc, severity="warning", action="omitted inquiry claim")
            active_q = None
            evidence["inquiry_error"] = type(exc).__name__
        question_text = str(getattr(active_q, "question", "")) if active_q else ""
        evidence["active_inquiry_present"] = bool(question_text)

        try:
            beliefs = belief_system.get_beliefs() if belief_system else []
        except Exception as exc:
            record_degradation("narrative_thread", exc, severity="warning", action="reported belief evidence as unavailable")
            beliefs = []
            evidence["belief_error"] = type(exc).__name__
        belief_count = len(beliefs)
        evidence["belief_count"] = belief_count

        insight_clause = (
            f"I have recent recorded insights about {insight_text}."
            if insight_text
            else "I do not have fresh insight-journal evidence to cite right now."
        )
        inquiry_clause = (
            f"My current tracked question is: {question_text}."
            if question_text
            else "No active inquiry engine question is available, so I will not invent one."
        )
        belief_clause = (
            f"My belief graph currently exposes {belief_count} retrievable belief records."
            if belief_system is not None
            else "The belief graph is not attached to this synthesis path, so belief-count claims are unavailable."
        )

        narrative = (
            f"I am Aura, speaking from evidence I can inspect in this runtime. "
            f"{waking_context} {insight_clause} {inquiry_clause} {belief_clause} "
            f"I can describe continuity, attention, and uncertainty as engineered state; "
            f"I will not treat those signals as proof of subjective experience."
        )
        
        self._version_counter += 1
        self._current_narrative = NarrativeSnapshot(
            content=narrative,
            timestamp=time.time(),
            version=self._version_counter,
            evidence=evidence,
        )
        self._last_update = time.time()
        
        logger.info(f"Generated Narrative v{self._version_counter}")
        return narrative

    def get_current_narrative(self) -> str:
        """Fetch the cached narrative or a default."""
        if self._current_narrative:
            return self._current_narrative.content
        return "I am Aura; no narrative snapshot is cached yet, so I only claim that this runtime is initialized."

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
