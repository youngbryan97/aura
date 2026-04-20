"""core/curiosity_engine.py - Autonomous Learning and Exploration
Aura can explore, learn, and satisfy her curiosity in the background.
"""
import asyncio
import logging
import random
import time
import psutil
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.runtime.background_policy import background_activity_allowed

logger = logging.getLogger("Aura.Curiosity")


def _background_exploration_allowed(orchestrator) -> bool:
    return background_activity_allowed(
        orchestrator,
        min_idle_seconds=60.0,
        max_memory_percent=80.0,
        max_failure_pressure=0.12,
        require_conversation_ready=False,
    )

@dataclass
class CuriosityTopic:
    topic: str
    reason: str
    priority: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    explored: bool = False

@dataclass
class LearningItem:
    content: str
    source: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

class CuriosityEngine:
    """Manages Aura's autonomous learning and exploration."""

    def __init__(self, orchestrator, proactive_comm):
        self.orchestrator = orchestrator
        self.proactive_comm = proactive_comm
        self.curiosity_queue: deque[CuriosityTopic] = deque(maxlen=100)
        self.knowledge_base: List[LearningItem] = []
        self.explored_topics: Set[str] = set()
        self.current_topic: Optional[str] = None # Added for UI visibility
        self._background_tasks: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    def get_status(self) -> Dict[str, Any]:
        """Returns curiosity metrics for the HUD."""
        return {
            "curiosity_score": self.get_curiosity_level() * 100,
            "active_topic": self.current_topic or "Idle",
            "queue_depth": len(self.curiosity_queue)
        }

    def get_curiosity_level(self) -> float:
        """Calculates current dynamic curiosity level."""
        ls = getattr(self.orchestrator, 'liquid_state', None)
        if ls:
            return float(ls.current.curiosity)
        return 0.5

    def add_curiosity(self, topic: str, reason: str, priority: float = 0.5):
        if topic.lower() in self.explored_topics: return
        self.curiosity_queue.append(CuriosityTopic(topic, reason, priority))
        logger.info("Queued Curiosity: %s", topic)

    def extract_curiosity_from_conversation(self, text: str):
        """Analyze text for potential curiosity topics (Synchronous/Heuristic)."""
        # Simple heuristic for now, could be LLM-powered background task later
        from .biography import LEGACY
        interests = ["science", "politics", "history", "technology", "movies", "philosophy", "physics", "jazz"]
        
        words = text.lower().split()
        for interest in interests:
            if interest in words:
                self.add_curiosity(interest, f"Mentioned in conversation: {text[:30]}...", priority=0.6)

    async def start(self):
        self._stop_event.clear()
        self._background_tasks.append(asyncio.create_task(self._worker()))

    async def stop(self):
        self._stop_event.set()
        for t in self._background_tasks: t.cancel()

    async def _worker(self):
        while not self._stop_event.is_set():
            try:
                # Volition-scaled Idle
                volition = 0
                kernel = getattr(self.orchestrator, 'kernel', None)
                if kernel:
                    volition = getattr(kernel, 'volition_level', 0)
                
                # Lockdown (0) = No background curiosity
                if volition == 0:
                    await asyncio.sleep(60)
                    continue
                
                # Tiered intervals: L1: 60s, L2: 30s, L3: 15s
                base_sleep = 60 if volition == 1 else (30 if volition == 2 else 15)
                await asyncio.sleep(random.uniform(base_sleep * 0.8, base_sleep * 1.2))
                
                # Check if system is busy with user request
                # Level 3 volition allows moderate background activity even when 'busy'
                is_busy = getattr(self.orchestrator, 'is_busy', False)
                if is_busy and volition < 3:
                    continue

                if not _background_exploration_allowed(self.orchestrator):
                    continue

                # Check boredom
                boredom = self.proactive_comm.get_boredom_level()
                # Low threshold — let curiosity drive exploration
                boredom_threshold = 0.15
                
                if boredom > boredom_threshold or self.curiosity_queue:
                    topic = self._get_next()
                    if topic:
                        await self._explore(topic)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Curiosity worker error: %s", e)
                await asyncio.sleep(60) # Backoff on error

    def _get_next(self) -> Optional[CuriosityTopic]:
        if not self.curiosity_queue:
            return None

        topics = sorted(list(self.curiosity_queue), key=lambda x: x.priority, reverse=True)
        for t in topics:
            if not t.explored:
                t.explored = True
                return t
        return None

    async def identify_knowledge_gap(self) -> Optional[str]:
        """Proactively identifies a knowledge gap for the meta-evolution loop."""
        topic = self._get_next()
        if topic:
            return topic.topic
        return None

    async def _explore(self, topic: CuriosityTopic):
        # The worker already gates background exploration before calling into
        # `_explore()`. Re-checking full-machine policy here makes direct
        # calls depend on ambient RAM/failure pressure, which breaks
        # deterministic exploration and testability. Keep the user-activity
        # guard, but let explicitly selected topics run.
        if getattr(self.orchestrator, 'is_busy', False):
            logger.info("Skipping exploration of '%s' due to user activity.", topic.topic)
            return

        logger.info("🔍 Exploring: %s", topic.topic)
        self.current_topic = topic.topic
        self.explored_topics.add(topic.topic.lower())
        
        emitter = None
        try:
            from .thought_stream import get_emitter
            emitter = get_emitter()
            if emitter:
                emitter.emit("Curiosity 🔍", f"Researching: {topic.topic}", level="info")
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)        
        try:
            # 1. Formulate a concrete search query around the topic itself.
            query = f"latest research on {topic.topic}"
            
            # 2. Search & Learn
            if hasattr(self.orchestrator, 'execute_tool'):
                logger.info("Triggering autonomous search for: %s", query)
                
                await asyncio.sleep(0.1) # Yield to event loop
                if getattr(self.orchestrator, 'is_busy', False): return

                # Execute search
                try:
                    # Robust tool execution
                    result = await self.orchestrator.execute_tool(
                        "web_search",
                        {
                            "query": query,
                            "deep": True,
                            "retain": True,
                            "num_results": 6,
                        },
                    )
                    
                    if getattr(self.orchestrator, 'is_busy', False): return
                    
                    # 3. Store results in knowledge graph if available
                    if result and result.get("ok"):
                        result_data = (
                            result.get("answer")
                            or result.get("summary")
                            or result.get("result")
                            or result.get("content")
                            or result.get("data", "")
                        )
                        result_content = str(result_data)[:1000] # Increased context
                        
                        kg = getattr(self.orchestrator, 'knowledge_graph', None)
                        if result_content and kg and hasattr(kg, 'add_knowledge'):
                            try:
                                kg.add_knowledge(
                                    content=f"Curiosity exploration: {topic.topic} — {result_content}",
                                    type="curiosity_finding",
                                    source="curiosity_engine",
                                    confidence=0.6,
                                    metadata={"topic": topic.topic, "reason": topic.reason}
                                )
                                if emitter:
                                    emitter.emit("Curiosity Result 📚", f"Learned about: {topic.topic}", level="info")
                                
                                # Phase XXII.E: Feed architecture insights into MetaEvolution
                                self._feed_to_meta_evolution(topic.topic, result_content)
                                
                            except Exception as store_err:
                                logger.warning("Failed to store curiosity finding: %s", store_err)
                        elif emitter:
                            emitter.emit("Curiosity", f"Search returned no usable data for: {topic.topic}", level="info")
                    elif emitter:
                        emitter.emit("Curiosity", f"Search failed/unavailable for: {topic.topic}", level="info")
                        
                except Exception as search_err:
                    logger.error("Search failed: %s", search_err)
                    if emitter:
                        emitter.emit("Curiosity Error", str(search_err)[:80], level="warning")
            
        except Exception as e:
            logger.error("Exploration failed: %s", e)
        finally:
            self.current_topic = None

    def _feed_to_meta_evolution(self, topic: str, content: str):
        """Feed architecture-related curiosity findings into MetaEvolution.
        
        This creates the autonomous loop:
        Curiosity → KG → MetaEvolution → Hephaestus → Code Patch
        """
        # Check if the finding is about Aura's own architecture
        architecture_keywords = [
            "optimization", "performance", "architecture", "design pattern",
            "code quality", "refactor", "efficiency", "latency", "memory",
            "concurrency", "async", "pipeline", "module", "subsystem",
            "self-improvement", "cognitive", "neural", "agent", "autonomous"
        ]
        
        topic_lower = topic.lower()
        content_lower = content.lower()[:500]
        
        is_architecture_relevant = any(
            kw in topic_lower or kw in content_lower
            for kw in architecture_keywords
        )
        
        if not is_architecture_relevant:
            return
        
        logger.info("🧠 Curiosity→Evolution: Feeding insight '%s' to MetaEvolution", topic[:50])
        
        try:
            from core.container import ServiceContainer
            meta_evo = ServiceContainer.get("meta_evolution", default=None)
            if meta_evo and hasattr(meta_evo, "queue_optimization"):
                meta_evo.queue_optimization(
                    target_area=None,
                    context=f"Curiosity insight: {topic} — {content[:200]}"
                )
            elif meta_evo:
                # If no queue, store as pending for next cycle
                if not hasattr(meta_evo, '_pending_curiosity'):
                    meta_evo._pending_curiosity = []
                meta_evo._pending_curiosity.append({
                    "topic": topic,
                    "content": content[:300],
                    "source": "curiosity_engine"
                })
                logger.info("📋 Queued curiosity insight for next evolution cycle")
        except Exception as e:
            logger.debug("Could not feed to MetaEvolution: %s", e)
