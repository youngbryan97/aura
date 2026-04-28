from core.runtime.errors import record_degradation
import asyncio
import logging
from core.utils.exceptions import capture_and_log
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.dual_memory import DualMemorySystem
from core.config import config

logger = logging.getLogger("Aura.MemoryBridge")

@dataclass
class ConversationTurn:
    timestamp: datetime
    conversation_id: str
    user_message: str
    aura_response: str
    context: Dict[str, Any] = field(default_factory=dict)

class EnhancedMemorySystem:
    """A bridge that maps the legacy Memory interface to the new DualMemorySystem."""
    
    def __init__(self, db_path=None):
        # We ignore db_path and use config-driven directories from DualMemorySystem
        self.dual = DualMemorySystem()
        self._tasks = set()

    async def store_turn(self, conversation_id: str, user_message: str, aura_response: str, context: Dict[str, Any]):
        """Stores a conversation turn in the episodic store of the DualMemorySystem."""
        # Dynamically derive valence/importance from context or AffectEngine
        valence = context.get("emotional_valence")
        importance = context.get("importance")
        
        if valence is None or importance is None:
            from core.container import ServiceContainer
            affect = ServiceContainer.get("affect_engine", default=None)
            if affect and hasattr(affect, 'get_state'):
                # Heuristic: Higher intensity = Higher importance
                current_state = affect.get_state() if hasattr(affect, 'get_state') else {}
                if isinstance(current_state, dict):
                    valence = current_state.get("valence", 0.0) if valence is None else valence
                    importance = current_state.get("intensity", 0.5) if importance is None else importance
        
        # Fallbacks
        valence = valence if valence is not None else 0.0
        importance = importance if importance is not None else 0.5
        
        # Store as an episode
        description = f"User: {user_message}\nAura: {aura_response}"
        self.dual.store_experience(
            description=description,
            emotional_valence=valence,
            importance=importance,
            tags=[conversation_id]
        )
        
        # Proactive learning using the LLM (Knowledge Extraction).
        # A+ contract: route through the canonical task tracker so this
        # background work has lifecycle ownership, supervised cancellation,
        # and a named trace.
        from core.utils.task_tracker import get_task_tracker

        task = get_task_tracker().create_task(
            self.learn_fact_from_interaction(user_message, aura_response),
            name="enhanced_memory.learn_fact_from_interaction",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def retrieve_context(self, message: str, limit=5) -> str:
        """Retrieves history from the DualMemorySystem."""
        # Use query-based retrieval to get relevant context
        return await self.dual.retrieve_context(message, max_episodes=limit)

    async def learn_fact_from_interaction(self, user_msg: str, aura_msg: str):
        """High-fidelity Knowledge Extraction via LLM (Cascading Success)."""
        prompt = f"""EXTRACT SEMANTIC KNOWLEDGE:
User said: "{user_msg}"
Aura replied: "{aura_msg}"

Identify any permanent facts, user preferences, or world knowledge shared in this interaction.
Output ONLY a JSON list of facts:
[
  {{"concept": "User", "predicate": "prefers", "value": "dark mode", "confidence": 0.9, "domain": "personal"}}
]
If no facts found, return [].
"""
        try:
            from core.container import get_container
            brain = get_container().get_service("cognitive_engine")
            if not brain: return
            
            # Use Fast mode for extraction
            from core.brain.cognitive_engine import ThinkingMode
            thought = await brain.think(prompt, mode=ThinkingMode.FAST)
            content = thought.content
            
            import json
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group(0))
                for f in facts:
                    self.dual.learn_fact(
                        concept=f.get("concept", "unknown"),
                        predicate=f.get("predicate", "is"),
                        value=f.get("value", ""),
                        confidence=f.get("confidence", 0.5),
                        domain=f.get("domain", "general")
                    )
        except Exception as e:
            record_degradation('memory', e)
            capture_and_log(e, {"context": "ConversationMemory.knowledge_extraction"})
            logger.error("Knowledge extraction failed: %s", e)

    # Internal API compatibility
    async def learn_fact(self, message: str):
        """Legacy compatibility method."""
        await self.learn_fact_from_interaction(message, "")