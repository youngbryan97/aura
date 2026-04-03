import asyncio
import logging
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState
from core.utils.queues import role_for_origin, unpack_priority_message

logger = logging.getLogger(__name__)

class SensoryIngestionPhase(BasePhase):
    """
    Phase 1: Sensory Ingestion.
    Responsible for pulling environmental stimuli (user messages, system alerts, impulses)
    from the orchestrator's queues and integrating them into the AuraState.
    """
    
    def __init__(self, container: Any):
        self.container = container

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Pull the next message from the queue and append it to working memory.
        """
        try:
            orchestrator = self.container.get("orchestrator", default=None)
            if not orchestrator or not hasattr(orchestrator, "message_queue"):
                return state
                
            # Try to get a message without blocking across legacy/current queue shapes.
            try:
                raw = orchestrator.message_queue.get_nowait()
                raw_msg, queued_origin = unpack_priority_message(raw)
            except asyncio.QueueEmpty:
                return state
            if raw_msg is None:
                return state
                
            logger.info("📥 SensoryIngestion: Ingested stimulus: %s...", str(raw_msg)[:100])
            
            # Normalize message into working memory format
            new_entry = {
                "role": role_for_origin(queued_origin),
                "content": str(raw_msg),
            }
            if isinstance(raw_msg, dict):
                # If it's already a dict, merge it (handles roles, origins, etc.)
                new_entry.update(raw_msg)
            elif queued_origin:
                new_entry["origin"] = queued_origin

            if "origin" in new_entry and (not isinstance(raw_msg, dict) or "role" not in raw_msg):
                new_entry["role"] = role_for_origin(new_entry.get("origin"))
            
            # Zenith-v6.3 Fix: Deduplication (Prevention of Echo Loops)
            history = state.cognition.working_memory
            recent = history[-3:] if len(history) >= 3 else history
            if any(
                m.get("content") == new_entry["content"]
                and m.get("origin") == new_entry.get("origin")
                for m in recent
            ):
                logger.debug("⏭️ SensoryIngestion: Duplicate stimulus ignored: %s...", new_entry['content'][:50])
                return state
                
            # Derive new state with the added message
            new_state = state.derive("sensory_ingestion")
            new_state.cognition.working_memory.append(new_entry)
            
            # Optional: Update orchestrator's internal time markers to prevent boredom
            if hasattr(orchestrator, "_last_thought_time"):
                import time
                orchestrator._last_thought_time = time.time()
                
            return new_state
            
        except (asyncio.QueueEmpty, AttributeError):
            # No message in queue, return state unchanged
            return state
        except Exception as e:
            logger.error("❌ SensoryIngestion: Error pulling from queue: %s", e)
            return state
