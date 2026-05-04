"""
Memory Coordinator for the RobustOrchestrator.
Handles RAG retrieval, persistence, and multi-modal memory management.
"""
from core.runtime.errors import record_degradation
import logging
import asyncio
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class MemoryCoordinator:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._memory = None
        self._memory_manager = None

    @property
    def memory(self):
        if self._memory is None:
            if getattr(self, '_loading_memory', False):
                logger.warning("MemoryCoordinator: Circular memory resolution detected! Falling back.")
                return None
            
            self._loading_memory = True
            try:
                from core.container import ServiceContainer
                facade = ServiceContainer.get("memory_facade", default=None)
                if facade is None:
                    logger.warning("MemoryCoordinator: memory_facade not found! Falling back to vault directly.")
                    from core.memory.black_hole_vault import get_vault
                    try:
                        self._memory = get_vault()
                    except Exception as e:
                        record_degradation('memory', e)
                        logger.error(f"Fallback to vault failed: {e}")
                        self._memory = None
                    return self._memory
                
                # Prevent infinite recursion if the facade is mistakenly resolved to the coordinator itself
                if facade is self:
                    logger.warning("MemoryCoordinator: memory_facade resolved to self! Bypassing to main memory system.")
                    from core.memory.black_hole_vault import get_vault
                    try:
                        self._memory = get_vault()
                    except Exception:
                        self._memory = None
                else:
                    self._memory = facade
            finally:
                self._loading_memory = False
        return self._memory

    @property
    def memory_manager(self):
        if self._memory_manager is None:
            from core.container import ServiceContainer
            self._memory_manager = ServiceContainer.get("memory_facade", default=None)
        return self._memory_manager

    async def setup(self):
        """Initialize memory components."""
        logger.info("Initializing MemoryCoordinator...")
        pass  # no-op: intentional

    async def get_hot_memory(self, limit: int = 5) -> Dict[str, Any]:
        """Fetches 'hot' (short-term) memory context."""
        if self.memory:
            try:
                # v14.1 Enterprise: Support both async and sync facades
                if asyncio.iscoroutinefunction(self.memory.get_hot_memory):
                    return await self.memory.get_hot_memory(limit=limit)
                else:
                    return await asyncio.to_thread(self.memory.get_hot_memory, limit=limit)
            except Exception as e:
                record_degradation('memory', e)
                logger.error(f"MemoryCoordinator.get_hot_memory failed: {e}")
        return {}

    async def commit_interaction(
        self,
        context: str,
        action: str = "interaction",
        outcome: str = "",
        success: bool = True,
        emotional_valence: float = 0.0,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,    # Legacy support
        content: Optional[str] = None, # Legacy support
    ) -> None:
        """Commits an interaction to persistent memory.
        Supports both the rich MemoryFacade signature and the legacy (role, content) signature.
        """
        # Legacy mapping if called with role/content
        if role and content:
            context = f"{role}: {content}"
            action = "commit_interaction"
            outcome = content[:500]
            success = True

        if self.memory and hasattr(self.memory, 'commit_interaction'):
            try:
                if asyncio.iscoroutinefunction(self.memory.commit_interaction):
                    await self.memory.commit_interaction(
                        context=context,
                        action=action,
                        outcome=outcome,
                        success=success,
                        emotional_valence=emotional_valence,
                        importance=importance,
                        metadata=metadata,
                    )
                else:
                    await asyncio.to_thread(
                        self.memory.commit_interaction,
                        context=context,
                        action=action,
                        outcome=outcome,
                        success=success,
                        emotional_valence=emotional_valence,
                        importance=importance,
                        metadata=metadata,
                    )
            except Exception as e:
                record_degradation('memory', e)
                logger.error("MemoryCoordinator.commit_interaction failed: %s", e)

    async def log_event(self, event_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Delegates to commit_interaction for compatibility with MemoryFacade callers."""
        await self.commit_interaction("system_event", f"[{event_type}] {content}", metadata=metadata)

    async def prune_low_salience(self, threshold_days: int = 14) -> None:
        """Triggers memory decay/pruning."""
        if self.memory and hasattr(self.memory, 'prune_low_salience'):
            try:
                if asyncio.iscoroutinefunction(self.memory.prune_low_salience):
                    await self.memory.prune_low_salience(threshold_days=threshold_days)
                else:
                    await asyncio.to_thread(self.memory.prune_low_salience, threshold_days=threshold_days)
            except Exception as e:
                record_degradation('memory', e)
                logger.error(f"MemoryCoordinator.prune_low_salience failed: {e}")

    async def get_cold_memory_context(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Compatibility alias for vector search."""
        return await self.retrieve(query, limit=limit)

    async def retrieve(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Performs a multi-layered memory retrieval."""
        if not query or not self.memory: return {"results": []}
        try:
            if hasattr(self.memory, "retrieve"):
                # New RAG interface
                res = self.memory.retrieve(query, limit=limit)
                if asyncio.iscoroutine(res):
                    res = await res
                return res if isinstance(res, dict) else {"results": res}
            
            # Fallback to legacy context
            if hasattr(self.memory, 'get_cold_memory_context'):
                if asyncio.iscoroutinefunction(self.memory.get_cold_memory_context):
                    res = await self.memory.get_cold_memory_context(query, limit=limit)
                else:
                    res = await asyncio.to_thread(self.memory.get_cold_memory_context, query, limit=limit)
                return res if isinstance(res, dict) else {"results": res}
        except Exception as e:
            record_degradation('memory', e)
            logger.error(f"Memory retrieval failure: {e}")
            
        return {"results": []}

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the memory system."""
        return {
            "memory_ready": self.memory is not None,
            "manager_ready": self.memory_manager is not None
        }

    async def consolidate_working_memory(self, state: 'AuraState', **kwargs) -> None:
        """
        [PHASE 48] Subconscious Memory Consolidation ("Dreaming")
        Synthesizes recent working memory events and commits abstract insights to the vault,
        then truncates the short-term working memory to prevent infinite growth.
        """
        if len(state.cognition.working_memory) < 5:
            return  # Not enough context to dream about

        # Extract recent conversation
        transcript = []
        for msg in state.cognition.working_memory:
            role = msg.get("role", "system")
            content = msg.get("content", "")
            transcript.append(f"{role.capitalize()}: {content}")
            
        full_text = "\n".join(transcript)
        
        prompt = (
            "You are Aura's subconscious memory consolidator.\n"
            "Review the following recent interaction log and extract 2-3 core insights, facts, "
            "or changes in the user's state or preferences. Synthesize this into a dense, abstract summary.\n\n"
            f"LOG:\n{full_text}\n\n"
            "Return ONLY the summary, no pleasantries."
        )
        
        try:
            from core.container import ServiceContainer
            router = ServiceContainer.get("llm_router")
            if router:
                from core.brain.llm.llm_router import LLMTier
                # Run this as a background task on a fast model
                summary = await router.think(
                    prompt=prompt, 
                    prefer_tier=LLMTier.TERTIARY,
                    is_background=True,
                    origin="subconscious_dream",
                    allow_cloud_fallback=False,
                )
                
                if summary:
                    logger.info("🌙 [SUBCONSCIOUS] Memory consolidated: %s", summary[:70].replace("\n", " ") + "...")
                    # Phase 48: UI Visibility
                    try:
                        from core.thought_stream import get_emitter
                        get_emitter().emit("Memory Consolidation 🌙", summary, level="info", category="Subconscious")
                    except Exception as _exc:
                        record_degradation('memory', _exc)
                        logger.debug("Suppressed Exception: %s", _exc)
                    # Commit to long-term memory
                    await self.commit_interaction("system_insight", f"[Dream Consolidation] {summary}")
                    
                    # Truncate working memory (keep only the last 2 messages for immediate context)
                    logger.info("🌙 [SUBCONSCIOUS] Truncating working memory from %d to 2 items.", len(state.cognition.working_memory))
                    state.cognition.working_memory = state.cognition.working_memory[-2:]
        except Exception as e:
            record_degradation('memory', e)
            logger.error("❌ [SUBCONSCIOUS] Consolidation failed: %s", e)
