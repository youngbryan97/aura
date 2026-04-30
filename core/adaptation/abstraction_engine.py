"""core/adaptation/abstraction_engine.py

Asynchronous First Principles Extractor.
Analyzes specific, successful problem resolutions and distills them into 
universal, generalized rules for zero-shot application in novel domains.
"""
from core.runtime.atomic_writer import atomic_write_text
import asyncio
import logging
import json
import time
from pathlib import Path
from core.container import ServiceContainer

logger = logging.getLogger("Aura.AbstractionEngine")

class AbstractionEngine:
    def __init__(self, storage_path: str = "data/first_principles.json"):
        # Use workspace-relative path if not absolute
        if not storage_path.startswith("/"):
            from core.config import config
            self.storage_path = config.paths.data_dir / "first_principles.json"
        else:
            self.storage_path = Path(storage_path)
            
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        
        # Initialize the file if it doesn't exist
        if not self.storage_path.exists():
            atomic_write_text(self.storage_path, "[]")

    async def abstract_from_success(self, context: str, successful_resolution: str) -> str:
        """
        Takes a specific solved problem and forces the local model to extract 
        the generalized underlying logic.
        """
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if not engine:
            logger.warning("AbstractionEngine: No cognitive engine found.")
            return ""

        prompt = f"""[SYSTEM ROLE: EPISTEMIC ARCHITECT]
You have just successfully solved a specific problem. Your task is to extract the underlying FIRST PRINCIPLE so it can be applied to entirely different domains in the future.

SPECIFIC CONTEXT:
"{context}"

SUCCESSFUL RESOLUTION:
"{successful_resolution}"

Task: Strip away all the specific nouns, entities, and situational details. Extract the pure, universal logical rule or structural truth that made this resolution work. 
Format your response as a single, highly condensed generalized heuristic. 
Example: "When a specialized resource is abruptly depleted, systemic adaptation must favor agility over direct substitution."
"""
        from core.brain.cognitive_engine import ThinkingMode
        # Deep thinking mode for high-level abstraction
        res = await engine.think(objective=prompt, mode=ThinkingMode.DEEP, block_user=False)
        abstracted_principle = res.content if hasattr(res, 'content') else str(res)

        if abstracted_principle:
            logger.info("🧠 First Principle Abstracted: %s...", abstracted_principle[:50])
            await self._commit_principle(abstracted_principle)
            
        return abstracted_principle

    async def _commit_principle(self, principle: str):
        """Asynchronously appends the new principle to the JSON store."""
        async with self._lock:
            try:
                # Read existing principles
                content = await asyncio.to_thread(self.storage_path.read_text)
                principles_list = json.loads(content) if content else []
                
                # Append new principle with timestamp
                principles_list.append({
                    "timestamp": time.time(),
                    "principle": principle,
                    "application_count": 0
                })
                
                # Write back to disk
                updated_content = json.dumps(principles_list, indent=2)
                await asyncio.to_thread(self.storage_path.write_text, updated_content)
                
                # Optionally: Inject into the BlackHoleVault for semantic retrieval
                memory_facade = ServiceContainer.get("memory_facade", default=None)
                if memory_facade and hasattr(memory_facade, 'store'):
                    await asyncio.to_thread(
                        memory_facade.store, 
                        content=f"[FIRST PRINCIPLE] {principle}", 
                        metadata={"type": "abstract_heuristic"}
                    )
                
                # Phase 15.2: Instant Swarm Propagation
                belief_sync = ServiceContainer.get("belief_sync", default=None)
                if belief_sync and hasattr(belief_sync, 'broadcast_attention_spike'):
                    # Using attention spike as a proxy for "everyone look at this new principle"
                    # Or we could call a specific broadcast method if we add it
                    await belief_sync.broadcast_attention_spike(
                        context=f"NEW_FIRST_PRINCIPLE: {principle[:100]}",
                        urgency=0.9
                    )
                    
            except Exception as e:
                logger.error(f"Failed to commit first principle: {e}")

    async def get_core_principles(self, limit: int = 5) -> str:
        """Retrieves the most frequently applied principles to inject into the system prompt."""
        try:
            if not self.storage_path.exists():
                return ""
                
            content = await asyncio.to_thread(self.storage_path.read_text)
            principles = json.loads(content)
            if not principles:
                return ""
                
            # Sort by application count (most useful principles bubble to the top)
            sorted_principles = sorted(principles, key=lambda x: x.get("application_count", 0), reverse=True)
            
            formatted = "\n[ACTIVE FIRST PRINCIPLES]\n"
            for p in sorted_principles[:limit]:
                formatted += f"- {p['principle']}\n"
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to load principles: {e}")
            return ""

def register_abstraction_engine():
    """Register the abstraction engine in the service container."""
    from core.container import ServiceContainer
    ServiceContainer.register("abstraction_engine", lambda: AbstractionEngine(), singleton=True)
