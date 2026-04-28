from core.runtime.errors import record_degradation
import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Aura.ParallelThought")

class ParallelThoughtStream:
    """
    [MOTO TRANSIMAL] Explores multiple hypotheses in parallel to avoid cognitive anchoring.
    """
    
    def __init__(self, container: Any):
        self.container = container
        
    async def branch(self, objective: str, context: str) -> List[Dict[str, str]]:
        """Generate 3 parallel thoughts/hypotheses for the current context."""
        from core.brain.llm.llm_router import LLMTier
        router = self.container.get("llm_router", default=None)
        
        if not router:
            return []
            
        prompt = (
            f"Objective: {objective}\n"
            f"Context: {context}\n"
            f"Aura, generate 3 radically different perspectives or internal hypotheses "
            f"on how to address this. Be brief for each.\n"
            f"Format: 1. [Hypothesis A] ...\n2. [Hypothesis B] ...\n3. [Hypothesis C] ..."
        )
        
        branches = []
        try:
            # Parallel branching is background cognition and must stay on the 7B brainstem.
            resp = await router.think(
                prompt,
                prefer_tier=LLMTier.TERTIARY,
                is_background=True,
                origin="parallel_thought",
                allow_cloud_fallback=False,
            )
            if not resp:
                return []
            # ISSUE-86: Resilient Branch Parsing
            for line in resp.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match 1., 2., 3. OR - Hypothesis A: OR * [Hypothesis A]
                content = None
                if any(line.startswith(p) for p in ("1.", "2.", "3.")):
                     content = line[3:].strip()
                elif any(line.startswith(p) for p in ("- ", "* ")):
                     content = line[2:].strip()
                elif line.lower().startswith("hypothesis"):
                     content = line.split(":", 1)[-1].strip() if ":" in line else line
                
                if content:
                    branches.append({"content": content})
            
            # Broadcast to UI
            if branches:
                from core.event_bus import get_event_bus
                get_event_bus().publish_threadsafe("aura/ui/parallel_thoughts", {
                    "objective": objective,
                    "branches": branches[:3]
                })
            
            return branches[:3]
        except Exception as e:
            record_degradation('parallel_thought', e)
            logger.warning(f"Parallel thought branching failed: {e}")
            return []
