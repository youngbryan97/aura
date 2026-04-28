"""
core/agency/latent_distiller.py
===============================
MIST'S LATENT SPACE DISTILLER

Implements background cognitive synthesis (Temporal Dilation).
When the system is idle, MIST scans transient logs and session histories, 
running them through a distillation process to extract permanent 
insights, beliefs, and vector memories.
"""

from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("Aura.LatentDistiller")

class LatentSpaceDistiller:
    """
    Distills raw experience into permanent semantic wisdom.
    """

    def __init__(self, memory_provider=None):
        self.memory = memory_provider
        self._is_distilling = False
        logger.info("🌫️ LatentSpaceDistiller initialized (MIST/Pantheon pattern)")

    async def distill_session(self, session_history: List[Dict[str, Any]]):
        """Asynchronously distills a session into core insights."""
        if self._is_distilling: return
        self._is_distilling = True
        
        try:
            logger.info("🌫️ MIST: Beginning temporal dilation / background distillation...")
            # Simulate heavy lifting: consolidating multiple turns into a single summary
            full_text = " ".join([m.get("content", "") for m in session_history])
            
            # Distill session via LLM if available, else use extractive summary
            summary = None
            if len(full_text) > 200:
                try:
                    from core.container import ServiceContainer
                    brain = ServiceContainer.get("cognitive_engine", default=None)
                    if brain and hasattr(brain, "think"):
                        summary = await brain.think(
                            f"Summarize the key insights and decisions from this conversation in 2-3 sentences:\n\n{full_text[:3000]}"
                        )
                except Exception as e:
                    record_degradation('latent_distiller', e)
                    logger.debug("MIST: LLM distillation unavailable: %s", e)

                # Extractive fallback if LLM unavailable
                if not summary or len(str(summary).strip()) < 10:
                    sentences = [s.strip() for s in full_text.split(".") if len(s.strip()) > 20]
                    summary = ". ".join(sentences[:5]) + "." if sentences else full_text[:500]

                # Store in vector memory if provider exists
                if self.memory and summary:
                    await self.memory.store_memory(
                        content=str(summary),
                        metadata={"type": "distilled_wisdom", "timestamp": time.time()}
                    )
                    logger.info("MIST: Distilled session into permanent memory.")
        except Exception as e:
            record_degradation('latent_distiller', e)
            logger.error("❌ MIST: Distillation failed: %s", e)
        finally:
            self._is_distilling = False

    async def find_associative_leaps(self, query: str) -> List[str]:
        """Identifies non-obvious links across different memory domains."""
        # This would use the vector DB's similarity search with high diversity
        return ["Potential link between [Topic A] and [Topic B] identified."]
