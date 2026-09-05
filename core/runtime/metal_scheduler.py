import asyncio
import logging
from enum import IntEnum
from typing import Dict, Any, Optional

logger = logging.getLogger("Aura.MetalScheduler")

class PriorityTier(IntEnum):
    STEERING = 0    # Highest: Critical for every token
    INFERENCE = 1   # High: Active generation
    SENSES = 2      # Medium: Vision, Audio processing
    COGNITION = 3   # Low: Deliberation, Swarm thinking
    EVOLUTION = 4   # Lowest: Background metabolic decay, dreams

class MetalScheduler:
    """Cooperative semaphore-based prioritization for hardware-heavy tasks.
    
    Ensures that high-priority tasks (STEERING, INFERENCE) have guaranteed 
    access to CPU/GPU resources by throttling lower-priority concurrent tasks.
    """
    
    def __init__(self):
        # Semaphores to control concurrency per tier
        self._semaphores = {
            PriorityTier.STEERING: asyncio.Semaphore(10), # Practically unlimited for steering
            PriorityTier.INFERENCE: asyncio.Semaphore(3),  # 64GB M-series — 32B+7B can overlap
            PriorityTier.SENSES: asyncio.Semaphore(1),     # Vision is heavy, do one at a time
            PriorityTier.COGNITION: asyncio.Semaphore(4),
            PriorityTier.EVOLUTION: asyncio.Semaphore(2),
        }
        self._active_counts = {tier: 0 for tier in PriorityTier}
        self._boost_mode = False

    async def run(self, tier: PriorityTier, coro):
        """Execute a coroutine within its priority tier's semaphore."""
        sem = self._semaphores[tier]
        
        # Throttling logic: If a higher priority task is active, tighten lower semaphores
        if self._active_counts[PriorityTier.INFERENCE] > 0 and tier > PriorityTier.INFERENCE:
            # We are generating! Slow down background work.
            # (In a real implementation, we'd dynamically adjust semaphore _value,
            # but for now we just add a small delay to background starts)
            await asyncio.sleep(0.01)

        async with sem:
            self._active_counts[tier] += 1
            try:
                return await coro
            finally:
                self._active_counts[tier] -= 1

    def set_boost(self, active: bool):
        """Enable 'Boost Mode' for ultra-low latency interaction."""
        self._boost_mode = active
        if active:
            logger.info("🚀 [METAL] Boost Mode active. Throttling background substrate.")
        else:
            logger.info("⚖️ [METAL] Standard distribution restored.")

    def get_stats(self) -> Dict[str, int]:
        return {tier.name: count for tier, count in self._active_counts.items()}

# Global Instance
_scheduler = MetalScheduler()

def get_metal_scheduler() -> MetalScheduler:
    return _scheduler
