"""core/system_monitor.py — Recursive Stability & Health Auditing for Aura Zenith.

Monitoring system health, technical debt, and recursive stability.
"""

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.SystemMonitor")

@dataclass
class SystemHealthState:
    """Current health snapshot."""
    uptime: float
    memory_usage: float  # Simulated
    active_shards: int
    unresolved_refinements: int
    cognitive_stability: float  # 0.0 to 1.0 (inverse of entropy/contradiction density)
    timestamp: float = field(default_factory=time.time)

class SystemStateMonitor:
    """Auditing Aura's own stability."""
    
    def __init__(self):
        self.start_time = time.time()
    async def audit_stability(self) -> SystemHealthState:
        """Perform a deep audit of current system state."""
        import asyncio
        from core.container import ServiceContainer
        swarm = ServiceContainer.get("sovereign_swarm", default=None)
        refiner = ServiceContainer.get("code_refiner", default=None)
        
        # Heuristic for stability: inverse of pending refinements and active shards
        active_shards = len(swarm.shards) if swarm and hasattr(swarm, 'shards') else 0
        pending_refinements = len(refiner.proposals) if refiner and hasattr(refiner, 'proposals') else 0
        
        # More refinements needed + more shards = higher risk of instability
        stability = max(0.0, 1.0 - (pending_refinements * 0.05) - (active_shards * 0.02))

        # Phase 11.3: Queue Monitoring & Overwhelm Reflex
        from core.state_registry import get_registry
        s = get_registry().get_state()
        queue_size = s.reasoning_queue_size
        
        OVERWHELM_THRESHOLD = 20
        if queue_size > OVERWHELM_THRESHOLD:
            from core.brain.reasoning_queue import get_reasoning_queue, ReasoningPriority
            logger.warning("🚨 [COGNITIVE OVERWHELM] Queue size %d exceeds threshold %d. Triggering reflex.", queue_size, OVERWHELM_THRESHOLD)
            rq = get_reasoning_queue()
            # Drop everything below HIGH (i.e. keep CRITICAL and HIGH)
            dropped = await rq.prune_low_priority(threshold_priority=ReasoningPriority.HIGH.value)
            logger.info("🗑️ Overwhelm reflex dropped %d low-priority reasoning tasks.", dropped)
            # Re-fetch stability after pruning
            stability *= 0.8  # Temporary penalty for being overwhelmed
        
        state = SystemHealthState(
            uptime=time.time() - self.start_time,
            memory_usage=0.0, # Placeholder
            active_shards=active_shards,
            unresolved_refinements=pending_refinements,
            cognitive_stability=stability
        )
        
        self.health_history.append(state)
        
        # Sync final stability back to registry
        t = get_task_tracker().create_task(get_registry().update(coherence=stability))
        t.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
        
        if stability < 0.6:
            logger.warning(f"Cognitive stability low: {stability:.2f}. Triggering dream cycle.")
            try:
                from core.scheduler import scheduler, TaskSpec
                from core.maintenance.dream_cycle import run_dream_cycle
                
                # Check if already registered to avoid duplicates
                if not any(t.name == "dream_cycle" for t in scheduler.tasks):
                    await scheduler.register(TaskSpec(
                        name="dream_cycle",
                        coro=run_dream_cycle,
                        tick_interval=3600 # Run once per hour if stability stays low
                    ))
            except Exception as e:
                record_degradation('system_monitor', e)
                logger.debug("Failed to register dream cycle: %s", e)
            
        return state

# Service Registration
def register_system_monitor():
    """Register the system monitor."""
    from core.container import ServiceContainer, ServiceLifetime
    ServiceContainer.register(
        "system_monitor",
        factory=lambda: SystemStateMonitor(),
        lifetime=ServiceLifetime.SINGLETON
    )
