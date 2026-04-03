"""core/state_registry.py

Unified State Registry for Aura.
Acts as the single source of truth for Affect, Agency, and Substrate states 
to prevent desynchronization and runtime contradictions.
"""

import asyncio
import logging
import time
import threading
import queue
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger("Aura.StateRegistry")

@dataclass
class UnifiedState:
    # Affect (Emotional)
    frustration: float = 0.0
    curiosity: float = 0.5
    energy: float = 1.0
    valence: float = 0.0
    arousal: float = 0.3
    
    # Agency (Operational)
    engagement_mode: str = "attentive_idle"
    initiative_energy: float = 0.7
    curiosity_pressure: float = 0.5
    active_shards: int = 0
    reasoning_queue_size: int = 0
    
    # Substrate (Consciousness Proxies)
    phi: float = 0.0
    coherence: float = 1.0
    em_field: float = 0.0
    
    # Metabolism (Vitals)
    health_score: float = 1.0
    cpu_load: float = 0.0
    memory_usage: float = 0.0
    
    # Global Indices (Internal Metrics)
    free_energy: float = 0.0        # Predictive surprise
    phi_estimate: float = 0.0       # Integrated Information
    loop_cycle: int = 0             # MindTick iteration count
    bonding_level: float = 0.0      # Relationship depth
    stability: float = 1.0          # Identity stability
    resonance: Dict[str, float] = field(default_factory=dict) # Persona blend
    
    # Situational Context
    current_goal: str = "Maintain homeostasis and observe."
    
    # Metadata
    version: int = 0
    timestamp: float = field(default_factory=time.time)

class UnifiedStateRegistry:
    """
    Central repository for all cross-engine state variables.
    Engines push their local state here, and other engines (or the prompt builder) 
    pull a consistent snapshot.
    """
    def __init__(self):
        self._state = UnifiedState()
        self._listeners: List[Callable[[UnifiedState], None]] = []
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._notify_queue: queue.Queue[Optional[UnifiedState]] = queue.Queue()
        self._dispatcher_task: Optional[asyncio.Task] = None
        
        # Instrumentation
        self.update_count = 0
        self.failed_notifications = 0
        
        logger.info("📡 UnifiedStateRegistry initialized (Hardened Dispatcher).")

    def ensure_dispatcher(self):
        """Lazy start the dispatcher in the active event loop."""
        if self._dispatcher_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._dispatcher_task = loop.create_task(self._notification_dispatcher())
                logger.info("🚀 StateRegistry: Notification Dispatcher started.")
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in state_registry.py: %s', _e)

    async def _notification_dispatcher(self):
        """Single consumer task for state updates."""
        while True:
            try:
                # get() is blocking, use to_thread to avoid blocking loop
                snapshot = await asyncio.to_thread(lambda: self._notify_queue.get())
                if snapshot is None: break # Shutdown signal
                
                for listener in list(self._listeners):
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(snapshot)
                        else:
                            # Run sync listeners in threads to avoid blocking the dispatcher
                            await asyncio.to_thread(lambda l=listener, s=snapshot: l(s))
                    except Exception as e:
                        self.failed_notifications += 1
                        logger.error(f"StateRegistry listener failed: {e}")
                
                self._notify_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"StateRegistry dispatcher error: {e}")
                await asyncio.sleep(0.1)

    async def update(self, **kwargs):
        """Async update for asyncio-based engines."""
        async with self._async_lock:
            self._apply_update(**kwargs)
            self.ensure_dispatcher()

    def sync_update(self, **kwargs):
        """Synchronous update for background threads (e.g., Metabolism)."""
        with self._sync_lock:
            self._apply_update(**kwargs)
            # Notify the dispatcher (thread-safe queue)
            # Note: We can't ensure_dispatcher() if we aren't in a thread with a loop
            # But usually the main thread/orchestrator will have started it.

    def _apply_update(self, **kwargs):
        """Internal logic to apply state changes (must be called under lock)."""
        self.update_count += 1
        has_changes = False
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                if getattr(self._state, key) != value:
                    setattr(self._state, key, value)
                    has_changes = True
            else:
                logger.warning(f"Attempted to update unknown state field: {key}")
        
        if has_changes:
            self._state.version += 1
            self._state.timestamp = time.time()
            # Push a deep copy or snapshot to the queue
            self._notify_queue.put(self.get_state_copy())

    def get_state_copy(self) -> UnifiedState:
        """Return a deep copy of current state."""
        return copy.deepcopy(self._state)

    def get_state(self) -> UnifiedState:
        """Get a copy of the current unified state to prevent reference leak."""
        return copy.deepcopy(self._state)

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns a dict snapshot of the current state."""
        return asdict(self._state)

    def subscribe(self, listener: Callable[[UnifiedState], None]):
        """Subscribe to state changes."""
        self._listeners.append(listener)

# Singleton
_instance: Optional[UnifiedStateRegistry] = None

def get_registry() -> UnifiedStateRegistry:
    global _instance
    if _instance is None:
        _instance = UnifiedStateRegistry()
    return _instance
