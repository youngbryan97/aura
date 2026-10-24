"""core/state/state_registry.py

Unified State Registry for Aura.
Acts as the single source of truth for Affect, Agency, and Substrate states 
to prevent desynchronization and runtime contradictions.
"""

import asyncio
import copy
import inspect
import logging
import queue
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.utils.task_tracker import get_task_tracker
from core.verify import Violation, invariant

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
    resonance: dict[str, float] = field(default_factory=dict) # Persona blend
    
    # Situational Context
    current_goal: str = "Maintain homeostasis and observe."
    
    # Metadata
    version: int = 0
    timestamp: float = field(default_factory=lambda: time.time())

class UnifiedStateRegistry:
    """
    Central repository for all cross-engine state variables.
    Engines push their local state here, and other engines (or the prompt builder) 
    pull a consistent snapshot.
    """
    def __init__(self, *, notification_capacity: int = 64):
        if type(notification_capacity) is not int or not 1 <= notification_capacity <= 256:
            raise ValueError("notification_capacity must be an integer from 1 to 256")
        self._state = UnifiedState()
        self._listeners: list[Callable[[UnifiedState], None]] = []
        self._sync_lock = checked_lock("state_registry.state", reentrant=True)
        self._notify_queue: queue.Queue[UnifiedState | None] = queue.Queue(maxsize=notification_capacity)
        self._dispatcher_task: asyncio.Task | None = None
        self._dispatcher_loop: asyncio.AbstractEventLoop | None = None
        self._notify_ready: asyncio.Event | None = None
        self._shutting_down = False
        
        # Instrumentation
        self.update_count = 0
        self.failed_notifications = 0
        self.coalesced_notifications = 0
        
        logger.info("📡 UnifiedStateRegistry initialized (Hardened Dispatcher).")

    def ensure_dispatcher(self):
        """Lazy start the dispatcher in the active event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self._sync_lock:
            if self._shutting_down:
                return
            if self._dispatcher_task is None or self._dispatcher_task.done():
                self._dispatcher_loop = loop
                self._notify_ready = asyncio.Event()
                self._dispatcher_task = get_task_tracker().create_task(
                    self._notification_dispatcher(),
                    name="state_registry.notification_dispatcher",
                )
                logger.info("🚀 StateRegistry: Notification Dispatcher started.")

    def _enqueue(self, snapshot: UnifiedState | None) -> None:
        """Keep recent state when a consumer falls behind. Caller holds the state lock."""
        needs_wakeup = self._notify_queue.empty()
        try:
            self._notify_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._notify_queue.get_nowait()
            except queue.Empty:
                # The consumer made room between Full and get_nowait.
                pass
            else:
                self._notify_queue.task_done()
                self.coalesced_notifications += 1
            self._notify_queue.put_nowait(snapshot)
        loop, ready = self._dispatcher_loop, self._notify_ready
        if needs_wakeup and loop is not None and ready is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(ready.set)
            except RuntimeError as exc:
                logger.debug("State notification loop closed before wakeup: %s", exc)

    async def _notification_dispatcher(self):
        """Single consumer task for state updates."""
        with self._sync_lock:
            if self._notify_ready is None:
                self._dispatcher_loop = asyncio.get_running_loop()
                self._notify_ready = asyncio.Event()
            ready = self._notify_ready
        while True:
            try:
                with self._sync_lock:
                    try:
                        snapshot = self._notify_queue.get_nowait()
                    except queue.Empty:
                        ready.clear()
                        waiting = True
                    else:
                        waiting = False
                    listeners = tuple(self._listeners)
                if waiting:
                    await ready.wait()
                    continue
                try:
                    if snapshot is None:
                        break
                    for listener in listeners:
                        try:
                            # A listener owns its copy, including nested mutable fields.
                            delivered = copy.deepcopy(snapshot)
                            if inspect.iscoroutinefunction(listener):
                                await listener(delivered)
                            else:
                                await asyncio.to_thread(listener, delivered)
                        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                            record_degradation('state_registry', e)
                            with self._sync_lock:
                                self.failed_notifications += 1
                            logger.error("StateRegistry listener failed: %s", e)
                finally:
                    self._notify_queue.task_done()
            except asyncio.CancelledError:
                break
            except (OSError, ConnectionError, TimeoutError) as e:
                record_degradation('state_registry', e)
                logger.error("StateRegistry dispatcher error: %s", e)
                await asyncio.sleep(0.1)

    async def shutdown_dispatcher(
        self,
        *,
        timeout_s: float = 1.0,
        clear_listeners: bool = False,
    ) -> None:
        """Stop the notification dispatcher without leaving event-loop work behind."""
        with self._sync_lock:
            self._shutting_down = True
            task = self._dispatcher_task
        if task is None:
            self._finish_shutdown(clear_listeners)
        elif not task.done():
            with self._sync_lock:
                self._enqueue(None)
            try:
                await asyncio.wait_for(task, timeout=timeout_s)
            except TimeoutError as exc:
                record_degradation("state_registry", exc)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._finish_shutdown(clear_listeners)
        else:
            self._finish_shutdown(clear_listeners)

    def _finish_shutdown(self, clear_listeners: bool) -> None:
        with self._sync_lock:
            while True:
                try:
                    self._notify_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._notify_queue.task_done()
            self._dispatcher_task = None
            self._dispatcher_loop = None
            self._notify_ready = None
            self._shutting_down = False
            if clear_listeners:
                self._listeners.clear()

    async def update(self, **kwargs):
        """Async update for asyncio-based engines."""
        await asyncio.to_thread(self._apply_update, **kwargs)
        self.ensure_dispatcher()

    def sync_update(self, **kwargs):
        """Synchronous update for background threads (e.g., Metabolism)."""
        self._apply_update(**kwargs)

    def _apply_update(self, **kwargs):
        """Serialize thread and async writers with snapshot readers."""
        with self._sync_lock:
            self._apply_update_locked(**kwargs)

    def _apply_update_locked(self, **kwargs):
        self.update_count += 1
        has_changes = False
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                if getattr(self._state, key) != value:
                    setattr(self._state, key, copy.deepcopy(value))
                    has_changes = True
            else:
                logger.warning("Attempted to update unknown state field: %s", key)
        
        if has_changes:
            self._state.version += 1
            self._state.timestamp = time.time()
            # Push a deep copy or snapshot to the queue
            if self._shutting_down:
                self.coalesced_notifications += 1
            else:
                self._enqueue(copy.deepcopy(self._state))

    def get_state_copy(self) -> UnifiedState:
        """Return a deep copy of current state."""
        with self._sync_lock:
            return copy.deepcopy(self._state)

    def get_state(self) -> UnifiedState:
        """Get a copy of the current unified state to prevent reference leak."""
        return self.get_state_copy()

    def get_snapshot(self) -> dict[str, Any]:
        """Returns a dict snapshot of the current state."""
        with self._sync_lock:
            return asdict(self._state)

    def subscribe(self, listener: Callable[[UnifiedState], None]):
        """Subscribe to state changes."""
        with self._sync_lock:
            self._listeners.append(listener)

# Singleton
_instance: UnifiedStateRegistry | None = None
_instance_lock = checked_lock("state_registry.singleton")

def get_registry() -> UnifiedStateRegistry:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = UnifiedStateRegistry()
        return _instance


@invariant("state_registry.bounded_notifications", scope="state", owner=__name__)
def _bounded_notifications():
    registry = _instance
    if registry is not None and not 1 <= registry._notify_queue.maxsize <= 256:
        yield Violation(subject="state_registry", message="Notification backlog exceeds its admitted capacity")
