import asyncio
import enum
import time
import logging
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, List

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Scheduler")

class Lifecycle(enum.Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    MODIFYING = "modifying"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class TaskSpec:
    name: str
    coro: Callable[[], Any]  # Awaitable factory or coroutine function
    tick_interval: Optional[float] = None  # seconds. If None, it's a one-shot or event-driven task.
    last_run: float = field(default_factory=lambda: 0.0)
    running_task: Optional[asyncio.Task] = None
    critical: bool = False
    priority: int = 0 # Higher = more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)

class Scheduler:
    """
    Central Scheduler for Aura's autonomic nervous system.
    Manages background loops, heartbeats, and metabolic tasks with 
    deterministic concurrency and error isolation.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Scheduler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Audit-38: Ensure __init__ only runs once across singleton access
        if getattr(self, "_initialized", False):
            return
        self._tasks: Dict[str, TaskSpec] = {}
        self.state = Lifecycle.INITIALIZING
        self._lock: Optional[asyncio.Lock] = None
        self._stop: Optional[asyncio.Event] = None
        self._health: Dict[str, str] = {}
        self._main_loop_task: Optional[asyncio.Task] = None
        self._initialized = True
        logger.info("Scheduler substrate initialized.")

    def _ensure_async_primitives(self) -> tuple[asyncio.Lock, asyncio.Event]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._stop is None:
            self._stop = asyncio.Event()
        return self._lock, self._stop

    async def register(self, spec: TaskSpec):
        """Register a subsystem task with the scheduler."""
        lock, _stop = self._ensure_async_primitives()
        async with lock:
            if spec.name in self._tasks:
                logger.warning(f"Task {spec.name} already registered. Updating spec.")
            self._tasks[spec.name] = spec
            self._health[spec.name] = "registered"
            logger.debug(f"Registered task: {spec.name} (interval={spec.tick_interval})")

    async def start(self):
        """Ignite the scheduling loop."""
        if self._main_loop_task and not self._main_loop_task.done():
            logger.warning("Scheduler already running.")
            return

        _lock, stop_event = self._ensure_async_primitives()
        self.state = Lifecycle.IDLE
        stop_event.clear()
        self._main_loop_task = get_task_tracker().create_task(
            self._main_loop(),
            name="aura.scheduler.main_loop",
        )
        logger.info("🚀 Scheduler started.")

    async def _main_loop(self):
        """The heartbeat of the scheduler."""
        lock, stop_event = self._ensure_async_primitives()
        try:
            while not stop_event.is_set():
                now = time.monotonic()
                async with lock:
                    pending_tasks = sorted(
                        self._tasks.values(), 
                        key=lambda x: (x.critical, x.priority), 
                        reverse=True
                    )
                
                for spec in pending_tasks:
                    if spec.tick_interval is None:
                        continue
                    
                    # Check if it's time to run and not already running
                    if now - spec.last_run >= spec.tick_interval:
                        if spec.running_task is None or spec.running_task.done():
                            spec.last_run = now
                            spec.running_task = get_task_tracker().create_task(
                                self._run_task(spec),
                                name=f"scheduler.{spec.name}",
                            )
                
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logger.info("Scheduler main loop cancelled.")
        except Exception as e:
            logger.error(f"Scheduler Fatal Crash: {e}")
            logger.error(traceback.format_exc())
            self.state = Lifecycle.RECOVERING

    async def _run_task(self, spec: TaskSpec):
        """Safely execute a task with structured monitoring."""
        try:
            self._health[spec.name] = "running"
            
            # Handle both coro functions and direct coroutines
            res = spec.coro()
            if asyncio.iscoroutine(res):
                await res
            elif hasattr(res, '__await__'):
                await res
                
            self._health[spec.name] = "ok"
        except asyncio.CancelledError:
            self._health[spec.name] = "cancelled"
            raise
        except Exception as e:
            self._health[spec.name] = f"error: {type(e).__name__}"
            logger.error(f"Task {spec.name} failed: {e}")
            if spec.critical:
                logger.critical(f"CRITICAL Task {spec.name} failed! Triggering recovery.")
                self.state = Lifecycle.RECOVERING
        finally:
            spec.running_task = None

    async def stop(self):
        """Gracefully shut down all scheduled tasks."""
        logger.info("Shutting down scheduler...")
        lock, stop_event = self._ensure_async_primitives()
        stop_event.set()
        self.state = Lifecycle.SHUTTING_DOWN
        
        async with lock:
            for spec in self._tasks.values():
                if spec.running_task:
                    spec.running_task.cancel()
        
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in scheduler.py: %s', _e)
        
        logger.info("Scheduler disengaged.")

    def get_health(self):
        """Return structured health check data for the system API."""
        return {
            "state": self.state.value,
            "tasks": dict(self._health),
            "active_tasks": len([t for t in self._tasks.values() if t.running_task and not t.running_task.done()])
        }

# Global Instance
scheduler = Scheduler()
