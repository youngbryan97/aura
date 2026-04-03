import asyncio
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

class TaskTracker:
    """Track and manage background asyncio tasks to ensure graceful shutdown.
    Prevents 'Task was destroyed but it is pending!' errors and ensures all
    background work is either completed or cleanly cancelled.
    
    Includes a concurrency semaphore to prevent task accumulation.
    """
    
    def __init__(self, name: str = "Global", max_concurrent: int = 20):
        self.name = name
        self.tasks: Set[asyncio.Task] = set()
        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None  # Lazy init
        self._high_water = 0
        self._total_tracked = 0
        
    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore (must be in event loop context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    def track(self, coro_or_task, name: Optional[str] = None) -> asyncio.Task:
        """Track a new task or coroutine (no concurrency limit)."""
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        else:
            task = asyncio.create_task(coro_or_task, name=name)
        self._mark_supervised(task)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        self._total_tracked += 1
        self._high_water = max(self._high_water, len(self.tasks))
        return task

    # Alias for compatibility with components calling track_task or create_task
    track_task = track
    create_task = track

    def bounded_track(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """Track a task WITH concurrency limiting via semaphore.
        Use this for short-lived tasks (maintenance, learning, reflection).
        Long-running loops should use track() directly.
        """
        async def _bounded():
            sem = self._get_semaphore()
            async with sem:
                if asyncio.iscoroutine(coro):
                    return await coro
                elif asyncio.iscoroutinefunction(coro):
                    return await coro()
                else:
                    return await coro

        task = asyncio.create_task(_bounded(), name=name)
        self._mark_supervised(task)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        self._total_tracked += 1
        self._high_water = max(self._high_water, len(self.tasks))
        return task

    def _mark_supervised(self, task: asyncio.Task) -> None:
        try:
            setattr(task, "_aura_supervised", True)
            setattr(task, "_aura_task_tracker", self.name)
        except Exception as e:
            logger.debug("TaskTracker[%s]: failed to mark task supervised: %s", self.name, e)

    @property
    def active_count(self) -> int:
        """Number of currently active (not done) tasks."""
        return len(self.tasks)

    async def shutdown(self, timeout: float = 5.0):
        """Cancel and wait for all tracked tasks."""
        if not self.tasks:
            return
            
        logger.info("Shutting down TaskTracker[%s]: %s tasks pending.", self.name, len(self.tasks))
        
        # 1. Signal cancellation
        for task in self.tasks:
            task.cancel()
            
        # 2. Wait for completion with timeout
        try:
            await asyncio.wait(self.tasks, timeout=timeout)
        except Exception as e:
            logger.error("Error during TaskTracker shutdown: %s", e)
            
        # 3. Final cleanup
        remaining = [t for t in self.tasks if not t.done()]
        if remaining:
            logger.warning("%d tasks still pending after timeout. Forcing abandonment.", len(remaining))
        
        self.tasks.clear()

    def get_stats(self) -> dict:
        return {
            "active": self.active_count,
            "high_water": self._high_water,
            "total_tracked": self._total_tracked,
            "max_concurrent": self._max_concurrent,
        }

# Global Instance
_task_tracker = None

def get_task_tracker() -> TaskTracker:
    global _task_tracker
    if _task_tracker is None:
        _task_tracker = TaskTracker(name="Global")
    return _task_tracker

# For backward compatibility if any module imports task_tracker directly
# Note: it's better to use get_task_tracker() in async contexts.
task_tracker = TaskTracker()

def fire_and_track(coro, name: Optional[str] = None) -> asyncio.Task:
    """Convenience function to create and track a task in one go."""
    tracker = get_task_tracker()
    return tracker.track(coro, name=name)
