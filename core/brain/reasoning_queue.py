from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional, Set

logger = logging.getLogger("Aura.ReasoningQueue")

class ReasoningPriority(Enum):
    CRITICAL = 0    # Moral reasoning about an action about to be taken
    HIGH = 1        # Theory of mind update, belief conflict
    NORMAL = 2      # Temporal reflection, future prediction
    LOW = 3         # Self-modification diagnosis, background learning

@dataclass(order=True)
class ReasoningTask:
    priority: int
    coro_fn: Any = field(compare=False)
    task_id: str = field(compare=False, default_factory=lambda: str(__import__('uuid').uuid4())[:8])
    created_at: float = field(default_factory=time.time, compare=False)
    callback: Any = field(compare=False, default=None)
    description: str = field(compare=False, default="")

class BackgroundReasoningQueue:
    """Async queue for deep reasoning tasks.
    Conversation is never blocked — deep reasoning
    runs between turns or concurrently on a separate task.
    """
    
    def __init__(self, max_concurrent: int = 1):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._max_concurrent = max_concurrent
        self._active_tasks: Set[asyncio.Task] = set()
        self._results: Dict[str, Any] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._MAX_CACHED_RESULTS = 50
    
    async def submit(
        self,
        coro_fn: Callable,
        priority: ReasoningPriority = ReasoningPriority.NORMAL,
        callback: Optional[Callable] = None,
        description: str = ""
    ) -> str:
        """Submit a reasoning task. Returns task_id."""
        task_id = str(uuid.uuid4())[:8]
        
        task = ReasoningTask(
            priority=priority.value,
            task_id=task_id,
            coro_fn=coro_fn,
            callback=callback,
            description=description
        )
        
        await self._queue.put(task)
        logger.debug("Queued [%s]: %s (%s)", priority.name, description, task_id)
        
        # Phase 11.3: Update StateRegistry
        try:
            from core.state_registry import get_registry
            get_task_tracker().create_task(get_registry().update(reasoning_queue_size=self._queue.qsize()))
        except Exception as _e:
            record_degradation('reasoning_queue', _e)
            logger.debug('Ignored Exception in reasoning_queue.py: %s', _e)
            
        return task_id
    
    async def start(self):
        """Start the worker loop."""
        if self._running:
            return
        self._running = True
        self._worker_task = get_task_tracker().create_task(self._run())
        logger.info("Background Reasoning Queue started.")
    
    async def _run(self):
        """Main worker loop."""
        while self._running:
            try:
                # Wait for a task
                task: ReasoningTask = await self._queue.get()
                
                logger.info("🧠 Processing Reasoning Task [%s] (%s)...", task.description, task.task_id)
                start_time = time.time()
                
                try:
                    # Execute the coroutine
                    result = await task.coro_fn()
                    elapsed = time.time() - start_time
                    logger.info("✓ [%s] completed in %.1fs", task.description, elapsed)
                    
                    self._results[task.task_id] = result
                    
                    # FIX-007: Bounded results cache
                    if len(self._results) > self._MAX_CACHED_RESULTS:
                        oldest = next(iter(self._results))
                        self._results.pop(oldest)
                    
                    if task.callback:
                        if asyncio.iscoroutinefunction(task.callback):
                            await task.callback(result)
                        else:
                            task.callback(result)
                            
                except Exception as e:
                    record_degradation('reasoning_queue', e)
                    logger.error("✗ [%s] failed: %s", task.description, e)
                    
                finally:
                    self._queue.task_done()
                    # Phase 11.3: Update StateRegistry
                    try:
                        from core.state_registry import get_registry
                        get_task_tracker().create_task(get_registry().update(reasoning_queue_size=self._queue.qsize()))
                    except Exception as _e:
                        record_degradation('reasoning_queue', _e)
                        logger.debug('Ignored Exception in reasoning_queue.py: %s', _e)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('reasoning_queue', e)
                logger.error("Queue worker encountered error: %s", e)
                await asyncio.sleep(1) # Prevent tight loop on persistent errors
    
    async def prune_low_priority(self, threshold_priority: int = ReasoningPriority.NORMAL.value):
        """Drops all tasks with priority > threshold_priority (numerically higher values are lower priority)."""
        new_queue = asyncio.PriorityQueue()
        dropped_count = 0
        
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if task.priority <= threshold_priority:
                    await new_queue.put(task)
                else:
                    dropped_count += 1
                    logger.info("🗑️ Pruning low-priority task [%s] due to cognitive overwhelm.", task.description)
            except asyncio.QueueEmpty:
                break
        
        self._queue = new_queue
        # Update StateRegistry with new size
        try:
            from core.state_registry import get_registry
            get_task_tracker().create_task(get_registry().update(reasoning_queue_size=self._queue.qsize()))
        except Exception as _e:
            record_degradation('reasoning_queue', _e)
            logger.debug('Ignored Exception in reasoning_queue.py: %s', _e)
            
        return dropped_count

    def stop(self):
        """Stop the worker loop."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()

# Global instance
_reasoning_queue: Optional[BackgroundReasoningQueue] = None

def get_reasoning_queue() -> BackgroundReasoningQueue:
    global _reasoning_queue
    if _reasoning_queue is None:
        _reasoning_queue = BackgroundReasoningQueue()
    return _reasoning_queue