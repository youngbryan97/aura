from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class CognitiveGovernor:
    """Ensures Aura's core loops remain stable during API outages or cognitive overload."""
    
    def __init__(self, max_concurrent_tasks: int = 5, base_backoff: float = 1.0):
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.base_backoff = base_backoff
        self.circuit_state = "CLOSED"
        self.error_count = 0
        self.max_errors = 3
        
    async def execute_safely(self, task_name: str, coroutine: Callable, *args, **kwargs) -> Any:
        """Wraps critical cognitive tasks in a protective execution layer."""
        if self.circuit_state == "OPEN":
            logger.warning(f"Circuit OPEN. Rejecting task {task_name}. Falling back to internal state.")
            return {"status": "bypassed", "reason": "circuit_open"}

        async with self.semaphore:
            try:
                # Execute the actual cognitive task (e.g., LLM generation, tool use)
                result = await asyncio.wait_for(coroutine(*args, **kwargs), timeout=30.0)
                self._record_success()
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Task {task_name} timed out.")
                await self._record_failure()
                return {"status": "timeout", "error": "Operation took too long"}
                
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}")
                await self._record_failure()
                return {"status": "failed", "error": str(e)}

    async def _record_failure(self):
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.circuit_state = "OPEN"
            logger.critical("Cognitive Governor tripped! Entering cool-down phase.")
            # Trigger Aura's background reflection or sleep cycle here instead of crashing
            get_task_tracker().create_task(self._cooldown_recovery())

    def _record_success(self):
        self.error_count = max(0, self.error_count - 1)
        if self.circuit_state == "HALF_OPEN":
            self.circuit_state = "CLOSED"

    async def _cooldown_recovery(self):
        """Asynchronously manages the recovery state without blocking the main loop."""
        backoff_time = self.base_backoff * (2 ** self.error_count) # Exponential backoff
        await asyncio.sleep(min(backoff_time, 60.0))
        self.circuit_state = "HALF_OPEN"
        logger.info("Cognitive Governor attempting recovery (HALF_OPEN).")
