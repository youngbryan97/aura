"""Shared types and dataclasses for the Orchestrator."""
from core.runtime.errors import record_degradation
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """System status tracking"""
    
    initialized: bool = False
    running: bool = False
    healthy: bool = False
    start_time: Optional[float] = None
    cycle_count: int = 0
    last_error: Optional[str] = None
    skills_loaded: int = 0
    dependencies_ok: bool = False
    is_processing: bool = False
    is_throttled: bool = False
    agency: float = 0.8
    curiosity: float = 0.5
    last_active: Optional[float] = None
    acceleration_factor: float = 1.0 # Phase 21: Cognitive Acceleration
    singularity_threshold: bool = False # Phase 21: Convergence State
    
    def add_error(self, error: str):
        self.last_error = error
        self.healthy = False

def _bg_task_exception_handler(task: asyncio.Task):
    """Callback for background tasks to log exceptions instead of losing them."""
    try:
        exc = task.exception()
        if exc:
            logger.warning("Background task %s failed: %s", task.get_name(), exc)
            try:
                from .container import ServiceContainer
                immune = ServiceContainer.get("immune_system", None)
                if immune:
                    immune.on_error(exc, {"task": task.get_name()})
            except Exception as e:
                record_degradation('orchestrator_types', e)
                logger.debug("Immune system unavailable for background task error logging: %s", e)
    except asyncio.CancelledError:
        pass  # Expected: task was cancelled
    except asyncio.InvalidStateError:
        pass  # Expected: task hasn't finished yet
