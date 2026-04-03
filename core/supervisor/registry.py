import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger("Aura.TaskRegistry")

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    task_id: str
    owner_actor: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: Optional[str] = None

class TaskRegistry:
    """
    Centralized registry for tracking asynchronous work across the Actor network.
    Provides observability into what the kernel is currently doing.
    """
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}

    def register_task(self, owner: str, description: str, metadata: Optional[Dict] = None) -> str:
        """Register a new cross-process task."""
        task_id = str(uuid.uuid4())
        task = TaskInfo(
            task_id=task_id,
            owner_actor=owner,
            description=description,
            metadata=metadata or {}
        )
        self._tasks[task_id] = task
        logger.debug(f"📋 Task Registered: {description} (ID: {task_id})")
        return task_id

    def update_task(self, task_id: str, status: Optional[TaskStatus] = None, 
                   progress: Optional[float] = None, error: Optional[str] = None,
                   metadata_update: Optional[Dict] = None):
        """Update task status or progress."""
        task = self._tasks.get(task_id)
        if not task:
            logger.warning(f"❓ Attempted to update non-existent task: {task_id}")
            return

        if status:
            task.status = status
        if progress is not None:
            task.progress = progress
        if error:
            task.error = error
        if metadata_update:
            task.metadata.update(metadata_update)
            
        task.updated_at = time.time()
        
        if status == TaskStatus.COMPLETED:
             logger.debug(f"✅ Task Completed: {task.description}")
        elif status == TaskStatus.FAILED:
             logger.error(f"❌ Task Failed: {task.description} -> {error}")

    def get_task(self, task_id: Optional[str]) -> Optional[TaskInfo]:
        if not task_id:
            return None
        return self._tasks.get(task_id)

    def get_active_tasks(self) -> List[TaskInfo]:
        """Return all tasks currently in non-terminal states."""
        return [t for t in self._tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]

    def prune_completed(self, max_age_seconds: int = 3600):
        """Remove old terminal tasks from the registry."""
        now = time.time()
        to_remove = [
            tid for tid, t in self._tasks.items() 
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            and now - t.updated_at > max_age_seconds
        ]
        for tid in to_remove:
            del self._tasks[tid]
        if to_remove:
            logger.debug(f"🧹 Pruned {len(to_remove)} tasks from registry.")

_registry_instance: Optional[TaskRegistry] = None

def get_task_registry() -> TaskRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = TaskRegistry()
    return _registry_instance
