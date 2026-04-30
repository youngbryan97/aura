"""infrastructure/services.py - Core Aura services.
"""
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("Infra.Services")

class SimpleInputBus:
    """Real in-memory event bus implementation of InputBus protocol."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._logger = logging.getLogger("Infra.InputBus")

    async def publish(self, message: Dict[str, Any]) -> bool:
        topic = message.get("topic", "default")
        if topic in self._subscribers:
            async def _run_cb(cb):
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(message)
                    else:
                        cb(message)
                except Exception as e:
                    self._logger.error("Error in subscriber %s: %s", cb, e)

            await asyncio.gather(*[_run_cb(cb) for cb in self._subscribers[topic]])
            return True
        return False

    def subscribe(self, topic: str, callback: Callable) -> str:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        return f"{topic}:{len(self._subscribers[topic])}"

    async def shutdown(self):
        self._subscribers.clear()

class SimpleProcessManager:
    """Real asyncio task manager implementation of ProcessManager protocol."""

    def __init__(self):
        self._processes: Dict[str, asyncio.Task] = {}
        self._logger = logging.getLogger("Infra.ProcessManager")

    async def start_process(self, process_id: str, config: Dict[str, Any]) -> bool:
        target = config.get("target")
        if not target or not asyncio.iscoroutinefunction(target):
            self._logger.error("Invalid target for process %s", process_id)
            return False
            
        args = config.get("args", [])
        kwargs = config.get("kwargs", {})
        
        task = get_task_tracker().create_task(target(*args, **kwargs), name=process_id)
        self._processes[process_id] = task
        self._logger.info("Started process %s", process_id)
        return True

    async def stop_process(self, process_id: str) -> bool:
        if process_id in self._processes:
            self._processes[process_id].cancel()
            try:
                await self._processes[process_id]
            except asyncio.CancelledError:
                pass
            del self._processes[process_id]
            self._logger.info("Stopped process %s", process_id)
            return True
        return False

    async def cleanup(self):
        for pid in list(self._processes.keys()):
            await self.stop_process(pid)

class KeyValueMemory:
    """Real in-memory Key-Value store implementation of MemoryStoreV2 protocol."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._ttls: Dict[str, float] = {}
        self._logger = logging.getLogger("Infra.Memory")

    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._store[key] = value
        if ttl:
            self._ttls[key] = time.time() + ttl
        elif key in self._ttls:
            del self._ttls[key]
        return True

    async def retrieve(self, key: str) -> Optional[Any]:
        if key in self._ttls and time.time() > self._ttls[key]:
            del self._store[key]
            del self._ttls[key]
            return None
        return self._store.get(key)
        
    async def cleanup(self):
        self._store.clear()
        self._ttls.clear()
