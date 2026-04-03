import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List

logger = logging.getLogger("Aura.HookManager")

class HookManager:
    """Manages system hooks for extending Aura without monkey-patching."""
    
    def __init__(self):
        self.hooks: Dict[str, List[Callable]] = {
            "pre_think": [],
            "post_think": [],
            "pre_action": [],
            "post_action": [],
            "on_message": [],
            "on_cycle": []
        }
        self._lock = threading.Lock()

    def register(self, event: str, callback: Callable):
        """Register a callback for a specific event."""
        with self._lock:
            if event in self.hooks:
                self.hooks[event].append(callback)
                logger.debug("Registered hook: %s for %s", callback.__name__ if hasattr(callback, '__name__') else 'lambda', event)
            else:
                logger.warning("Attempted to register unknown hook event: %s", event)

    async def trigger(self, event: str, *args, **kwargs) -> List[Any]:
        """Trigger all callbacks for a specific event and return results."""
        callbacks = []
        with self._lock:
            callbacks = list(self.hooks.get(event, []))
        
        results = []
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    res = await cb(*args, **kwargs)
                else:
                    res = cb(*args, **kwargs)
                results.append(res)
            except Exception as e:
                logger.error("Error in hook %s: %s", event, e, exc_info=True)
                results.append(None)
        return results