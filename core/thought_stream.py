"""Thread-safe ThoughtEmitter — the broadcast backbone for Aura's internal thought stream.
"""
import asyncio
import logging
import threading
from datetime import datetime

logger = logging.getLogger("Kernel.ThoughtStream")


class ThoughtEmitter:
    """Thread-safe thought broadcast singleton with bounded queues."""

    _instance = None
    _creation_lock = threading.Lock()
    _QUEUE_SIZE = 200

    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = super(ThoughtEmitter, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        with self._creation_lock:
            if getattr(self, 'initialized', False):
                return

            self.listeners = set()
            self._lock = threading.Lock()
            self._loop = None
            self.initialized = True
            logger.info("ThoughtEmitter initialized.")

    async def register(self, websocket):
        """Register a new listener queue."""
        q = asyncio.Queue(maxsize=self._QUEUE_SIZE)
        with self._lock:
            self.listeners.add(q)
            # Capture the loop of the registering client (Main Server Loop)
            if self._loop is None:
                self._loop = asyncio.get_running_loop()
        return q

    async def unregister(self, queue):
        """Remove a listener."""
        with self._lock:
            if queue in self.listeners:
                self.listeners.discard(queue)

    def emit(self, title: str, content: str, level: str = "info", category: str = "General", **kwargs):
        """Broadcast a thought/event to all listeners.
        Thread-safe: Can be called from sync threads (Orchestrator).
        """
        message = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "content": content,
            "level": level,
            "category": category
        }
        message.update(kwargs)

        with self._lock:
            loop = self._loop
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                    self._loop = loop
                except RuntimeError as exc:
                    logger.debug('No running event loop available: %s', exc)
            listeners_snapshot = list(self.listeners)

        if loop and listeners_snapshot:
            dead = []
            for q in listeners_snapshot:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, message)
                except asyncio.QueueFull:
                    dead.append(q)
                except RuntimeError:
                    # Event loop closed — log and mark dead
                    logger.debug("Event loop closed; removing dead listener")
                    dead.append(q)
            
            # Clean up dead listeners
            if dead:
                with self._lock:
                    for q in dead:
                        self.listeners.discard(q)

        # Bridge to unified AuraEventBus
        try:
            from .event_bus import get_event_bus
            get_event_bus().publish_threadsafe("thoughts", message)
        except Exception as e:
            logger.debug("Failed to bridge thought to EventBus: %s", e)


# Global singleton accessor
_emitter = ThoughtEmitter()


def get_emitter():
    return _emitter