"""Thread-safe ThoughtEmitter — the broadcast backbone for Aura's internal thought stream.
"""
import asyncio
import logging
import threading
from datetime import datetime

from core.runtime.errors import record_degradation

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
                    cls._instance = super().__new__(cls)
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

    @staticmethod
    def _also_tell_the_interface(message: dict) -> None:
        """Put the thought where the interface is actually looking.

        This emitter broadcasts to listeners that register with it, and nothing
        in the codebase registers — seventy-two modules have been emitting into
        a channel with no reader. The neural feed is fed from the event bus,
        which the interface bridge subscribes to with a wildcard.

        Bridging here rather than at the call sites means every existing
        emitter becomes visible at once, and anything written later reaches the
        interface without having to know this.
        """

        try:
            from core.event_bus import get_event_bus

            bus = get_event_bus()
            if bus is None:
                return
            payload = {
                "content": str(message.get("content") or ""),
                "phase": str(message.get("category") or "cognition"),
                "title": str(message.get("title") or ""),
                "urgency": "NORMAL" if message.get("level") != "warning" else "HIGH",
            }
            publish = getattr(bus, "publish_threadsafe", None)
            if callable(publish):
                publish("thoughts", payload)
        except Exception as exc:
            # A thought that cannot be shown is not worth failing the work that
            # produced it.
            record_degradation("thought_stream", exc, action="thought not bridged to the interface")

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

        self._also_tell_the_interface(message)

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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('thought_stream', e)
            logger.debug("Failed to bridge thought to EventBus: %s", e)


# Global singleton accessor
_emitter = ThoughtEmitter()


def get_emitter():
    return _emitter