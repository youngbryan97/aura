import asyncio
import json
import logging
import sys
import time
import threading
from collections import defaultdict
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import redis.asyncio as redis
    _REDIS_AVAILABLE = True
except ImportError:
    redis = None
    _REDIS_AVAILABLE = False

from core.config import config
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Kernel.EventBus")


class EventPriority(IntEnum):
    """Event priority tiers. Lower number = higher priority."""
    CRITICAL = 0    # System emergencies, stall recovery
    USER = 1        # User messages (typed + voice), direct interaction
    COGNITIVE = 2   # LLM responses, cognitive processing results
    AUTONOMIC = 3   # Health pulses, subsystem heartbeats, agency actions
    BACKGROUND = 4  # Dreams, curiosity exploration, self-modification


class BoundedPriorityQueue(asyncio.PriorityQueue):
    """Custom PriorityQueue that drops the LOWEST priority item when full."""
    
    def put_nowait(self, item: Any):
        """Standard put_nowait with overflow management.

        When the queue is full the lowest-priority (highest numeric value)
        item is replaced if the new item has higher priority.
        """
        if self.full():
            # Guard: _queue is a CPython implementation detail.
            queue_list = getattr(self, "_queue", None)
            if not isinstance(queue_list, list) or not queue_list:
                return super().put_nowait(item) # Should not happen if full()

            # Find the item with the highest numeric priority value (lowest logical priority)
            # Tuple: (priority, seq, data)
            max_idx = 0
            max_p = -1.0 # Use float for numeric comparison
            
            for i, val in enumerate(queue_list):
                try:
                    # Coerce priority to float to handle potential mixed types safely
                    p = float(val[0]) if isinstance(val, (tuple, list)) else 99.0
                except (Exception, IndexError, TypeError):
                    p = 99.0
                
                if p > max_p:
                    max_p = p
                    max_idx = i
            
            # Only replace if the new item has a lower numeric priority value (higher logical priority)
            try:
                new_p = float(item[0]) if isinstance(item, (tuple, list)) else 0.0
            except (Exception, IndexError, TypeError):
                new_p = 0.0
                
            if new_p < max_p:
                # Use the internal list directly if available, otherwise fallback
                ql = getattr(self, "_queue", None)
                if isinstance(ql, list):
                    ql[max_idx] = item
                    import heapq
                    # Use a try-except to catch non-comparable items during heapify
                    try:
                        heapq.heapify(ql)
                    except TypeError:
                        # If heapify fails due to comparison errors, we sort with a key
                        # to force numeric-only comparison for the heap property.
                        ql.sort(key=lambda x: (float(x[0]) if isinstance(x, (tuple, list)) else 99.0))
            return

        return super().put_nowait(item)


class AuraEventBus:
    """Topic-based Asynchronous Event Bus for unified messaging across sub-systems."""

    def __init__(self):
        # Store tuples of (PriorityQueue, asyncio.AbstractEventLoop)
        self._subscribers: Dict[str, Set[tuple]] = defaultdict(set)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_error: Optional[Exception] = None
        # M-12 FIX: Initialize lock in __init__, not as lazy property
        self._lock = threading.Lock()
        self._seq = 0  # Monotonic counter for stable priority ordering
        self._seq_lock = threading.Lock()
        
        import uuid
        self._bus_id = str(uuid.uuid4())
        
        # Redis integration (C-07/H-12 FIX)
        self._redis: Optional[Any] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._redis_url = config.redis.url if hasattr(config, "redis") else "redis://localhost:6379/0"
        self._use_redis = (_REDIS_AVAILABLE and getattr(config.redis, "use_for_events", False))

        if not _REDIS_AVAILABLE and getattr(config.redis, "use_for_events", False):
            logger.warning("redis package not installed — EventBus running in local-only mode.")

        # Self-Diagnostic State
        self.degraded = False
        self._error_count = 0
        self._delivered_count = 0
        self._dropped_count = 0
        self._stats_lock = threading.Lock()
        
        logger.info("AuraEventBus initialized (Redis: %s).", self._use_redis)
        logger.info("✅ [EVENT_BUS] Kernel signaling READY.")

    def get_status(self) -> Dict[str, Any]:
        """Return a diagnostic report of the event bus health."""
        return {
            "bus_id": self._bus_id,
            "redis_connected": self._redis is not None,
            "use_redis": self._use_redis,
            "degraded": self.degraded, # Patch 13
            "subscribers": {topic: len(subs) for topic, subs in self._subscribers.items()},
            "stats": {
                "delivered": self._delivered_count,
                "dropped": self._dropped_count,
                "errors": self._error_count,
                "last_error": str(self._last_error) if self._last_error else None
            },
            "alive": self._loop is not None and self._loop.is_running()
        }

    async def diagnose(self):
        """Actively check and report health, attempting self-repair if needed."""
        status = self.get_status()
        
        # Self-Repair: Redis
        if self._use_redis and self._redis is None:
            logger.info("EventBus: Redis reconnection triggered during diagnosis.")
            await self._setup_redis()
            
        # Tell the system outright
        await self.publish("system/event_bus/status", {
            "type": "diagnostic_report",
            "status": status,
            "timestamp": time.time()
        }, priority=EventPriority.AUTONOMIC)
        
        if self._error_count > 0:
            logger.warning("🚨 [EVENT_BUS] Degradation detected: %s errors. Report: %s", 
                           self._error_count, status)
        else:
            logger.info("✓ [EVENT_BUS] Health check passed: %s topics active.", len(self._subscribers))

    async def _setup_redis(self):
        """Initialize Redis connection and start listener task."""
        if not self._use_redis or self._redis:
            return
            
        try:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            self._pubsub_task = get_task_tracker().create_task(
                self._redis_listener(),
                name="event_bus.redis_listener",
            )
            logger.info("AuraEventBus: Redis Pub/Sub connection established.")
        except Exception as e:
            logger.error("AuraEventBus: Failed to connect to Redis: %s", e)
            if self._redis:
                try:
                    await self._redis.aclose()
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                self._redis = None
            self._use_redis = False
            self.degraded = True # Patch 13

    async def _redis_listener(self):
        """Listen for events from other processes via Redis."""
        if not self._redis:
            return

        pubsub = self._redis.pubsub()
        
        try:
            await pubsub.psubscribe("aura/events/*")
            async for message in pubsub.listen():
                if message["type"] == "pmessage":
                    channel = message["channel"]
                    topic = channel.split("/")[-1]
                    try:
                        data = json.loads(message["data"])
                        
                        # Prevent echoing our own events
                        if isinstance(data, dict) and data.get("_bus_id") == self._bus_id:
                            continue
                            
                        # Publish locally to this process's subscribers
                        await self._publish_local(topic, data)
                    except json.JSONDecodeError:
                        logger.warning("AuraEventBus: Received malformed JSON from Redis for topic %s", topic)
        except asyncio.CancelledError as _e:
            logger.debug('Ignored asyncio.CancelledError in event_bus.py: %s', _e)
        except Exception as e:
            self._last_error = e
            self._error_count += 1
            self.degraded = True
            self._use_redis = False
            logger.warning("AuraEventBus: Redis listener unavailable, falling back to local-only mode: %s", e)
        finally:
            self._pubsub_task = None
            try:
                await pubsub.punsubscribe("aura/events/*")
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
            try:
                await pubsub.aclose()
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
            if not self._use_redis and self._redis:
                try:
                    await self._redis.aclose()
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                self._redis = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Bind the bus to a specific event loop."""
        self._loop = loop

    async def shutdown(self):
        """Best-effort teardown for tests and controlled process shutdown."""
        pubsub_task = self._pubsub_task
        self._pubsub_task = None
        if pubsub_task:
            pubsub_task.cancel()
            try:
                await pubsub_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
            except Exception as exc:
                logger.debug("AuraEventBus: pubsub shutdown failed: %s", exc)

        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as exc:
                logger.debug("AuraEventBus: redis close failed: %s", exc)
            finally:
                self._redis = None

        self._subscribers.clear()
        self._loop = None

    async def subscribe(self, topic: str) -> asyncio.Queue:
        """Subscribe to a topic and receive a queue for events."""
        # Auto-capture the running loop for threadsafe publishing
        if not self._loop:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in event_bus.py: %s', _e)
                
        if self._use_redis and not self._redis:
            await self._setup_redis()
            
        q = BoundedPriorityQueue(maxsize=1000)
        current_loop = asyncio.get_running_loop()
        
        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            logger.error("🚨 [EVENTBUS] DEADLOCK DETECTED in subscribe(%s)!", topic)
            return q
        try:
            self._subscribers[topic].add((q, current_loop))
            logger.debug("New subscriber for topic: %s", topic)
        finally:
            self._lock.release()
            
        return q

    async def unsubscribe(self, topic: str, q: asyncio.PriorityQueue):
        """Remove a subscriber from a topic."""
        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            logger.error("🚨 [EVENTBUS] DEADLOCK DETECTED in unsubscribe(%s)!", topic)
            return
        
        try:
            if topic in self._subscribers:
                # Find and remove the tuple containing this queue
                to_remove = [tup for tup in self._subscribers[topic] if tup[0] == q]
                for tup in to_remove:
                    self._subscribers[topic].discard(tup)
                logger.debug("Subscriber removed from topic: %s", topic)
        finally:
            self._lock.release()

    async def publish(self, topic: str, data: Any, priority: int = EventPriority.COGNITIVE):
        """Publish an event to all subscribers (local and remote).
        
        Args:
            priority: EventPriority tier. Lower = higher priority.
        """
        # Tag with bus ID to prevent our own redis listener from echoing it back
        if isinstance(data, dict) and "_bus_id" not in data:
            data["_bus_id"] = self._bus_id
            
        # 1. Local delivery with priority
        await self._publish_local(topic, data, priority)
        
        # 2. Remote delivery via Redis (H-12)
        if self._use_redis:
            if not self._redis:
                await self._setup_redis()
            
            if self._redis:
                try:
                    # Offload JSON serialization to thread to avoid event loop lag
                    payload = await asyncio.to_thread(json.dumps, data)
                    # [STABILITY] Wrap Redis publish in a 2.0s timeout to prevent external stalls.
                    await asyncio.wait_for(self._redis.publish(f"aura/events/{topic}", payload), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.debug("AuraEventBus: Redis publish STALLED (timeout).")
                except Exception as e:
                    logger.debug("AuraEventBus: Redis publish failed: %s", e)

    async def _publish_local(self, topic: str, data: Any, priority: int = EventPriority.COGNITIVE):
        """Asynchronously publish an event to all local subscribers with priority."""
        
        # --- 🛑 PREVENT EVENT ECHOES (H-21 FIX: Proper cloning) ---
        if isinstance(data, dict):
            # Create a shallow copy to prevent sub-scribers from mutating the shared data
            data = data.copy() 
            bounce_count = data.get("_bounce_count", 0)
            if bounce_count > 5: # Relaxed slightly for complex multi-hop routing
                logger.debug("Dropped event on topic %s - Max bounce depth reached.", topic)
                return
            data["_bounce_count"] = bounce_count + 1
        # -------------------------------

        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            logger.error("🚨 [EVENTBUS] DEADLOCK DETECTED in _publish_local(%s)! Dropping event.", topic)
            return
            
        try:
            subscribers = list(self._subscribers.get(topic, []))
            subscribers.extend(list(self._subscribers.get("*", [])))
        finally:
            self._lock.release()

        if not subscribers:
            return

        # Moderate modulo to prevent integer expansion slowing down PriorityQueue sort
        with self._seq_lock:
            self._seq = (self._seq + 1) % 10_000_000
            sequence = self._seq
        # PriorityQueue tuple: (priority, sequence, event)
        item = (priority, sequence, {"topic": topic, "data": data})
        
        for q, loop in subscribers:
            try:
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(self._safe_put_direct, q, item)
                    with self._stats_lock:
                        self._delivered_count += 1
                else:
                    self._safe_put_direct(q, item)
                    with self._stats_lock:
                        self._delivered_count += 1
            except Exception as e:
                with self._stats_lock:
                    self._error_count += 1
                    self._last_error = e
                logger.error("EventBus delivery failure on topic '%s': %s", topic, e)

    def _safe_put_direct(self, queue, itm):
        """Synchronously puts item into the subscriber queue. 
        BoundedPriorityQueue handles the 'drop-worst' policy if full.
        """
        try:
            queue.put_nowait(itm)
        except asyncio.QueueFull:
            with self._stats_lock:
                self._dropped_count += 1
        except Exception as e:
            with self._stats_lock:
                self._error_count += 1
                self._last_error = e

    def publish_threadsafe(self, topic: str, data: Any, priority: int = EventPriority.COGNITIVE):
        """Safely fire events from background threads to the main asyncio loop."""
        # C-09 FIX: Use run_coroutine_threadsafe for consistent background -> loop transition
        target_loop = self._loop
        
        # 🛡️ Hardening: Aggressively look for a running loop if the bound one is stale
        if not target_loop or not target_loop.is_running():
            try:
                target_loop = asyncio.get_running_loop()
            except RuntimeError as _e:
                # If no running loop in current thread, try to find the main loop if we can
                # (This is often 'None' in background threads unless we set it globally)
                logger.debug('Ignored RuntimeError in event_bus.py: %s', _e)

        if target_loop and target_loop.is_running():
            # Schedule the async publish call on the target loop
            asyncio.run_coroutine_threadsafe(self.publish(topic, data, priority), target_loop)
            
            # Periodic health report for diagnostics
            if self._delivered_count % 100 == 0:
                logger.debug("📡 [EVENT_BUS:%s] Status: %d delivered, %d subs active.", 
                            self._bus_id[:8], self._delivered_count, len(self._subscribers))
        else:
            # No loop during sync init (e.g., skill discovery at import time) — not an error
            logger.debug("[EVENT_BUS:%s] No running loop for '%s'; delivery deferred (expected during sync init).",
                         self._bus_id[:8], topic)

    # _inject_threadsafe is now retired in favor of run_coroutine_threadsafe



# Global singleton instance
_bus = AuraEventBus()


def get_event_bus():
    return _bus


async def reset_event_bus() -> AuraEventBus:
    """Replace the global event bus with a fresh instance."""
    global _bus
    try:
        await _bus.shutdown()
    except Exception as exc:
        logger.debug("EventBus reset shutdown failed: %s", exc)
    _bus = AuraEventBus()
    return _bus
