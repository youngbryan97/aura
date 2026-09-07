"""core/event_bus.py — Topic-based Asynchronous Event Bus."""
import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import CancelledError as FutureCancelledError
from enum import IntEnum
from typing import Any

try:
    import redis.asyncio as redis
    from redis.exceptions import RedisError
    _REDIS_AVAILABLE = True
except ImportError:
    redis = None
    RedisError = None
    _REDIS_AVAILABLE = False

from core.config import config
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Kernel.EventBus")

_REDIS_ERRORS = (RedisError,) if RedisError is not None else ()
_EVENT_BUS_RECOVERABLE_ERRORS = (
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    ValueError,
    TypeError,
    AttributeError,
) + _REDIS_ERRORS


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(str(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))



#: Set while this thread is inside the payload check, so recording a
#: degradation — which can publish — cannot re-enter it.
_CHECKING = threading.local()



#: Times the payload checker itself raised, by exception name.
#:
#: A checker that fails silently reports exactly like a bus with no violations
#: to report, and those are opposite conditions. It cannot record a degradation
#: — that publishes, and publishing checks — so the count is the report.
WHEN_CHECKING_ITSELF_FAILED: dict[str, int] = {}

def _the_payload_matches_what_the_topic_declared(topic: Any, data: Any) -> None:
    """Check a declared topic's payload, and never raise doing it.

    The topic string is an address and stays one. What a consumer could not
    know is what arrives on it, and the only way to find out was to read every
    producer. A topic that declares its payload is checked here; one that says
    nothing is counted and left alone.

    A violation is a degradation, not an exception. A bus that raises inside
    publish turns a consumer's schema mistake into a producer's crash, which
    is a worse failure than the one being caught.
    """

    # Reentrancy is the whole difficulty here.
    #
    # Recording a degradation can publish, and publishing checks, and checking
    # can record — so the first mismatched payload turns into a loop that eats
    # the process. The guard is per thread rather than global: two threads
    # publishing at once are two independent checks and neither is inside the
    # other.
    if getattr(_CHECKING, "inside", False):
        return
    _CHECKING.inside = True
    try:
        from core.runtime.what_an_event_carries import check

        ok, reasons = check(str(topic or ""), data)
        if ok:
            return
        from core.runtime.errors import record_degradation

        record_degradation(
            "event_bus.payload",
            ValueError(f"{topic}: {'; '.join(reasons)}"),
            severity="warning",
            action="delivered the event anyway; the topic's declaration and its payload disagree",
        )
    except Exception as exc:  # noqa: BLE001 — checking must never stop a publish
        # Counted, not recorded. Recording a degradation publishes, publishing
        # checks, and checking is what just failed — so reporting this the
        # ordinary way is the loop the reentrancy guard above exists to stop.
        # A number the health surface can read is the one report that cannot
        # re-enter.
        kind = type(exc).__name__
        WHEN_CHECKING_ITSELF_FAILED[kind] = WHEN_CHECKING_ITSELF_FAILED.get(kind, 0) + 1
        return
    finally:
        _CHECKING.inside = False


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
                except (IndexError, TypeError, ValueError):
                    p = 99.0
                
                if p > max_p:
                    max_p = p
                    max_idx = i
            
            # Only replace if the new item has a lower numeric priority value (higher logical priority)
            try:
                new_p = float(item[0]) if isinstance(item, (tuple, list)) else 0.0
            except (IndexError, TypeError, ValueError):
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


def _record_to_bag(topic: str, data: Any) -> None:
    """Feed the always-on bus ring (core/observability/bus_recorder.py).

    Both local delivery paths funnel through here, which makes this the
    one place that sees every message. The ring is bounded and in memory,
    so the cost is a deque append; the payoff is that when something goes
    wrong the last minute of what the runtime actually SAW is already
    captured, rather than having to be predicted in advance.
    """
    try:
        from core.observability.bus_recorder import record

        record(topic, data)
    except Exception:  # noqa: BLE001 — the recorder is never load-bearing
        logger.debug("bus recorder capture failed for topic %s", topic, exc_info=True)


class AuraEventBus:
    """Topic-based Asynchronous Event Bus for unified messaging across sub-systems."""

    def __init__(self):
        # Store tuples of (PriorityQueue, asyncio.AbstractEventLoop)
        self._subscribers: dict[str, set[tuple]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_error: Exception | None = None
        # M-12 FIX: Initialize lock in __init__, not as lazy property
        self._lock = threading.Lock()
        self._seq = 0  # Monotonic counter for stable priority ordering
        self._seq_lock = threading.Lock()
        
        import uuid
        self._bus_id = str(uuid.uuid4())
        
        # Redis integration (C-07/H-12 FIX)
        self._redis: Any | None = None
        self._pubsub_task: asyncio.Task | None = None
        self._redis_loop: asyncio.AbstractEventLoop | None = None
        self._redis_url = config.redis.url if hasattr(config, "redis") else "redis://localhost:6379/0"
        self._use_redis = (_REDIS_AVAILABLE and getattr(config.redis, "use_for_events", False))
        self._redis_required = bool(getattr(config.redis, "required_for_events", False))
        self._remote_degraded = False
        self._remote_error_count = 0
        self._remote_last_error: BaseException | None = None
        self._remote_publish_lock: asyncio.Lock | None = None
        self._redis_publish_timeout_s = _env_float(
            "AURA_EVENT_BUS_REDIS_PUBLISH_TIMEOUT_S",
            2.0 if self._redis_required else 0.75,
            low=0.05,
            high=10.0,
        )
        self._closing = False

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

    def get_status(self) -> dict[str, Any]:
        """Return a diagnostic report of the event bus health."""
        alive = self.is_alive()
        return {
            "bus_id": self._bus_id,
            "redis_connected": self._redis is not None,
            "use_redis": self._use_redis,
            "redis_required": self._redis_required,
            "remote_degraded": self._remote_degraded,
            "degraded": self.degraded, # Patch 13
            "healthy": alive,
            "subscribers": {topic: len(subs) for topic, subs in self._subscribers.items()},
            "stats": {
                "delivered": self._delivered_count,
                "dropped": self._dropped_count,
                "errors": self._error_count,
                "remote_errors": self._remote_error_count,
                "last_error": str(self._last_error) if self._last_error else None,
                "remote_last_error": (
                    str(self._remote_last_error) if self._remote_last_error else None
                ),
            },
            "alive": alive,
        }

    def _record_error(self, exc: BaseException, message: str, *args: Any, degraded: bool = True) -> None:
        """Record event-bus degradation with counters and a visible log line.

        The event bus is used from callbacks and background threads. Recording
        must update the canonical degradation/health registries, but cannot
        raise out of a Future done-callback where fail-closed would become
        "exception calling callback" noise. Health contracts fail through
        get_status()/is_alive() instead.
        """
        with self._stats_lock:
            self._error_count += 1
            self._last_error = exc
        if degraded:
            self.degraded = True
        try:
            action = message % ((*args, exc) if args else (exc,))
        except (TypeError, ValueError):
            action = message
        record_degradation(
            "event_bus",
            exc,
            severity="degraded",
            action=action,
            enforce_failure_policy=False,
        )
        logger.warning(message, *args, exc)

    def _record_remote_error(self, exc: BaseException, message: str, *args: Any) -> None:
        """Record Redis transport failure without poisoning local bus health.

        Redis is useful for cross-process fanout, but Aura's canonical in-process
        event bus remains operational when local delivery succeeds. Installations
        that need Redis as a hard dependency can set
        AURA_REDIS_REQUIRED_FOR_EVENTS=1, which preserves fail-closed behavior.
        """
        with self._stats_lock:
            self._remote_error_count += 1
            self._remote_last_error = exc
            self._last_error = exc
        self._remote_degraded = True
        if self._redis_required:
            self._record_error(exc, message, *args, degraded=True)
            return
        try:
            action = message % ((*args, exc) if args else (exc,))
        except (TypeError, ValueError):
            action = message
        record_degradation(
            "event_bus.remote_redis",
            exc,
            severity="warning",
            action=action,
            enforce_failure_policy=False,
        )
        logger.warning(message, *args, exc)

    def is_alive(self) -> bool:
        """Return true only when local event delivery is usable and not degraded.

        The bus binds its loop lazily, on the first subscribe or publish. So a
        bus nobody has used yet has ``_loop is None``, and requiring a bound
        loop reported it DEAD — which the health checker treats as a critical
        component that stopped answering, tainting a clean boot with "an organ
        crashed and was restarted in-process" before anything had gone wrong.

        An unused bus is not a wedged bus. What the question actually asks is
        whether local delivery would work right now, so an unbound bus is alive
        when there is a running loop for it to bind to.
        """
        if self._redis_required and (self._remote_degraded or self._use_redis and self._redis is None):
            return False
        if self.degraded:
            return False
        if self._loop is not None:
            return bool(self._loop.is_running())
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop here to bind to and none bound already. Asked from a
            # plain thread, this cannot be answered in the affirmative.
            return False
        return True

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

    def _check_loop_mismatch(self):
        """Check if the current running loop matches the loop Redis was initialized on.
        If not, reset Redis connection state so it gets recreated on the current loop.
        """
        if not self._use_redis or not self._redis:
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # ONLY perform loop mismatch check on the main loop.
        # Secondary loops should not touch or reset the main loop's Redis connection.
        if self._loop and current_loop is not self._loop:
            return

        redis_loop = getattr(self, "_redis_loop", None)
        if redis_loop is None:
            self._redis_loop = current_loop
            return

        if redis_loop is not current_loop:
            logger.info(
                "AuraEventBus: Event loop changed from %s to %s. Re-initializing Redis client.",
                redis_loop,
                current_loop,
            )
            old_redis = self._redis
            old_task = self._pubsub_task
            self._redis = None
            self._pubsub_task = None
            self._redis_loop = None

            if old_task:
                try:
                    old_task.cancel()
                except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
                    self._record_remote_error(
                        exc,
                        "AuraEventBus: failed to cancel stale Redis listener task: %s",
                    )
            if old_redis:
                self._dispose_stale_redis_client(old_redis, redis_loop)

    def _dispose_stale_redis_client(
        self,
        redis_client: Any,
        redis_loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Dispose a Redis client without awaiting it from the wrong event loop."""
        async def safe_close(r: Any) -> None:
            try:
                await r.aclose()
            except asyncio.CancelledError:
                raise
            except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
                if self._closing:
                    logger.debug("AuraEventBus: stale Redis close skipped during shutdown: %s", exc)
                    return
                self._record_remote_error(
                    exc,
                    "AuraEventBus: stale Redis close failed: %s",
                )

        if redis_loop and redis_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(safe_close(redis_client), redis_loop)
                return
            except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
                if not self._closing:
                    self._record_remote_error(
                        exc,
                        "AuraEventBus: error scheduling stale Redis close on owner loop: %s",
                    )
                    return
                logger.debug("AuraEventBus: stale Redis owner-loop close skipped during shutdown: %s", exc)
                return

        logger.debug(
            "AuraEventBus: dropping stale Redis client bound to closed or unknown loop without cross-loop close."
        )

    async def _setup_redis(self):
        """Initialize Redis connection and start listener task."""
        if not self._use_redis or self._redis:
            return
            
        try:
            current_loop = asyncio.get_running_loop()
            if self._loop and current_loop is not self._loop:
                # Do not set up Redis on a secondary loop
                return
            self._redis_loop = current_loop
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            self._pubsub_task = get_task_tracker().create_task(
                self._redis_listener(),
                name="event_bus.redis_listener",
            )
            logger.info("AuraEventBus: Redis Pub/Sub connection established.")
        except _EVENT_BUS_RECOVERABLE_ERRORS as e:
            self._record_remote_error(
                e,
                "AuraEventBus: failed to connect to Redis; falling back to local-only mode: %s",
            )
            if self._redis:
                try:
                    await self._redis.aclose()
                except _EVENT_BUS_RECOVERABLE_ERRORS as _exc:
                    self._record_remote_error(
                        _exc,
                        "AuraEventBus: Redis cleanup after setup failure failed: %s",
                    )
                self._redis = None
            self._use_redis = False

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
        except _EVENT_BUS_RECOVERABLE_ERRORS as e:
            self._record_remote_error(
                e,
                "AuraEventBus: Redis listener unavailable; falling back to local-only mode: %s",
            )
            self._use_redis = False
        finally:
            self._pubsub_task = None
            try:
                await pubsub.punsubscribe("aura/events/*")
            except _EVENT_BUS_RECOVERABLE_ERRORS as _exc:
                self._record_remote_error(
                    _exc,
                    "AuraEventBus: Redis listener unsubscribe failed: %s",
                )
            try:
                await pubsub.aclose()
            except _EVENT_BUS_RECOVERABLE_ERRORS as _exc:
                self._record_remote_error(
                    _exc,
                    "AuraEventBus: Redis listener pubsub close failed: %s",
                )
            if not self._use_redis and self._redis:
                try:
                    await self._redis.aclose()
                except _EVENT_BUS_RECOVERABLE_ERRORS as _exc:
                    self._record_remote_error(
                        _exc,
                        "AuraEventBus: Redis listener client close failed: %s",
                    )
                self._redis = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Bind the bus to a specific event loop."""
        self._loop = loop

    async def shutdown(self):
        """Best-effort teardown for tests and controlled process shutdown."""
        self._closing = True
        current_loop = asyncio.get_running_loop()
        pubsub_task = self._pubsub_task
        self._pubsub_task = None
        if pubsub_task:
            task_loop = None
            try:
                task_loop = pubsub_task.get_loop()
            except _EVENT_BUS_RECOVERABLE_ERRORS:
                task_loop = None
            if task_loop is not None and task_loop is not current_loop:
                try:
                    if task_loop.is_running():
                        task_loop.call_soon_threadsafe(pubsub_task.cancel)
                    else:
                        pubsub_task.cancel()
                    logger.debug(
                        "AuraEventBus: Redis listener cancellation delegated to owner loop."
                    )
                except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
                    logger.debug(
                        "AuraEventBus: Redis listener owner-loop cancellation skipped during shutdown: %s",
                        exc,
                    )
            else:
                pubsub_task.cancel()
                try:
                    await pubsub_task
                except asyncio.CancelledError as _exc:
                    logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
                except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
                    self._record_error(
                        exc,
                        "AuraEventBus: pubsub shutdown failed: %s",
                        degraded=True,
                    )

        if self._redis:
            redis_loop = getattr(self, "_redis_loop", None)
            if (
                redis_loop is not None
                and redis_loop is not current_loop
                and hasattr(redis_loop, "is_running")
            ):
                self._dispose_stale_redis_client(self._redis, redis_loop)
                self._redis = None
                self._redis_loop = None
                self._subscribers.clear()
                self._loop = None
                return
            try:
                await self._redis.aclose()
            except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
                if self._closing and "Event loop is closed" in str(exc):
                    logger.debug(
                        "AuraEventBus: redis close skipped during shutdown; owner loop is already closed."
                    )
                else:
                    self._record_error(
                        exc,
                        "AuraEventBus: redis close failed: %s",
                        degraded=True,
                    )
            finally:
                self._redis = None
                self._redis_loop = None

        self._subscribers.clear()
        self._loop = None

    async def _acquire_lock_async(self, timeout: float, where: str) -> bool:
        """Take the subscriber lock without blocking the event loop.

        CP126 (critical): "Async subscription paths block on a threading
        lock. subscribe, unsubscribe, and local publish acquire a process
        threading lock with a multi-second timeout from coroutine code,
        allowing contention to freeze the event loop."

        ``threading.Lock.acquire(timeout=5.0)`` called from a coroutine
        blocks the loop THREAD, not the coroutine — so under contention
        every task on that loop stopped for up to five seconds, including
        the heartbeats that decide whether the runtime is alive.

        The uncontended case still takes the fast path: a non-blocking
        acquire, no thread hop, no measurable cost. Only actual contention
        goes to a worker thread, where waiting is free.
        """
        if self._lock.acquire(blocking=False):
            return True
        acquired = await asyncio.to_thread(self._lock.acquire, True, timeout)
        if not acquired:
            logger.error("🚨 [EVENTBUS] lock contention timeout in %s", where)
        return acquired

    async def subscribe(self, topic: str) -> asyncio.Queue:
        """Subscribe to a topic and receive a queue for events."""
        # Auto-capture the running loop for threadsafe publishing
        if not self._loop:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError as _e:
                logger.debug('Ignored RuntimeError in event_bus.py: %s', _e)
                
        self._check_loop_mismatch()
        if self._use_redis and not self._redis:
            await self._setup_redis()
            
        q = BoundedPriorityQueue(maxsize=1000)
        current_loop = asyncio.get_running_loop()
        
        acquired = await self._acquire_lock_async(5.0, f"subscribe({topic})")
        if not acquired:
            return q
        try:
            self._subscribers[topic].add((q, current_loop))
            logger.debug("New subscriber for topic: %s", topic)
        finally:
            self._lock.release()
            
        return q

    async def unsubscribe(self, topic: str, q: asyncio.PriorityQueue):
        """Remove a subscriber from a topic."""
        acquired = await self._acquire_lock_async(5.0, f"unsubscribe({topic})")
        if not acquired:
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
        _the_payload_matches_what_the_topic_declared(topic, data)
        # Ensure we're on the correct loop before doing anything
        try:
            current_loop = asyncio.get_running_loop()
            if self._loop is None or not self._loop.is_running():
                self._loop = current_loop
        except RuntimeError:
            current_loop = None

        # Redis clients are loop-bound, so their path must return to the owner
        # loop. Local delivery is not: _publish_local snapshots subscribers
        # under a thread-safe lock and schedules each queue on its own loop.
        # Delegating a local-only publish anyway made MindTick wait five seconds
        # on an unrelated busy loop and report a false EventBus stall even with
        # Redis disabled.
        if (
            self._use_redis
            and current_loop
            and self._loop
            and self._loop is not current_loop
            and self._loop.is_running()
        ):
            fut = asyncio.run_coroutine_threadsafe(
                self.publish(topic, data, priority), self._loop
            )
            await asyncio.wrap_future(fut)
            return

        self._check_loop_mismatch()

        # Tag with bus ID to prevent our own redis listener from echoing it back
        if isinstance(data, dict) and "_bus_id" not in data:
            data["_bus_id"] = self._bus_id
            
        # 1. Local delivery with priority
        await self._publish_local(topic, data, priority)
        
        # 2. Remote delivery via Redis (H-12)
        if self._closing:
            return
        if self._remote_degraded and not self._redis_required:
            return
        if self._use_redis:
            self._check_loop_mismatch()
            if not self._redis:
                await self._setup_redis()

            redis_client = self._redis
            if redis_client:
                if self._remote_publish_lock is None:
                    self._remote_publish_lock = asyncio.Lock()
                if self._remote_publish_lock.locked() and not self._redis_required:
                    return
                async with self._remote_publish_lock:
                    if self._closing:
                        return
                    if self._remote_degraded and not self._redis_required:
                        return
                    await self._publish_remote_redis(topic, data, redis_client)

    async def _publish_remote_redis(self, topic: str, data: Any, redis_client: Any) -> None:
        try:
            # Offload JSON serialization to thread to avoid event loop lag
            payload = await asyncio.to_thread(json.dumps, data)
            publish = getattr(redis_client, "publish", None)
            if not callable(publish):
                raise AttributeError(
                    f"{type(redis_client).__name__} has no callable publish"
                )
            await asyncio.wait_for(
                publish(f"aura/events/{topic}", payload),
                timeout=self._redis_publish_timeout_s,
            )
        except TimeoutError as e:
            self._record_remote_error(
                e,
                "AuraEventBus: Redis publish stalled; switching to local-only mode: %s",
            )
            self._use_redis = False
            self._redis = None
        except _EVENT_BUS_RECOVERABLE_ERRORS as e:
            self._record_remote_error(
                e,
                "AuraEventBus: Redis publish failed; switching to local-only mode: %s",
            )
            self._use_redis = False
            self._redis = None

    async def _publish_local(self, topic: str, data: Any, priority: int = EventPriority.COGNITIVE):
        """Asynchronously publish an event to all local subscribers with priority."""

        _record_to_bag(topic, data)

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

        acquired = await self._acquire_lock_async(5.0, f"publish_local({topic})")
        if not acquired:
            self._record_error(
                RuntimeError(f"event bus local publish lock timeout for topic {topic!r}"),
                "EventBus local publish lock timeout for topic '%s': %s",
                topic,
                degraded=True,
            )
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
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        
        stale_subscribers = []
        for q, loop in subscribers:
            try:
                if loop and loop.is_running():
                    if current_loop is loop:
                        loop.call_soon(self._safe_put_direct, q, item)
                    else:
                        loop.call_soon_threadsafe(self._safe_put_direct, q, item)
                    with self._stats_lock:
                        self._delivered_count += 1
                elif loop and loop.is_closed():
                    # Stale subscriber from a closed loop — mark for removal
                    stale_subscribers.append((q, loop))
                else:
                    self._safe_put_direct(q, item)
                    with self._stats_lock:
                        self._delivered_count += 1
            except RuntimeError as e:
                if "attached to a different loop" in str(e) or "is closed" in str(e):
                    # Stale loop reference — mark for removal
                    stale_subscribers.append((q, loop))
                    logger.debug("EventBus: Removing stale subscriber (loop mismatch) on topic '%s'", topic)
                else:
                    self._record_error(
                        e,
                        "EventBus delivery failure on topic '%s': %s",
                        topic,
                        degraded=True,
                    )
            except _EVENT_BUS_RECOVERABLE_ERRORS as e:
                self._record_error(
                    e,
                    "EventBus delivery failure on topic '%s': %s",
                    topic,
                    degraded=True,
                )

        # Clean up stale subscribers
        if stale_subscribers:
            acquired = await self._acquire_lock_async(2.0, "publish_local.stale_cleanup")
            if acquired:
                try:
                    for tup in stale_subscribers:
                        for t_topic in list(self._subscribers.keys()):
                            self._subscribers[t_topic].discard(tup)
                    logger.info("EventBus: Removed %d stale subscriber(s).", len(stale_subscribers))
                finally:
                    self._lock.release()

    def _safe_put_direct(self, queue, itm):
        """Synchronously puts item into the subscriber queue. 
        BoundedPriorityQueue handles the 'drop-worst' policy if full.
        """
        try:
            queue.put_nowait(itm)
        except asyncio.QueueFull:
            with self._stats_lock:
                self._dropped_count += 1
        except _EVENT_BUS_RECOVERABLE_ERRORS as e:
            self._record_error(
                e,
                "EventBus direct queue delivery failed: %s",
                degraded=True,
            )

    def _publish_local_now(self, topic: str, data: Any, priority: int = EventPriority.COGNITIVE) -> None:
        """Deliver a local event synchronously when already on the owning loop.

        ``publish_threadsafe`` is intentionally fire-and-forget. If it is called
        from the same event loop and schedules ``publish`` as another coroutine,
        short-lived loops can close before the coroutine gets a turn. This helper
        keeps local delivery deterministic without blocking on Redis.
        """
        _record_to_bag(topic, data)

        if isinstance(data, dict):
            data = data.copy()
            data.setdefault("_bus_id", self._bus_id)
            bounce_count = data.get("_bounce_count", 0)
            if bounce_count > 5:
                logger.debug("Dropped event on topic %s - Max bounce depth reached.", topic)
                return
            data["_bounce_count"] = bounce_count + 1

        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            self._record_error(
                RuntimeError(f"event bus local publish lock timeout for topic {topic!r}"),
                "EventBus local publish lock timeout for topic '%s': %s",
                topic,
                degraded=True,
            )
            return
        try:
            subscribers = list(self._subscribers.get(topic, []))
            subscribers.extend(list(self._subscribers.get("*", [])))
        finally:
            self._lock.release()

        if not subscribers:
            return

        with self._seq_lock:
            self._seq = (self._seq + 1) % 10_000_000
            sequence = self._seq
        item = (priority, sequence, {"topic": topic, "data": data})
        stale_subscribers = []
        for q, loop in subscribers:
            try:
                if loop and loop.is_closed():
                    stale_subscribers.append((q, loop))
                    continue
                self._safe_put_direct(q, item)
                with self._stats_lock:
                    self._delivered_count += 1
            except RuntimeError as e:
                if "attached to a different loop" in str(e) or "is closed" in str(e):
                    stale_subscribers.append((q, loop))
                    logger.debug("EventBus: Removing stale subscriber (loop mismatch) on topic '%s'", topic)
                else:
                    self._record_error(
                        e,
                        "EventBus delivery failure on topic '%s': %s",
                        topic,
                        degraded=True,
                    )
            except _EVENT_BUS_RECOVERABLE_ERRORS as e:
                self._record_error(
                    e,
                    "EventBus delivery failure on topic '%s': %s",
                    topic,
                    degraded=True,
                )

        if stale_subscribers:
            acquired = self._lock.acquire(timeout=2.0)
            if acquired:
                try:
                    for tup in stale_subscribers:
                        for t_topic in list(self._subscribers.keys()):
                            self._subscribers[t_topic].discard(tup)
                    logger.info("EventBus: Removed %d stale subscriber(s).", len(stale_subscribers))
                finally:
                    self._lock.release()

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
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is target_loop:
                self._publish_local_now(topic, data, priority)
                return
            # Schedule the async publish call on the target loop
            future = asyncio.run_coroutine_threadsafe(self.publish(topic, data, priority), target_loop)
            future.add_done_callback(
                lambda completed, loop=target_loop: self._threadsafe_publish_done(completed, loop)
            )
            
            # Periodic health report for diagnostics
            if self._delivered_count % 100 == 0:
                logger.debug("📡 [EVENT_BUS:%s] Status: %d delivered, %d subs active.", 
                            self._bus_id[:8], self._delivered_count, len(self._subscribers))
        else:
            # No loop during sync init (e.g., skill discovery at import time) — not an error
            logger.debug("[EVENT_BUS:%s] No running loop for '%s'; delivery deferred (expected during sync init).",
                         self._bus_id[:8], topic)

    # _inject_threadsafe is now retired in favor of run_coroutine_threadsafe

    def _threadsafe_publish_done(self, future, target_loop: asyncio.AbstractEventLoop | None = None) -> None:
        if future.cancelled():
            if self._closing or self._loop_is_tearing_down(target_loop):
                logger.debug("EventBus threadsafe publish cancelled during controlled shutdown.")
                return
            self._record_error(
                asyncio.CancelledError("threadsafe publish cancelled"),
                "EventBus threadsafe publish did not complete: %s",
                degraded=True,
            )
            return
        try:
            future.result()
        except (asyncio.CancelledError, FutureCancelledError) as exc:
            if self._closing or self._loop_is_tearing_down(target_loop):
                logger.debug("EventBus threadsafe publish cancelled during loop teardown: %s", exc)
                return
            self._record_error(
                exc,
                "EventBus threadsafe publish did not complete: %s",
                degraded=True,
            )
        except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
            self._record_error(
                exc,
                "EventBus threadsafe publish failed: %s",
                degraded=True,
            )

    @staticmethod
    def _loop_is_tearing_down(loop: asyncio.AbstractEventLoop | None) -> bool:
        """Return true when a threadsafe callback was cancelled by loop teardown."""
        return loop is None or loop.is_closed() or not loop.is_running()



# Global singleton instance
_bus = AuraEventBus()


def get_event_bus():
    return _bus


async def reset_event_bus() -> AuraEventBus:
    """Replace the global event bus with a fresh instance."""
    global _bus
    try:
        await _bus.shutdown()
    except _EVENT_BUS_RECOVERABLE_ERRORS as exc:
        _bus._record_error(exc, "EventBus reset shutdown failed: %s", degraded=True)
    _bus = AuraEventBus()
    return _bus
