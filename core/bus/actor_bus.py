import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

from .local_pipe_bus import LocalPipeBus, _is_connection_pair

logger = logging.getLogger("Kernel.ActorBus")


_ACTOR_BUS_SEND_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    BrokenPipeError,
    ConnectionResetError,
)


class BusDegraded(Exception):  # noqa: N818 - public compatibility name.
    """Raised when the bus health probe fails or congestion is too high."""


def _is_shutdown_commit_request(actor: str, msg_type: str, payload: Any) -> bool:
    return (
        str(actor or "") == "state_vault"
        and str(msg_type or "") == "commit"
        and isinstance(payload, dict)
        and str(payload.get("cause") or "").lower() == "shutdown"
    )


class ActorBus:
    """Unified Actor Bus abstraction with health gating and congestion control.
    Manages multiple LocalPipeBus transports indexed by actor name.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._transports: dict[str, LocalPipeBus] = {}
        self._last_health_check: dict[str, float] = {}
        self._health_timeout = 0.1  # 100ms spec
        self._high_water_mark = 50  # Max pending requests before degradation
        self._is_running = False
        
        # ZENITH: Backpressured Telemetry Queue
        self._telemetry_queue: asyncio.Queue | None = None
        self._telemetry_broadcaster_task = None
        self._telemetry_drops = 0
        self._send_drops = 0
        self._last_drop: dict[str, Any] | None = None
        self._initialized = True

    @staticmethod
    def _should_report_drop(count: int) -> bool:
        return count in {1, 10, 50, 100} or (count > 0 and count % 250 == 0)

    def _record_drop(
        self,
        *,
        kind: str,
        reason: str,
        actor: str | None = None,
        topic: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        if kind == "telemetry":
            self._telemetry_drops += 1
            count = self._telemetry_drops
        else:
            self._send_drops += 1
            count = self._send_drops

        self._last_drop = {
            "kind": kind,
            "reason": reason,
            "actor": actor,
            "topic": topic,
            "count": count,
            "at": time.time(),
        }
        if self._should_report_drop(count):
            exc = error if error is not None else RuntimeError(reason)
            record_degradation(
                "actor_bus",
                exc,
                severity="warning",
                action=f"{kind}_drop_visible:{reason}",
                extra={k: v for k, v in self._last_drop.items() if v is not None},
                enforce_failure_policy=False,
            )

    def get_status(self) -> dict[str, Any]:
        queue_size = self._telemetry_queue.qsize() if self._telemetry_queue is not None else 0
        transport_status = {
            name: self._transport_status(transport)
            for name, transport in self._transports.items()
        }
        return {
            "running": self._is_running,
            "actors": sorted(self._transports),
            "healthy": self.is_alive(),
            "telemetry_task": self._task_status(self._telemetry_broadcaster_task),
            "telemetry_task_alive": self._telemetry_task_alive(),
            "transports": transport_status,
            "telemetry_queue_size": queue_size,
            "telemetry_drops": self._telemetry_drops,
            "send_drops": self._send_drops,
            "last_drop": dict(self._last_drop or {}),
        }

    def is_alive(self) -> bool:
        """Return true when the bus and every registered transport are usable."""

        if not self._is_running:
            return False
        if self._telemetry_queue is None or not self._telemetry_task_alive():
            return False
        return all(self._transport_alive(transport) for transport in self._transports.values())

    @staticmethod
    def _task_status(task: asyncio.Task | None) -> dict[str, Any]:
        if task is None:
            return {
                "present": False,
                "done": None,
                "cancelled": None,
                "failed": None,
                "exception": None,
            }
        done = bool(task.done())
        cancelled = bool(task.cancelled()) if done else False
        exception: str | None = None
        failed = False
        if done and not cancelled:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                cancelled = True
                exc = None
            except (RuntimeError, AttributeError, TypeError, ValueError) as status_exc:
                exc = status_exc
            if exc is not None:
                failed = True
                exception = f"{type(exc).__name__}: {exc}"
        return {
            "present": True,
            "done": done,
            "cancelled": cancelled,
            "failed": failed,
            "exception": exception,
        }

    def _telemetry_task_alive(self) -> bool:
        status = self._task_status(self._telemetry_broadcaster_task)
        return bool(status["present"]) and not bool(status["done"])

    @staticmethod
    def _transport_alive(transport: Any) -> bool:
        is_alive = getattr(transport, "is_alive", None)
        if callable(is_alive):
            return bool(is_alive())
        if not bool(getattr(transport, "_is_running", False)):
            return False
        write_conn = getattr(transport, "write_conn", None)
        if write_conn is not None and bool(getattr(write_conn, "closed", False)):
            return False
        if bool(getattr(transport, "_pipe_broken", False)):
            return False
        return True

    @classmethod
    def _transport_status(cls, transport: Any) -> dict[str, Any]:
        get_status = getattr(transport, "get_status", None)
        if callable(get_status):
            status = get_status()
            if isinstance(status, dict):
                return status
        return {
            "alive": cls._transport_alive(transport),
            "running": bool(getattr(transport, "_is_running", False)),
            "pipe_broken": bool(getattr(transport, "_pipe_broken", False)),
            "write_closed": bool(
                getattr(getattr(transport, "write_conn", None), "closed", False)
            ),
            "legacy_transport_status": True,
        }

    def _transport_stop_timeout_s(self) -> float:
        try:
            value = float(os.getenv("AURA_ACTOR_BUS_STOP_TIMEOUT_S", "1.5") or 1.5)
        except (TypeError, ValueError):
            value = 1.5
        return min(10.0, max(0.25, value))

    def add_actor(self, name: str, connection: Any, is_child: bool = False):
        """Register and start a new actor transport."""
        if connection is None:
            logger.warning("Refusing to register actor '%s' without a live transport.", name)
            return False
        if not _is_connection_pair(connection):
            logger.warning(
                "Refusing to register actor '%s' with legacy shared transport; "
                "expected an explicit (read_conn, write_conn) pair.",
                name,
            )
            return False
        transport = LocalPipeBus(is_child=is_child, connection=connection)
        try:
            from core.container import ServiceContainer
            supervisor = ServiceContainer.get("supervisor", default=None)
            if supervisor and hasattr(supervisor, "record_activity"):
                transport.set_activity_callback(
                    lambda actor_name=name, sup=supervisor: sup.record_activity(actor_name)
                )
                supervisor.record_activity(name)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('actor_bus', e)
            logger.debug("ActorBus activity monitor hookup failed for %s: %s", name, e)
        transport.start()
        self._transports[name] = transport
        logger.info("Registered Actor Transport: %s", name)
        return True

    def has_actor(self, name: str) -> bool:
        """Return whether a live transport is registered for the actor."""
        return name in self._transports

    def is_actor_usable(self, name: str) -> bool:
        """Return whether an actor transport is present, running, and writable."""
        if not self._is_running:
            return False
        transport = self._transports.get(name)
        if not transport or not self._transport_alive(transport):
            return False
        return True

    async def update_actor(self, name: str, connection: Any):
        """Hot-swap an actor's transport with a new connection (e.g. after a restart)."""
        old_transport = self._transports.get(name)
        if old_transport:
            logger.info("Hot-swapping transport for %s...", name)
            await old_transport.stop()
        
        # Register new transport
        self.add_actor(name, connection)

    def start(self):
        """Global bus start (transports are started individually on add)."""
        self._is_running = True
        if self._telemetry_queue is None:
            self._telemetry_queue = asyncio.Queue(maxsize=100)
        self.start_transports()
        
        # Start or repair Telemetry Broadcaster
        if self._telemetry_broadcaster_task is not None and not self._telemetry_task_alive():
            record_degradation(
                "actor_bus",
                RuntimeError("ActorBus telemetry broadcaster restart required"),
                severity="warning",
                action="restarting_dead_telemetry_broadcaster",
                extra={"telemetry_task": self._task_status(self._telemetry_broadcaster_task)},
            )
            self._telemetry_broadcaster_task = None
        if self._telemetry_broadcaster_task is None:
            self._telemetry_broadcaster_task = get_task_tracker().create_task(
                self._telemetry_broadcaster(),
                name="actor_bus.telemetry_broadcaster",
            )
            
        logger.info("ActorBus (Unified Layer) ONLINE.")

    async def _telemetry_broadcaster(self):
        """ZENITH: Non-blocking telemetry delivery with backpressure."""
        while self._is_running:
            item_taken = False
            try:
                if self._telemetry_queue is None:
                    await asyncio.sleep(0.05)
                    continue
                item = await self._telemetry_queue.get()
                item_taken = True
                topic, payload = item
                # Broadcast to all transports that handle telemetry
                for _name, transport in self._transports.items():
                    try:
                        await transport.send(topic, payload)
                    except _ACTOR_BUS_SEND_ERRORS as exc:
                        self._record_drop(
                            kind="telemetry",
                            reason="transport_send_failed",
                            actor=_name,
                            topic=topic,
                            error=exc,
                        )
                        continue
            except asyncio.CancelledError:
                break
            except (ValueError, TypeError) as e:
                record_degradation(
                    'actor_bus',
                    e,
                    action="dropped malformed telemetry queue item",
                )
                logger.warning("Malformed telemetry queue item dropped: %s", e)
                await asyncio.sleep(0)
            except (_ACTOR_BUS_SEND_ERRORS + (asyncio.QueueEmpty,)) as e:
                try:
                    from core.runtime import resource_psutil as psutil
                    if psutil.virtual_memory().percent < 90:
                        from core.runtime.self_healing import get_healer
                        logger.warning("Active repair triggered for telemetry broadcast error: %s", e)
                        record_degradation('actor_bus', e, action="scheduled_deep_repair", receipt_required=True)
                        get_healer().schedule_deep_repair(
                            "core/bus/actor_bus.py",
                            reason=f"telemetry_broadcast_exception: {e}",
                            metadata={"error_type": type(e).__name__}
                        )
                    else:
                        record_degradation('actor_bus', e, action="suppressed_repair_due_to_memory_pressure", receipt_required=True)
                except (ImportError, AttributeError, RuntimeError):
                    record_degradation('actor_bus', e)
                    
                logger.error("Telemetry broadcast error: %s", e)
                await asyncio.sleep(0.1)
            finally:
                if item_taken and self._telemetry_queue is not None:
                    try:
                        self._telemetry_queue.task_done()
                    except ValueError as exc:
                        record_degradation(
                            'actor_bus',
                            exc,
                            action="telemetry queue task accounting was already settled",
                        )

    async def broadcast_telemetry(self, topic: str, payload: Any) -> bool:
        """Submit telemetry to the backpressured queue. Drops if full."""
        if not self._is_running:
            self._record_drop(kind="telemetry", reason="bus_not_running", topic=topic)
            return False
        if self._telemetry_queue is None:
            self._record_drop(kind="telemetry", reason="queue_not_initialized", topic=topic)
            return False
            
        try:
            # use put_nowait to ensure we never block the caller
            self._telemetry_queue.put_nowait((topic, payload))
            return True
        except asyncio.QueueFull:
            # Overwrite oldest if full
            self._record_drop(kind="telemetry", reason="queue_full_overwrite", topic=topic)
            try:
                self._telemetry_queue.get_nowait()
                self._telemetry_queue.task_done()
                self._telemetry_queue.put_nowait((topic, payload))
                return True
            except (asyncio.QueueEmpty, RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                self._record_drop(
                    kind="telemetry",
                    reason="queue_overwrite_failed",
                    topic=topic,
                    error=_exc,
                )
                return False

    async def publish(self, topic: str, payload: Any):
        """Alias for broadcast_telemetry to satisfy legacy Orchestrator calls."""
        return await self.broadcast_telemetry(topic, payload)

    def start_transports(self):
        """Ensure all registered transports are started."""
        for _name, transport in self._transports.items():
            transport.start()

    async def stop(self):
        self._is_running = False
        telemetry_task = self._telemetry_broadcaster_task
        self._telemetry_broadcaster_task = None
        if telemetry_task:
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('actor_bus', e)
                logger.debug("ActorBus telemetry shutdown failed: %s", e)
        transports = list(self._transports.items())
        timeout_s = self._transport_stop_timeout_s()

        async def _stop_transport(actor_name: str, transport: LocalPipeBus) -> None:
            try:
                await asyncio.wait_for(transport.stop(), timeout=timeout_s)
            except TimeoutError as exc:
                record_degradation(
                    'actor_bus',
                    exc,
                    action=(
                        "bounded ActorBus shutdown timed out for one actor transport; "
                        "continuing shutdown of remaining transports"
                    ),
                    extra={"actor": actor_name, "timeout_s": timeout_s},
                )
                logger.warning("ActorBus transport stop timed out for %s", actor_name)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('actor_bus', e)
                logger.debug("ActorBus transport shutdown failed for %s: %s", actor_name, e)

        if transports:
            await asyncio.gather(
                *(_stop_transport(name, transport) for name, transport in transports)
            )
        self._transports.clear()
        self._last_health_check.clear()
        cleaned_segments = LocalPipeBus.cleanup_owned_shared_memory()
        if cleaned_segments:
            logger.info(
                "ActorBus shutdown released %d retained outbound shared-memory segment(s).",
                cleaned_segments,
            )
        LocalPipeBus.shutdown_executor()

    @classmethod
    async def reset_singleton(cls):
        """Best-effort singleton reset for tests and controlled warm reboots."""
        inst = cls._instance
        cls._instance = None
        if inst is None:
            return
        try:
            await inst.stop()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('actor_bus', e)
            logger.debug("ActorBus reset encountered a shutdown error: %s", e)
        inst._initialized = False

    async def _health_ping(self, actor: str) -> bool:
        """One-shot TCP/Pipe probe to verify actor responsiveness."""
        transport = self._transports.get(actor)
        if not transport or not self._transport_alive(transport):
            return False
            
        # Congestion Check (High Water Mark)
        pending = len(transport._pending_requests)
        if pending > self._high_water_mark:
            logger.warning("Bus Congested: %s pending requests for %s", pending, actor)
            return False
            
        return True

    async def request(  # noqa: ASYNC109 - timeout is part of the public bus API.
        self,
        actor: str,
        msg_type: str,
        payload: Any,
        timeout: float = 5.0,  # noqa: ASYNC109
    ) -> Any:
        """Send a request with sub-100ms health gating."""
        transport = self._transports.get(actor)
        if not transport:
            # Routing: Forward to kernel if it's a child process
            if "kernel" in self._transports:
                logger.debug("Routing request for '%s' via kernel...", actor)
                return await self.request("kernel", "route_request", {
                    "target": actor,
                    "type": msg_type,
                    "payload": payload
                }, timeout=timeout)
            raise BusDegraded(f"Unknown actor: {actor}")

        try:
            # 1. Health Gate
            if not await self._health_ping(actor):
                raise BusDegraded(f"Bus degraded or congested for {actor}")
            
            # 2. Performance Tracking
            start = time.time()
            result = await asyncio.wait_for(
                transport.request(msg_type, payload, timeout=timeout),
                timeout=timeout
            )
            
            latency = (time.time() - start) * 1000
            if latency > 100:
                logger.debug("Slow bus request to %s: %sms", actor, f"{latency:.1f}")
                
            return result
            
        except (TimeoutError, BusDegraded, BrokenPipeError, ConnectionResetError) as e:
            if _is_shutdown_commit_request(actor, msg_type, payload):
                logger.debug("StateVault bus closed during shutdown commit; caller will handle snapshot fallback.")
            else:
                logger.warning("Bus degraded for %s → %s", actor, e)
            raise

    async def send(self, actor: str, msg_type: str, payload: Any) -> bool:
        """Fire-and-forget send with health gate."""
        transport = self._transports.get(actor)
        if not transport:
            # Routing: Forward to kernel if it's a child process
            if "kernel" in self._transports:
                logger.debug("Routing send for '%s' via kernel...", actor)
                return await self.send("kernel", "route_send", {
                    "target": actor,
                    "type": msg_type,
                    "payload": payload
                })
            logger.error("Unknown actor: %s", actor)
            self._record_drop(kind="send", reason="unknown_actor", actor=actor, topic=msg_type)
            return False

        if not await self._health_ping(actor):
            logger.error("Cannot send to %s: Bus degraded", actor)
            self._record_drop(kind="send", reason="actor_degraded", actor=actor, topic=msg_type)
            return False

        try:
            await transport.send(msg_type, payload)
            return True
        except _ACTOR_BUS_SEND_ERRORS as exc:
            self._record_drop(
                kind="send",
                reason="transport_send_failed",
                actor=actor,
                topic=msg_type,
                error=exc,
            )
            return False

    def register_handler(self, actor: str, msg_type: str, handler: Callable):
        """Register a handler on a specific actor's transport."""
        transport = self._transports.get(actor)
        if transport:
            transport.register_handler(msg_type, handler)

# Factory for creating a generic bus (e.g. for child actors)
def create_actor_bus(is_child: bool = False, connection: Any = None) -> ActorBus:
    bus = ActorBus()
    if connection is not None:
        bus.add_actor("SensoryGate", connection, is_child=is_child)
    return bus
