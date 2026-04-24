import asyncio
import logging
import time
from typing import Any, Dict, Optional, Callable
from .local_pipe_bus import LocalPipeBus

logger = logging.getLogger("Kernel.ActorBus")

class BusDegraded(Exception):
    """Raised when the bus health probe fails or congestion is too high."""
    pass

class ActorBus:
    """Unified Actor Bus abstraction with health gating and congestion control.
    Manages multiple LocalPipeBus transports indexed by actor name.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ActorBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._transports: Dict[str, LocalPipeBus] = {}
        self._last_health_check: Dict[str, float] = {}
        self._health_timeout = 0.1  # 100ms spec
        self._high_water_mark = 50  # Max pending requests before degradation
        self._is_running = False
        
        # ZENITH: Backpressured Telemetry Queue
        self._telemetry_queue: Optional[asyncio.Queue] = None
        self._telemetry_broadcaster_task = None
        self._initialized = True

    def add_actor(self, name: str, connection: Any, is_child: bool = False):
        """Register and start a new actor transport."""
        if connection is None:
            logger.warning("📡 Refusing to register actor '%s' without a live transport.", name)
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
        except Exception as e:
            logger.debug("ActorBus activity monitor hookup failed for %s: %s", name, e)
        transport.start()
        self._transports[name] = transport
        logger.info(f"📡 Registered Actor Transport: {name}")
        return True

    def has_actor(self, name: str) -> bool:
        """Return whether a live transport is registered for the actor."""
        return name in self._transports
    async def update_actor(self, name: str, connection: Any):
        """Hot-swap an actor's transport with a new connection (e.g. after a restart)."""
        old_transport = self._transports.get(name)
        if old_transport:
            logger.info(f"🔄 Hot-swapping transport for {name}...")
            await old_transport.stop()
        
        # Register new transport
        self.add_actor(name, connection)

    def start(self):
        """Global bus start (transports are started individually on add)."""
        self._is_running = True
        if self._telemetry_queue is None:
            self._telemetry_queue = asyncio.Queue(maxsize=100)
        self.start_transports()
        
        # Start Telemetry Broadcaster
        if self._telemetry_broadcaster_task is None:
            self._telemetry_broadcaster_task = asyncio.create_task(self._telemetry_broadcaster())
            
        logger.info("📡 ActorBus (Unified Layer) ONLINE.")

    async def _telemetry_broadcaster(self):
        """ZENITH: Non-blocking telemetry delivery with backpressure."""
        while self._is_running:
            try:
                if self._telemetry_queue is None:
                    await asyncio.sleep(0.05)
                    continue
                topic, payload = await self._telemetry_queue.get()
                # Broadcast to all transports that handle telemetry
                for name, transport in self._transports.items():
                    try:
                        await transport.send(topic, payload)
                    except Exception:
                        continue
                self._telemetry_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry broadcast error: {e}")
                await asyncio.sleep(0.1)

    async def broadcast_telemetry(self, topic: str, payload: Any):
        """Submit telemetry to the backpressured queue. Drops if full."""
        if not self._is_running:
            return
        if self._telemetry_queue is None:
            return
            
        try:
            # use put_nowait to ensure we never block the caller
            self._telemetry_queue.put_nowait((topic, payload))
        except asyncio.QueueFull:
            # Overwrite oldest if full
            try:
                self._telemetry_queue.get_nowait()
                self._telemetry_queue.put_nowait((topic, payload))
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

    async def publish(self, topic: str, payload: Any):
        """Alias for broadcast_telemetry to satisfy legacy Orchestrator calls."""
        await self.broadcast_telemetry(topic, payload)

    def start_transports(self):
        """Ensure all registered transports are started."""
        for name, transport in self._transports.items():
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
            except Exception as e:
                logger.debug("ActorBus telemetry shutdown failed: %s", e)
        for name, transport in self._transports.items():
            await transport.stop()
        self._transports.clear()
        self._last_health_check.clear()
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
        except Exception as e:
            logger.debug("ActorBus reset encountered a shutdown error: %s", e)
        inst._initialized = False

    async def _health_ping(self, actor: str) -> bool:
        """One-shot TCP/Pipe probe to verify actor responsiveness."""
        transport = self._transports.get(actor)
        if not transport or not transport._is_running:
            return False
            
        # Congestion Check (High Water Mark)
        pending = len(transport._pending_requests)
        if pending > self._high_water_mark:
            logger.warning(f"⚠️ Bus Congested: {pending} pending requests for {actor}")
            return False
            
        return True

    async def request(self, actor: str, msg_type: str, payload: Any, timeout: float = 5.0) -> Any:
        """Send a request with sub-100ms health gating."""
        transport = self._transports.get(actor)
        if not transport:
            # Routing: Forward to kernel if it's a child process
            if "kernel" in self._transports:
                logger.debug(f"🔀 Routing request for '{actor}' via kernel...")
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
                logger.debug(f"🐢 Slow bus request to {actor}: {latency:.1f}ms")
                
            return result
            
        except (asyncio.TimeoutError, BusDegraded, BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"📡 Bus degraded for {actor} → {e}")
            raise

    async def send(self, actor: str, msg_type: str, payload: Any):
        """Fire-and-forget send with health gate."""
        transport = self._transports.get(actor)
        if not transport:
            # Routing: Forward to kernel if it's a child process
            if "kernel" in self._transports:
                logger.debug(f"🔀 Routing send for '{actor}' via kernel...")
                await self.send("kernel", "route_send", {
                    "target": actor,
                    "type": msg_type,
                    "payload": payload
                })
                return
            logger.error(f"❌ Unknown actor: {actor}")
            return

        if not await self._health_ping(actor):
            logger.error(f"❌ Cannot send to {actor}: Bus degraded")
            return

        await transport.send(msg_type, payload)

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
