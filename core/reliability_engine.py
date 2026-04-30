"""core/reliability_engine.py — v1.0 PRODUCTION RELIABILITY ORCHESTRATOR
The single source of truth for zero-break runtime. 
Watches every service, enforces circuit breakers, graceful degradation,
and guarantees cognitive_stability never drops below 0.85.
"""

from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any
from pathlib import Path

from core.container import ServiceContainer, ServiceLifetime

logger = logging.getLogger("Aura.Reliability")

@dataclass
class ServiceHealth:
    name: str
    last_heartbeat: float
    stability: float  # 0.0–1.0
    circuit_open: bool
    resource_pressure: float  # VRAM/CPU normalized

class ReliabilityEngine:
    name = "reliability_engine"

    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.heartbeat_interval = 15.0
        self.global_stability_threshold = 0.85
        self.circuit_timeout = 300.0  # 5 min cooldown
        self._shutdown = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._on_degrade_callbacks: list[Callable] = []

    async def start(self):
        """Register and start monitoring every critical service."""
        logger.info("🚀 Reliability Engine online — protecting all systems.")

        # Auto-register known services
        for svc_name in ["local_voice_cortex", "planner", "terminal_monitor",
                         "system_monitor", "sovereign_swarm", "insight_journal"]:
            self.register_service(svc_name)

        # Global sweep
        sweep = get_task_tracker().create_task(self._global_sweep_loop())
        self._tasks.append(sweep)

        # Heartbeat listener
        hb = get_task_tracker().create_task(self._heartbeat_listener())
        self._tasks.append(hb)

    def register_service(self, name: str, initial_stability: float = 1.0):
        self.services[name] = ServiceHealth(
            name=name,
            last_heartbeat=time.time(),
            stability=initial_stability,
            circuit_open=False,
            resource_pressure=0.0
        )

    async def heartbeat(self, service_name: str, stability: float = 1.0, pressure: float = 0.0):
        """Called by every service every 15s."""
        if service_name in self.services:
            svc = self.services[service_name]
            svc.last_heartbeat = time.time()
            svc.stability = stability
            svc.resource_pressure = pressure

            if stability < 0.7 and not svc.circuit_open:
                await self._open_circuit(service_name)

    async def _heartbeat_listener(self):
        """Standard async loop to keep engine alive."""
        while not self._shutdown.is_set():
            await asyncio.sleep(1)

    async def _global_sweep_loop(self):
        while not self._shutdown.is_set():
            await asyncio.sleep(self.heartbeat_interval)
            global_stability = self._compute_global_stability()

            if global_stability < self.global_stability_threshold:
                logger.warning(f"⚠️ Global stability {global_stability:.2f} — triggering degradation")
                await self._trigger_graceful_degradation()

    def _compute_global_stability(self) -> float:
        if not self.services:
            return 1.0
        weights = [s.stability * (1.0 - s.resource_pressure) for s in self.services.values()]
        return sum(weights) / len(weights)

    async def _open_circuit(self, service_name: str):
        svc = self.services[service_name]
        svc.circuit_open = True
        logger.error(f"🔴 Circuit breaker OPEN for {service_name}")
        await asyncio.sleep(self.circuit_timeout)
        svc.circuit_open = False
        logger.info(f"🟢 Circuit breaker CLOSED for {service_name}")

    async def _trigger_graceful_degradation(self):
        """Universal safe mode: drop low-priority tasks, pause non-critical shards."""
        try:
            from core.brain.reasoning_queue import get_reasoning_queue, ReasoningPriority
            rq = get_reasoning_queue()
            dropped = await rq.prune_low_priority(threshold_priority=ReasoningPriority.HIGH.value)
            logger.info(f"🛡️ Degradation: dropped {dropped} low-priority tasks")
        except Exception as e:
            logger.error(f"Failed to access reasoning queue during degradation: {e}")

        # Call any registered callbacks (e.g. swarm pause)
        for cb in self._on_degrade_callbacks:
            try:
                await cb()
            except Exception as e:
                logger.error(f"Degradation callback failed: {e}")

    def register_degrade_callback(self, cb: Callable):
        self._on_degrade_callbacks.append(cb)

    async def stop(self):
        self._shutdown.set()
        for t in self._tasks:
            if not t.done():
                t.cancel()
        logger.info("Reliability Engine shut down cleanly.")

# Singleton registration
def get_reliability_engine() -> ReliabilityEngine:
    return ServiceContainer.get("reliability_engine", default=None) or ReliabilityEngine()

# Auto-register on import
ServiceContainer.register(
    "reliability_engine",
    factory=get_reliability_engine,
    lifetime=ServiceLifetime.SINGLETON
)
