from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import logging
logger = logging.getLogger(__name__)

import asyncio
from dataclasses import dataclass

import psutil

from core.agency_bus import AgencyBus
from core.container import ServiceContainer
from core.resilience.state_manager import StateManager


def _get_mx():
    try:
        import mlx.core as mx
    except Exception:
        return None
    return mx


@dataclass(frozen=True)
class CoreRuntime:
    _instance = None
    _lock = asyncio.Lock()

    container: ServiceContainer
    agency_bus: AgencyBus
    state_manager: StateManager

    @classmethod
    async def get(cls) -> "CoreRuntime":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    if psutil.virtual_memory().total < 15_000_000_000:
                        raise RuntimeError("M5 Pro 64 GB recommended")

                    from core.service_registration import register_all_services

                    register_all_services()

                    instance = cls.__new__(cls)
                    container = ServiceContainer()
                    agency_bus = AgencyBus.get()
                    state_manager = StateManager()

                    object.__setattr__(instance, "container", container)
                    object.__setattr__(instance, "agency_bus", agency_bus)
                    object.__setattr__(instance, "state_manager", state_manager)

                    mx = _get_mx()
                    if mx:
                        if psutil.sensors_battery() and psutil.sensors_battery().percent < 12:
                            mx.set_default_device(mx.cpu())
                        try:
                            from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel

                            sentinel = get_gpu_sentinel()
                            if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                                try:
                                    mx.clear_cache()
                                finally:
                                    sentinel.release()
                        except Exception as _exc:
                            record_degradation('core_runtime', _exc)
                            logger.debug("Suppressed Exception: %s", _exc)

                    cls._instance = instance

                    from core.eternal_lifecycle import eternal_lifecycle

                    get_task_tracker().create_task(eternal_lifecycle())
        return cls._instance

    @classmethod
    def get_sync(cls) -> "CoreRuntime":
        if cls._instance is None:
            raise RuntimeError("CoreRuntime not initialized. Call 'await CoreRuntime.get()' first.")
        return cls._instance
