from core.utils.task_tracker import get_task_tracker
import asyncio
import psutil
try:
    import mlx.core as mx
except ImportError:
    mx = None
from dataclasses import dataclass
from core.agency_bus import AgencyBus
from core.resilience.state_manager import StateManager
from core.container import ServiceContainer
from core.eternal_lifecycle import eternal_lifecycle

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
                    # Enforce M5 Pro budget before instantiation
                    if psutil.virtual_memory().total < 15_000_000_000:
                        raise RuntimeError("M5 Pro 64 GB recommended")
                    
                    # 1. Register everything once
                    from core.service_registration import register_all_services
                    register_all_services()
                    
                    instance = cls.__new__(cls)
                    # Initialize components from container
                    container = ServiceContainer()
                    agency_bus = AgencyBus.get()
                    state_manager = StateManager()
                    
                    # Store in immutable slots
                    object.__setattr__(instance, 'container', container)
                    object.__setattr__(instance, 'agency_bus', agency_bus)
                    object.__setattr__(instance, 'state_manager', state_manager)
                    
                    # Hardware optimization
                    if mx:
                        if psutil.sensors_battery() and psutil.sensors_battery().percent < 12:
                            mx.set_default_device(mx.cpu())
                        try:
                            from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
                            sentinel = get_gpu_sentinel()
                            if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                                try: mx.clear_cache()
                                finally: sentinel.release()
                        except Exception: pass
                    
                    cls._instance = instance
                    # Start the eternal loop
                    get_task_tracker().create_task(eternal_lifecycle())
        return cls._instance

    @classmethod
    def get_sync(cls) -> "CoreRuntime":
        """Synchronous access for non-async contexts (use with caution)."""
        if cls._instance is None:
            raise RuntimeError("CoreRuntime not initialized. Call 'await CoreRuntime.get()' first.")
        return cls._instance
