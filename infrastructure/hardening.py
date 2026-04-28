"""infrastructure/hardening.py
Makes Aura resilient, non-brittle, and production-ready.

Includes:
- Circuit breakers for failing components
- Automatic retry with exponential backoff
- Graceful degradation
- Health monitoring
- State management and recovery
- Resource management
"""
import asyncio
import functools
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

logger = logging.getLogger("Infrastructure.Hardening")


class ComponentState(Enum):
    """Health state of a component"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class HealthCheck:
    """Health check result"""

    component: str
    state: ComponentState
    timestamp: float
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        # Satisfy type checkers that this is a dataclass
        if is_dataclass(self):
            d = asdict(self) # type: ignore
            d['state'] = self.state.value
            return d
        return {"component": self.component, "state": self.state.value, "timestamp": self.timestamp}


class CircuitBreaker:
    """Circuit breaker pattern for failing components.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Component failing, reject all requests
    - HALF_OPEN: Testing if component recovered
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0.0
        
        logger.info("CircuitBreaker '%s' initialized", name)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker (Async)"""
        if self.state == "OPEN":
            # Check if we should try recovery
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit '%s' entering HALF_OPEN state", self.name)
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise Exception(f"Circuit breaker OPEN for {self.name}. Subsystem is cooling down.")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # If it's a regular function, run it in a thread to keep async loop free
                result = await asyncio.to_thread(func, *args, **kwargs)
                
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(str(e))
            raise
    
    def allow_request(self) -> bool:
        """Compatibility alias for can_execute"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                return True
            return False
        return True

    def can_execute(self) -> bool:
        """Compatibility alias for allow_request"""
        return self.allow_request()

    def record_success(self):
        """Public method to record success"""
        self._on_success()

    def record_failure(self, error: Optional[str] = None):
        """Public method to record failure"""
        if error:
            logger.debug("Circuit '%s' recorded failure: %s", self.name, error)
        self._on_failure()

    def _on_success(self):
        """Handle successful call"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info("Circuit '%s' recovered - CLOSED", self.name)
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            logger.error("Circuit '%s' OPEN after %d failures", self.name, self.failure_count)
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            logger.warning("Circuit '%s' failed during recovery - back to OPEN", self.name)
            self.state = "OPEN"


class RetryPolicy:
    """Automatic retry with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry (Async)"""
        last_exception: Optional[BaseException] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, self.max_retries + 1, e, delay
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("All %d attempts failed", self.max_retries + 1)
        
        if last_exception is not None:
            raise cast(BaseException, last_exception)
        raise Exception("Unknown error in RetryPolicy")


class StateManager:
    """Manages system state with checkpointing and recovery.
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            # Portability fix — use relative or config-based paths
            try:
                # Add project root to sys path for robust importing
                root = Path(__file__).resolve().parent.parent
                if str(root) not in sys.path:
                    sys.path.append(str(root))
                from core.config import config
                self.checkpoint_dir = Path(config.paths.data_dir) / "checkpoints"
            except (ImportError, AttributeError):
                self.checkpoint_dir = Path("data/checkpoints")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state: Dict[str, Any] = {}
        self.checkpoint_interval = 300  # 5 minutes
        self.last_checkpoint: float = 0.0
        
        # Auto-checkpoint in background
        self.checkpoint_thread: Optional[threading.Thread] = None
        self.running = False
        
        logger.info("StateManager initialized")
    
    def set(self, key: str, value: Any):
        """Set state value"""
        self.state[key] = value
        
        # Auto-checkpoint if interval passed
        if time.time() - self.last_checkpoint > self.checkpoint_interval:
            self.checkpoint()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        return self.state.get(key, default)
    
    def checkpoint(self) -> bool:
        """Save current state to disk"""
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{int(time.time())}.json"
            
            # Create backup of current state
            with open(checkpoint_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
            
            # Keep only last 10 checkpoints
            checkpoints = sorted(list(self.checkpoint_dir.glob("checkpoint_*.json")))
            if len(checkpoints) > 10:
                for old_checkpoint in checkpoints[:-10]:
                    old_checkpoint.unlink()
            
            self.last_checkpoint = time.time()
            logger.info("State checkpoint saved: %s", checkpoint_file.name)
            return True
            
        except Exception as e:
            logger.error("Checkpoint failed: %s", e)
            return False
    
    def restore(self) -> bool:
        """Restore from latest checkpoint"""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            
            if not checkpoints:
                logger.warning("No checkpoints found")
                return False
            
            latest = checkpoints[-1]
            
            with open(latest, 'r') as f:
                loaded_state = json.load(f)
                
            # Basic schema/type validation to prevent unsafe deserialization
            if not isinstance(loaded_state, dict):
                logger.error("Restore failed: Checkpoint data is not a valid JSON object.")
                return False
                
            self.state = loaded_state
            
            logger.info("State restored from %s", latest.name)
            return True
            
        except Exception as e:
            logger.error("Restore failed: %s", e)
            return False

    def create_snapshot(self, orchestrator: Any) -> Dict[str, Any]:
        """Watchdog compatibility: capture system snapshot"""
        task_queue = getattr(orchestrator, 'task_queue', None)
        container = getattr(orchestrator, 'container', None)
        
        snapshot = {
            "timestamp": time.time(),
            "cycle_count": getattr(orchestrator, 'cycle_count', 0),
            "state": str(getattr(orchestrator, 'state', 'UNKNOWN')),
            "active_tasks": task_queue.qsize() if task_queue else 0,
            "skills_loaded": len(getattr(orchestrator, 'skills', {})),
            "memory_usage": container.get('status', {}).get('memory', 0) if container and hasattr(container, 'get') else 0
        }
        return snapshot

    def push_checkpoint(self, snapshot: Dict[str, Any]):
        """Watchdog compatibility: push snapshot as state"""
        self.set("last_system_snapshot", snapshot)
        self.checkpoint()
    
    def start_auto_checkpoint(self):
        """Start background checkpointing"""
        if self.running:
            return
        
        self.running = True
        thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True,
            name="StateCheckpoint"
        )
        self.checkpoint_thread = thread
        thread.start()
        logger.info("Auto-checkpoint enabled")
    
    def stop_auto_checkpoint(self):
        """Stop background checkpointing"""
        self.running = False
        if isinstance(self.checkpoint_thread, threading.Thread) and self.checkpoint_thread.is_alive():
            self.checkpoint_thread.join(timeout=5)
    
    async def _checkpoint_loop(self):
        """Background checkpoint loop (Async)."""
        while self.running:
            await asyncio.sleep(self.checkpoint_interval)
            if self.running:
                # Wrap sync checkpoint in to_thread to prevent event loop blocking
                await asyncio.to_thread(self.checkpoint)


class HealthMonitor:
    """Monitors health of all system components.
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentState] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.monitors: Dict[str, Callable] = {}
        
        # Background monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.check_interval = 30  # 30 seconds
        
        logger.info("HealthMonitor initialized")
    
    def record_execution(self, component: str, success: bool, latency_ms: float = 0.0, error: Optional[str] = None):
        """Record a manual execution result for health tracking."""
        state = ComponentState.HEALTHY if success else ComponentState.DEGRADED
        if not success and error:
            logger.warning("Health Monitor: %s execution failed: %s", component, error)
            
        result = HealthCheck(
            component=component,
            state=state,
            timestamp=time.time(),
            latency_ms=latency_ms,
            error=error
        )
        self.components[component] = state
        self.health_history.append(result)
    
    def register_component(self, name: str, health_check: Callable):
        """Register a component for health monitoring.
        """
        self.monitors[name] = health_check
        self.components[name] = ComponentState.HEALTHY
        logger.info("Registered component: %s", name)
    
    def check_health(self, component: str) -> HealthCheck:
        """Check health of a specific component"""
        if component not in self.monitors:
            return HealthCheck(
                component=component,
                state=ComponentState.FAILED,
                timestamp=time.time(),
                error="Component not registered"
            )
        
        start = time.time()
        
        try:
            health_func = self.monitors[component]
            is_healthy = health_func()
            
            latency = (time.time() - start) * 1000
            
            if is_healthy:
                state = ComponentState.HEALTHY
            else:
                state = ComponentState.DEGRADED
            
            result = HealthCheck(
                component=component,
                state=state,
                timestamp=time.time(),
                latency_ms=latency
            )
            
        except Exception as e:
            result = HealthCheck(
                component=component,
                state=ComponentState.FAILED,
                timestamp=time.time(),
                error=str(e)
            )
        
        # Update component state
        self.components[component] = result.state
        self.health_history.append(result)
        
        return result
    
    def check_all(self) -> Dict[str, HealthCheck]:
        """Check health of all components"""
        results = {}
        
        for component in self.monitors:
            results[component] = self.check_health(component)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        all_checks = self.check_all()
        
        healthy = sum(1 for c in all_checks.values() if c.state == ComponentState.HEALTHY)
        degraded = sum(1 for c in all_checks.values() if c.state == ComponentState.DEGRADED)
        failed = sum(1 for c in all_checks.values() if c.state == ComponentState.FAILED)
        
        total = len(all_checks)
        
        # Overall state
        if failed > 0:
            overall = ComponentState.FAILING
        elif total > 0 and degraded > total / 2:
            overall = ComponentState.DEGRADED
        else:
            overall = ComponentState.HEALTHY
        
        return {
            "overall_state": overall.value,
            "healthy": healthy,
            "degraded": degraded,
            "failed": failed,
            "total": total,
            "components": {k: v.to_dict() for k, v in all_checks.items()}
        }
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self.running:
            return
        
        self.running = True
        thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self.monitor_thread = thread
        thread.start()
        logger.info("Background health monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        if isinstance(self.monitor_thread, threading.Thread) and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    async def _monitor_loop(self):
        """Background monitoring loop (Async)."""
        import concurrent.futures
        while self.running:
            # Execute health checks as tasks
            try:
                # check_all is synchronous, so we run it in a thread
                await asyncio.to_thread(self.check_all)
            except Exception as e:
                logger.error("Health Monitor loop error: %s", e)
                
            await asyncio.sleep(self.check_interval)


class ResourceManager:
    """Manages system resources to prevent leaks and exhaustion.
    """
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.limits: Dict[str, int] = {
            "max_memory_mb": 1024,
            "max_connections": 100,
            "max_threads": 50
        }
        
        logger.info("ResourceManager initialized")
    
    def acquire(self, resource_type: str, key: str, resource: Any):
        """Acquire a resource"""
        if resource_type not in self.resources:
            self.resources[resource_type] = {}
        
        # Check limits
        current_count = len(self.resources[resource_type])
        limit_key = f"max_{resource_type}"
        
        if limit_key in self.limits and current_count >= self.limits[limit_key]:
            raise Exception(f"Resource limit exceeded: {resource_type} (limit: {self.limits[limit_key]})")
        
        self.resources[resource_type][key] = {
            "resource": resource,
            "acquired_at": time.time()
        }
        
        logger.debug("Acquired %s: %s", resource_type, key)
    
    def release(self, resource_type: str, key: str):
        """Release a resource"""
        if resource_type in self.resources and key in self.resources[resource_type]:
            resource_info = self.resources[resource_type][key]
            resource = resource_info["resource"]
            
            # Call cleanup if available
            if hasattr(resource, 'close'):
                try:
                    resource.close()
                except Exception:
                    pass
            
            del self.resources[resource_type][key]
            logger.debug("Released %s: %s", resource_type, key)
    
    def release_all(self, resource_type: str):
        """Release all resources of a type"""
        if resource_type not in self.resources:
            return
        
        keys = list(self.resources[resource_type].keys())
        for key in keys:
            self.release(resource_type, key)
        
        logger.info("Released all %s", resource_type)
    
    def cleanup_stale(self, resource_type: str, max_age: float = 3600):
        """Clean up resources older than max_age seconds"""
        if resource_type not in self.resources:
            return
        
        current_time = time.time()
        stale_keys = []
        
        for key, info in self.resources[resource_type].items():
            if current_time - info["acquired_at"] > max_age:
                stale_keys.append(key)
        
        for key in stale_keys:
            self.release(resource_type, key)
        
        if stale_keys:
            logger.info("Cleaned up %d stale %s", len(stale_keys), resource_type)


class InfrastructureHardeningSystem:
    """Complete infrastructure hardening system.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policy = RetryPolicy()
        self.state_manager = StateManager()
        self.health_monitor = HealthMonitor()
        self.resource_manager = ResourceManager()
        
        logger.info("InfrastructureHardeningSystem initialized")
    
    async def resilient_call(
        self,
        component_name: str,
        func: Callable,
        *args,
        use_retry: bool = True,
        use_circuit_breaker: bool = True,
        **kwargs
    ):
        """Execute a function with full resilience features (Async).
        """
        # Get or create circuit breaker
        if use_circuit_breaker:
            if component_name not in self.circuit_breakers:
                self.circuit_breakers[component_name] = CircuitBreaker(component_name)
            
            cb = self.circuit_breakers[component_name]
        
        # Execute with resilience
        if use_circuit_breaker and use_retry:
            return await cb.call(self.retry_policy.execute, func, *args, **kwargs)
        elif use_circuit_breaker:
            return await cb.call(func, *args, **kwargs)
        elif use_retry:
            return await self.retry_policy.execute(func, *args, **kwargs)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)
    
    def start_background_services(self):
        """Start all background services"""
        self.state_manager.start_auto_checkpoint()
        self.health_monitor.start_monitoring()
        logger.info("Background services started")
    
    def stop_background_services(self):
        """Stop all background services"""
        self.state_manager.stop_auto_checkpoint()
        self.health_monitor.stop_monitoring()
        logger.info("Background services stopped")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        return self.health_monitor.get_status()


# Global instance for decorators
_hardening_system = None

def set_global_hardening_system(system: InfrastructureHardeningSystem):
    """Set global hardening system for decorators"""
    global _hardening_system
    _hardening_system = system

def resilient(component_name: str, retry: bool = True, circuit_breaker: bool = True):
    """Decorator to make any function resilient. Supports both sync and async.
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                global _hardening_system
                if _hardening_system is None:
                    return await func(*args, **kwargs)
                
                return await _hardening_system.resilient_call(
                    component_name,
                    func,
                    *args,
                    use_retry=retry,
                    use_circuit_breaker=circuit_breaker,
                    **kwargs
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                global _hardening_system
                if _hardening_system is None:
                    return func(*args, **kwargs)
                
                # For sync functions, delegate to a thread if in a running loop
                # Alternatively, create a short-lived loop safely.
                try:
                    loop = asyncio.get_running_loop()
                    # If we're already in a loop, we must delegate the resilient call to a separate thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            lambda: asyncio.run(_hardening_system.resilient_call(
                                component_name, func, *args, use_retry=retry, use_circuit_breaker=circuit_breaker, **kwargs
                            ))
                        )
                        return future.result()
                except RuntimeError:
                    # No running loop — safe to create one for this sync wrapper
                    pass
                
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(_hardening_system.resilient_call(
                            component_name, func, *args, use_retry=retry, use_circuit_breaker=circuit_breaker, **kwargs
                        ))
                    finally:
                        loop.close()
                except Exception as e:
                    logger.debug("Resilience wrapper fallback for %s: %s", func.__name__, e)
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator
