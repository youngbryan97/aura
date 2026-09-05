"""core/providers/engine_connection_pool.py — CognitiveEngine Connection Pooling & Persistence

Provides persistent connection pooling for CognitiveEngine with automatic retry,
health monitoring, and graceful fallback mechanisms.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.EngineConnectionPool")

_CONNECTION_POOL_RECOVERABLE_ERRORS = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    LookupError,
    asyncio.TimeoutError,
)


class ConnectionHealth(Enum):
    """Health status of a connection."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class ConnectionStats:
    """Track connection statistics for monitoring."""
    created_at: float = field(default_factory=time.time)
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    current_health: ConnectionHealth = ConnectionHealth.HEALTHY
    consecutive_failures: int = 0
    
    def record_success(self):
        self.last_success_at = time.time()
        self.success_count += 1
        self.consecutive_failures = 0
        self.current_health = ConnectionHealth.HEALTHY
        
    def record_failure(self, is_timeout: bool = False):
        self.last_failure_at = time.time()
        self.failure_count += 1
        if is_timeout:
            self.timeout_count += 1
        self.consecutive_failures += 1
        
        # Mark degraded or unhealthy based on failure count
        if self.consecutive_failures >= 3:
            self.current_health = ConnectionHealth.UNHEALTHY
        elif self.consecutive_failures >= 1:
            self.current_health = ConnectionHealth.DEGRADED
    
    def mark_recovering(self):
        self.current_health = ConnectionHealth.RECOVERING
    
    def uptime_seconds(self) -> float:
        """Time since connection was created."""
        return time.time() - self.created_at
    
    def time_since_success(self) -> float:
        """Seconds since last successful operation."""
        if self.last_success_at == 0:
            return float('inf')
        return time.time() - self.last_success_at


@dataclass
class ConnectionRetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 10.0
    backoff_multiplier: float = 2.0
    timeout_multiplier: float = 1.5
    
    def get_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for an attempt."""
        delay = min(
            self.max_backoff_seconds,
            self.initial_backoff_seconds * (self.backoff_multiplier ** attempt)
        )
        return delay
    
    def get_timeout_for_attempt(self, base_timeout: float, attempt: int) -> float:
        """Get timeout with multiplier for retry attempts."""
        return base_timeout * (1.0 + (self.timeout_multiplier * (attempt + 1)))

    @staticmethod
    def coerce_total_timeout(timeout: float) -> float:
        try:
            return max(0.1, float(timeout))
        except (TypeError, ValueError):
            return 120.0


class CognitiveEngineConnectionPool:
    """
    Connection pool for CognitiveEngine with persistence and resilience.
    
    Key features:
    - Maintains permanent connections to reduce handshake overhead
    - Implements exponential backoff retry strategy
    - Tracks connection health and auto-recovers
    - Provides graceful fallback when engine is unavailable
    - Monitors desktop chat path connectivity
    """
    
    def __init__(self, max_connections: int = 1):
        self.max_connections = max_connections
        self.connections: dict[str, Any] = {}
        self.stats: dict[str, ConnectionStats] = {}
        self.retry_config = ConnectionRetryConfig()
        self._lock = asyncio.Lock()
        self._recovery_tasks: dict[str, asyncio.Task] = {}
        self._health_check_interval = 30.0  # seconds
        
    async def acquire_engine_connection(
        self,
        engine: Any,
        connection_id: str = "default",
        force_refresh: bool = False,
    ) -> Any | None:
        """
        Acquire a connection to the CognitiveEngine with persistence.
        
        Args:
            engine: The CognitiveEngine instance
            connection_id: Unique identifier for this connection
            force_refresh: Force creation of new connection
            
        Returns:
            The engine connection or None if unavailable
        """
        async with self._lock:
            # Return existing healthy connection if available
            if not force_refresh and connection_id in self.connections:
                conn = self.connections[connection_id]
                stats = self.stats.get(connection_id)
                if stats and stats.current_health == ConnectionHealth.HEALTHY:
                    logger.debug(
                        "🔌 Reusing existing CognitiveEngine connection (id=%s, uptime=%.1fs)",
                        connection_id,
                        stats.uptime_seconds(),
                    )
                    return conn
            
            # Create new connection if needed
            if connection_id not in self.connections or force_refresh:
                logger.info(
                    "🔗 Establishing persistent CognitiveEngine connection (id=%s, force_refresh=%s)",
                    connection_id,
                    force_refresh,
                )
                self.connections[connection_id] = engine
                if connection_id not in self.stats:
                    self.stats[connection_id] = ConnectionStats()
                else:
                    self.stats[connection_id] = ConnectionStats()  # Reset stats
                    
            return self.connections[connection_id]
    
    async def execute_with_retry(
        self,
        operation_name: str,
        coro_factory: callable,
        connection_id: str = "default",
        timeout: float = 120.0,
    ) -> Any:
        """
        Execute an operation on the CognitiveEngine with automatic retry and backoff.
        
        Args:
            operation_name: Name of the operation for logging
            coro_factory: Callable that returns the coroutine to execute
            connection_id: Connection identifier
            timeout: Hard wall-clock budget in seconds for all attempts, including
                retry backoff. The pool must never expand a foreground caller's
                timeout because the desktop UI uses this value as its fail-closed
                SLA.
            
        Returns:
            Result of the operation or None if all retries fail
        """
        stats = self.stats.get(connection_id)
        if stats is None:
            stats = ConnectionStats()
            self.stats[connection_id] = stats
        
        last_exception = None
        total_timeout = self.retry_config.coerce_total_timeout(timeout)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + total_timeout
        
        for attempt in range(self.retry_config.max_retries):
            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{operation_name} exceeded total budget {total_timeout:.1f}s"
                    )
                # Calculate timeout for this attempt without exceeding the caller's
                # total wall-clock budget.
                attempt_timeout = min(
                    self.retry_config.get_timeout_for_attempt(total_timeout, attempt),
                    remaining,
                )
                
                logger.debug(
                    "🔄 %s attempt %d/%d (timeout=%.1fs)",
                    operation_name,
                    attempt + 1,
                    self.retry_config.max_retries,
                    attempt_timeout,
                )
                
                # Execute the operation
                coro = coro_factory()
                result = await asyncio.wait_for(coro, timeout=attempt_timeout)
                
                # Success
                stats.record_success()
                logger.debug(
                    "✅ %s succeeded on attempt %d",
                    operation_name,
                    attempt + 1,
                )
                return result
                
            except TimeoutError as e:
                stats.record_failure(is_timeout=True)
                last_exception = e
                logger.warning(
                    "⏱️  %s timed out on attempt %d (%.1fs)",
                    operation_name,
                    attempt + 1,
                    attempt_timeout,
                )
                
                # Trigger connection recovery on timeout
                await self._trigger_recovery(connection_id)
                
            except _CONNECTION_POOL_RECOVERABLE_ERRORS as e:
                stats.record_failure(is_timeout=False)
                last_exception = e
                logger.warning(
                    "❌ %s failed on attempt %d: %s",
                    operation_name,
                    attempt + 1,
                    str(e),
                )
            
            # Apply exponential backoff before retry (except on last attempt)
            if attempt < self.retry_config.max_retries - 1:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                backoff_delay = min(
                    self.retry_config.get_backoff_delay(attempt),
                    max(0.0, remaining),
                )
                if backoff_delay <= 0:
                    break
                logger.info(
                    "⏳ Waiting %.1fs before retry (attempt %d/%d)",
                    backoff_delay,
                    attempt + 1,
                    self.retry_config.max_retries,
                )
                await asyncio.sleep(backoff_delay)
        
        # All retries exhausted
        logger.error(
            "🔴 %s failed after %d attempts. Last error: %s",
            operation_name,
            self.retry_config.max_retries,
            str(last_exception),
        )
        stats.mark_recovering()
        return None
    
    async def _trigger_recovery(self, connection_id: str):
        """Trigger automatic recovery for a failed connection."""
        if connection_id in self._recovery_tasks:
            task = self._recovery_tasks[connection_id]
            if not task.done():
                return  # Recovery already in progress
        
        logger.info("🔧 Triggering connection recovery (id=%s)", connection_id)
        
        async def recovery_task():
            try:
                stats = self.stats.get(connection_id)
                if stats:
                    stats.mark_recovering()
                
                # Wait before attempting recovery
                await asyncio.sleep(5.0)
                
                # Force refresh the connection
                engine = self.connections.get(connection_id)
                if engine:
                    await self.acquire_engine_connection(
                        engine,
                        connection_id=connection_id,
                        force_refresh=True,
                    )
                    logger.info("✅ Connection recovery completed (id=%s)", connection_id)
                    if stats:
                        stats.record_success()
            except _CONNECTION_POOL_RECOVERABLE_ERRORS as e:
                record_degradation("engine_connection_pool.recovery", e)
                logger.error("🛑 Connection recovery failed: %s", str(e))
        
        task = get_task_tracker().create_task(
            recovery_task(),
            name=f"engine_connection_recovery_{connection_id}",
        )
        self._recovery_tasks[connection_id] = task
    
    def get_health_status(self, connection_id: str = "default") -> dict[str, Any]:
        """Get detailed health status of a connection."""
        stats = self.stats.get(connection_id)
        if not stats:
            return {"status": "unknown", "message": "Connection not found"}
        
        return {
            "connection_id": connection_id,
            "health": stats.current_health.value,
            "uptime_seconds": stats.uptime_seconds(),
            "success_count": stats.success_count,
            "failure_count": stats.failure_count,
            "timeout_count": stats.timeout_count,
            "consecutive_failures": stats.consecutive_failures,
            "time_since_success": stats.time_since_success(),
            "last_success_at": stats.last_success_at,
            "last_failure_at": stats.last_failure_at,
        }
    
    async def close_connection(self, connection_id: str = "default"):
        """Close a connection and cancel any recovery tasks."""
        async with self._lock:
            if connection_id in self.connections:
                del self.connections[connection_id]
            
            if connection_id in self._recovery_tasks:
                task = self._recovery_tasks[connection_id]
                task.cancel()
                del self._recovery_tasks[connection_id]
            
            logger.info("🔌 Closed CognitiveEngine connection (id=%s)", connection_id)


# Global instance
_global_pool: CognitiveEngineConnectionPool | None = None


def get_engine_connection_pool() -> CognitiveEngineConnectionPool:
    """Get or create the global engine connection pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = CognitiveEngineConnectionPool()
    return _global_pool
