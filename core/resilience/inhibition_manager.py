import asyncio
import logging
import threading
import time
import uuid

logger = logging.getLogger("Aura.Resilience.Inhibition")


class InhibitionManager:
    """
    Global Inhibition Manager (Nervous System Protection).
    
    Provides a centralized mechanism to inhibit specific subsystems or behaviors
    to prevent recursive loops, attention seizures, or resource exhaustion.
    
    This is a 'Biological' primitive that mimics neural inhibition.
    """
    
    def __init__(self) -> None:
        self.instance_id = f"inhibition-{uuid.uuid4()}"
        self._inhibited_sources: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        logger.info("🚫 InhibitionManager initialized. (Global Cross-Process Protection).")

    async def inhibit(
        self,
        source: str,
        duration: float = 5.0,
        reason: str | None = None,
    ) -> None:
        """
        Place a source into the inhibited state.
        
        Args:
            source: The name of the subsystem or process to inhibit (e.g. 'thought_loop', 'voice_cortex')
            duration: How long (seconds) to inhibit.
            reason: Optional reason for logging.
        """
        async with self._lock:
            expiry = time.time() + duration
            with self._thread_lock:
                current_expiry = self._inhibited_sources.get(source, 0)
                self._inhibited_sources[source] = max(current_expiry, expiry)
            
            logger.warning(
                "🛑 [INHIBITION] Source '%s' inhibited for %.1fs. Reason: %s",
                source, duration, reason or "unspecified"
            )

    async def is_inhibited(self, source: str) -> bool:
        """Check if a source is currently inhibited."""
        async with self._lock:
            with self._thread_lock:
                expiry = self._inhibited_sources.get(source, 0)
                if expiry > time.time():
                    return True
                if source in self._inhibited_sources:
                    del self._inhibited_sources[source]
                return False

    async def get_inhibited_sources(self) -> set[str]:
        """Return a set of all currently inhibited source names."""
        now = time.time()
        async with self._lock:
            with self._thread_lock:
                active = {
                    source
                    for source, expiry in self._inhibited_sources.items()
                    if expiry > now
                }
                expired = set(self._inhibited_sources) - active
                for source in expired:
                    del self._inhibited_sources[source]
                return active

    async def release(self, source: str) -> None:
        """Manually lift an inhibition before its expiry."""
        async with self._lock:
            with self._thread_lock:
                removed = self._inhibited_sources.pop(source, None) is not None
            if removed:
                logger.info("🔓 [INHIBITION] Manual release for source: %s", source)

    def get_remaining_time(self, source: str) -> float:
        """Synchronous check for remaining time (returns 0 if not inhibited)."""
        # Use threading lock for sync access
        with self._thread_lock:
            expiry = self._inhibited_sources.get(source, 0)
            remaining = expiry - time.time()
            return max(0, remaining)

    def snapshot(self) -> dict[str, object]:
        with self._thread_lock:
            now = time.time()
            active = {
                source: max(0.0, expiry - now)
                for source, expiry in self._inhibited_sources.items()
                if expiry > now
            }
        return {
            "instance_id": self.instance_id,
            "ready": True,
            "active_sources": active,
        }

    def is_alive(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True

    def get_status(self) -> dict[str, object]:
        return self.snapshot()


_MANAGER: InhibitionManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_inhibition_manager() -> InhibitionManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = InhibitionManager()
    return _MANAGER


def reset_inhibition_manager_for_test() -> None:
    """Drop the process-global manager and its ServiceContainer registration.

    ``GlobalWorkspace`` lazily registers this on first competition, so a test
    that runs one leaves it behind and the next test to evict it is reported as
    having removed shared state.
    """
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
    try:
        from core.container import ServiceContainer

        services = getattr(ServiceContainer, "_services", None)
        if isinstance(services, dict):
            services.pop("inhibition_manager", None)
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        logger.debug(
            "%s unavailable (%s: %s); the manager stayed in the container after being torn down",
            "ServiceContainer",
            type(exc).__name__,
            exc,
        )
