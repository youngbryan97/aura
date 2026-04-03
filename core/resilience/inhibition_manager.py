import time
import logging
import asyncio
import threading
from typing import Dict, Optional, Set

logger = logging.getLogger("Aura.Resilience.Inhibition")

class InhibitionManager:
    """
    Global Inhibition Manager (Nervous System Protection).
    
    Provides a centralized mechanism to inhibit specific subsystems or behaviors
    to prevent recursive loops, attention seizures, or resource exhaustion.
    
    This is a 'Biological' primitive that mimics neural inhibition.
    """
    
    def __init__(self):
        self._inhibited_sources: Dict[str, float] = {}  # source_name -> expiry_timestamp
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock() # For sync access
        logger.info("🚫 InhibitionManager initialized. (Global Cross-Process Protection).")

    async def inhibit(self, source: str, duration: float = 5.0, reason: Optional[str] = None):
        """
        Place a source into the inhibited state.
        
        Args:
            source: The name of the subsystem or process to inhibit (e.g. 'thought_loop', 'voice_cortex')
            duration: How long (seconds) to inhibit.
            reason: Optional reason for logging.
        """
        async with self._lock:
            expiry = time.time() + duration
            # If already inhibited, we take the MAX of current and new expiry
            current_expiry = self._inhibited_sources.get(source, 0)
            self._inhibited_sources[source] = max(current_expiry, expiry)
            
            logger.warning(
                "🛑 [INHIBITION] Source '%s' inhibited for %.1fs. Reason: %s",
                source, duration, reason or "unspecified"
            )

    async def is_inhibited(self, source: str) -> bool:
        """Check if a source is currently inhibited."""
        async with self._lock:
            expiry = self._inhibited_sources.get(source, 0)
            if expiry > time.time():
                return True
            
            # Cleanup expired entry
            if source in self._inhibited_sources:
                del self._inhibited_sources[source]
            return False

    async def get_inhibited_sources(self) -> Set[str]:
        """Return a set of all currently inhibited source names."""
        now = time.time()
        async with self._lock:
            # Active filter and cleanup
            active = set()
            expired = []
            for src, exp in self._inhibited_sources.items():
                if exp > now:
                    active.add(src)
                else:
                    expired.append(src)
            
            for src in expired:
                del self._inhibited_sources[src]
                
            return active

    async def release(self, source: str):
        """Manually lift an inhibition before its expiry."""
        async with self._lock:
            if source in self._inhibited_sources:
                del self._inhibited_sources[source]
                logger.info("🔓 [INHIBITION] Manual release for source: %s", source)

    def get_remaining_time(self, source: str) -> float:
        """Synchronous check for remaining time (returns 0 if not inhibited)."""
        # Use threading lock for sync access
        with self._thread_lock:
            expiry = self._inhibited_sources.get(source, 0)
            remaining = expiry - time.time()
            return max(0, remaining)
