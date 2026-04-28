from core.runtime.errors import record_degradation
import logging
import os
import time

import psutil

logger = logging.getLogger("Aura.Governor")

class SystemGovernor:
    def __init__(self, cpu_limit: float = 95, ram_limit: float = 95) -> None:
        self.cpu_limit = cpu_limit
        self.ram_limit = ram_limit
        # Track Aura's OWN process, not system-wide
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent()  # prime the counter (first call always 0)
        self._last_check = 0.0
        self._cached_result = True
        self._check_interval = 5.0  # only re-check every 5 seconds

    def can_think_deeply(self) -> bool:
        """Safety Check: Returns False if Aura's OWN process is overloaded.
        Uses per-process CPU (not system-wide) to avoid interference from system tasks.
        Caches results for 5s to avoid blocking the event loop.
        """
        now = time.time()
        if now - self._last_check < self._check_interval:
            return self._cached_result

        try:
            # Non-blocking: returns CPU% since last call for THIS process only
            cpu = self._process.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            self._last_check = now
            
            if cpu > self.cpu_limit:
                logger.warning("⚠️ High Aura CPU (%.1f%%). Throttle engaged.", cpu)
                self._cached_result = False
                return False
            if ram > self.ram_limit:
                logger.warning("⚠️ High RAM (%.1f%%). Throttle engaged.", ram)
                self._cached_result = False
                return False
            
            self._cached_result = True
            return True
        except Exception as e:
            record_degradation('governor', e)
            # Fail-safe: If sensors break, allow operation but log error
            logger.error("Governor sensor failure: %s", e)
            self._cached_result = True
            return True
