import logging
import time
from typing import Optional

logger = logging.getLogger("Aura.HealthMonitor")

class HealthMonitor:
    """Monitors system health, manages circuit breakers, and tracks errors.
    """
    
    def __init__(self, max_consecutive_errors: int = 5):
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0
        self.last_error: Optional[str] = None
        self.healthy = True
        self.start_time = time.time()
        self.total_errors = 0

    def track_error(self, error: Exception):
        """Track an error and update health status."""
        self.consecutive_errors += 1
        self.total_errors += 1
        self.last_error = str(error)
        
        if self.consecutive_errors >= self.max_consecutive_errors:
            if self.healthy:
                logger.critical("Circuit breaker tripped: %d consecutive errors.", self.consecutive_errors)
                self.healthy = False
                # Circuit breaker tripped. System continues in Safe Mode.
                pass
        
        return self.healthy

    def reset_failure_counter(self):
        """Reset the consecutive error counter on success."""
        if not self.healthy:
            logger.info("System recovered. Circuit breaker reset.")
        self.consecutive_errors = 0
        self.healthy = True

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    def get_status_report(self) -> dict:
        return {
            "healthy": self.healthy,
            "consecutive_errors": self.consecutive_errors,
            "total_errors": self.total_errors,
            "last_error": self.last_error,
            "uptime_seconds": round(self.uptime, 2)
        }

    def get_status(self) -> dict:
        """Alias for standardized system reporting."""
        return self.get_status_report()