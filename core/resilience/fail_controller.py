import logging
import time
from collections import defaultdict

logger = logging.getLogger("Kernel.FailControl")

class FailureController:
    """Prevents insanity (doing the same thing expecting different results).
    """

    def __init__(self, max_retries: int = 3, cooldown: int = 60):
        self.max_retries = max_retries
        self.cooldown = cooldown
        self.failures = defaultdict(int)
        self.last_failure_time = defaultdict(float)

    def register_failure(self, goal_id: str):
        self.failures[goal_id] += 1
        self.last_failure_time[goal_id] = time.time()
        logger.warning("Failure count for %s: %d", goal_id, self.failures[goal_id])

    def should_abort(self, goal_id: str) -> bool:
        if self.failures[goal_id] >= self.max_retries:
            # Check if cooldown passed
            if (time.time() - self.last_failure_time[goal_id]) > self.cooldown:
                self.failures[goal_id] = 0 # Reset
                return False
            return True
        return False