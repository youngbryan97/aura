"""core/sovereign/safety.py — SafetyGuard Singleton
=================================================
Provides process-wide safety blocks and emergency shutdown mechanisms.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

class SafetyGuard:
    _instance: Optional['SafetyGuard'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SafetyGuard, cls).__new__(cls)
                cls._instance._blocks = set()
                cls._instance._emergency_mode = False
            return cls._instance

    def add_block(self, reason: str):
        """Register a safety block preventing certain operations."""
        logger.warning("🛡️ SAFETY BLOCK ADDED: %s", reason)
        self._blocks.add(reason)

    def remove_block(self, reason: str):
        """Remove a safety block."""
        if reason in self._blocks:
            logger.info("🛡️ SAFETY BLOCK REMOVED: %s", reason)
            self._blocks.discard(reason)

    def is_blocked(self) -> bool:
        """Check if any safety blocks are active."""
        return len(self._blocks) > 0 or self._emergency_mode

    def trigger_emergency(self):
        """Trigger emergency lockdown mode."""
        logger.critical("🛑 EMERGENCY LOCKDOWN TRIGGERED")
        self._emergency_mode = True

    @property
    def blocks(self):
        return list(self._blocks)
