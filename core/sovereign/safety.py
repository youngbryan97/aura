"""core/sovereign/safety.py — SafetyGuard Singleton
=================================================
Provides process-wide safety blocks and emergency shutdown mechanisms.
"""
from __future__ import annotations


import logging
import threading

logger = logging.getLogger(__name__)


class SafetyGuard:
    _instance: SafetyGuard | None = None
    _lock = threading.Lock()
    _blocks: set[str]
    _emergency_mode: bool
    
    def __new__(cls) -> SafetyGuard:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._blocks = set()
                cls._instance._emergency_mode = False
            return cls._instance

    def add_block(self, reason: str) -> None:
        """Register a safety block preventing certain operations."""
        logger.warning("🛡️ SAFETY BLOCK ADDED: %s", reason)
        self._blocks.add(reason)

    def remove_block(self, reason: str) -> None:
        """Remove a safety block."""
        if reason in self._blocks:
            logger.info("🛡️ SAFETY BLOCK REMOVED: %s", reason)
            self._blocks.discard(reason)

    def is_blocked(self) -> bool:
        """Check if any safety blocks are active."""
        return len(self._blocks) > 0 or self._emergency_mode

    def trigger_emergency(self) -> None:
        """Trigger emergency lockdown mode."""
        logger.critical("🛑 EMERGENCY LOCKDOWN TRIGGERED")
        self._emergency_mode = True

    @property
    def blocks(self) -> list[str]:
        return list(self._blocks)
