import logging
import threading
from typing import Any, Dict, Optional

from .drive_engine import DriveEngine

logger = logging.getLogger("Kernel.Drives")

class DriveSystem:
    """Legacy Shim for DriveSystem -> DriveEngine (Resource Budgets).
    Maintains backward compatibility for 'drives' dict access and 'time' decay.
    """

    def __init__(self):
        self.engine = DriveEngine()
        self._lock = threading.Lock() # Final safety lock for sync mutations
        
    @property
    def drives(self) -> Dict[str, float]:
        # Map budget levels to simple float dict for legacy access
        return {name: b.level for name, b in self.engine.budgets.items()}

    def update(self) -> None:
        """Forward update tick."""
        self.engine.update()

    async def satisfy_async(self, drive: str, amount: float):
        """Async-safe satisfaction."""
        await self.engine.satisfy(drive, amount)

    async def punish_async(self, drive: str, amount: float):
        """Async-safe punishment."""
        await self.engine.punish(drive, amount)

    def satisfy(self, drive: str, amount: float):
        """Satisfy a drive (Sync wrapper for Async engine).
        CAUTION: Mutates directly under lock to prevent race conditions.
        """
        with self._lock:
            b = self.engine.budgets.get(drive)
            if b:
                b.tick()
                b.level = min(b.capacity, b.level + amount)
                logger.debug("❤️ Drive Satisfied (Legacy-Safe): %s +%.1f -> %.1f", drive, amount, b.level)

    def punish(self, drive: str, amount: float):
        """Punish a drive (Sync wrapper)."""
        with self._lock:
            b = self.engine.budgets.get(drive)
            if b:
                b.tick()
                b.level = max(0.0, b.level - amount)
                logger.debug("💔 Drive Damaged (Legacy-Safe): %s -%.1f -> %.1f", drive, amount, b.level)

    def get_imperative(self) -> Optional[str]:
        return self.engine.get_imperative()

    async def get_status(self) -> Dict[str, Any]:
        """Forward to engine's async get_status to return full dict structure
        expected by server.py (e.g. {'energy': {'level': ...}}).
        """
        return await self.engine.get_status()
