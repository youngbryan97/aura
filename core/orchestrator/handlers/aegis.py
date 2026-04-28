"""core/orchestrator/handlers/aegis.py
Extracted AEGIS Sentinel monitoring loop from RobustOrchestrator.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.orchestrator.main import RobustOrchestrator

logger = logging.getLogger("Aura.Core.Orchestrator.Aegis")


async def aegis_sentinel_loop(orch: "RobustOrchestrator") -> None:
    """Phase XXIII: True-Lock Reality Maintenance Loop.

    Monitors core Mycelial integrity and prevents narrative collapse
    or unauthorized structure deletion during runtime.
    """
    from core.container import ServiceContainer

    logger.info("🛡️ AEGIS SENTINEL: Narrative Integrity Guard Active")

    while not orch._stop_event.is_set():
        await asyncio.sleep(10.0)

        try:
            from core.mycelium import MycelialNetwork
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if not mycelium:
                logger.warning(
                    "🛡️ AEGIS ALERT: Mycelial Network UNRESOLVABLE. Attempting restoration..."
                )
                restored = MycelialNetwork()
                try:
                    ServiceContainer.register_instance("mycelial_network", restored)
                except Exception:
                    logger.debug("AEGIS: Mycelial registration failed after lock")
                continue

            # Check Object Integrity
            if not getattr(mycelium, "_aegis_locked", False):
                logger.critical(
                    "🛡️ AEGIS ALERT: True-Lock disabled! Possible tampering."
                )
                if hasattr(MycelialNetwork, "restore_from_vault"):
                    await MycelialNetwork.restore_from_vault()
                continue

            # Periodic Vault Sync
            if hasattr(mycelium, "vault_sync"):
                last_sync = getattr(orch, "_last_vault_sync", 0.0)
                if time.monotonic() - last_sync > 60.0:
                    await mycelium.vault_sync()
                    orch._last_vault_sync = time.monotonic()

        except Exception as exc:
            record_degradation('aegis', exc)
            logger.debug("🛡️ AEGIS Sentinel Pulse Error: %s", exc)
