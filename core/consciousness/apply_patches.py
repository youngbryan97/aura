"""Legacy compatibility hook for consciousness boot wiring.

The continuity persistence and audit-driven self-development behaviors now
live natively in their primary modules. This entry point remains so older boot
paths can keep calling it safely; it only starts the loop monitor.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.ConsciousnessPatches")


def apply_consciousness_patches(orchestrator: Any) -> None:
    """Compatibility hook for native consciousness runtime wiring."""

    if getattr(orchestrator, "_consciousness_patches_applied", False):
        logger.debug("apply_consciousness_patches: already applied — skipping")
        return

    # The behavioral upgrades that used to be patched are now implemented
    # natively. The remaining boot-time concern is starting the loop monitor.
    try:
        from core.consciousness.loop_monitor import get_loop_monitor

        monitor = get_loop_monitor(orchestrator)
        monitor.start()
        orchestrator.loop_monitor = monitor
    except Exception as exc:
        logger.error("ConsciousnessPatches: loop monitor failed — %s", exc, exc_info=True)

    orchestrator._consciousness_patches_applied = True
    logger.info("🧠 Consciousness runtime wiring is native; compatibility hook completed")
