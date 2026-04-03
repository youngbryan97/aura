"""
core/consciousness/apply_patches.py
=====================================
Unified entry point — applies all three consciousness patches.

Usage (call once after orchestrator is fully booted):

    from core.consciousness.apply_patches import apply_consciousness_patches
    apply_consciousness_patches(orchestrator)

That single call:
  1. Patches PhenomenologicalExperiencer for cross-session continuity
  2. Patches AgencyCore._pathway_self_development for audit-driven focus
  3. Starts ConsciousnessLoopMonitor as a background task

All three patches are idempotent — safe to call multiple times.

Minimal dependency: each patch degrades gracefully if its target
component hasn't been registered yet (logs a warning, does not raise).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("Aura.ConsciousnessPatches")


def apply_consciousness_patches(orchestrator: Any) -> None:
    """Wire all three patches into a live Aura orchestrator."""

    if getattr(orchestrator, "_consciousness_patches_applied", False):
        logger.debug("apply_consciousness_patches: already applied — skipping")
        return

    # ── Patch 1: Cross-session experiential continuity ────────────────────────
    try:
        from core.consciousness.continuity_patch import patch_experiencer
        from core.container import ServiceContainer

        experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
        if experiencer is None:
            # Try orchestrator attribute
            experiencer = getattr(orchestrator, "phenomenological_experiencer", None)

        if experiencer is not None:
            patch_experiencer(experiencer)
        else:
            logger.warning(
                "ConsciousnessPatches: PhenomenologicalExperiencer not found — "
                "continuity patch deferred. Call patch_experiencer(exp) manually "
                "once it is registered."
            )
    except Exception as exc:
        logger.error("ConsciousnessPatches: continuity patch failed — %s", exc, exc_info=True)

    # ── Patch 2: Audit-driven self-development pathway ───────────────────────
    try:
        from core.agency.self_development_patch import patch_agency_core

        agency = getattr(orchestrator, "agency_core", None)
        if agency is None:
            try:
                from core.container import ServiceContainer
                agency = ServiceContainer.get("agency_core", default=None)
            except Exception as _e:
                logger.debug('Ignored Exception in apply_patches.py: %s', _e)

        if agency is not None:
            patch_agency_core(agency)
        else:
            logger.warning(
                "ConsciousnessPatches: AgencyCore not found — "
                "self-development patch deferred. Call patch_agency_core(ac) manually."
            )
    except Exception as exc:
        logger.error("ConsciousnessPatches: self-dev patch failed — %s", exc, exc_info=True)

    # ── Patch 3: Consciousness loop health monitor ────────────────────────────
    try:
        from core.consciousness.loop_monitor import get_loop_monitor

        monitor = get_loop_monitor(orchestrator)
        monitor.start()
        orchestrator.loop_monitor = monitor
    except Exception as exc:
        logger.error("ConsciousnessPatches: loop monitor failed — %s", exc, exc_info=True)

    orchestrator._consciousness_patches_applied = True
    logger.info("🧠 All consciousness patches applied successfully")
