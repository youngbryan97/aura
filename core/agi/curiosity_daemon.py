"""core/agi/curiosity_daemon.py — Decoupled Epistemic Curiosity Daemon
====================================================================
A background actor daemon that runs out-of-band and periodically
queries the EpistemicTracker for knowledge gaps. It sends explorations through
the canonical orchestrator or CapabilityEngine authority chain without
clogging the main conversational thread.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.epistemics.epistemic_tracker import EpistemicTracker, get_epistemic_tracker
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_orchestrator
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.AGI.CuriosityDaemon")


class AutonomousCuriosityDaemon:
    """Runs periodic background curiosity exploration based on the epistemic profile."""

    def __init__(self, tracker: EpistemicTracker | None = None, interval_seconds: int = 300):
        self.tracker = tracker or get_epistemic_tracker()
        self.interval = interval_seconds
        self._is_running = False
        self._task: asyncio.Task | None = None
        logger.info("AutonomousCuriosityDaemon initialized (interval: %ds).", interval_seconds)

    async def start(self, capability_engine: Any = None, will_gate: Any = None):
        """Start exploration; ``will_gate`` remains only for caller compatibility.

        Authorization belongs to the canonical tool path. Calling a separate
        token gate here would create an uncorrelated, double-authorization flow.
        """
        if self._is_running:
            return
        self._is_running = True

        self._task = get_task_tracker().create_task(
            self.start_exploration_loop(capability_engine),
            name="AutonomousCuriosityDaemon",
        )
        logger.info("🚀 AutonomousCuriosityDaemon background task started.")

    async def stop(self):
        """Stop the background exploration loop."""
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed %s in core.agi.curiosity_daemon: %s", type(_exc).__name__, _exc)
            self._task = None
        logger.info("AutonomousCuriosityDaemon background task stopped.")

    async def start_exploration_loop(self, capability_engine: Any = None):
        """Periodic curiosity drive checks the epistemic tracker for missing domains or gaps."""
        while self._is_running:
            try:
                # 1. Fetch the latest epistemic profile
                profile = self.tracker.get_profile(force_refresh=True)

                # 2. Extract urgent gaps or uncertain domains
                if profile.gaps:
                    # Sort by urgency
                    gaps = sorted(profile.gaps, key=lambda g: g.urgency, reverse=True)
                    target_gap = gaps[0]
                    target_domain = target_gap.domain
                    query = target_gap.seed_question

                    logger.info(
                        "🚀 Curiosity drive triggered: Investigating missing domain '%s' (gap: %s)",
                        target_domain,
                        target_gap.description,
                    )

                    # 3. Use the canonical orchestrator path so origin, standing
                    # authority, Will, capability, execution, and closure stay one chain.
                    from core.container import ServiceContainer

                    resolved_engine = capability_engine or ServiceContainer.get(
                        "capability_engine", default=None
                    )
                    orchestrator = resolve_orchestrator()
                    execution_context = {
                        "origin": "curiosity_daemon",
                        "source": "curiosity_daemon",
                        "objective": f"research epistemic gap: {target_domain}",
                    }
                    if orchestrator is not None and hasattr(orchestrator, "execute_tool"):
                        await orchestrator.execute_tool(
                            "web_search",
                            {"query": query},
                            origin="curiosity_daemon",
                            payload_context=execution_context,
                        )
                    elif resolved_engine:
                        await resolved_engine.execute(
                            "web_search",
                            {"query": query},
                            context=execution_context,
                        )
                    else:
                        logger.warning(
                            "Canonical tool orchestrator and CapabilityEngine unavailable; "
                            "background exploration skipped."
                        )

            except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
                record_degradation(
                    "curiosity_daemon",
                    e,
                    severity="warning",
                    action="skipped one background exploration iteration and kept curiosity loop alive",
                )
                logger.error("Error in background curiosity loop: %s", e)

            await asyncio.sleep(self.interval)
