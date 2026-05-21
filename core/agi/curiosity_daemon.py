"""core/agi/curiosity_daemon.py — Decoupled Epistemic Curiosity Daemon
====================================================================
A background actor daemon that runs out-of-band and periodically
queries the EpistemicTracker for knowledge gaps. It triggers
background explorations using an isolated token gate without
clogging the main conversational thread.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.epistemic_tracker import EpistemicTracker, get_epistemic_tracker
from core.runtime.errors import record_degradation
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
        """Start the background exploration loop."""
        if self._is_running:
            return
        self._is_running = True

        self._task = get_task_tracker().create_task(
            self.start_exploration_loop(capability_engine, will_gate),
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
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("AutonomousCuriosityDaemon background task stopped.")

    async def start_exploration_loop(self, capability_engine: Any = None, will_gate: Any = None):
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

                    # 3. Retrieve or resolve capability_engine and will_gate / authority_gateway
                    from core.container import ServiceContainer

                    resolved_engine = capability_engine or ServiceContainer.get(
                        "capability_engine", default=None
                    )
                    resolved_gate = will_gate or ServiceContainer.get(
                        "authority_gateway", default=None
                    )

                    if not resolved_gate:
                        try:
                            from core.executive.authority_gateway import get_authority_gateway

                            resolved_gate = get_authority_gateway()
                        except ImportError:
                            pass

                    # 4. Generate/request capability token if we have a gate
                    token = None
                    if resolved_gate:
                        if hasattr(resolved_gate, "request_background_token"):
                            token = await resolved_gate.request_background_token(
                                f"research:{target_domain}"
                            )
                        else:
                            auth = await resolved_gate.authorize_tool_execution(
                                "web_search",
                                {"query": query},
                                source="curiosity_daemon",
                                priority=0.5,
                                is_critical=False,
                            )
                            if auth.approved:
                                token = getattr(auth, "capability_token_id", None)

                    # 5. Execute search/exploration out-of-band via CapabilityEngine
                    if resolved_engine:
                        context = {"capability_token_id": token} if token else {}
                        await resolved_engine.execute(
                            "web_search", {"query": query}, context=context
                        )
                    else:
                        logger.warning(
                            "CapabilityEngine unavailable; background exploration skipped."
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
