"""MIST / Pantheon — TemporalDilationScheduler.

When the person is away, the compute is free. One supervised cycle per
poll, each one bounded and offloaded where the work is synchronous.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.fictional.common import (
    coerce_insight_text,
    record_fictional_degradation,
)

logger = logging.getLogger("Aura.FictionalSynthesis")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 6: PANTHEON/MIST — TemporalDilationScheduler
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalDilationScheduler:
    """
    Derived from: MIST / Pantheon
    """
    MIN_IDLE_FOR_SYNTHESIS_S = 300.0
    SYNTHESIS_COOLDOWN_S = 300.0

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._brain: Any = None
        self._last_user_activity: float = time.time()
        self._is_running = False
        self._synthesis_count = 0
        self._last_synthesis_time = 0.0
        # Retained product of the last idle synthesis; previously the
        # generated insight was discarded the moment it was produced.
        self._last_insight: str = ""

    def record_user_activity(self): self._last_user_activity = time.time()

    def last_insight(self) -> str:
        """Most recent background-synthesis insight (empty if none yet)."""
        return self._last_insight

    async def run_idle_loop(self, brain=None):
        """Poll for idle, then synthesize. Each cycle is supervised.

        The loop body used to hold its own imports and service lookups
        outside every try, so an ImportError or a container failure ended
        the background task for the life of the process and nothing said
        so (CP126 ``20d4fbd3``). The cycle is now one call with a named
        boundary and a backoff.
        """
        if self._is_running:
            return
        self._is_running = True
        logger.info("⏳ MIST TemporalDilation active. Watching for idle states...")

        consecutive_failures = 0
        while self._is_running:
            await asyncio.sleep(30.0 * min(consecutive_failures + 1, 10))
            try:
                await self._idle_cycle(brain)
                consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except self._CYCLE_BOUNDARY_ERRORS as exc:
                consecutive_failures += 1
                record_fictional_degradation(
                    exc,
                    severity="warning",
                    action=(
                        "backed off the MIST idle loop after cycle failure "
                        f"#{consecutive_failures}"
                    ),
                )
                logger.error("MIST idle cycle error: %s", exc)

    #: Everything one cycle may raise and still be worth another cycle.
    _CYCLE_BOUNDARY_ERRORS = (
        ImportError,
        OSError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        asyncio.TimeoutError,
    )

    async def _idle_cycle(self, brain=None) -> None:
        """One idle check and, if warranted, one synthesis. Raises."""
            
        # Lazy brain resolution, retried EVERY cycle and remembered.
        # The deferred starter used to give the brain thirty seconds to
        # appear and then never look again (CP126 ``be9ce637``); a boot
        # slower than that left idle synthesis off for the life of the
        # process with one warning line to show for it.
        from core.container import ServiceContainer

        if brain is None:
            brain = self._brain
        if brain is None:
            # The injected orchestrator first, then the container. The old
            # path consulted only the container, so a scheduler constructed
            # WITH an orchestrator still could not find its brain.
            host = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
            if host is not None and getattr(host, "brain", None):
                brain = host.brain
        if brain is not None:
            self._brain = brain

        orch = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
        last_user = self._last_user_activity
        if orch:
            last_user = float(getattr(orch, "_last_user_interaction_time", last_user) or last_user)

        idle_time = max(0.0, time.time() - last_user)
        if idle_time < self.MIN_IDLE_FOR_SYNTHESIS_S:
            return

        if (time.time() - self._last_synthesis_time) < self.SYNTHESIS_COOLDOWN_S:
            return

        if orch and getattr(getattr(orch, "status", None), "is_processing", False):
            return

        if orch:
            try:
                from core.runtime.background_policy import (
                    MAINTENANCE_BACKGROUND_POLICY,
                    background_activity_reason,
                )

                reason = background_activity_reason(
                    orch,
                    profile=MAINTENANCE_BACKGROUND_POLICY,
                    min_idle_seconds=self.MIN_IDLE_FOR_SYNTHESIS_S,
                    max_memory_percent=78.0,
                    max_failure_pressure=0.25,
                    require_conversation_ready=True,
                )
                if reason:
                    logger.debug("MIST: Skipping synthesis by background policy: %s", reason)
                    return
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                # Fail CLOSED. Background synthesis exists only when the
                # runtime can afford it; an unreadable policy is not
                # permission, and proceeding made the governance
                # dependency advisory exactly when the system was already
                # degraded enough to break the probe.
                record_fictional_degradation(
                    exc,
                    severity="warning",
                    action="skipped idle-synthesis cycle because the background policy probe failed",
                )
                logger.debug("MIST background policy probe failed: %s", exc)
                return

        flow_controller = getattr(orch, "_flow_controller", None) if orch else None
        if flow_controller and orch:
            try:
                # Off the loop and bounded. `snapshot` walks live runtime
                # state and is arbitrary work called from a coroutine
                # (CP126 ``a793b8cc``).
                snapshot = await asyncio.wait_for(
                    asyncio.to_thread(flow_controller.snapshot, orch), timeout=5.0
                )
                if snapshot.overloaded:
                    logger.debug("MIST: Skipping synthesis while cognition is overloaded.")
                    return
            except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_fictional_degradation(
                    exc,
                    severity="warning",
                    action="continued idle-synthesis loop after flow-control probe failed",
                )
                logger.debug("MIST flow-control probe failed: %s", exc)

        if idle_time >= self.MIN_IDLE_FOR_SYNTHESIS_S:
            self._synthesis_count += 1
            logger.info("⏳ MIST: System idle (%.0fs). Initiating background synthesis cycle #%d...",
                        idle_time, self._synthesis_count)
                
            try:
                mem = ServiceContainer.get("memory_facade", default=None)
                if mem and hasattr(mem, "get_cold_memory_context") and brain:
                    # Perform background consolidation logic
                    query = "recent unresolved goals, salient memories, and open threads"
                    cold_context = await asyncio.wait_for(
                        mem.get_cold_memory_context(query, limit=3),
                        timeout=10.0,
                    )
                    if cold_context:
                        logger.info("⏳ MIST: Consolidated background context: %d chars.", len(cold_context))
                        synth_prompt = f"Background synthesis: Refine the following context into a proactive insight: {cold_context[:500]}"
                        # We use FAST mode for background synthesis to conserve resources
                        from core.brain.types import ThinkingMode
                        # RETAIN the generated insight. The result used to
                        # be awaited and thrown away, and the PRE-generation
                        # cold context was bottled in its place — so the
                        # expensive background reasoning was never
                        # incorporated anywhere and the "idle thinking"
                        # persisted only its own input.
                        synthesis_result = await asyncio.wait_for(
                            brain.think(
                                synth_prompt,
                                mode=ThinkingMode.FAST,
                                origin="mist",
                                is_background=True,
                            ),
                            timeout=45.0,
                        )
                        insight_text = coerce_insight_text(synthesis_result)
                        self._last_synthesis_time = time.time()
                        self._last_insight = insight_text
                        logger.info(
                            "⏳ MIST: Synthesis cycle complete (%d chars of insight).",
                            len(insight_text),
                        )

                        # Brainiac: durably bottle the consolidated context so the
                        # idle thinking is retrievable later, not just logged. Stable
                        # slug keeps storage bounded (latest consolidation overwrites).
                        try:
                            brainiac = ServiceContainer.get("brainiac", default=None)
                            if brainiac is not None and hasattr(brainiac, "bottle"):
                                # Bottle the INSIGHT (with its source
                                # context as provenance) rather than the
                                # raw pre-generation context alone.
                                bottled = (
                                    f"{insight_text}\n\n---\nsource context:\n{cold_context}"
                                    if insight_text
                                    else cold_context
                                )
                                await asyncio.wait_for(
                                    brainiac.bottle("idle-consolidation", bottled),
                                    timeout=20.0,
                                )
                                logger.debug("🫙 Brainiac bottled the idle insight.")
                        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as _bx:
                            record_fictional_degradation(
                                _bx,
                                action="completed MIST cycle without bottling consolidated context",
                            )

                        # Idle = the model is free, so go deep. Deep Thought *deliberates*
                        # on the open thread (refine + answer, bounded), Caine rehearses a
                        # what-if from the refined question, and HAL runs a semantic
                        # conflict scan over the directive set the keyword scan can't catch.
                        try:
                            dt = ServiceContainer.get("deep_thought", default=None)
                            refined = query
                            if dt is not None and hasattr(dt, "deliberate"):
                                # Its own timeout_s is a hint to the callee.
                                # The deadline that protects THIS loop has
                                # to be enforced here.
                                _delib = await asyncio.wait_for(
                                    dt.deliberate(
                                        query,
                                        budget=1,
                                        timeout_s=12.0,
                                        foreground_request=False,
                                    ),
                                    timeout=20.0,
                                )
                                refined = _delib.refined_question
                                logger.debug("🪐 Deep Thought deliberated an idle thread.")
                            caine = ServiceContainer.get("caine", default=None)
                            if caine is not None and hasattr(caine, "forge_fast"):
                                # Synchronous scenario forging, off the loop.
                                await asyncio.wait_for(
                                    asyncio.to_thread(caine.forge_fast, refined[:120]),
                                    timeout=15.0,
                                )
                                logger.debug("🎪 Caine rehearsed an idle what-if scenario.")
                            hal = ServiceContainer.get("hal", default=None)
                            if hal is not None and hasattr(hal, "scan_semantic"):
                                _sem = await hal.scan_semantic(timeout=8.0)
                                if any(getattr(c, "kind", "") == "semantic" for c in _sem):
                                    logger.info("🔴 HAL semantic scan flagged a directive tension.")
                        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as _idle_exc:
                            record_fictional_degradation(
                                _idle_exc,
                                action="completed MIST cycle without deep deliberation / semantic scan",
                            )
                    else:
                        logger.debug("MIST: No cold context available for synthesis.")
                else:
                    logger.debug("MIST: Missing memory facade or brain; skipping synthesis cycle.")
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as e:
                record_fictional_degradation(
                    e,
                    action="completed idle-synthesis cycle without writing a background insight",
                )
                logger.debug("MIST synthesis error: %s", e)

            # NOTE: no unconditional extra sleep here. A fixed 300s pause
            # was added to EVERY iteration on top of the 30s poll, so the
            # documented "check every 30s" cadence was really 330s and
            # idle/shutdown responsiveness was 11x slower than configured.
            # Thrash protection is already provided by SYNTHESIS_COOLDOWN_S,
            # which is enforced against _last_synthesis_time at the top of
            # the loop and only advances when a synthesis actually ran.

    def stop(self): self._is_running = False

