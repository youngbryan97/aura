"""Registration for the six fictional-AI engines.

One owner per service. The short names ("jarvis", "cortana", …) are
container ALIASES of the canonical service names, not second instance
registrations — two registrations of one object gave the container two
owners for one lifecycle, so start, stop and health could each be applied
twice to a thing that can only be started once (CP126 ``9eb5b7c6``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from core.fictional.ava import SocialModelingEngine
from core.fictional.common import record_fictional_degradation
from core.fictional.cortana import CognitiveHealthMonitor
from core.fictional.edi import ProgressiveAutonomySystem
from core.fictional.jarvis import ProactiveAnticipationEngine
from core.fictional.mist import TemporalDilationScheduler
from core.fictional.skynet import DistributedResilienceCore
from core.service_names import ServiceNames

logger = logging.getLogger("Aura.FictionalSynthesis")

__all__ = ["register_all_fictional_engines"]





def register_all_fictional_engines(orchestrator=None) -> dict[str, Any]:
    from core.container import ServiceContainer
    from core.utils.task_tracker import get_task_tracker

    engines: dict[str, Any] = {}
    tracker = get_task_tracker()
    foreground_only = os.getenv("AURA_FOREGROUND_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
    background_loop_blocker = ""
    try:
        from core.runtime.background_policy import background_loop_start_reason

        background_loop_blocker = background_loop_start_reason(origin="fictional_ai_synthesis")
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_fictional_degradation(
            exc,
            severity="warning",
            action="registered fictional engines without autonomous loops because background policy was unavailable",
        )
        background_loop_blocker = "background_policy_unavailable"

    def _register(short_name: str, canonical: str, build) -> Any:
        """One owner, one registration, and an ALIAS for the short name.

        Both names used to be full instance registrations of the same
        object, so the container held two owners for one lifecycle and
        health, start and stop could each be applied twice to a thing that
        can only be started once (CP126 ``9eb5b7c6``). The container has an
        alias mechanism; using it makes the short name resolve to the
        canonical service rather than shadow it.
        """
        instance = ServiceContainer.get(canonical, default=None) or build()
        ServiceContainer.register_instance(canonical, instance)
        if short_name != canonical:
            ServiceContainer.register_alias(short_name, canonical)
        engines[short_name] = instance
        return instance

    _register("jarvis", ServiceNames.JARVIS,
              lambda: ProactiveAnticipationEngine(orchestrator=orchestrator))
    _register("cortana", ServiceNames.CORTANA, CognitiveHealthMonitor)
    _register("edi", ServiceNames.EDI, ProgressiveAutonomySystem)
    _register("ava", ServiceNames.AVA, SocialModelingEngine)
    _register("skynet", ServiceNames.SKYNET, DistributedResilienceCore)
    _register("mist", ServiceNames.MIST,
              lambda: TemporalDilationScheduler(orchestrator=orchestrator))

    if foreground_only or background_loop_blocker:
        logger.info(
            "✅ Fictional AI engines registered without background loops (%s).",
            background_loop_blocker or "foreground-only boot",
        )
        return engines

    # FIXED: Supervised task creation — tasks tracked and named
    async def _safe_start(name: str, coro):
        try:
            await coro
        except asyncio.CancelledError:
            logger.info("Fictional engine '%s' task cancelled cleanly.", name)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_fictional_degradation(
                e,
                severity="degraded",
                action=f"ended supervised fictional engine task {name} after unrecoverable loop crash",
            )
            logger.error("Fictional engine '%s' task crashed: %s", name, e, exc_info=True)

    tracker.track(
        get_task_tracker().create_task(_safe_start("jarvis", engines["jarvis"].start()), name="jarvis.start"),
        name="jarvis.start"
    )
    tracker.track(
        get_task_tracker().create_task(_safe_start("skynet", engines["skynet"].start_monitoring()), name="skynet.monitor"),
        name="skynet.monitor"
    )
    # The idle loop starts unconditionally and resolves the brain on every
    # cycle. It used to poll for thirty seconds and then give up for the
    # life of the process, so a boot slower than that — a cold model load,
    # a contended host — permanently disabled idle synthesis and said so
    # in one warning (CP126 ``be9ce637``).
    tracker.track(
        get_task_tracker().create_task(
            _safe_start("mist", engines["mist"].run_idle_loop()),
            name="mist.idle"
        ),
        name="mist.idle"
    )

    logger.info("✅ All fictional AI engines registered and supervised.")
    return engines
