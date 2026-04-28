from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from core.container import ServiceContainer

logger = logging.getLogger(__name__)


class BootBackgroundMixin:
    """Provides initialization for background coroutines and metabolic loops."""

    belief_sync: Any
    meta_cognition: Any

    async def _cognitive_heartbeat_task(self):
        """A background coroutine that constantly updates the heartbeat file for the Lazarus brainstem."""
        heartbeat_file = Path.home() / ".aura" / "run" / "heartbeat.pulse"
        logger.info("💓 Heartbeat monitor starting (Lazarus Protocol active)")
        _last_continuity_save = 0.0
        while True:
            try:
                # 'touch' the file, updating its modification time.
                # Use a thread for file I/O to avoid blocking the event loop (ASYNC240)
                await asyncio.to_thread(heartbeat_file.touch, exist_ok=True)

                # Save continuity record every 5 minutes
                if time.time() - _last_continuity_save > 300:
                    cont = ServiceContainer.get("continuity", default=None)
                    if cont:
                        last_msg = ""
                        if (
                            hasattr(self, "conversation_history")
                            and self.conversation_history
                        ):
                            last_msg = str(self.conversation_history[-1])[:200]
                        cont.save(reason="checkpoint", last_exchange=last_msg)
                    _last_continuity_save = time.time()
            except Exception as _e:
                record_degradation('boot_background', _e)
                # Fail-silent internally, brainstem will eventually reboot us if this fails repeatedly
                logger.debug("Ignored Exception in boot.py: %s", _e)
            await asyncio.sleep(10)

    async def _init_subconscious_loop_subsystem(self, tracker):
        """Initialize the Subconscious Loop."""
        try:
            from core.consciousness.subconscious_loop import register_subconscious_loop

            scl = register_subconscious_loop(self)
            tracker.create_task(scl.start(), name="subconscious_loop")
        except Exception as e:
            record_degradation('boot_background', e)
            logger.error("Subconscious Loop init failed: %s", e)
            ServiceContainer.register_instance("subconscious_loop", None)

    async def _start_meta_evolution(self):
        """Initializes the Meta-Evolution Engine for recursive self-optimization."""
        try:
            from core.meta_cognition import MetaEvolutionEngine

            self.meta_cognition = MetaEvolutionEngine()
            ServiceContainer.register_instance(
                "meta_cognition_shard", self.meta_cognition
            )
            ServiceContainer.register_instance("meta_evolution", self.meta_cognition)
            logger.info(
                "🌀 [BOOT] Meta-Evolution Engine (meta_cognition_shard) initialized."
            )
        except Exception as e:
            record_degradation('boot_background', e)
            logger.error("🛑 [BOOT] Failed to initialize Meta-Evolution Engine: %s", e)

    async def _start_belief_sync_at_boot(self, tracker):
        """Start the Belief Sync loop if available."""
        try:
            if (
                hasattr(self, "belief_sync")
                and self.belief_sync
                and hasattr(self.belief_sync, "start")
            ):
                tracker.create_task(self.belief_sync.start(), name="belief_sync")
        except Exception as e:
            record_degradation('boot_background', e)
            logger.error("BeliefSync start failed: %s", e)

    async def _proactive_notify_callback(self, content: str, urgency: int):
        """Callback for proactive system messages."""
        def _constitutional_runtime_live() -> bool:
            try:
                return (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
            except Exception:
                return False

        try:
            from core.consciousness.executive_authority import get_executive_authority

            decision = await get_executive_authority(self).release_expression(
                content,
                source="proactive_callback",
                urgency=max(0.1, min(1.0, float(urgency) / 5.0)),
                metadata={
                    "voice": False,
                    "trigger": "proactive_callback",
                },
            )
            if decision.get("action") in {"released", "suppressed"}:
                return
        except Exception as exc:
            record_degradation('boot_background', exc)
            logger.debug("Proactive callback executive route failed: %s", exc)

        if _constitutional_runtime_live():
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "boot_background",
                    "proactive_callback_suppressed_without_authority",
                    detail=content[:120],
                    severity="warning",
                    classification="background_degraded",
                    context={"urgency": urgency},
                )
            except Exception as degraded_exc:
                record_degradation('boot_background', degraded_exc)
                logger.debug("Proactive callback degraded-event logging failed: %s", degraded_exc)
            return

        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event(
                "boot_background",
                "proactive_callback_dropped",
                detail=content[:120],
                severity="warning",
                classification="background_degraded",
                context={"urgency": urgency},
            )
        except Exception as exc:
            record_degradation('boot_background', exc)
            logger.debug("Proactive callback degraded-event logging failed: %s", exc)

    async def _init_metabolism(self):
        """Initialize system resource awareness and optimization."""
        try:
            from core.ops.metabolic_monitor import MetabolicMonitor

            # Phase 5: OptimizationEngine replaced by AutonomicCore

            monitor = MetabolicMonitor(ram_threshold_mb=8192, cpu_threshold=85.0)
            monitor.start()  # Phase 21: Decoupled ANS Thread

            # Use top-level import to avoid shadowing
            ServiceContainer.register_instance("metabolic_monitor", monitor)

            # --- Phase 21: Metabolic Coordinator (High-level background manager) ---
            try:
                from core.coordinators.metabolic_coordinator import MetabolicCoordinator

                coordinator = MetabolicCoordinator(self)
                ServiceContainer.register_instance("metabolic_coordinator", coordinator)
                ServiceContainer.register_instance("metabolism", coordinator)
                logger.info("✓ Metabolic Coordinator ACTIVE (High-level pacing enabled)")
            except Exception as e:
                record_degradation('boot_background', e)
                logger.error("Failed to initialize Metabolic Coordinator: %s", e)

            logger.info("✓ Metabolic Monitor ACTIVE (Decoupled ANS Thread Online)")

            # Phase 21: Singularity State Tracking
            from core.config import config

            archive_dir = config.paths.home_dir / "eternal_archive"
            if archive_dir.exists() and any(archive_dir.iterdir()):
                self.status.singularity_threshold = True
                # H-28 FIX: Restoring original color scheme (Purple/Cyan) as requested by user
                # Acceleration factor > 1.2 triggers the 'Gold' Zenith theme
                self.status.acceleration_factor = 1.0
                logger.info(
                    "✨ Singularity Threshold DETECTED. Subsurface Resonance active."
                )

            # Phase 21: Dream Cycle (DLQ Re-ingestion)
            try:
                from core.resilience.dream_cycle import DreamCycle

                dlq_path = config.paths.data_dir / "dlq.jsonl"
                self.dream_cycle = DreamCycle(self, dlq_path)
                cycle = self.dream_cycle
                if cycle:
                    cycle.start()
            except Exception as e:
                record_degradation('boot_background', e)
                logger.error("Failed to initialize Dream Cycle: %s", e)

        except Exception as e:
            record_degradation('boot_background', e)
            logger.error("Failed to initialize Metabolic systems: %s", e)
