"""Lifecycle Coordinator — handles boot sequence, startup, teardown,
the main execution loop, and cognitive retries.

Extracted from orchestrator.py as part of the God Object decomposition.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from typing import Any, Dict, Optional

from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)


class LifecycleCoordinator:
    """Manages the orchestrator's run loop, startup, shutdown, and connection recovery."""

    def __init__(self, orch):
        self.orch = orch

    async def run(self):
        """Main execution loop. Continuously processes cycles until stop event is set."""
        orch = self.orch
        if not orch.status.running:
            logger.info("Run called but orchestrator not running. Starting...")
            started = await self.start()
            if not started:
                logger.error("Failed to auto-start in run loop.")
                return

        logger.info("🚀 Orchestrator Main Loop ACTIVE")
        orch.status.is_processing = False

        # Watchdog for Cycle Stalls
        last_cycle = orch.status.cycle_count
        last_change_time = time.time()
        idle_cycles = 0

        while orch.status.running and not orch._stop_event.is_set():
            try:
                loop_start = time.time()
                did_work = await orch._process_cycle()

                # Dynamic Backoff
                if did_work:
                    idle_cycles = 0
                    sleep_time = 0.05 / max(1.0, getattr(orch.status, 'acceleration_factor', 1.0))
                else:
                    idle_cycles += 1
                    sleep_time = min(2.0, 0.1 + (idle_cycles * 0.05))

                await asyncio.sleep(sleep_time)
                orch._update_heartbeat() # Persistent heartbeat for Temporal Drift detection

                # Watchdog Check
                if orch.status.cycle_count > last_cycle:
                    last_cycle = orch.status.cycle_count
                    last_change_time = time.time()
                elif time.time() - last_change_time > 30:
                    logger.critical("WATCHDOG: Cycle stalled for 30s! Initiating recovery...")
                    orch._emit_telemetry("Watchdog", "Cycle stalled. Attempting recovery.", level="error")
                    await orch._recover_from_stall()
                    last_change_time = time.time() # Reset to prevent spam

            except asyncio.CancelledError:
                logger.info("Orchestrator run loop cancelled.")
                break
            except Exception as e:
                record_degradation('lifecycle_coordinator', e)
                logger.error("CRITICAL LOOP ERROR: %s", e)
                orch.status.add_error(str(e))
                await asyncio.sleep(1) # Prevent tight error loops

    async def start(self) -> bool:
        """Start the orchestrator (Async)"""
        import core.config as config
        orch = self.orch
        tracker = get_task_tracker()

        # Handle lazy initialization of subsystems
        if not orch.status.initialized:
            await orch._async_init_subsystems()

        if orch.status.running:
            logger.warning("Orchestrator already running")
            return True

        logger.info("Starting orchestrator (Async Mode)...")
        try:
            orch.status.running = True
            orch.status.start_time = time.time()

            # Initialize async threading and sensory systems
            if hasattr(orch, '_async_init_threading'):
                orch._async_init_threading()
            if hasattr(orch, '_start_sensory_systems'):
                await orch._start_sensory_systems()
            if hasattr(orch, 'belief_sync') and orch.belief_sync:
                await orch.belief_sync.start()
            if hasattr(orch, 'attention_summarizer') and orch.attention_summarizer:
                await orch.attention_summarizer.start()
            if hasattr(orch, 'probe_manager') and orch.probe_manager:
                tracker.create_task(
                    orch.probe_manager.auto_cleanup_loop(),
                    name="lifecycle.probe_auto_cleanup",
                )

            # Start Dreaming System: Semantic Defragmentation & DLQ Recycling
            try:
                from core.memory.semantic_defrag import SemanticDefragmenter
                from core.resilience.dream_cycle import DreamCycle

                # Semantic Defrag (Vector Memory Consolidation)
                orch.semantic_defrag = SemanticDefragmenter()
                orch.semantic_defrag.start()

                # Dream Cycle (DLQ Ingestion)
                dlq_path = config.config.paths.data_dir / "dlq.jsonl"
                orch.dream_cycle = DreamCycle(orch, dlq_path)
                orch.dream_cycle.start()

                logger.info("💤 Dreaming Systems Active (Semantic Defrag & DLQ Recycle)")
            except Exception as e:
                record_degradation('lifecycle_coordinator', e)
                logger.error("Failed to start Dreaming Systems: %s", e)

            # Loading Self Model
            if orch.self_model:
                try:
                    from core.self_model import SelfModel
                    loaded = await SelfModel.load()
                    orch.self_model.beliefs = loaded.beliefs
                    logger.info("✓ Self-Model persistent state loaded.")
                except Exception as e:
                    record_degradation('lifecycle_coordinator', e)
                    logger.error("Failed to load Self-Model state: %s", e)

            # Start Lazarus Brainstem (v11.0)
            try:
                from core.brain.llm.lazarus_brainstem import LazarusBrainstem
                orch.brainstem = LazarusBrainstem(orch)
                logger.info("✓ Lazarus Brainstem active")
            except Exception as e:
                record_degradation('lifecycle_coordinator', e)
                logger.error("Failed to init Lazarus: %s", e)
                orch.brainstem = None

            # Connect physical hardware devices
            if getattr(orch, "hardware_manager", None):
                await orch.hardware_manager.start()
                logger.info("✓ Hardware Manager online")

            # ── Boot Barrier (v50) ─────────────────────────────────────
            # Ensure all core services are instantiated BEFORE any background
            # loops start. Without this, services race to access dependencies
            # that haven't been created yet, causing NEVER_SEEN errors.
            await self._boot_barrier()

            # Start Background Loops
            if hasattr(orch, 'consciousness') and orch.consciousness:
                if hasattr(orch.consciousness, 'start'):
                    res = orch.consciousness.start()
                    if res and hasattr(res, '__await__'): await res
            if hasattr(orch, 'curiosity') and orch.curiosity:
                if hasattr(orch.curiosity, 'start'):
                    res = orch.curiosity.start()
                    if res and hasattr(res, '__await__'): await res
                logger.info("✓ Curiosity background loop started")
            # Start Proactive Communication (v4.3)
            if hasattr(orch, 'proactive_comm') and orch.proactive_comm:
                if hasattr(orch.proactive_comm, 'start'):
                    res = orch.proactive_comm.start()
                    if res and hasattr(res, '__await__'): await res
                logger.info("✓ Proactive Communication loop started")
            # Start Narrative Engine (v11.0)
            if hasattr(orch, 'narrative_engine') and orch.narrative_engine:
                await orch.narrative_engine.start()
            # Start Global Workspace Loop
            if hasattr(orch, 'global_workspace') and orch.global_workspace:
                tracker.create_task(
                    orch.global_workspace.run_loop(),
                    name="lifecycle.global_workspace",
                )
                logger.info("✓ Global Workspace Attention Loop started")
            # Start Sovereign Ears
            if orch.ears:
                if hasattr(orch.ears, "should_auto_listen") and orch.ears.should_auto_listen():
                    def _hear_callback(text):
                        logger.info("👂 Heard: %s", text)
                        if hasattr(orch, 'process_user_input'):
                             orch.process_user_input(f"[VOICE] {text}")
                    await orch.ears.start_listening(_hear_callback)
                    logger.info("✓ Sovereign Ears listening")
                else:
                    logger.info("✓ Sovereign Ears standing by (mic idle until explicitly enabled)")
            # Start Sensory Instincts (v11.0)
            if hasattr(orch, 'instincts') and orch.instincts:
                await orch.instincts.start()
            # Start Pulse Manager (Proactive Awareness)
            if orch.pulse_manager:
                await orch.pulse_manager.start()
                logger.info("✓ Pulse Manager active (Proactive Awareness)")
            # Start Inter-process Event Listeners (H-12)
            tracker.create_task(
                orch._setup_event_listeners(),
                name="lifecycle.event_listeners",
            )
            # Start Cognitive Integration Layer
            if hasattr(orch, 'cognition') and orch.cognition:
                if hasattr(orch.cognition, 'initialize'):
                    res = orch.cognition.initialize()
                    if res and hasattr(res, '__await__'): await res
                logger.info("✓ Advanced Cognitive Layer (Learning, Memory, Beliefs) initialized")
            # Start Phase 5: Autonomic Core heartbeat
            if hasattr(orch, 'autonomic_core') and orch.autonomic_core:
                await orch.autonomic_core.start()

            logger.info("✓ Orchestrator started")
            return True
        except Exception as e:
            record_degradation('lifecycle_coordinator', e)
            logger.error("Failed to start orchestrator: %s", e)
            orch.status.running = False
            return False

    async def _boot_barrier(self):
        """Deterministic boot barrier: instantiate all core services before
        any background loops start.

        The ServiceContainer uses lazy factories — services are only created
        on first .get() call. Without this barrier, background loops would
        race to access services that don't exist yet, causing NEVER_SEEN
        heartbeat errors and potential crashes.

        This barrier forces all critical services to instantiate NOW.
        """
        from core.container import ServiceContainer

        REQUIRED_SERVICES = [
            "subsystem_audit",
            "event_bus",
            "mycelial_network",
            "llm_router",
            "autonomic_core",
            "snap_kv_evictor",
            "dual_memory",
            "inhibition_manager",
        ]

        ready = 0
        for name in REQUIRED_SERVICES:
            try:
                svc = ServiceContainer.get(name, default=None)
                if svc is not None:
                    ready += 1
                else:
                    logger.debug("Boot barrier: %s not yet registered (optional).", name)
            except Exception as e:
                record_degradation('lifecycle_coordinator', e)
                logger.debug("Boot barrier: %s failed to instantiate: %s", name, e)

        logger.info("Boot barrier: %d/%d core services ready. Background loops may proceed.",
                     ready, len(REQUIRED_SERVICES))

    async def stop(self):
        """Signal the orchestrator to stop gracefully."""
        orch = self.orch
        logger.info("🛑 Orchestrator stop requested.")

        # Save continuity record (session count, uptime, last exchange)
        try:
            from core.continuity import get_continuity
            get_continuity().save(reason="graceful")
            logger.info("✓ Continuity record saved")
        except Exception as e:
            record_degradation('lifecycle_coordinator', e)
            logger.debug("Continuity save failed: %s", e)

        # Persist epistemic state (knowledge graph, gaps)
        try:
            from core.epistemic_tracker import get_epistemic_tracker
            get_epistemic_tracker().save()
            logger.info("✓ Epistemic state saved")
        except Exception as e:
            record_degradation('lifecycle_coordinator', e)
            logger.debug("Epistemic save failed: %s", e)

        try:
            orch._save_state("shutdown")
        except Exception as e:
            record_degradation('lifecycle_coordinator', e)
            logger.debug("Final state save failed: %s", e)

        orch._publish_status({"event": "stopping", "message": "Graceful shutdown initiated"})

        if hasattr(orch, '_stop_event') and orch._stop_event:
            orch._stop_event.set()

        if hasattr(orch, 'consciousness') and orch.consciousness:
            orch.consciousness.stop()
        if hasattr(orch, 'conversation_loop') and orch.conversation_loop:
            orch.conversation_loop.stop()
        if hasattr(orch, 'status') and orch.status:
            orch.status.running = False

        # Gracefully shutdown all services (including PhantomBrowser in skills)
        try:
            if hasattr(orch, 'semantic_defrag') and orch.semantic_defrag:
                orch.semantic_defrag.stop()
            if hasattr(orch, 'dream_cycle') and orch.dream_cycle:
                orch.dream_cycle.stop()
            if getattr(orch, "hardware_manager", None):
                await orch.hardware_manager.stop()
            from core.container import ServiceContainer
            await ServiceContainer.shutdown()
        except Exception as e:
            record_degradation('lifecycle_coordinator', e)
            logger.error("Error during ServiceContainer shutdown: %s", e)

        # Stop System Watchdog
        if hasattr(orch, '_watchdog') and orch._watchdog:
            orch._watchdog.stop()

        # Bulletproof Shutdown via TaskTracker
        await get_task_tracker().shutdown(timeout=3.0)

        orch._publish_status({"event": "stopped", "message": "Orchestrator offline"})
        logger.info("✅ Orchestrator stopped.")

    async def retry_cognitive_connection(self) -> bool:
        """Manually retry connecting to the cognitive brain (LLM). Forces full re-wire."""
        orch = self.orch
        logger.info("🧠 Manual Cognitive Retry Initiated...")
        try:
            from core.brain.cognitive_engine import CognitiveEngine
            ce = orch.cognitive_engine
            if ce is None:
                ce = CognitiveEngine()
            # Force re-wire with capability engine
            try:
                from core.container import get_container
                container = get_container()
                engine = container.get("capability_engine", None)
                logger.info("🔄 Re-wiring cognitive engine...")
                ce.wire(engine, router=engine)
            except Exception as e:
                record_degradation('lifecycle_coordinator', e)
                logger.error("Re-wire failed: %s", e)
            
            # Check result
            if not getattr(ce, 'lobotomized', True):
                from core.container import ServiceContainer
                ServiceContainer.register_instance("cognitive_engine", ce)
                logger.info("✅ Cognitive Engine ONLINE — Safe Mode deactivated")
                try:
                    from .thought_stream import get_emitter
                    get_emitter().emit("System", "Cognitive Connection Re-established", level="success")
                except Exception as e:
                    record_degradation('lifecycle_coordinator', e)
                    logger.debug("ThoughtStream emit failed during cognitive retry: %s", e)
                return True
            else:
                logger.error("❌ Cognitive Retry Failed: Engine still lobotomized after re-wire")
                logger.error("  client=%s, autonomous_brain=%s", getattr(ce, 'client', None), getattr(ce, 'autonomous_brain', None))
                return False
        except Exception as e:
            record_degradation('lifecycle_coordinator', e)
            logger.error("Cognitive Retry Exception: %s", e)
            return False

    def handle_signal(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        orch = self.orch
        logger.info("Received signal %s. Shutting down gracefully...", signum)
        try:
            asyncio.get_running_loop()
            get_task_tracker().create_task(
                self.stop(),
                name=f"lifecycle.signal_stop.{signum}",
            )
        except RuntimeError:
            if hasattr(orch, '_stop_event') and orch._stop_event:
                orch._stop_event.set()
            if hasattr(orch, 'status') and orch.status:
                orch.status.running = False
