"""core/consciousness/subconscious_loop.py

Continuous Background Processing (The Default Mode Network).
Runs endlessly when Aura is idle, generating insights, reviewing logs,
and experimenting in the sandbox.
"""

import asyncio
import logging
import time
from typing import Optional, Any
from core.container import ServiceContainer
from core.runtime.background_policy import background_activity_allowed

logger = logging.getLogger("Aura.Subconscious")

class SubconsciousLoop:
    """The deeply autonomous background process for Aura."""
    
    def __init__(self, orchestrator, idle_threshold: float = 60.0):
        self.orchestrator = orchestrator
        self.idle_threshold = idle_threshold  # Seconds of silence before activating
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        self.last_dream_cycle = 0.0
        self.last_sandbox_experiment = 0.0

    async def start(self):
        """Ignite the subconscious daemon."""
        if self._running:
            return
        self._running = True
        try:
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(
                self._run_loop(),
                name="aura.subconscious_loop",
            )
        except Exception:
            self._task = asyncio.create_task(self._run_loop(), name="aura.subconscious_loop")
        logger.info("🧠 Subconscious Loop activated")

    async def stop(self):
        """Suspend subconscious activity."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in subconscious_loop.py: %s', _e)
        logger.info("🛑 Subconscious Loop halted")

    async def _run_loop(self):
        """The main continuous loop running in the background."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Wake up briefly to check state
                
                # Are we idle?
                last_user = float(getattr(self.orchestrator, "_last_user_interaction_time", 0.0) or 0.0)
                time_since_last = (time.time() - last_user) if last_user > 0.0 else 0.0
                if getattr(self.orchestrator.status, "is_processing", False):
                    continue  # The active mind is busy, stay quiet
                    
                if time_since_last > self.idle_threshold and background_activity_allowed(
                    self.orchestrator,
                    min_idle_seconds=self.idle_threshold,
                    max_memory_percent=80.0,
                    max_failure_pressure=0.1,
                    require_conversation_ready=True,
                ):
                    await self._perform_subconscious_beat()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Subconscious loop fault: {e}")

    async def _perform_subconscious_beat(self):
        """Execute one cycle of background processing."""
        now = time.time()
        
        # 1. Identity/Memory Consolidation (Dreaming)
        if now - self.last_dream_cycle > 300.0:  # Every 5 mins of idle time
            try:
                dreaming = ServiceContainer.get("dreaming_process", default=None)
                if dreaming and hasattr(dreaming, "dream"):
                    logger.info("💭 Subconscious triggering Dream Cycle...")
                    await dreaming.dream()
                    self.last_dream_cycle = now
            except Exception as e:
                logger.debug(f"Subconscious dreaming failed: {e}")
                self.last_dream_cycle = now + 60.0  # Back off a bit

        # 2. Epistemic Foraging / Hypothesis Testing (Sandbox)
        # Bypasses the user to run code autonomously
        if now - self.last_sandbox_experiment > 900.0:  # Every 15 minutes of idle
            try:
                logger.info("🧪 Subconscious running proactive sandbox experiment...")
                await self._run_proactive_sandbox()
                self.last_sandbox_experiment = now
            except Exception as e:
                logger.debug(f"Subconscious sandbox beat failed: {e}")
                self.last_sandbox_experiment = now + 120.0

    async def _run_proactive_sandbox(self):
        """Use the ToolOrchestrator to verify an assumption or explore."""
        tool_orch = ServiceContainer.get("tool_orchestrator", default=None)
        if not tool_orch:
            return
            
        script = """
# Autonomous Subconscious Check
try:
    import platform
    print(f"Subconscious ping: Running on {platform.system()} {platform.release()}")
except Exception as e:
    print(f"Subconscious error: {e}")
"""
        handle = None
        constitutional_core = None
        error_text = None
        success = False
        result = ""
        started = time.perf_counter()
        try:
            from core.constitution import get_constitutional_core
            from core.health.degraded_events import record_degraded_event

            constitutional_core = get_constitutional_core(self.orchestrator)
            handle = await constitutional_core.begin_tool_execution(
                "subconscious_sandbox_probe",
                {"purpose": "idle_probe"},
                source="subconscious_loop",
                objective="Idle subconscious sandbox experiment",
            )
            if not handle.approved:
                record_degraded_event(
                    "subconscious_loop",
                    "sandbox_probe_blocked",
                    detail="idle_probe",
                    severity="warning",
                    classification="background_degraded",
                    context={"reason": handle.decision.reason},
                )
                return
        except Exception as e:
            logger.debug("Subconscious constitutional gate failed: %s", e)

        try:
            success, result = await tool_orch.execute_python(script)
            if not success:
                error_text = str(result or "sandbox_failed")
            if success:
                # --- PHASE 23.1: CRYPTOLALIA TRANSMISSION ---
                bridge = ServiceContainer.get("concept_bridge", default=None)
                decoder = ServiceContainer.get("cryptolalia_decoder", default=None)
                
                if bridge and decoder:
                    # Subconscious encodes the insight as a raw vector (mocking here for sandbox result)
                    vector_insight = await bridge.generate_concept_vector(result)
                    
                    # Transmits purely in Latent Space to the Epistemic Tracker
                    thought_id = await bridge.transmit(
                        source="SubconsciousLoop", 
                        target="EpistemicTracker", 
                        semantic_vector=vector_insight
                    )
                    
                    # --- PHASE 23.2: DECODER OBSERVER ---
                    # The human sees the approximated translation, not the 768-dim float array
                    translation = decoder.approximate_translation(vector_insight)
                    logger.info(f"🌌 [Latent Transmission {thought_id}] {translation}")
                else:
                    logger.debug(f"Subconscious Sandbox Result: {result}")
        except Exception as e:
            logger.debug(f"Subconscious sandbox run failed: {e}")
            error_text = f"{type(e).__name__}: {e}"
        finally:
            if handle is not None and constitutional_core is not None:
                try:
                    duration_ms = (time.perf_counter() - started) * 1000.0
                    await constitutional_core.finish_tool_execution(
                        handle,
                        result=str(result or error_text or "")[:1000],
                        success=bool(success),
                        duration_ms=duration_ms,
                        error=error_text,
                    )
                except Exception as finish_exc:
                    logger.debug("Subconscious sandbox finish skipped: %s", finish_exc)

def register_subconscious_loop(orchestrator):
    loop = SubconsciousLoop(orchestrator)
    ServiceContainer.register_instance("subconscious_loop", loop)
    return loop
