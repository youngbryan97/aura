"""Aura Zenith Cognitive Loop Service.
Decouples the cognitive cycle from the Orchestrator.
"""
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from typing import Any, Optional

from core.runtime.service_access import (
    optional_service,
    resolve_affect_engine,
    resolve_belief_graph,
    resolve_curiosity_engine,
    resolve_identity_model,
    resolve_motivation_engine,
    resolve_self_prediction,
)
from core.utils.queues import unpack_priority_message

logger = logging.getLogger("Aura.CognitiveLoop")

class CognitiveLoop:
    """Service responsible for the cognitive thinking cycle.
    """
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.cycle_count = 0
        self.last_cycle_time = time.time()
        self.stall_threshold = 30.0  # Seconds

    async def start(self):
        """Start the cognitive cycle."""
        if self.is_running:
            return
        self.is_running = True
        self._task = get_task_tracker().create_task(self.run())
        logger.info("🧠 Cognitive Loop service started.")

    async def stop(self):
        """Stop the cognitive cycle."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug('Ignored asyncio.CancelledError in cognitive_loop.py: %s', _e)
        logger.info("🧠 Cognitive Loop service stopped.")

    async def run(self):
        """Main cognitive cycle."""
        while self.is_running:
            try:
                start_time = time.monotonic()
                # v27 Hardening: Explicit 5-minute timeout for infinite loops
                async with asyncio.timeout(300):
                    await self._process_cycle()
                self.last_cycle_time = time.monotonic()
                
                # Dynamic pacing based on load
                await asyncio.sleep(0.05)
                
                # Watchdog Check
                if time.monotonic() - self.last_cycle_time > self.stall_threshold:
                    logger.critical(f"🚨 COGNITIVE STALL: No cycle for {self.stall_threshold}s. Recovering...")
                    await self._recover_from_stall()
            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('cognitive_loop', e)
                logger.error(f"Error in Cognitive Loop cycle: {e}", exc_info=True)
                await asyncio.sleep(1.0) # Error backoff

    async def _process_cycle(self):
        """Single cognitive cycle iteration."""
        self.cycle_count += 1
        if hasattr(self.orchestrator, "status"):
            self.orchestrator.status.cycle_count = self.cycle_count

        # 1. Free Energy Heartbeat (Metabolic Metric)
        await self._tick()
        
        # Unified Cognitive Pipeline: Health Check every 20 cycles
        if self.cycle_count % 20 == 0:
            await self._check_coordinators_health()

        # 2. Message Acquisition (Priority Weighted)
        message = await self._acquire_next_message()
        if message:
            await self._dispatch_message(message)
            
        # 3. Autonomous Thinking (Legacy fallthrough)
        await self._trigger_autonomous_thought(bool(message))
        
        # 4. Health & Heartbeat via Orchestrator
        if hasattr(self.orchestrator, "_update_heartbeat"):
            self.orchestrator._update_heartbeat()

    async def _tick(self):
        """
        The real cognitive heartbeat. Every cycle, Aura:
        1. Computes her free energy (replaces phi)
        2. Derives her emotional state from it
        3. Decides what autonomous action (if any) to take
        4. Updates her self-model
        """
        from core.consciousness.free_energy import get_free_energy_engine

        fe_engine = get_free_energy_engine()
        spl = resolve_self_prediction(default=None)
        belief_system = resolve_belief_graph(default=None)
        affect = resolve_affect_engine(default=None)

        # 1. Get prediction error from SelfPredictionLoop
        prediction_error = spl.get_surprise_signal() if spl and hasattr(spl, 'get_surprise_signal') else 0.2

        # 2. Compute free energy
        recent_actions = getattr(self.orchestrator, '_recent_action_count', 0)
        user_interaction_gap = time.time() - getattr(self.orchestrator, '_last_user_interaction_time', 0)
        user_present = user_interaction_gap < 30

        fe_state = fe_engine.compute(
            prediction_error=prediction_error,
            belief_system=belief_system,
            recent_action_count=recent_actions,
            user_present=user_present,
        )

        # 3. Push emotional state to affect engine (grounded in actual internal state)
        if affect and hasattr(affect, 'update_from_free_energy'):
            affect.update_from_free_energy(fe_state.valence, fe_state.arousal)
        elif affect and hasattr(affect, 'update_emotion'):
            # Fallback for AffectEngineV2
            affect.update_emotion("free_energy", fe_state.valence, fe_state.arousal)

        # 4. Autonomous action — driven by FE (Active Inference), not a timer
        if not user_present and fe_state.dominant_action in ("act_on_world", "explore", "update_beliefs"):
            await self._autonomous_action(fe_state)

        # 5. Update self-model with real internal telemetry
        self._update_self_telemetry(fe_state, prediction_error)

    async def _autonomous_action(self, fe_state):
        """
        Aura acts because her free energy demands it, not because a timer fired.
        """
        spl = resolve_self_prediction(default=None)
        motivation = resolve_motivation_engine(default=None)

        if fe_state.dominant_action == "explore":
            # Find the most unpredictable dimension and explore it
            if spl and hasattr(spl, 'get_most_unpredictable_dimension'):
                unpredictable_dim = spl.get_most_unpredictable_dimension()
                curiosity = resolve_curiosity_engine(default=None)
                if curiosity and hasattr(curiosity, 'explore_dimension'):
                    await curiosity.explore_dimension(unpredictable_dim)

        elif fe_state.dominant_action == "act_on_world":
            # High FE + inaction = generate an intention from the belief system
            if motivation and hasattr(motivation, 'pulse'):
                intention = await motivation.pulse()
                if intention and hasattr(motivation, '_dispatch_intention'):
                    await motivation._dispatch_intention(intention)

    def _update_self_telemetry(self, fe_state, prediction_error: float):
        """Update Aura's self-model with ACTUAL internal readings."""
        self_model = resolve_identity_model(default=None)
        if not self_model:
            return

        telemetry = {
            "free_energy": fe_state.free_energy,
            "prediction_error": prediction_error,
            "emotional_valence": fe_state.valence,
            "emotional_arousal": fe_state.arousal,
            "processing_intent": fe_state.dominant_action,
            "is_distressed": fe_state.free_energy > 0.7,
            "timestamp": time.time(),
        }

        if hasattr(self_model, 'update_telemetry'):
            self_model.update_telemetry(telemetry)

    async def _acquire_next_message(self) -> Optional[dict]:
        """Get next message from orchestrator queue (PriorityQueue support)."""
        try:
            q = self.orchestrator.message_queue
            if q.empty():
                return None
            
            item = await q.get()
            payload, origin = unpack_priority_message(item)
            
            # Unpack the wrapped dict (content, origin)
            if isinstance(payload, dict) and "content" in payload:
                return payload
            elif isinstance(payload, str):
                return {"content": payload, "origin": origin or "unknown"}
            
            return {"content": str(payload), "origin": origin or "raw"}
        except Exception as e:
            record_degradation('cognitive_loop', e)
            logger.error("Failed to acquire message from queue: %s", e)
            return None

    async def _dispatch_message(self, message: dict):
        """Process incoming message using immediate priority bypass.
        Bypassing process_user_input prevents infinite recursion.
        """
        content = message.get("content", "")
        origin = message.get("origin", "user")

        if hasattr(self.orchestrator, "process_user_input_priority"):
             await self.orchestrator.process_user_input_priority(content, origin=origin)
        elif hasattr(self.orchestrator, "process_user_input"):
             # Legacy fallback (may still recurse if orchestrator doesn't have priority method)
             await self.orchestrator.process_user_input(content)

    async def _check_coordinators_health(self):
        """Audit critical coordinators and trigger restarts if needed."""
        # Coordinator names in container
        critical_components = {
            "agency_core": "Agency Core",
            "memory_coordinator": "Memory Coordinator",
            "affect_engine": "Affect Engine",
            "personality_engine": "Personality Engine"
        }
        
        for key, name in critical_components.items():
            comp = optional_service(key, default=None)
            if comp is None:
                logger.warning(f"🚨 [RECOVERY] Critical component '{name}' ({key}) missing from container. Triggering Orchestrator repair...")
                if hasattr(self.orchestrator, "retry_cognitive_connection"):
                    # This often fixes the container state
                    await self.orchestrator.retry_cognitive_connection()
                
    async def _trigger_autonomous_thought(self, was_responding: bool):
        """
        [DEPRECATED] v15: Autonomous thoughts are now driven by _tick's FE logic.
        Legacy boredom timers are disabled to prevent periodic 'cron' behavior.
        """
        logger.debug("🧠 CognitiveLoop: Legacy autonomous thought trigger ignored (deprecated).")

    async def _recover_from_stall(self):
        """Emergency recovery logic for cognitive stalls."""
        if hasattr(self.orchestrator, "_recover_from_stall"):
             await self.orchestrator._recover_from_stall()
        else:
             # Basic internal recovery
             logger.warning("Falling back to basic cognitive recovery.")
             self.cycle_count = 0
             self.last_cycle_time = time.time()
             # Maybe reset LLM circuit breakers
             if hasattr(self.orchestrator, "cognitive_engine"):
                  ce = self.orchestrator.cognitive_engine
                  if ce and hasattr(ce, "unload_models"):
                       try:
                           await ce.unload_models()
                       except Exception as e:
                           record_degradation('cognitive_loop', e)
                           logger.error(f"Failed to unload models during recovery: {e}")
