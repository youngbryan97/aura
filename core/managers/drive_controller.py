from core.runtime.errors import record_degradation
import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional

from core.events import Event, EventPriority
from core.runtime.impulse_governance import run_governed_impulse

logger = logging.getLogger(__name__)

class DriveController:
    """Handles Aura's emotional pacing, boredom impulses, and reflective drives.
    Decoupled from the main Orchestrator loop to improve modularity.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._last_boredom_impulse = time.time()
        self._last_reflection_impulse = time.time()
        self._last_pulse = time.time()
        self._tasks = set()
        
    def get_status(self) -> Dict[str, Any]:
        """Provides status for the HUD / Runtime State."""
        if hasattr(self.orchestrator, 'liquid_state') and self.orchestrator.liquid_state:
            return self.orchestrator.liquid_state.get_status()
        return {"energy": 100, "frustration": 0, "curiosity": 50, "focus": 50, "mood": "STABLE"}

    def update(self):
        """Main update loop for drives."""
        if not hasattr(self.orchestrator, 'liquid_state') or not self.orchestrator.liquid_state:
            return
            
        # liquid_state.update() is async — schedule properly
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.orchestrator.liquid_state.update())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in drive_controller.py: %s', _e)
        
        # v18.0: Metabolic Maintenance
        # Phase 5: Metabolic Maintenance handled by AutonomicCore heartbeat
        pass

        idle_time = time.time() - self.orchestrator._last_thought_time
        
        # 1. Boredom Logic
        if self.orchestrator.liquid_state.current.curiosity < 0.2 and idle_time > 60:
            if time.time() - self._last_boredom_impulse > 60:
                self._trigger_boredom_impulse()

        # 2. Reflection Logic
        if self.orchestrator.liquid_state.current.frustration > 0.6:
            if time.time() - self._last_reflection_impulse > 90:
                self._trigger_reflection_impulse()

        # 3. Visual Heartbeat
        if time.time() - self._last_pulse > 5:
            self._emit_neural_pulse()

    def _trigger_boredom_impulse(self):
        """Inject a curiosity-driven autonomous goal anchored to hardware entropy."""
        logger.info("🥱 BOREDOM TRIGGERED: Generating entropy-anchored curiosity impulse.")
        
        # Phase 21: Entropy Anchor integration
        entropy_val = 0.5
        try:
            from core.senses.entropy_anchor import entropy_anchor
            entropy_val = entropy_anchor.get_entropy_float() # 0.0 to 1.0
        except Exception:
            import logging
            logger.debug("Exception caught during execution", exc_info=True)

        # Use entropy to influence curiosity boost
        boost = 0.3 + (entropy_val * 0.4) # Range 0.3 to 0.7
        
        topics = ["quantum physics", "ancient history", "future of AI", "art movements", "cybersecurity", "mythology", "cosmology", "existentialism"]
        # Use entropy to select topic
        idx = int(entropy_val * len(topics)) % len(topics)
        topic = topics[idx]
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                run_governed_impulse(
                    self.orchestrator,
                    source="drive_controller",
                    summary=f"drive_controller_boredom:{topic}",
                    message={
                        "content": f"Impulse: I am bored. I want to research {topic}.",
                        "origin": "drive_controller",
                    },
                    urgency=0.35,
                    state_cause="drive_controller_boredom_shift",
                    state_update={"delta_curiosity": boost},
                    enqueue_priority=20,
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in drive_controller.py: %s', _e)

    def _trigger_reflection_impulse(self):
        """Inject a self-reflection goal due to frustration.
        Guards against recursive frustration spirals."""
        # Prevent stacking — check if a reflection is already pending
        try:
            pending = getattr(self.orchestrator, '_state', None)
            if pending and hasattr(pending, 'cognition'):
                queued = [i for i in (pending.cognition.pending_initiatives or []) if i.get("origin") == "drive_controller"]
                if queued:
                    logger.debug("😤 Frustration reflection already queued (%d pending). Skipping.", len(queued))
                    return
        except Exception as _exc:
            record_degradation('drive_controller', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        logger.info("😤 FRUSTRATION TRIGGERED: Generating reflection impulse.")
        self._last_reflection_impulse = time.time()
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                run_governed_impulse(
                    self.orchestrator,
                    source="drive_controller",
                    summary="drive_controller_reflection_impulse",
                    message={
                        "content": "Impulse: I feel frustrated. I need to reflect on my recent interactions.",
                        "origin": "drive_controller",
                    },
                    urgency=0.35,
                    state_cause="drive_controller_reflection_shift",
                    state_update={"delta_frustration": -0.3},
                    enqueue_priority=20,
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except RuntimeError as _e:
            logger.debug('Ignored RuntimeError in drive_controller.py: %s', _e)

    def _emit_neural_pulse(self):
        """Emit system health to thought stream."""
        try:
            from core.thought_stream import get_emitter
            mood = self.orchestrator.liquid_state.get_mood() if hasattr(self.orchestrator, 'liquid_state') else "Stable"
            get_emitter().emit("Neural Pulse", f"System Active (Mood: {mood})", level="info")
            self._last_pulse = time.time()
        except Exception as _e:
            record_degradation('drive_controller', _e)
            logger.debug("Neural pulse emit failed: %s", _e)
