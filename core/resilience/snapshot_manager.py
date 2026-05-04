"""core/resilience/snapshot_manager.py

Snapshot Manager for Architectural Resilience
============================================

Handles freezing and thawing the entire cognitive, emotional, and 
phenomenal state of Aura to disk. This mitigates the "House of Cards"
risk where a crash loses the continuous stream of consciousness.

Features:
- Versioned JSON serialization
- Atomic writes
- Coordinates extracting state from deeply entangled subsystems
"""

from core.runtime.errors import record_degradation
import json
import logging
import os
import time
from typing import Any

from core.config import config

logger = logging.getLogger("Aura.SnapshotManager")

class SnapshotManager:
    """Coordinates freezing and thawing cognitive state."""
    
    VERSION = "1.0"

    def __init__(self, orchestrator=None):
        self._orch = orchestrator
        self.snapshot_dir = config.paths.data_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.snapshot_dir / "latest.aura_snapshot"
        logger.info("📸 Cognitive Snapshot Manager ONLINE")

    def freeze(self) -> bool:
        """Freeze current state to disk."""
        if not self._orch:
            logger.error("SnapshotManager requires orchestrator reference to freeze state.")
            return False

        logger.info("🥶 Freezing cognitive state...")
        
        try:
            state = {
                "version": self.VERSION,
                "timestamp": time.time(),
                "subsystems": {}
            }

            # 1. Qualia State
            qualia = getattr(self._orch, "qualia", None)
            if qualia and hasattr(qualia, "get_state"):
                logger.debug("Freezing Qualia Synthesizer...")
                state["subsystems"]["qualia"] = qualia.get_state()

            # 2. Affect State
            affect = getattr(self._orch, "affect", None)
            if affect:
                logger.debug("Freezing Affect Engine...")
                if hasattr(affect, "get_snapshot"):
                    state["subsystems"]["affect"] = affect.get_snapshot()
                elif hasattr(affect, "get"):
                    # Fallback for older versions if they were sync
                    res = affect.get()
                    import inspect
                    if not inspect.isawaitable(res):
                        state["subsystems"]["affect"] = res

            # 3. Conversation Context (Short-term memory)
            engine = getattr(self._orch, "conversation_engine", None)
            if engine and hasattr(engine, "get_context"):
                # Grab the default text chat and voice chat contexts
                ctx = engine.get_context("default")
                voice_ctx = engine.get_context("voice")
                
                state["subsystems"]["conversation"] = {
                    "default_emotional_state": ctx.emotional_state.value if hasattr(ctx, 'emotional_state') else "neutral",
                    "default_mode": ctx.mode.value if hasattr(ctx, 'mode') else "chat",
                    "voice_emotional_state": voice_ctx.emotional_state.value if hasattr(voice_ctx, 'emotional_state') else "neutral",
                    "voice_mode": voice_ctx.mode.value if hasattr(voice_ctx, 'mode') else "chat"
                }

            # 4. Orchestrator Volatiles
            state["subsystems"]["orchestrator"] = {
                "current_mode": getattr(self._orch, "current_mode", "autonomous"),
                "cognitive_load": getattr(self._orch, "cognitive_load", 0.0)
            }

            # Write atomically
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            temp_file = str(self.snapshot_file) + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_file, self.snapshot_file)
            
            logger.info("✅ Cognitive state frozen to disk: %s", self.snapshot_file)
            return True

        except Exception as e:
            record_degradation('snapshot_manager', e)
            logger.error("Failed to freeze state: %s", e, exc_info=True)
            return False

    def _governance_approve_thaw(self) -> bool:
        """Request governance approval before restoring state from disk.

        State restoration is a significant mutation — it replaces the current
        cognitive/affective state with values from a file. This must be
        authorized through the same governance path as any other state change.

        Returns True if approved, False if denied. If governance is unavailable
        (early boot before Will is initialized), logs a warning and permits
        the thaw with an explicit audit note — the system needs its state to
        function, and early boot is a known governance gap.
        """
        try:
            from core.will import get_will, ActionDomain
            will = get_will()
            if will is None:
                logger.warning(
                    "SnapshotManager: Will unavailable during thaw (early boot). "
                    "Permitting with audit note."
                )
                return True

            decision = will.decide(
                content="snapshot_thaw",
                source="snapshot_manager",
                domain=ActionDomain.STATE_MUTATION,
                context={
                    "file": str(self.snapshot_file),
                    "operation": "snapshot_thaw",
                },
                priority=0.9,  # High priority — system needs its state
            )
            approved = decision.is_approved() if hasattr(decision, "is_approved") else True
            if not approved:
                logger.warning(
                    "SnapshotManager: Governance DENIED state thaw: %s",
                    getattr(decision, "reason", "unknown"),
                )
            return approved
        except Exception as exc:
            record_degradation('snapshot_manager', exc)
            # During early boot, governance may not be available.
            # Log and permit — the system needs state to function.
            logger.warning(
                "SnapshotManager: Governance check failed during thaw (%s). "
                "Permitting with audit note (early boot assumption).",
                exc,
            )
            return True

    def thaw(self) -> bool:
        """Thaw state from disk and inject into running subsystems.

        GOVERNANCE: Requests Will approval before mutating live state.
        During early boot (before Will is initialized), thaw is permitted
        with an explicit audit warning — this is a known governance gap
        that closes once the kernel finishes initialization.
        """
        if not self._orch:
            logger.error("SnapshotManager requires orchestrator reference to thaw state.")
            return False

        if not self.snapshot_file.exists():
            logger.info("No snapshot found. Starting fresh.")
            return False

        # Governance gate: request approval before mutating state
        if not self._governance_approve_thaw():
            logger.warning("SnapshotManager: State thaw BLOCKED by governance.")
            return False

        logger.info("🔥 Thawing cognitive state from disk...")
        
        try:
            with open(self.snapshot_file, "r") as f:
                state = json.load(f)

            if state.get("version") != self.VERSION:
                logger.warning("Snapshot version mismatch (%s != %s). Discarding.", 
                               state.get("version"), self.VERSION)
                return False

            subsystems = state.get("subsystems", {})

            # 1. Qualia State
            qualia = getattr(self._orch, "qualia", None)
            q_state = subsystems.get("qualia")
            if qualia and q_state:
                logger.debug("Thawing Qualia Synthesizer...")
                # Restore the last known qualia vector and history
                if hasattr(qualia, "q_vector") and "q_vector" in q_state:
                    qualia.q_vector = q_state["q_vector"]
                if hasattr(qualia, "history") and "history" in q_state:
                    qualia.history = q_state["history"]

            # 2. Affect State
            affect = getattr(self._orch, "affect", None)
            a_state = subsystems.get("affect")
            if affect and a_state:
                logger.debug("Thawing Affect Engine...")
                # AffectEngineV2 restoration
                if hasattr(affect, "markers"):
                    markers = affect.markers
                    if "emotions" in a_state:
                        markers.emotions.update(a_state["emotions"])
                    if "physiology" in a_state:
                        p = a_state["physiology"]
                        markers.heart_rate = p.get("heart_rate", markers.heart_rate)
                        markers.gsr = p.get("gsr", markers.gsr)
                        markers.cortisol = p.get("cortisol", markers.cortisol)
                        markers.adrenaline = p.get("adrenaline", markers.adrenaline)
                    if "mood_baselines" in a_state:
                        markers.mood_baselines.update(a_state["mood_baselines"])
                else:
                    # Legacy fallback
                    if hasattr(affect, "valence") and "valence" in a_state:
                        affect.valence = a_state["valence"]
                    if hasattr(affect, "arousal") and "arousal" in a_state:
                        affect.arousal = a_state["arousal"]

            # 3. Conversation Context
            engine = getattr(self._orch, "conversation_engine", None)
            c_state = subsystems.get("conversation")
            if engine and c_state:
                logger.debug("Thawing Conversation context...")
                try:
                    from core.conversation.engine import EmotionalState, ConversationMode
                    ctx = engine.get_context("default")
                    voice_ctx = engine.get_context("voice")
                    
                    if "default_emotional_state" in c_state:
                        ctx.emotional_state = EmotionalState(c_state["default_emotional_state"])
                    if "default_mode" in c_state:
                        ctx.mode = ConversationMode(c_state["default_mode"])
                        
                    if "voice_emotional_state" in c_state:
                        voice_ctx.emotional_state = EmotionalState(c_state["voice_emotional_state"])
                    if "voice_mode" in c_state:
                        voice_ctx.mode = ConversationMode(c_state["voice_mode"])
                except Exception as e:
                    record_degradation('snapshot_manager', e)
                    logger.debug("Could not fully restore conversation enums: %s", e)

            # 4. Orchestrator Volatiles
            o_state = subsystems.get("orchestrator")
            if o_state:
                if "current_mode" in o_state:
                    self._orch.current_mode = o_state["current_mode"]
                if "cognitive_load" in o_state:
                    self._orch.cognitive_load = o_state["cognitive_load"]

            logger.info("✅ Cognitive state thawed successfully.")
            return True

        except Exception as e:
            record_degradation('snapshot_manager', e)
            logger.error("Failed to thaw state: %s", e, exc_info=True)
            return False
