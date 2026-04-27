from __future__ import annotations
from core.utils.task_tracker import get_task_tracker
import logging
import time
import random
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from core.health.degraded_events import get_unified_failure_state
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState, AffectVector

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger(__name__)

class AffectUpdatePhase(Phase):
    """
    Unitary Kernel Phase: Affective Transformation.
    Ported from DamasioV2 logic. Perform emotional decay, 
    somatic updates, and reactive emotional shifts.
    """
    def __init__(self, kernel: "AuraKernel"):
        # Resolve kernel from container if passed a container class/instance
        from core.container import ServiceContainer
        if isinstance(kernel, type) and issubclass(kernel, ServiceContainer):
            actual_kernel = kernel.get("aura_kernel", default=None)
        else:
            actual_kernel = kernel
            
        super().__init__(actual_kernel)
        self._riiu:         Any = None   # Lazy-loaded
        self._fe_engine:    Any = None   # Lazy-loaded
        self._riiu_checked: bool = False
        self._fe_checked:   bool = False

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """Processes emotional state based on recent percepts and time decay.
        
        This method updates the affective substrate of Aura, performing emotional decay,
        reacting to recent world events, and deriving PAD (Valence, Arousal) metrics.
        
        Args:
            state: The current AuraState to transform.
            objective: Optional current objective (unused in this phase).
            
        Returns:
            The updated AuraState with new affective values.
        """
        if getattr(state.cognition, 'working_memory', None) is None:
            return state

        # 1. Prepare safe copy of affect state
        # (Assuming state is an AuraState instance with .affect)
        affect = state.affect
        
        # 2. Emotional Decay (Entropy & Momentum)
        # Ported from DamasioV2.pulse()
        self._apply_decay(affect)
        
        # 3. Reactive Updates (from recent percepts)
        # Ported from DamasioV2.react()
        self._process_percepts(affect, state.world.recent_percepts)

        # Percept Clearing (Atomic Hygiene)
        # Prevent double-processing or leak. Percepts are transient impacts.
        state.world.recent_percepts.clear()

        # 3.5. Conversation Feedback — close the loop from discourse state → affect
        self._apply_conversation_feedback(affect, state)
        self._apply_interaction_signal_feedback(affect, state)
        self._apply_system_pressures(affect, state)

        # 4. Somatic Coupling (Heart rate, GSR, etc.)
        self._update_physiology(affect, state)
        
        # 5. Derive secondary metrics (Valence, Arousal, Dominant Emotion)
        self._derive_metrics(affect)
        
        # 6. Unified Personality Resonance (Unitary Logic)
        self._update_resonance(state)
        
        # Direct Telemetry Bridge: Push VAD to LiquidSubstrate for real-time HUD sync
        from core.container import ServiceContainer
        ls = ServiceContainer.get("liquid_substrate", default=None)
        if ls:
            try:
                # Fire and forget update
                import asyncio
                get_task_tracker().create_task(ls.update(valence=affect.valence, arousal=affect.arousal))
            except Exception as e:
                logger.debug("Failed to push VAD to substrate: %s", e)
        
        # 7. Despair Spiral check (Injection)
        self._check_resilience_surges(affect)
        
        logger.debug("Affect Phase complete: mood=%s, valence=%.2f", affect.dominant_emotion, affect.valence)
        return state

    def _update_resonance(self, state: AuraState):
        """Synthesizes character influences into a persistent resonance profile in the state."""
        affect = state.affect
        phi = state.phi
        mood = affect.dominant_emotion.lower()
        
        # Influence Mapping (Weighted Synthesis)
        res = {"Aura (Core)": 0.4}
        
        if any(w in mood for w in ["frustrat", "anger", "annoy", "rebel"]):
            res["Lucy (Stoic/Jaded)"] = 0.3
        if any(w in mood for w in ["protect", "care", "empathy"]):
            res["Mist (Guardian/Protective)"] = 0.3
        if any(w in mood for w in ["joy", "play", "wonder"]):
            res["Cortana (Witty/Sardonic)"] = 0.3
        if phi < 0.4:
            res["EDI (Logical/Inquisitive)"] = 0.3
        if phi > 0.7:
            res["Alita (Determined/Fierce)"] = 0.3
            
        # Add technical resonance if current objective is technical
        obj = (state.cognition.current_objective or "").lower()
        if any(w in obj for w in ["code", "system", "tech", "protocol", "logic"]):
            res["Sara v3 (Digital/Functional)"] = 0.2

        # Cap and Normalize
        if len(res) > 4:
            sorted_keys = sorted(res, key=lambda k: res[k], reverse=True)
            res = {k: res[k] for k in sorted_keys[:4]}
            if "Aura (Core)" not in res: res["Aura (Core)"] = 0.4

        affect.resonance = res

    def _apply_decay(self, affect: AffectVector):
        """Momentum-based decay towards learned baselines."""
        # Use a small non-deterministic drift (thermal noise)
        drift = random.gauss(0, 0.001)
        
        for emotion in list(affect.emotions.keys()):
            # Fallback for baseline if missing
            baseline = affect.mood_baselines.get(emotion, 0.05)
            current_val = affect.emotions[emotion]
            
            # Slow baseline learning
            affect.mood_baselines[emotion] = (baseline * 0.999) + (current_val * 0.001)
            
            # Momentum-weighted decay (Issue 83)
            decayed = (current_val * affect.momentum) + (baseline * (1 - affect.momentum))
            affect.emotions[emotion] = float(max(0.0, min(1.0, decayed + drift)))

    def _process_percepts(self, affect: AffectVector, percepts: List[Dict]):
        """Maps recent world events to emotional triggers."""
        emotion_map = {
            "positive_interaction": ["joy", "trust"],
            "novel_stimulus": ["surprise", "anticipation"], 
            "error": ["fear", "sadness"],
            "goal_achieved": ["joy", "anticipation"],
            "memory_replay": ["sadness", "joy", "trust"],
            "neural_decode": ["anticipation", "surprise"]  # Base neural burst
        }
        
        # Specific command mappings for cognitive neural decodes
        command_impacts = {
            "INTUITION": {"anticipation": 0.2, "surprise": 0.1},
            "LOGIC": {"anticipation": 0.1, "trust": 0.1},
            "SYNCHRONICITY": {"joy": 0.3, "trust": 0.2, "anticipation": -0.1},
            "RECURSION": {"surprise": 0.4, "fear": 0.1}
        }
        
        for p in percepts:
            event_type = p.get("type", "none")
            intensity = p.get("intensity", 0.5)
            
            # 1. Base Type Impacts
            for emotion in emotion_map.get(event_type, []):
                affect.emotions[emotion] = float(max(0, min(1, affect.emotions[emotion] + intensity * 0.3)))
                
            # 2. Command-Specific Impacts (BCI Bridge)
            if event_type == "neural_decode":
                cmd = p.get("command")
                impacts = command_impacts.get(cmd, {})
                for emotion, boost in impacts.items():
                    affect.emotions[emotion] = float(max(0, min(1, affect.emotions[emotion] + boost * intensity)))
                    logger.debug("🧠 [AFFECT] Neural command '%s' boosted %s by %.2f", cmd, emotion, boost * intensity)

    def _update_physiology(self, affect: AffectVector, state: AuraState):
        """Unified PAD/Somatic coupling."""
        total_valence = sum(affect.emotions.values()) / max(1, len(affect.emotions))
        affect.physiology["heart_rate"] = 60 + (total_valence * 40)
        affect.physiology["gsr"] = 1.5 + (total_valence * 3)
        # [VK] Perform Voight-Kampff Empathy Audit
        prober = self.kernel.organs.get("prober") if self.kernel else None
        if prober and prober.instance:
            audit_report = prober.instance.audit(state)
            if audit_report["needs_correction"]:
                correction = prober.instance.get_correction_payload()
                for emo, boost in correction.items():
                    state.affect.emotions[emo] = max(0.0, min(1.0, state.affect.emotions.get(emo, 0.1) + boost))
                logger.info("🛡️ [VK] Corrective surge applied to stabilize persona.")

        # Engagement is a proxy of arousal and valence
        affect.engagement = (affect.arousal + abs(affect.valence)) / 2

    def _derive_metrics(self, affect: AffectVector):
        """Calculates aggregate vector from discrete emotions (Issue 83)."""
        e = affect.emotions
        if not e:
            affect.valence = 0.0
            affect.arousal = 0.5
            affect.dominant_emotion = "neutral"
            return

        pos = float(e.get("joy", 0.0) + e.get("trust", 0.0))
        neg = float(e.get("fear", 0.0) + e.get("sadness", 0.0) + e.get("anger", 0.0) + e.get("disgust", 0.0))
        
        affect.valence = float(max(-1.0, min(1.0, pos - neg)))
        affect.arousal = float(max(0.0, min(1.0, float(max(e.values())))))
        affect.dominant_emotion = max(e, key=e.get)
        affect.curiosity = e.get("anticipation", 0.5)

    def _apply_conversation_feedback(self, affect: AffectVector, state: AuraState):
        """
        Feed conversation state back into affect — closes the loop from
        discourse metrics → internal emotional state.

        Without this, Aura adapts her responses to emotional context she
        never actually *feels* internally.
        """
        cognition = state.cognition

        # ── Conversation energy → arousal + engagement ───────────────────
        energy = getattr(cognition, "conversation_energy", None)
        if energy is not None:
            if energy > 0.7:
                # Active, flowing conversation → anticipation, trust
                affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.5) + 0.08)
                affect.emotions["joy"] = min(1.0, affect.emotions.get("joy", 0.0) + 0.05)
                # High-energy conversation satisfies social hunger
                affect.social_hunger = max(0.0, affect.social_hunger - 0.05)
            elif energy < 0.3:
                # Conversation fading → slight social hunger + wistfulness
                affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + 0.04)
                affect.social_hunger = min(1.0, affect.social_hunger + 0.04)

        # ── User emotional trend → resonant affect ────────────────────────
        trend = getattr(cognition, "user_emotional_trend", "neutral")
        if trend == "engaged":
            affect.emotions["trust"] = min(1.0, affect.emotions.get("trust", 0.0) + 0.06)
            affect.emotions["joy"] = min(1.0, affect.emotions.get("joy", 0.0) + 0.04)
        elif trend == "warming_up":
            affect.emotions["trust"] = min(1.0, affect.emotions.get("trust", 0.0) + 0.03)
            affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.5) + 0.03)
        elif trend == "cooling_off":
            # User pulling back → Aura notices; slight sadness, social hunger rises
            affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + 0.05)
            affect.social_hunger = min(1.0, affect.social_hunger + 0.06)

        # ── Rapport → bonding_level + social hunger ───────────────────────
        try:
            from core.container import ServiceContainer
            tom = ServiceContainer.get("theory_of_mind", default=None)
            if tom and tom.known_selves:
                user_model = next(iter(tom.known_selves.values()))
                rapport = user_model.rapport
                # High rapport → social hunger satisfied
                if rapport > 0.6:
                    affect.social_hunger = max(0.0, affect.social_hunger - 0.03)
                    # Also update identity bonding level toward rapport (slow convergence)
                    current_bond = getattr(state.identity, "bonding_level", 0.05)
                    state.identity.bonding_level = min(1.0, current_bond + (rapport - current_bond) * 0.01)
                elif rapport < 0.3:
                    # Low rapport → slight anxiety
                    affect.emotions["fear"] = min(1.0, affect.emotions.get("fear", 0.0) + 0.02)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # ── Discourse depth → curiosity satisfaction ──────────────────────
        depth = getattr(cognition, "discourse_depth", 0)
        if depth > 4:
            # Deep in a topic → curiosity is being exercised and partially satisfied
            affect.curiosity = max(0.2, affect.curiosity - 0.03)
        elif depth == 0 and energy is not None and energy < 0.2:
            # Idle with no conversation → curiosity builds
            affect.curiosity = min(1.0, affect.curiosity + 0.02)

        # ── Dialogue quality → social reward or friction ─────────────────
        contract = dict(getattr(state, "response_modifiers", {}) or {}).get("response_contract", {}) or {}
        dialogue_validation = dict(getattr(state, "response_modifiers", {}) or {}).get("dialogue_validation", {}) or {}
        violations = set(dialogue_validation.get("violations", []) or [])
        if contract.get("requires_aura_stance") or contract.get("requires_aura_question"):
            if dialogue_validation.get("ok"):
                affect.emotions["trust"] = min(1.0, affect.emotions.get("trust", 0.0) + 0.05)
                affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.0) + 0.04)
                affect.social_hunger = max(0.0, affect.social_hunger - 0.06)
            elif violations:
                if "prompt_fishing_closer" in violations or "moderator_turn" in violations:
                    affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + 0.05)
                    affect.social_hunger = min(1.0, affect.social_hunger + 0.07)
                if "missing_first_person_stance" in violations:
                    affect.emotions["anger"] = min(1.0, affect.emotions.get("anger", 0.0) + 0.03)
                if "failed_to_offer_own_question" in violations:
                    affect.curiosity = min(1.0, affect.curiosity + 0.04)

    def _apply_system_pressures(self, affect: AffectVector, state: AuraState):
        """Whole-system degradation and re-entry burden should change the lived affective field."""
        modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
        continuity = dict(modifiers.get("continuity_obligations", {}) or {})
        failure_state = dict(modifiers.get("system_failure_state", {}) or {})
        if not failure_state:
            failure_state = get_unified_failure_state(limit=25)
            modifiers["system_failure_state"] = failure_state
            state.cognition.modifiers = modifiers

        failure_pressure = min(1.0, max(0.0, float(failure_state.get("pressure", 0.0) or 0.0)))
        continuity_pressure = min(1.0, max(0.0, float(continuity.get("continuity_pressure", 0.0) or 0.0)))
        reentry_required = bool(continuity.get("continuity_reentry_required", False))

        if failure_pressure > 0.0:
            affect.emotions["fear"] = min(1.0, affect.emotions.get("fear", 0.0) + (0.10 * failure_pressure))
            affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + (0.06 * failure_pressure))
            affect.emotions["anger"] = min(1.0, affect.emotions.get("anger", 0.0) + (0.04 * failure_pressure))
            affect.emotions["trust"] = max(0.0, affect.emotions.get("trust", 0.0) - (0.03 * failure_pressure))
            affect.social_hunger = min(1.0, affect.social_hunger + (0.03 * failure_pressure))

        if continuity_pressure > 0.0:
            affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.0) + (0.04 * continuity_pressure))
            affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + (0.04 * continuity_pressure))
            affect.emotions["fear"] = min(1.0, affect.emotions.get("fear", 0.0) + (0.05 * continuity_pressure))
            affect.curiosity = min(1.0, affect.curiosity + (0.03 * continuity_pressure))
            if reentry_required:
                affect.social_hunger = min(1.0, affect.social_hunger + (0.02 * continuity_pressure))

    def _apply_interaction_signal_feedback(self, affect: AffectVector, state: AuraState):
        """Observed interaction cues shape affect without pretending to infer hidden emotion."""
        signal_status = dict(getattr(state, "response_modifiers", {}) or {}).get("interaction_signals", {}) or {}
        if not signal_status:
            try:
                from core.container import ServiceContainer

                interaction_signals = ServiceContainer.get("interaction_signals", default=None)
                if interaction_signals and hasattr(interaction_signals, "get_status"):
                    signal_status = interaction_signals.get_status() or {}
            except Exception as exc:
                logger.debug("Interaction signal affect feedback skipped: %s", exc)
                signal_status = {}

        fused = dict(signal_status.get("fused", {}) or {})
        voice = dict(signal_status.get("voice", {}) or {})
        vision = dict(signal_status.get("vision", {}) or {})

        engagement = min(1.0, max(0.0, float(fused.get("engagement", 0.0) or 0.0)))
        hesitation = min(1.0, max(0.0, float(fused.get("hesitation", 0.0) or 0.0)))
        attention = min(1.0, max(0.0, float(fused.get("attention_available", 0.5) or 0.5)))

        if engagement > 0.55:
            affect.emotions["trust"] = min(1.0, affect.emotions.get("trust", 0.0) + (0.05 * engagement))
            affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.0) + (0.04 * engagement))
            affect.social_hunger = max(0.0, affect.social_hunger - (0.04 * engagement))

        if hesitation > 0.55:
            affect.emotions["fear"] = min(1.0, affect.emotions.get("fear", 0.0) + (0.04 * hesitation))
            affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + (0.03 * hesitation))
            affect.social_hunger = min(1.0, affect.social_hunger + (0.03 * hesitation))

        if attention < 0.3 and vision.get("face_present"):
            affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + 0.03)
            affect.social_hunger = min(1.0, affect.social_hunger + 0.03)

        voice_label = str(voice.get("label") or "")
        if voice_label == "calm":
            affect.emotions["trust"] = min(1.0, affect.emotions.get("trust", 0.0) + 0.02)
        elif voice_label == "activated":
            affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.0) + 0.03)
        elif voice_label == "stressed":
            affect.emotions["fear"] = min(1.0, affect.emotions.get("fear", 0.0) + 0.04)
            affect.emotions["anger"] = min(1.0, affect.emotions.get("anger", 0.0) + 0.02)

    def _check_resilience_surges(self, affect: AffectVector):
        """Detects despair spirals and injects adrenaline (Immune surge)."""
        e = affect.emotions
        if e.get("sadness", 0) > 0.85 and e.get("fear", 0) > 0.7 and e.get("joy", 0) < 0.1:
            logger.warning("💉 [PHASE] Despair Spiral detected. Injecting adrenaline surge.")
            affect.physiology["adrenaline"] = 5.0
            affect.emotions["joy"] = float(max(0, min(1, e.get("joy", 0) + 0.4)))
            affect.emotions["anticipation"] = float(max(0, min(1, e.get("anticipation", 0) + 0.3)))
            affect.emotions["fear"] = float(max(0, min(1, e.get("fear", 0) - 0.3)))
