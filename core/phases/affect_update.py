from __future__ import annotations

import inspect
import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any

from core.health.degraded_events import get_unified_failure_state
from core.kernel.bridge import Phase
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.task_ownership import create_tracked_task
from core.state.aura_state import AffectVector, AuraState
from core.state.percepts import drop_consumed, fresh_for, mark_consumed

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger(__name__)

_USER_FACING_ORIGINS = {
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "http",
    "owner",
    "owner_session_cookie",
    "owner_sovereign",
    "direct",
}

_POSITIVE_AFFECT_WEIGHTS = {
    "joy": 1.0,
    "trust": 0.9,
    "anticipation": 0.35,
    "love": 0.8,
    "awe": 0.45,
    "happiness": 0.9,
    "interest": 0.8,
    "wonder": 0.6,
    "excitement": 0.55,
    "pride": 0.55,
    "curiosity": 0.55,
    "gratitude": 0.65,
    "warmth": 0.75,
    "hope": 0.8,
    "nostalgia": 0.25,
    "satisfaction": 0.6,
    "empathy": 0.5,
    "belonging": 0.75,
    "amusement": 0.45,
    "inspiration": 0.7,
    "relief": 0.55,
    "admiration": 0.45,
}

_NEGATIVE_AFFECT_WEIGHTS = {
    "fear": 1.0,
    "sadness": 0.85,
    "anger": 0.9,
    "disgust": 0.65,
    "terror": 1.1,
    "remorse": 0.55,
    "contempt": 0.55,
    "aggressiveness": 0.7,
    "cynicism": 0.45,
    "boredom": 0.35,
    "apathy": 0.45,
    "indifference": 0.25,
    "dread": 0.85,
    "unhappiness": 0.75,
    "upset": 0.9,
    "confused": 0.6,
    "loneliness": 0.55,
    "longing": 0.45,
    "frustration": 0.75,
    "vulnerability": 0.45,
}

_REASSURANCE_PERCEPTS = {
    "positive_interaction",
    "interaction",
    "extended_dialogue",
    "deep_expression",
    "goal_achieved",
    "discovery",
}

_THREAT_PERCEPTS = {
    "error",
    "threat_detected",
    "resource_pressure",
    "security_alert",
    "self_correction",
}

_STALE_NEGATIVE_EMOTIONS = (
    "fear",
    "sadness",
    "anger",
    "dread",
    "unhappiness",
    "upset",
    "frustration",
    "terror",
)

_SOCIAL_DISTRESS_EMOTIONS = (
    "loneliness",
    "longing",
    "apathy",
    "indifference",
)

_AFFECT_UPDATE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_affect_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "affect_update",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError as signature_exc:
        try:
            record_degradation("affect_update", error, severity=severity, action=action)
        except TypeError:
            logger.debug("AffectUpdate degradation could not be recorded: %s", signature_exc)


class AffectUpdatePhase(Phase):
    """
    Unitary Kernel Phase: Affective Transformation.
    Ported from DamasioV2 logic. Perform emotional decay, 
    somatic updates, and reactive emotional shifts.
    """
    def __init__(self, kernel: AuraKernel):
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

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _set_emotion(self, affect: AffectVector, emotion: str, value: Any) -> None:
        affect.emotions[emotion] = self._clip01(value)

    def _bump_emotion(self, affect: AffectVector, emotion: str, delta: float) -> None:
        self._set_emotion(affect, emotion, affect.emotions.get(emotion, 0.0) + delta)

    def _ensure_affect_schema(self, affect: AffectVector) -> None:
        """Backfill newer affect dimensions into persisted older AuraState snapshots."""
        defaults = AffectVector()
        for emotion, value in defaults.emotions.items():
            affect.emotions.setdefault(emotion, value)
        for emotion, baseline in defaults.mood_baselines.items():
            affect.mood_baselines.setdefault(emotion, baseline)

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
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
        
        self._ensure_affect_schema(affect)

        # 2. Emotional Decay (Entropy & Momentum)
        # Ported from DamasioV2.pulse()
        self._apply_decay(affect)
        
        # 3. Reactive Updates (from recent percepts)
        # Ported from DamasioV2.react()
        # A percept lives one turn. This phase used to take the whole list and
        # clear it, which does prevent double-processing and also deletes the
        # event for everyone downstream: the workspace competition, the world
        # model's observation, the phi estimate and the state's own reading of
        # perception all run after this phase, and all of them saw an empty
        # stream on every turn. Now what this phase felt on a previous turn is
        # dropped here, what arrived since is felt and marked, and the rest of
        # the turn can see what arrived. No clock and no growth: the stream
        # holds exactly one turn of perception by the time affect is done.
        drop_consumed(state.world, "affect")
        recent_percepts = fresh_for(state.world.recent_percepts, "affect")
        self._process_percepts(affect, recent_percepts)
        for item in recent_percepts:
            mark_consumed(item, "affect")
        state.world.trim_percepts()

        # 3.5. Conversation Feedback — close the loop from discourse state → affect
        self._apply_conversation_feedback(affect, state)
        self._apply_interaction_signal_feedback(affect, state)
        self._apply_system_pressures(affect, state)
        self._regulate_stale_negative_affect(affect, state, objective, recent_percepts)

        # 4. Somatic Coupling (Heart rate, GSR, etc.)
        self._update_physiology(affect, state)
        
        # 5. Derive secondary metrics (Valence, Arousal, Dominant Emotion)
        self._derive_metrics(affect)
        
        # 6. Unified Personality Resonance (Unitary Logic)
        self._update_resonance(state)

        # 6b. Advance the lifetime state on this moment, and let what it senses
        # colour the moment. The reservoir used to step only when a memory
        # retrieval happened to ask it something, so a day of conversation with
        # no retrieval left her developmental state exactly where it started.
        #
        # After the emotion channels are settled, not before. The first version
        # ran this at the top of the phase and the derived-affect step three
        # steps later recomputed curiosity from the emotions dictionary, which
        # overwrote the blend every time: displacing the developmental state
        # far enough to take novelty from 0.60 to 1.00 moved curiosity by five
        # ten-thousandths.
        self._advance_lifetime(state, affect)

        # 6c. What won the workspace, as arousal. Global workspace theory's
        # claim is that ignition makes content available to the specialised
        # processes, and affect is one of them; the blend weight is the
        # ignition level itself, so a competition that barely ignited moves
        # arousal barely and a full one moves it most of the way to the
        # winner's priority.
        self._blend_broadcast_into_affect(state, affect)
        
        # Direct Telemetry Bridge: Push VAD to LiquidSubstrate for real-time HUD sync
        from core.container import ServiceContainer
        ls = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get(
            "conscious_substrate", default=None
        )
        if ls:
            self._schedule_substrate_update(ls, affect, state)
            # And read it back. `HomeostaticCoupling` says the continuous
            # substrate is the ground truth for felt state and blends it at
            # thirty percent — into a local dictionary used to pick cognitive
            # modifiers, and never into the affect state itself. The push
            # existed and the return did not, so the substrate was thirty
            # percent of how hard she was allowed to think and none of how she
            # felt.
            self._blend_substrate_into_affect(ls, affect, state)
        
        # 7. Despair Spiral check (Injection)
        self._check_resilience_surges(affect)
        
        logger.debug("Affect Phase complete: mood=%s, valence=%.2f", affect.dominant_emotion, affect.valence)
        return state

    def _mark_phase_degraded(self, state: AuraState, stage: str, exc: BaseException) -> None:
        modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
        degraded = dict(modifiers.get("affect_update_degraded", {}) or {})
        degraded.update(
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc)[:240],
                "at": time.time(),
            }
        )
        modifiers["affect_update_degraded"] = degraded
        state.cognition.modifiers = modifiers

    def _record_phase_degradation(
        self,
        state: AuraState,
        exc: BaseException,
        *,
        stage: str,
        action: str,
        severity: Severity = "warning",
    ) -> None:
        self._mark_phase_degraded(state, stage, exc)
        _record_affect_degradation(
            exc,
            action=action,
            severity=severity,
            extra={"stage": stage},
        )

    def _advance_lifetime(self, state: AuraState, affect: AffectVector) -> None:
        """Step her lifetime state, then blend curiosity toward its novelty.

        The blend weight is the step's own displacement — how far this moment
        moved her — rather than a number chosen here. A large developmental
        update pulls the moment toward how unprecedented it is; a quiet step
        leaves curiosity alone. Nothing else in the runtime reads novelty, so
        without this the state was advancing and sensing into a void.
        """
        try:
            from core.ontogeny.lifetime import LIFETIME_SCHEMA, advance

            features = {
                "perception": float(len(state.world.recent_percepts or [])),
                "interoception": float(
                    (getattr(state.soma, "hardware", {}) or {}).get("cpu_usage", 0.0) or 0.0
                ),
                "affect_valence": float(affect.valence),
                "affect_arousal": float(affect.arousal),
                "workspace": float(getattr(state.cognition, "conversation_energy", 0.5) or 0.5),
                "cognition": float(getattr(state, "phi", 0.0) or 0.0),
                "self_state": float(getattr(state.identity, "stability", 1.0) or 1.0),
                "memory": float(len(state.cognition.working_memory or [])),
                "world": float(len(getattr(state.world, "facts", {}) or {})),
                "deliberation": float(len(state.cognition.active_goals or [])),
            }
            del LIFETIME_SCHEMA
            reading = advance(features)
            if reading is None:
                return
            weight = max(0.0, min(1.0, float(reading.displacement)))
            affect.curiosity = max(
                0.0,
                min(1.0, (1.0 - weight) * float(affect.curiosity) + weight * float(reading.novelty)),
            )
            state.response_modifiers["ontogenetic_novelty"] = round(float(reading.novelty), 4)
            state.response_modifiers["ontogenetic_displacement"] = round(weight, 4)
            self._ground_affect(state, affect, novelty=float(reading.novelty))
        except _AFFECT_UPDATE_ERRORS as exc:
            self._record_phase_degradation(
                state,
                exc,
                stage="lifetime_state",
                action="kept affect state after the lifetime reservoir did not advance",
                severity="warning",
            )

    def _ground_affect(self, state: AuraState, affect: AffectVector, *, novelty: float) -> None:
        """Let the measured signals name the feeling, when they have earned it.

        `AffectGroundingEngine` derives affect from sustained evidence —
        prediction error from the world model, nociceptive pressure from the
        body, novelty from the lifetime state — and refuses to assert a label
        until it has enough samples. It was registered as a service and never
        called, so the three channels it bridges were readers with nothing
        running them.

        What it returns informs rather than replaces. The grounded label is
        blended into the emotion of the same name with the engine's own
        confidence as the weight, so a tentative read moves almost nothing and
        a well-evidenced one moves most of the way. Nothing here overrides the
        dominant emotion outright: an evidence layer that can silence the rest
        of affect is not evidence, it is a second opinion with a veto.
        """
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("affect_grounding", default=None)
            if engine is None:
                return
            control = 0.5
            try:
                from core.agency.authorship import get_agency_ledger

                ledger = get_agency_ledger()
                if ledger.acted:
                    control = float(ledger.efficacy)
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                control = 0.5
            engine.observe(
                novelty=novelty,
                valence=float(affect.valence),
                arousal=float(affect.arousal),
                control=control,
                idle=0.0 if (state.cognition.current_objective or "").strip() else 1.0,
            )
            engine.gather()
            grounded = engine.dominant()
            if grounded is None:
                return
            weight = max(0.0, min(1.0, float(grounded.confidence)))
            channel = grounded.label
            if channel in affect.emotions:
                affect.emotions[channel] = max(
                    0.0,
                    min(
                        1.0,
                        (1.0 - weight) * float(affect.emotions[channel])
                        + weight * float(grounded.intensity),
                    ),
                )
            markers = dict(getattr(affect, "markers", {}) or {})
            markers["grounded"] = grounded.to_dict()
            affect.markers = markers
        except _AFFECT_UPDATE_ERRORS as exc:
            self._record_phase_degradation(
                state,
                exc,
                stage="affect_grounding",
                action="kept affect state after grounded affect could not be read",
                severity="warning",
            )

    def _blend_broadcast_into_affect(self, state: AuraState, affect: AffectVector) -> None:
        """Read the last broadcast's ignition off the workspace and feel it."""
        try:
            from core.runtime.service_registry import get_runtime_service

            workspace = get_runtime_service("global_workspace", default=None)
            reading = getattr(workspace, "last_broadcast_arousal", None)
            if not isinstance(reading, dict):
                return
            level = float(reading.get("ignition", 0.0))
            if level <= 0.0:
                return
            target = float(reading.get("priority", 0.0))
            affect.arousal = max(0.0, min(1.0, (1.0 - level) * float(affect.arousal) + level * target))
            state.response_modifiers["broadcast_ignition"] = round(level, 4)
        except _AFFECT_UPDATE_ERRORS as exc:
            self._record_phase_degradation(
                state,
                exc,
                stage="broadcast_arousal",
                action="kept affect state without the broadcast's ignition",
                severity="warning",
            )

    def _blend_substrate_into_affect(
        self, substrate: Any, affect: AffectVector, state: AuraState
    ) -> None:
        """Close the loop the coupling's own docstring describes.

        The share is the one `HomeostaticCoupling` already uses, imported
        rather than repeated, because two copies of a number like this drift
        and then two subsystems disagree about how much of her feeling is the
        substrate.

        A stale reading is skipped rather than blended. The substrate publishes
        how old its snapshot is, and a felt state built from a snapshot taken
        before the last thing that happened is worse than one built without it.
        """
        try:
            from core.consciousness.homeostatic_coupling import SUBSTRATE_SHARE

            reading = substrate.get_state_summary_nowait()
            if not isinstance(reading, dict):
                return
            if reading.get("snapshot_stale"):
                # Worth recording rather than skipping quietly. The snapshot is
                # marked fresh only by the substrate's own dynamics step, so a
                # loop that has died disconnects the substrate from affect with
                # no other symptom — the readings stay plausible and simply
                # stop arriving.
                state.response_modifiers["substrate_snapshot_age_s"] = round(
                    float(reading.get("snapshot_age_s", 0.0)), 3
                )
                record_degradation(
                    "affect_update",
                    RuntimeError("substrate snapshot stale"),
                    severity="info",
                    action="affect kept its own valence; the substrate is not integrating",
                )
                return
            keep = 1.0 - SUBSTRATE_SHARE
            affect.valence = max(
                -1.0,
                min(1.0, affect.valence * keep + float(reading.get("valence", 0.0)) * SUBSTRATE_SHARE),
            )
            affect.arousal = max(
                0.0,
                min(1.0, affect.arousal * keep + float(reading.get("arousal", 0.0)) * SUBSTRATE_SHARE),
            )
            state.response_modifiers["substrate_share_of_affect"] = SUBSTRATE_SHARE
        except _AFFECT_UPDATE_ERRORS as exc:
            self._record_phase_degradation(
                state,
                exc,
                stage="substrate_readback",
                action="kept affect state without the substrate's contribution",
                severity="warning",
            )

    def _schedule_substrate_update(self, substrate: Any, affect: AffectVector, state: AuraState) -> None:
        try:
            update = getattr(substrate, "update", None)
            if not callable(update):
                return
            result = update(valence=affect.valence, arousal=affect.arousal)
            if not inspect.isawaitable(result):
                return
            create_tracked_task(result, name="affect_update.liquid_substrate")
        except _AFFECT_UPDATE_ERRORS as exc:
            self._record_phase_degradation(
                state,
                exc,
                stage="substrate_telemetry",
                action="kept affect state update after liquid substrate telemetry scheduling failed",
            )
            logger.debug("Failed to push VAD to substrate: %s", exc)

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
            if "Aura (Core)" not in res:
                res["Aura (Core)"] = 0.4

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

    def _process_percepts(self, affect: AffectVector, percepts: list[dict]):
        """Maps recent world events to emotional triggers."""
        emotion_map = {
            "interaction": ["trust", "happiness", "interest", "warmth", "belonging"],
            "positive_interaction": ["joy", "trust", "happiness", "interest", "pride", "gratitude", "warmth", "hope", "satisfaction", "belonging"],
            "extended_dialogue": ["happiness", "interest", "curiosity", "warmth", "hope", "belonging"],
            "deep_expression": ["interest", "trust", "curiosity", "satisfaction", "inspiration"],
            "novel_stimulus": ["surprise", "anticipation", "wonder", "excitement", "curiosity"],
            "discovery": ["wonder", "excitement", "interest", "curiosity", "pride", "hope"],
            "error": ["fear", "sadness", "unhappiness", "dread", "upset", "frustration", "confused"],
            "threat_detected": ["fear", "dread", "upset", "vulnerability"],
            "goal_achieved": ["joy", "anticipation", "happiness", "excitement", "pride", "satisfaction", "hope", "relief"],
            "memory_replay": ["sadness", "joy", "trust", "nostalgia", "warmth", "belonging"],
            "monotony": ["boredom", "apathy", "loneliness", "indifference"],
            # Three types the tree emits that this map had no entry for, so a
            # phase crash, an apology and every stimulus injected through the
            # compatibility bridge arrived and moved nothing. An internal error
            # is an error; a self-correction is an error about her own output,
            # without the part that is afraid of the world.
            "internal_error": ["fear", "sadness", "unhappiness", "dread", "upset", "frustration", "confused"],
            # The body under strain. This type was already listed as a threat
            # a few lines up, and until the proprioceptive loop emitted one,
            # nothing in the tree ever produced it — so a machine at ninety
            # percent load reached her feeling through nothing at all.
            "resource_pressure": ["fear", "upset", "frustration", "vulnerability"],
            "self_correction": ["sadness", "unhappiness", "upset", "frustration", "confused"],
            "disconnection": ["unhappiness", "apathy", "loneliness", "longing"],
            "neural_decode": ["anticipation", "surprise"]  # Base neural burst
        }
        
        # Specific command mappings for cognitive neural decodes. Neural input is
        # advisory sensory context only; it must never become a hard dependency
        # for autonomous RSI/self-improvement loops.
        command_impacts = {
            "INTUITION": {"anticipation": 0.2, "surprise": 0.1},
            "LOGIC": {"anticipation": 0.1, "trust": 0.1},
            "SYNCHRONICITY": {"joy": 0.3, "trust": 0.2, "anticipation": -0.1},
            "RECURSION": {"surprise": 0.18, "anticipation": 0.12}
        }
        
        for p in percepts:
            event_type = p.get("type", "none")
            intensity = p.get("intensity", 0.5)
            if event_type == "neural_decode":
                raw_intensity = intensity
                intensity = min(float(intensity), 0.25)
                affect.markers["neural_decode_autonomy_cap"] = {
                    "raw_intensity": float(raw_intensity),
                    "applied_intensity": float(intensity),
                    "reason": "advisory_bci_not_autonomy_dependency",
                }
            
            # 1. Base Type Impacts
            for emotion in emotion_map.get(event_type, []):
                self._bump_emotion(affect, emotion, float(intensity) * 0.3)

            if event_type in _REASSURANCE_PERCEPTS:
                for emotion in ("fear", "sadness", "upset", "frustration", "loneliness", "longing"):
                    baseline = affect.mood_baselines.get(emotion, 0.0)
                    current = affect.emotions.get(emotion, 0.0)
                    relief = min(0.35, float(intensity) * 0.25)
                    self._set_emotion(affect, emotion, max(baseline, current - relief))
                
            # 2. Command-Specific Impacts (BCI Bridge)
            if event_type == "neural_decode":
                cmd = p.get("command")
                impacts = command_impacts.get(cmd, {})
                for emotion, boost in impacts.items():
                    self._bump_emotion(affect, emotion, float(boost) * float(intensity))
                    logger.debug("🧠 [AFFECT] Neural command '%s' boosted %s by %.2f", cmd, emotion, boost * intensity)

    def _update_physiology(self, affect: AffectVector, state: AuraState):
        """Unified PAD/Somatic coupling."""
        total_valence = sum(affect.emotions.values()) / max(1, len(affect.emotions))
        affect.physiology["heart_rate"] = 60 + (total_valence * 40)
        affect.physiology["gsr"] = 1.5 + (total_valence * 3)
        # [VK] Perform Voight-Kampff Empathy Audit
        prober = self.kernel.organs.get("prober") if self.kernel else None
        if prober and prober.instance:
            try:
                audit_report = prober.instance.audit(state)
                if audit_report["needs_correction"]:
                    correction = prober.instance.get_correction_payload()
                    for emo, boost in correction.items():
                        state.affect.emotions[emo] = max(0.0, min(1.0, state.affect.emotions.get(emo, 0.1) + boost))
                    logger.info("🛡️ [VK] Corrective surge applied to stabilize persona.")
            except _AFFECT_UPDATE_ERRORS as exc:
                self._record_phase_degradation(
                    state,
                    exc,
                    stage="vk_empathy_audit",
                    action="kept somatic affect update after empathy audit failed",
                )
                logger.debug("Voight-Kampff affect audit skipped: %s", exc)

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

        baselines = affect.mood_baselines or {}

        def activation(emotion: str) -> float:
            return max(0.0, float(e.get(emotion, 0.0) or 0.0) - float(baselines.get(emotion, 0.0) or 0.0))

        pos = sum(activation(emotion) * weight for emotion, weight in _POSITIVE_AFFECT_WEIGHTS.items())
        neg = sum(activation(emotion) * weight for emotion, weight in _NEGATIVE_AFFECT_WEIGHTS.items())

        affect.valence = float(max(-1.0, min(1.0, math.tanh((pos - neg) * 1.6))))
        raw_peak = max((float(value or 0.0) for value in e.values()), default=0.5)
        active_peak = max((activation(emotion) for emotion in e), default=0.0)
        affect.arousal = float(max(0.0, min(1.0, max(raw_peak * 0.55, active_peak))))
        dominant_by_activation = max(e, key=lambda emotion: activation(emotion))
        if activation(dominant_by_activation) < 0.025 and abs(affect.valence) < 0.08:
            affect.dominant_emotion = "neutral"
        else:
            affect.dominant_emotion = dominant_by_activation
        affect.curiosity = max(e.get("curiosity", 0.0), e.get("anticipation", 0.5))

    def _apply_conversation_feedback(self, affect: AffectVector, state: AuraState):
        """
        Feed conversation state back into affect — closes the loop from
        discourse metrics → internal emotional state.

        Without this, Aura adapts her responses to emotional context she
        never actually *feels* internally.
        """
        cognition = state.cognition
        trend = getattr(cognition, "user_emotional_trend", "neutral")

        # ── Conversation energy → arousal + engagement ───────────────────
        energy = getattr(cognition, "conversation_energy", None)
        if energy is not None:
            if energy > 0.7:
                # Active, flowing conversation → anticipation, trust
                affect.emotions["anticipation"] = min(1.0, affect.emotions.get("anticipation", 0.5) + 0.08)
                affect.emotions["joy"] = min(1.0, affect.emotions.get("joy", 0.0) + 0.05)
                # High-energy conversation satisfies social hunger
                affect.social_hunger = max(0.0, affect.social_hunger - 0.05)
            elif energy < 0.3 and trend == "cooling_off":
                # Low energy alone can be a short normal message. Only treat it as
                # social fading when the discourse tracker also sees withdrawal.
                affect.emotions["sadness"] = min(1.0, affect.emotions.get("sadness", 0.0) + 0.04)
                affect.social_hunger = min(1.0, affect.social_hunger + 0.04)

        # ── User emotional trend → resonant affect ────────────────────────
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
            except _AFFECT_UPDATE_ERRORS as exc:
                self._record_phase_degradation(
                    state,
                    exc,
                    stage="interaction_signals",
                    action="used neutral interaction signals after affect signal lookup failed",
                )
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

    def _regulate_stale_negative_affect(
        self,
        affect: AffectVector,
        state: AuraState,
        objective: str | None,
        recent_percepts: list[dict],
    ) -> None:
        """Release stale distress on foreground turns without masking real failures."""
        origin = str(getattr(state.cognition, "current_origin", "") or "").strip().lower().replace("-", "_")
        if origin not in _USER_FACING_ORIGINS and not objective:
            return

        modifiers = dict(getattr(state.cognition, "modifiers", {}) or {})
        failure_state = dict(modifiers.get("system_failure_state", {}) or {})
        failure_pressure = self._clip01(failure_state.get("pressure", 0.0))
        continuity = dict(modifiers.get("continuity_obligations", {}) or {})
        continuity_pressure = self._clip01(continuity.get("continuity_pressure", 0.0))
        percept_types = {str(p.get("type", "")).strip().lower() for p in recent_percepts if isinstance(p, dict)}
        dialogue_validation = dict(getattr(state, "response_modifiers", {}) or {}).get("dialogue_validation", {}) or {}
        violations = set(dialogue_validation.get("violations", []) or [])
        signal_status = dict(getattr(state, "response_modifiers", {}) or {}).get("interaction_signals", {}) or {}
        fused = dict(signal_status.get("fused", {}) or {})
        hesitation = self._clip01(fused.get("hesitation", 0.0))

        if (
            failure_pressure >= 0.15
            or continuity_pressure >= 0.85
            or percept_types.intersection(_THREAT_PERCEPTS)
            or violations
            or hesitation >= 0.7
        ):
            return

        negative_load = sum(float(affect.emotions.get(emotion, 0.0) or 0.0) for emotion in _STALE_NEGATIVE_EMOTIONS)
        social_load = sum(float(affect.emotions.get(emotion, 0.0) or 0.0) for emotion in _SOCIAL_DISTRESS_EMOTIONS)
        if negative_load < 0.9 and social_load < 0.7 and float(getattr(affect, "valence", 0.0) or 0.0) > -0.75:
            return

        relief = 0.24
        if percept_types.intersection(_REASSURANCE_PERCEPTS):
            relief = 0.34
        conversation_energy = self._clip01(getattr(state.cognition, "conversation_energy", 0.5))
        if conversation_energy >= 0.65:
            relief = max(relief, 0.3)

        for emotion in _STALE_NEGATIVE_EMOTIONS:
            current = float(affect.emotions.get(emotion, 0.0) or 0.0)
            baseline = float(affect.mood_baselines.get(emotion, 0.0) or 0.0)
            if current <= baseline:
                continue
            target = max(baseline, current - relief)
            self._set_emotion(affect, emotion, target)

        for emotion in _SOCIAL_DISTRESS_EMOTIONS:
            current = float(affect.emotions.get(emotion, 0.0) or 0.0)
            baseline = float(affect.mood_baselines.get(emotion, 0.0) or 0.0)
            if current <= baseline:
                continue
            self._set_emotion(affect, emotion, max(baseline, current - (relief * 0.65)))

        for emotion, delta in {
            "trust": 0.05,
            "interest": 0.04,
            "warmth": 0.04,
            "hope": 0.035,
        }.items():
            self._bump_emotion(affect, emotion, delta)

        affect.social_hunger = max(0.0, affect.social_hunger - 0.08)
        affect.markers["stale_negative_affect_regulated"] = {
            "at": time.time(),
            "origin": origin or "objective",
            "relief": round(relief, 3),
            "reason": "foreground_turn_without_active_failure_or_threat",
        }

    def _check_resilience_surges(self, affect: AffectVector):
        """Detects despair spirals and injects adrenaline (Immune surge)."""
        e = affect.emotions
        if e.get("sadness", 0) > 0.85 and e.get("fear", 0) > 0.7 and e.get("joy", 0) < 0.1:
            logger.warning("💉 [PHASE] Despair Spiral detected. Injecting adrenaline surge.")
            affect.physiology["adrenaline"] = 5.0
            affect.emotions["joy"] = float(max(0, min(1, e.get("joy", 0) + 0.4)))
            affect.emotions["anticipation"] = float(max(0, min(1, e.get("anticipation", 0) + 0.3)))
            affect.emotions["fear"] = float(max(0, min(1, e.get("fear", 0) - 0.3)))


# ─────────────────────────────────────────────────────────────────────────────
# Declared semantics. See core/runtime/cognitive_contract.py.
#
# `writes` is MEASURED — tools/observe_phase_writes.py ran this phase against a
# real AuraState and recorded which fields moved. It is not a reading of the
# code, which is how a declaration ends up describing what the author believed.
from core.runtime.cognitive_contract import (
    BranchSpec,
    CognitiveTransformContract,
    register_contract,
)

register_contract(
    CognitiveTransformContract(
        name="AffectUpdatePhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Advance the affect vector one tick from appraisal, decay and "
            "physiology, and publish the modifiers downstream phases read."
        ),
        reads=(
            "affect.valence",
            "affect.arousal",
            "affect.curiosity",
            "affect.emotions",
            "cognition.working_memory",
            "cognition.current_objective",
        ),
        writes=(
            "affect.arousal",
            "affect.curiosity",
            "affect.emotions",
            "affect.engagement",
            "affect.physiology",
            "affect.resonance",
            "affect.social_hunger",
            "affect.valence",
            "cognition.modifiers",
        ),
        preconditions=("state carries an affect vector",),
        branches=(
            BranchSpec(
                "appraised",
                "a turn is present to appraise",
                "update emotions, valence and arousal from it",
            ),
            BranchSpec(
                "decay_only",
                "no new turn since the last tick",
                "decay toward mood baselines without appraisal",
            ),
        ),
        invariants=(
            "valence stays within its declared range",
            "arousal stays within its declared range",
        ),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py; thresholds are "
            "judgement calls not yet extracted to named constants"
        ),
    )
)
