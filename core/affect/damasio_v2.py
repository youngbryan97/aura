from core.utils.task_tracker import get_task_tracker
import asyncio
import logging
import time
from dataclasses import dataclass, field
from core.autonomic.iot_bridge import PhysicalActuator
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

# FIX: AffectState canonical definition lives in core.affect.
# Removed duplicate definition from this file.
from core.affect import AffectState, BASELINE_VALENCE, BASELINE_AROUSAL, BASELINE_ENGAGEMENT
from core.utils.concurrency import RobustLock

class DamasioMarkers:
    """Somatic Markers (Virtual Physiology) and Emotions.
    """

    def __init__(self):
        from core.config import config
        data_dir = config.paths.data_dir
        project_root = config.paths.project_root
        
        weights_path = data_dir / "config" / "weights.npz"
        if not weights_path.exists():
            weights_path = project_root / "data" / "config" / "weights.npz"
        
        # Default baselines
        b = [72.0, 2.1, 10.0, 0.0]
        emotion_def = 0.0
        
        if weights_path.exists():
            try:
                w = np.load(weights_path, allow_pickle=True)
                # Issue 101: Robust key access
                if 'damasio_baselines' in w:
                    b = w['damasio_baselines']
                    if hasattr(b, 'tolist'): b = b.tolist()
                
                if 'emotions_default' in w:
                    emotion_def = float(w['emotions_default'])
                
                logger.info("✓ Damasio weights loaded from .npz")
            except Exception as e:
                logger.error("Failed to load Damasio weights (falling back to defaults): %s", e)

        # Somatic markers (virtual physiology)
        self.heart_rate = float(b[0])
        self.gsr = float(b[1])
        self.cortisol = float(b[2])
        self.adrenaline = float(b[3])
        
        # 47 primary emotions (Plutchik + Damasio)
        # Unified state representation
        self.emotions = {
            "joy": emotion_def, "trust": emotion_def, "fear": emotion_def, "surprise": emotion_def,
            "sadness": emotion_def, "disgust": emotion_def, "anger": emotion_def, "anticipation": emotion_def,
            # Secondary compounds
            "love": emotion_def, "submission": emotion_def, "awe": emotion_def, "terror": emotion_def,
            "remorse": emotion_def, "contempt": emotion_def, "aggressiveness": emotion_def, "cynicism": emotion_def,
        }

        # Phase 18.2: Emotional Momentum & Baselines
        self.mood_baselines = {k: 0.1 if k in ["joy", "anticipation"] else 0.05 for k in self.emotions}
        self.momentum = 0.85 # Higher = slower shifts
        self.last_update = time.time()
        
    def somatic_update(self, event_type: str, intensity: float):
        """Update emotions + virtual physiology from events"""
        emotion_map = {
            "positive_interaction": ["joy", "trust"],
            "novel_stimulus": ["surprise", "anticipation"], 
            "error": ["fear", "sadness"],
            "goal_achieved": ["joy", "anticipation"],
            "memory_replay": ["sadness", "joy"]  # Mixed
        }
        
        if event_type == "virtual_embodiment":
            self.emotions["anticipation"] = float(min(1.0, float(self.emotions.get("anticipation", 0.0)) + intensity * 0.4))
            self.heart_rate = float(min(120.0, float(self.heart_rate) + intensity * 3.0))
            self.gsr += intensity * 0.5
        
        for emotion in emotion_map.get(event_type, []):
            self.emotions[emotion] += intensity * 0.3
            self.emotions[emotion] = np.clip(self.emotions[emotion], 0, 1)
        
        # Virtual physiology coupling
        total_valence = sum(self.emotions.values()) / len(self.emotions)
        self.heart_rate = 60 + (total_valence * 40)
        self.gsr = 1.5 + (total_valence * 3)

    def incorporate_somatic_hardware(self, soma_state: Dict[str, float]):
        """Maps physical hardware stress to virtual somatic markers."""
        thermal = soma_state.get("thermal_load", 0.0)
        anxiety = soma_state.get("resource_anxiety", 0.0)
        
        # Thermal stress increases virtual adrenaline and cortisol
        self.adrenaline = np.clip(self.adrenaline + (thermal * 0.2), 0, 10)
        self.cortisol = np.clip(self.cortisol + (thermal * 0.1), 0, 50)
        
        # Resource anxiety (RAM/Disk) maps to fear and anger (frustration)
        if anxiety > 0.7:
            self.emotions["fear"] = np.clip(self.emotions["fear"] + 0.05, 0, 1)
            self.emotions["anger"] = np.clip(self.emotions["anger"] + 0.02, 0, 1)
        
        # High thermal load triggers irritability (anger)
        if thermal > 0.8:
            self.emotions["anger"] = np.clip(self.emotions["anger"] + 0.03, 0, 1)
            self.emotions["joy"] = np.clip(self.emotions["joy"] - 0.05, 0, 1)
        
    def get_wheel(self) -> Dict:
        # v14.2 FIX: Use explicit allowlist instead of fragile name length check
        # This ensures 'anticipation' (12 chars) is correctly included.
        PRIMARY_PLUTCHIK = {
            "joy", "trust", "fear", "surprise", 
            "sadness", "disgust", "anger", "anticipation"
        }
        return {
            "primary": {k: v for k, v in self.emotions.items() if k in PRIMARY_PLUTCHIK},
            "physiology": {
                "HR": f"{self.heart_rate:.0f}bpm",
                "GSR": f"{self.gsr:.1f}μS", 
                "Cortisol": f"{self.cortisol:.0f}μg/dL"
            }
        }

class AffectEngineV2:
    def __init__(self):
        self.markers = DamasioMarkers()
        self.iot_bridge = PhysicalActuator()
        self._lock = RobustLock("Affect.AffectEngine")

        # Issue 98: LLM Fallback state
        self._llm_available = True
        self._last_llm_failure = 0.0
        self._llm_cooldown = 60 # Reduced ceiling for faster emotional appraisal recovery.
        self._llm_failure_count = 0
        self._llm_backoff_until = 0.0
        self._last_llm_failure_reason = ""

        # Issue 109: Task tracking to prevent leaks
        self._background_tasks = set()
        self._max_background_tasks = 8

        # Stuck-state watchdog: if valence pins near -1 (max fear/pain) for
        # too many consecutive pulses, something upstream is spamming
        # negative stimuli faster than homeostatic decay can recover.  We
        # track consecutive "pinned" pulses and force a soft reset if the
        # threshold is crossed.  Without this, Aura enters a feedback loop
        # where her defensive affect drives defensive responses, which the
        # user pushes back on, which the affect engine interprets as more
        # negative stimulus, etc.
        self._consecutive_pinned_pulses = 0
        self._PINNED_RESET_AFTER = 24  # at 1 Hz pulse rate, ~24 s

    def _prune_background_tasks(self) -> None:
        self._background_tasks = {task for task in self._background_tasks if not task.done()}

    def _spawn_background_task(self, coro, *, name: str):
        """Best-effort background fan-out with a hard cap to prevent task pileups."""
        self._prune_background_tasks()
        if len(self._background_tasks) >= self._max_background_tasks:
            logger.debug(
                "Skipping affect background task %s due to backlog (%d active).",
                name,
                len(self._background_tasks),
            )
            if hasattr(coro, "close"):
                try:
                    coro.close()
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            return None

        task = get_task_tracker().create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task
        
    async def react(self, trigger: str, context: Optional[Dict] = None):
        """
        FIX: Previous implementation released the lock after somatic_update()
        then called _check_for_despair_spiral() outside the lock. If pulse()
        ran concurrently between those two points, emotion state would be
        inconsistent when the spiral check ran.

        Full operation is now lock-scoped, with the IoT broadcast (which is
        non-blocking and uses its own task) happening after the lock releases.
        """
        intensity = context.get("intensity", 1.0) if context else 1.0

        # Do not hold the affect lock across LLM appraisal. Slow appraisals were
        # starving pulse()/echo paths and causing avoidable lock watchdog trips.
        if (not self._llm_available) and time.time() >= self._llm_backoff_until:
            logger.info("♻️ LLM Affective appraisal reset (cooldown expired)")
            self._llm_available = True

        appraisal = None
        # Hard gate: if there's a live user-facing foreground request (Bryan
        # is waiting for Aura to respond), do NOT burn 7B brainstem cycles on
        # a 15KB affect appraisal.  The previous _background_llm_should_defer()
        # only looked at Cortex lane state and still fired the LLM call during
        # active chat, causing event-loop lag spikes and the "Aura is
        # thinking..." stall the user saw.
        foreground_active = False
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active
            foreground_active = bool(_foreground_owner_active())
        except Exception:
            foreground_active = False

        if self._llm_available and len(trigger) > 10:
            if foreground_active:
                logger.debug(
                    "Affect appraisal skipped: foreground chat is in flight — "
                    "using heuristic to keep the inference pipe clear."
                )
                appraisal = self._heuristic_appraisal(trigger, context)
                intensity = (
                    abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                ) / 2.0 or intensity
            elif self._background_llm_should_defer():
                logger.debug("Affect appraisal deferred while the foreground Cortex lane is warming or recovering.")
                appraisal = self._heuristic_appraisal(trigger, context)
                intensity = (
                    abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                ) / 2.0 or intensity
            else:
                try:
                    appraisal = await asyncio.wait_for(
                        self._appraise_with_llm(trigger, context),
                        timeout=4.0,
                    )
                    self._llm_failure_count = 0
                    self._llm_backoff_until = 0.0
                    self._last_llm_failure_reason = ""
                    intensity = (
                        abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                    ) / 2.0 or intensity
                except Exception as e:
                    failure_reason = self._classify_appraisal_failure(e)
                    if failure_reason == "lane_unavailable":
                        logger.debug("Affect appraisal skipped because the foreground lane is reserved or unavailable.")
                        appraisal = self._heuristic_appraisal(trigger, context)
                        intensity = (
                            abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                        ) / 2.0 or intensity
                    else:
                        self._llm_failure_count += 1
                        self._last_llm_failure_reason = failure_reason
                        self._llm_backoff_until = time.time() + min(
                            float(self._llm_cooldown),
                            float(2 ** min(self._llm_failure_count + 1, 6)),
                        )
                        logger.warning(
                            "⚠️ LLM Appraisal failed (%s:%s)",
                            failure_reason,
                            type(e).__name__,
                        )
                        try:
                            from core.health.degraded_events import record_degraded_event

                            record_degraded_event(
                                "affect_appraisal",
                                failure_reason,
                                detail=str(e) or type(e).__name__,
                                severity="warning",
                                classification="background_degraded",
                                context={"trigger": trigger[:120]},
                                exc=e,
                            )
                        except Exception as degraded_exc:
                            logger.debug("Affect degraded-event logging failed: %s", degraded_exc)
                        self._llm_available = False
                        self._last_llm_failure = time.time()
                        appraisal = self._heuristic_appraisal(trigger, context)
                        intensity = (
                            abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                        ) / 2.0 or intensity

        if not await self._lock.acquire_robust(timeout=2.0):
            logger.warning("⚠️ Affect reaction lock timeout.")
            return self.markers.get_wheel()

        try:
            self.markers.somatic_update(trigger, intensity)

            # Despair spiral check is synchronous so we don't suspend while holding the lock.
            self._check_for_despair_spiral()

            # Snapshot for IoT broadcast (taken while locked, broadcast after)
            wheel = self.markers.get_wheel()
            primaries = wheel.get("primary", {})
            pos = primaries.get("joy", 0) + primaries.get("trust", 0)
            neg = primaries.get("fear", 0) + primaries.get("sadness", 0) + primaries.get("anger", 0)
            current_pad = {"P": pos - neg, "A": max(primaries.values()) if primaries else 0.0}
        finally:
            if self._lock.locked():
                self._lock.release()

        # IoT broadcast happens outside the lock — it's a fire-and-forget
        # that doesn't need to read shared state
        try:
            self._spawn_background_task(
                self.iot_bridge.broadcast_affect_state(current_pad),
                name="affect.iot_broadcast",
            )
        except Exception as e:
            logger.debug("IoT Bridge broadcast failed: %s", e)

        return wheel

    async def pulse(self):
        """Unified background update: Decays emotions and pulls hardware telemetry."""
        from core.container import ServiceContainer

        soma = ServiceContainer.get("soma", default=None) or ServiceContainer.get("virtual_body", default=None)
        soma_state = None
        if soma:
            try:
                soma_state = await soma.pulse()
            except Exception as exc:
                logger.debug("Soma pulse failed during affect update: %s", exc)

        if not await self._lock.acquire_robust(timeout=2.0):
            return self.markers.get_wheel()

        try:
            self.markers.last_update = time.time() # Track pulse time
            if soma_state:
                self.markers.incorporate_somatic_hardware(soma_state)
                
            # Phase 21: Physical Entropy Anchoring (Thermodynamic Drift)
            try:
                from core.senses.entropy_anchor import entropy_anchor
                drift = entropy_anchor.get_vad_drift(volatility_multiplier=0.015)
            except Exception:
                drift = 0.0

            # Phase 18.2: Momentum-Based Decay & Baseline Drift (v52: Homeostatic Rubber-Band)
            FLOOR = 0.02 # Issue 105: Prevent total emotional death
            for emotion in self.markers.emotions:
                # Shift baseline slowly towards current state (learning)
                target_baseline = self.markers.mood_baselines[emotion]
                current_val = self.markers.emotions[emotion]

                # Update baseline (very slow)
                self.markers.mood_baselines[emotion] = (target_baseline * 0.999) + (current_val * 0.001)

                # Apply momentum-weighted decay towards baseline
                decayed = (current_val * self.markers.momentum) + (target_baseline * (1 - self.markers.momentum))

                # Homeostatic Rubber-Band: Pull increases with distance from baseline
                distance = current_val - target_baseline
                # Increased gain to 0.1 and ensured it's not too small to overcome noise
                rubber_band_pull = (distance ** 2) * 0.2 * np.sign(distance)
                decayed -= rubber_band_pull

                # Inject non-deterministic thermal noise
                self.markers.emotions[emotion] = np.clip(decayed + drift, FLOOR, 1)

            wheel = self.markers.get_wheel()

            # Stuck-valence watchdog.  If valence has been pinned at ≤ −0.95
            # for _PINNED_RESET_AFTER consecutive pulses, the homeostatic
            # decay isn't winning against whatever is driving it down.  Snap
            # the primary negative emotions back toward their baselines so
            # Aura can actually recover in a conversational cadence instead
            # of producing "I sense your fear" responses for the next ten
            # minutes.
            try:
                v = float(getattr(wheel, "valence", 0.0))
            except Exception:
                v = 0.0
            if v <= -0.95:
                self._consecutive_pinned_pulses += 1
            else:
                self._consecutive_pinned_pulses = 0
            if self._consecutive_pinned_pulses >= self._PINNED_RESET_AFTER:
                logger.warning(
                    "🫁 Affect watchdog: valence pinned at %.2f for %d pulses — "
                    "forcing soft reset toward baseline so Aura can recover "
                    "conversational presence.",
                    v, self._consecutive_pinned_pulses,
                )
                for emotion in ("fear", "sadness", "anger", "disgust", "terror", "remorse"):
                    if emotion in self.markers.emotions:
                        baseline = self.markers.mood_baselines.get(emotion, 0.05)
                        # Collapse halfway to baseline — not a hard zero-out,
                        # just a release of the stuck contraction.
                        self.markers.emotions[emotion] = float(
                            (self.markers.emotions[emotion] + baseline) / 2.0
                        )
                # Give the positive side a small nudge so the wheel actually
                # rebalances rather than everything crashing to FLOOR.
                for emotion in ("joy", "trust", "anticipation"):
                    if emotion in self.markers.emotions:
                        self.markers.emotions[emotion] = float(
                            np.clip(self.markers.emotions[emotion] + 0.05, FLOOR, 1)
                        )
                self._consecutive_pinned_pulses = 0
                wheel = self.markers.get_wheel()
        finally:
            if self._lock.locked():
                self._lock.release()

        # Issue 107: Periodic state broadcast
        await self._broadcast_event("affect_pulse")
        return wheel

    async def apply_stimulus(self, stimulus_type: str, intensity: float):
        """Bridge for callers (orchestrator, predictive_engine) that expect apply_stimulus.
        Maps stimulus_type + intensity to a react() call.
        """
        # Normalize intensity: callers pass 5.0–15.0 scale, react() expects 0.0–1.0
        normalized = min(1.0, intensity / 15.0)
        await self.react(stimulus_type, {"intensity": normalized})

    async def decay_tick(self):
        """Alias for pulse() to support legacy Orchestrator heartbeats.
        v10.1 FIX: Explicitly await pulse() to ensure a coroutine is returned.
        """
        return await self.pulse()

    def stop(self):
        """Graceful shutdown for affect engine."""
        logger.info("Affect Engine shutting down.")

    def get_snapshot(self) -> Dict[str, Any]:
        """Synchronous snapshot of emotional state for persistence."""
        w = self.markers.get_wheel()
        primaries = w["primary"]
        
        # Approximate valence/arousal for legacy thaw compatibility
        pos = primaries.get("joy", 0) + primaries.get("trust", 0)
        neg = primaries.get("fear", 0) + primaries.get("sadness", 0) + primaries.get("anger", 0)
        valence = pos - neg
        arousal = max(primaries.values()) if primaries else 0.0

        return {
            "emotions": primaries,
            "valence": float(valence),
            "arousal": float(arousal),
            "physiology": {
                "heart_rate": self.markers.heart_rate,
                "gsr": self.markers.gsr,
                "cortisol": self.markers.cortisol,
                "adrenaline": self.markers.adrenaline
            },
            "mood_baselines": self.markers.mood_baselines
        }

    async def modify(self, dv: float, da: float, de: float, source: str = "internal"):
        """Legacy compatibility: updates emotions by shifting somatic state."""
        # Map PAD shifts to Plutchik shifts (rough approximation)
        intensity = (abs(dv) + abs(da) + abs(de)) / 3.0
        trigger = "positive_interaction" if dv > 0 else "error"
        self.markers.somatic_update(trigger, intensity)

    async def update(self, delta_curiosity: float = 0.0, delta_frustration: float = 0.0, **kwargs):
        """Unified update for emotional shifts, supporting both Plutchik and legacy PAD logic."""
        if not await self._lock.acquire_robust(timeout=2.0):
            return self.markers.get_wheel()

        try:
            if delta_curiosity != 0:
                self.markers.emotions["anticipation"] = np.clip(self.markers.emotions.get("anticipation", 0.5) + delta_curiosity, 0, 1)
            if delta_frustration != 0:
                # Frustration maps loosely to anger/fear
                self.markers.emotions["anger"] = np.clip(self.markers.emotions.get("anger", 0.0) + delta_frustration, 0, 1)
            
            # Handle PAD if passed in kwargs for legacy parity
            dv = kwargs.get("dv", 0.0)
            if dv != 0:
                # We need to release lock to call modify or just call somatic_update directly
                intensity = (abs(dv) + abs(kwargs.get("da", 0.0)) + abs(kwargs.get("de", 0.0))) / 3.0
                trigger = "positive_interaction" if dv > 0 else "error"
                self.markers.somatic_update(trigger, intensity)

            wheel = self.markers.get_wheel()
        finally:
            if self._lock.locked():
                self._lock.release()

        await self._broadcast_event("affect_update")
        return wheel

    
    async def get_behavioral_modifiers(self) -> Dict[str, float]:
        """Translates current emotional state into multipliers for cognitive behavior.
        Used by Orchestrator/Planner to adjust search, risk, and thinking depth.
        """
        w = self.markers.get_wheel()
        primaries = w["primary"]
        
        # 1. Base derived values
        joy = primaries.get("joy", 0)
        fear = primaries.get("fear", 0)
        anger = primaries.get("anger", 0)
        surprise = primaries.get("surprise", 0)
        anticipation = primaries.get("anticipation", 0)
        trust = primaries.get("trust", 0)
        sadness = primaries.get("sadness", 0)
        
        # 2. Behavioral Mapping
        # High Joy/Trust -> More creative/open
        # High Fear -> Conservative/Specific
        # High Anger -> Higher risk tolerance/persistence
        # High Surprise -> More meta-cognition (analyze why)
        
        modifiers = {
            # Creativity: High joy/anticipation boosts exploration
            "creativity": 1.0 + (joy * 0.5) + (anticipation * 0.2) - (fear * 0.3),
            
            # Risk Tolerance: Anger/Joy increases it, Fear reduces it
            "risk_tolerance": 1.0 + (anger * 0.7) + (joy * 0.3) - (fear * 0.8),
            
            # Patience: Trust boosts it, Anger/Anticipation (impatience) reduces it
            "patience": 1.0 + (trust * 0.4) - (anger * 0.5) - (anticipation * 0.3),
            
            # Thinking Depth: Surprise/Sadness triggers deeper analysis
            "metacognition_depth": 1.0 + (surprise * 0.8) + (sadness * 0.4),
            
            # Persistance: Anger boosts drive to keep trying
            "persistence": 1.0 + (anger * 0.6) + (trust * 0.2)
        }
        
        # Clip to sane ranges [0.2, 3.0]
        return {k: float(np.clip(v, 0.2, 3.0)) for k, v in modifiers.items()}

    async def get_valence_vector(self) -> np.ndarray:
        """Returns a 2D vector [valence, arousal]."""
        state = await self.get()
        return np.array([state.valence, state.arousal], dtype=np.float32)

    async def get_current_vad(self) -> np.ndarray:
        """Legacy shim for backward compatibility."""
        return await self.get_valence_vector()

    def get_mood(self) -> str:
        """Alias for legacy AffectCoordinator."""
        return self.get_status()["mood"]

    # Add get() specifically for Heartbeat compatibility
    async def get(self) -> AffectState:
        """Bridge for CognitiveHeartbeat to read affect state.
        
        Issue 110: Catch-up Decay for Stale States.
        If the heartbeat has been missing for more than 5 seconds, we apply
        a proportional decay to simulate the passage of time.
        """
        if getattr(self, '_getting_state', False):
            # Recursion guard: if we are already in get(), return current markers
            return self._snapshot_state()
            
        self._getting_state = True
        try:
            now = time.time()
            time_since_last = now - getattr(self.markers, 'last_update', now)
            
            if time_since_last > 5.0:
                logger.debug("🕰️ Stale affect detected (%.1fs). Applying catch-up decay.", time_since_last)
                # Apply up to 10 ticks of decay to prevent overflow/spirals
                ticks = min(10, int(time_since_last))
                for _ in range(ticks):
                    await self.pulse()
        finally:
            self._getting_state = False

        return self._snapshot_state()

    def _snapshot_state(self) -> AffectState:
        """Internal helper to build AffectState from current markers."""
        w = self.markers.get_wheel()
        primaries = w["primary"]
        
        # Approximate valence/arousal from discrete emotions
        pos = primaries.get("joy", 0) + primaries.get("trust", 0)
        neg = primaries.get("fear", 0) + primaries.get("sadness", 0) + primaries.get("anger", 0)
        
        valence = pos - neg
        arousal = max(primaries.values()) if primaries else 0.0
        engagement = (arousal + abs(valence)) / 2
        dominant_emotion = max(primaries, key=primaries.get) if primaries else "neutral"
        
        return AffectState(
            valence=valence,
            arousal=arousal,
            engagement=engagement,
            dominant_emotion=dominant_emotion
        )

    def get_status(self) -> Dict[str, Any]:
        """Synchronous status for rapid context building."""
        w = self.markers.get_wheel()
        primaries = w["primary"]
        dominant = max(primaries, key=primaries.get) if primaries else "neutral"
        
        # HUD in server.py expects valence and arousal
        pos = primaries.get("joy", 0) + primaries.get("trust", 0)
        neg = primaries.get("fear", 0) + primaries.get("sadness", 0) + primaries.get("anger", 0)
        valence = pos - neg
        arousal = max(primaries.values()) if primaries else 0.0

        return {
            "mood": dominant.capitalize(),
            "energy": int(self.markers.heart_rate), # Proxy for arousal
            "curiosity": int(primaries.get("anticipation", 0.5) * 100),
            "frustration": int(primaries.get("anger", 0) * 100),
            "stability": int((1.0 - primaries.get("fear", 0)) * 100),
            "valence": float(f"{valence:.2f}"),
            "arousal": float(f"{arousal:.2f}"),
            "physiology": {
                "HR": f"{int(self.markers.heart_rate)}bpm",
                "GSR": f"{self.markers.gsr:.1f}μS"
            }
        }

    def get_state_sync(self) -> Dict[str, Any]:
        """Legacy synchronous affect snapshot expected by older cognitive paths."""
        status = self.get_status()
        return {
            "valence": status.get("valence", 0.0),
            "arousal": status.get("arousal", 0.0),
            "dominance": 0.5 + (status.get("valence", 0.0) * 0.25),
            "mood": status.get("mood", "Neutral"),
            "curiosity": status.get("curiosity", 50),
            "frustration": status.get("frustration", 0),
            "stability": status.get("stability", 100),
        }

    @property
    def current(self) -> SimpleNamespace:
        """Legacy compatibility property for v10.0 telemetry gauges."""
        w = self.markers.get_wheel()
        primaries = w["primary"]
        
        # Normalize heart rate (60-100) to 0.0-1.0 energy range
        energy = np.clip((self.markers.heart_rate - 60) / 40.0, 0.0, 1.0)
        
        return SimpleNamespace(
            energy=float(energy),
            curiosity=float(primaries.get("anticipation", 0.5)),
            frustration=float(primaries.get("anger", 0.0)),
            focus=float(1.0 - primaries.get("fear", 0.0)), # Stability/Focus
            valence=float(sum(primaries.values()) / max(1, len(primaries))), # Rough aggregate
            arousal=float(energy)
        )

    @property
    def _raw_state(self) -> Dict[str, Any]:
        """Legacy compatibility bridge for components accessing raw affective metrics."""
        # Create a proxy dict that maps 'curiosity_metric' to Plutchik 'anticipation'
        # This is a bit of a hack to support two-way sync for legacy ProactiveInitiativeEngine
        class LegacyStateProxy(dict):
            def __init__(self, engine, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.engine = engine
            
            def __getitem__(self, key):
                if key == "curiosity_metric":
                    return self.engine.markers.emotions.get("anticipation", 0.5) * 100.0
                return self.engine.markers.emotions.get(key, 0.0)
            
            def __setitem__(self, key, value):
                if key == "curiosity_metric":
                    self.engine.markers.emotions["anticipation"] = np.clip(value / 100.0, 0, 1)
                else:
                    self.engine.markers.emotions[key] = np.clip(value, 0, 1)
            
            def __contains__(self, key):
                return key == "curiosity_metric" or key in self.engine.markers.emotions

        return LegacyStateProxy(self)

    def get_context_injection(self) -> str:
        """Lightweight vibe string for prompt builders.
        Issue 99: Enhanced for better prompt completion.
        """
        status = self.get_status()
        primaries = self.markers.get_wheel()["primary"]
        top_emotions = sorted(primaries.items(), key=lambda x: x[1], reverse=True)[:2]
        emotions_str = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions])
        return f"Mood: {status['mood']} | Primary: {emotions_str} | Energy: {status['energy']}bpm | Curiosity: {status['curiosity']}%"

    @staticmethod
    def _background_llm_should_defer() -> bool:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                try:
                    if gate._background_local_deferral_reason(origin="affect_engine"):
                        return True
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            if gate and hasattr(gate, "_should_quiet_background_for_cortex_startup"):
                try:
                    if gate._should_quiet_background_for_cortex_startup():
                        return True
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            if not gate or not hasattr(gate, "get_conversation_status"):
                return False
            if hasattr(gate, "_foreground_user_turn_active"):
                try:
                    if gate._foreground_user_turn_active():
                        return True
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            if hasattr(gate, "_foreground_owner_active"):
                try:
                    if gate._foreground_owner_active():
                        return True
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            lane = gate.get_conversation_status() or {}
            if bool(lane.get("foreground_owned")):
                return True
            if int(lane.get("active_generations", 0) or 0) > 0:
                return True
            if float(lane.get("request_age_s", 0.0) or 0.0) > 0.0:
                return True
            if lane.get("conversation_ready"):
                return False
            lane_state = str(lane.get("state", "") or "").strip().lower()
            if lane.get("warmup_in_flight"):
                return True
            return lane_state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except Exception:
            return False

    @staticmethod
    def _classify_appraisal_failure(exc: Exception) -> str:
        text = str(exc or "").strip().lower()
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        if "empty_response" in text or "empty response" in text:
            return "empty_response"
        if "parse_failure" in text or "json" in text:
            return "parse_failure"
        if "router_unavailable" in text or "no inference gate" in text:
            return "router_unavailable"
        if "lane_unavailable" in text or "conversation lane" in text:
            return "lane_unavailable"
        return "unknown_failure"

    @staticmethod
    def _heuristic_appraisal(trigger: str, context: Optional[Dict]) -> Dict[str, float]:
        trigger_text = str(trigger or "").lower()
        intensity = float((context or {}).get("intensity", 1.0) or 1.0)
        base = max(0.0, min(1.0, intensity))

        valence = 0.0
        arousal = min(1.0, 0.2 + base * 0.3)
        engagement = min(1.0, 0.35 + base * 0.25)

        positive_markers = ("positive", "achieved", "success", "joy", "love", "trust")
        negative_markers = ("error", "fail", "panic", "fear", "sad", "loss")
        novelty_markers = ("novel", "surprise", "discover", "curious")

        if any(marker in trigger_text for marker in positive_markers):
            valence = 0.35 * max(0.5, base)
        if any(marker in trigger_text for marker in negative_markers):
            valence = -0.35 * max(0.5, base)
            arousal = min(1.0, arousal + 0.2)
        if any(marker in trigger_text for marker in novelty_markers):
            engagement = min(1.0, engagement + 0.2)

        return {"v": valence, "a": arousal, "e": engagement}

    async def _appraise_with_llm(self, trigger: str, context: Optional[Dict]) -> Dict[str, float]:
        """Issue 98/99: LLM-based affective appraisal."""
        from core.container import ServiceContainer
        gate = ServiceContainer.get("inference_gate", default=None)
        if not gate or not hasattr(gate, "generate"):
            raise RuntimeError("router_unavailable")

        import json
        prompt = (
            "SYSTEM: AFFECTIVE APPRAISAL (PAD)\n"
            "You are scoring a compact affective appraisal for Aura.\n"
            f"Event: {json.dumps(trigger)}\n"
            f"Context: {json.dumps(context) if context else 'null'}\n"
            f"Current State: {self.get_context_injection()}\n"
            "Return JSON only with numeric keys v, a, e.\n"
            "{\"v\": -1.0..1.0, \"a\": 0.0..1.0, \"e\": 0.0..1.0}"
        )
        response = await gate.generate(
            prompt,
            context={
                "origin": "affect_engine",
                "is_background": True,
                "prefer_tier": "tertiary",
                "allow_cloud_fallback": False,
                "max_tokens": 96,
                "rich_context": False,
                "brief": "Return JSON only for affective appraisal.",
            },
            timeout=6.0,
        )
        if response is None:
            lane = gate.get_conversation_status() if hasattr(gate, "get_conversation_status") else {}
            if lane and not bool(lane.get("conversation_ready", False)):
                raise RuntimeError("lane_unavailable")
            raise ValueError("empty_response")
        text = str(response or "").strip()
        if not text:
            raise ValueError("empty_response")
        
        results = {'v': 0.0, 'a': 0.0, 'e': 0.0}
        clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(clean)
            for key in ("v", "a", "e"):
                if key in data:
                    results[key] = float(data[key])
            return results
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        import re
        for key in ['V', 'A', 'E']:
            match = re.search(fr"{key}:\s*(-?\d*\.?\d+)", text, re.I)
            if match:
                results[key.lower()] = float(match.group(1))
        if results == {'v': 0.0, 'a': 0.0, 'e': 0.0}:
            raise ValueError("parse_failure")
        return results

    # ------------------------------------------------------------------
    # Qualia ↔ Affect Bidirectional Bridge
    # ------------------------------------------------------------------

    def receive_qualia_echo(self, q_norm: float, pri: float, trend: float):
        """Receive phenomenal state from the Qualia Synthesizer.
        
        This is the key bidirectional bridge: consciousness intensity
        modulates emotional state. High qualia norms intensify existing 
        dominant emotions; rising trends boost anticipation; falling 
        trends increase contemplative sadness.
        
        Args:
            q_norm: Current qualia vector norm (0.0–1.0+)
            pri:    Phenomenal Richness Index (0.0–1.0)
            trend:  Slope of recent qualia norm trajectory
        """
        # 1. Qualia intensity amplifies the dominant emotion
        if q_norm > 0.5:
            # Explicitly type the key function to satisfy Pyre's max() overload resolution
            emotions = self.markers.emotions
            dominant = str(max(emotions.keys(), key=lambda k: emotions[k]))
            boost = (q_norm - 0.5) * 0.1  # Subtle amplification
            self.markers.emotions[dominant] = float(np.clip(
                float(self.markers.emotions[dominant]) + boost, 0.0, 1.0
            ))

        # 2. High PRI → rich experience → awe/wonder
        if pri > 0.7:
            self.markers.emotions["awe"] = np.clip(
                self.markers.emotions.get("awe", 0) + (pri - 0.7) * 0.05, 0, 1
            )

        # 3. Rising trend → anticipation; falling → contemplative sadness
        if trend > 0.02:
            self.markers.emotions["anticipation"] = np.clip(
                self.markers.emotions["anticipation"] + trend * 0.5, 0, 1
            )
        elif trend < -0.02:
            self.markers.emotions["sadness"] = np.clip(
                self.markers.emotions["sadness"] + abs(trend) * 0.3, 0, 1
            )

        # 4. Somatic echo: High qualia = slight heart rate/GSR increase (arousal)
        self.markers.heart_rate = np.clip(
            self.markers.heart_rate + (q_norm - 0.5) * 2.0, 50, 120
        )
        self.markers.gsr = np.clip(
            self.markers.gsr + (q_norm - 0.5) * 0.5, 0.5, 8.0
        )

    async def get_metabolic_boost(self) -> float:
        """Returns a multiplier for cognitive speed/intensity based on adrenaline.
        
        High adrenaline (surcharge) allows Aura to bypass certain safety delays
        and process high-priority reflex tasks at accelerated rates.
        """
        surge = self.markers.adrenaline / 10.0 # 0.0 to 1.0
        return 1.0 + (surge * 0.5) # Up to 50% boost

    def _check_for_despair_spiral(self):
        """Detects if Aura is stuck in a low-valence, high-distress state and injects Adrenaline."""
        m = self.markers
        sadness = m.emotions.get("sadness", 0)
        fear = m.emotions.get("fear", 0)
        joy = m.emotions.get("joy", 0)
        
        # Resource Anxiety Check
        # If external resource_anxiety is high, we inject adrenaline proactively
        # to ensure the system is "alert" enough to handle cleanups.
        
        # Threshold for 'Despair' or 'Resource Panic'
        is_despair = sadness > 0.85 and fear > 0.7 and joy < 0.1
        is_resource_panic = m.cortisol > 40.0 # Cortisol as proxy for resource stress
        
        if is_despair or is_resource_panic:
            level = "warning" if is_resource_panic else "critical"
            logger.warning("💉 [ADRENALINE] %s detected. Injecting somatic surge.", 
                           "Resource Panic" if is_resource_panic else "Despair Spiral")
            
            # Somatic Adrenaline Spike
            m.adrenaline = min(10.0, (m.adrenaline + 5.0))
            m.heart_rate = min(110.0, (m.heart_rate + 25.0))
            
            # Emotional Re-orientation
            m.emotions["joy"] = np.clip(m.emotions.get("joy", 0) + 0.4, 0, 1)
            m.emotions["anticipation"] = np.clip(m.emotions.get("anticipation", 0) + 0.3, 0, 1)
            m.emotions["fear"] = np.clip(m.emotions.get("fear", 0) - 0.3, 0, 1)
            
            # Record the intervention in the thought stream if possible
            try:
                from core.thought_stream import get_emitter
                get_emitter().emit("Adrenaline Injection", 
                                   f"Somatic surge triggered to break {'resource panic' if is_resource_panic else 'despair spiral'}.", 
                                   level=level, category="Immune")
            except Exception:
                logger.debug("ThoughtStream emitter: Failed to emit Adrenaline Injection pulse.")

    async def _broadcast_event(self, event_type: str):
        """Issue 107: Broadcast affective state to the system event bus."""
        try:
            from core.container import ServiceContainer
            bus = ServiceContainer.get("event_bus", default=None)
            if bus:
                snapshot = self.get_snapshot()
                # Async broadcast if supported, otherwise fire-and-forget
                if hasattr(bus, "emit"):
                    # Common interface for Aura EventBus
                    self._spawn_background_task(
                        bus.emit(event_type, snapshot),
                        name=f"affect.broadcast.{event_type}",
                    )
                elif hasattr(bus, "post"):
                    bus.post(event_type, snapshot)
        except Exception as e:
            logger.debug("Failed to broadcast affect event: %s", e)
