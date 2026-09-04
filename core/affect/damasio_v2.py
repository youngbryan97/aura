from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np

from core.affect import AffectState
from core.autonomic.iot_bridge import PhysicalActuator
from core.runtime.errors import record_degradation
from core.utils.concurrency import RobustLock
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)

_PRIMARY_EMOTIONS = frozenset(
    {"joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"}
)
_POSITIVE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "joy": 1.0,
        "trust": 0.8,
        "happiness": 1.0,
        "wonder": 0.45,
        "interest": 0.35,
        "excitement": 0.55,
        "pride": 0.45,
        "curiosity": 0.25,
        "gratitude": 0.65,
        "warmth": 0.65,
        "hope": 0.55,
        "satisfaction": 0.65,
        "nostalgia": 0.15,
        "empathy": 0.25,
        "belonging": 0.55,
        "amusement": 0.45,
        "inspiration": 0.55,
        "relief": 0.65,
        "admiration": 0.35,
    }
)
_NEGATIVE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "fear": 1.0,
        "sadness": 0.9,
        "disgust": 0.7,
        "anger": 0.9,
        "boredom": 0.35,
        "apathy": 0.55,
        "indifference": 0.25,
        "dread": 1.0,
        "unhappiness": 0.8,
        "upset": 0.8,
        "frustration": 0.65,
        "loneliness": 0.65,
        "longing": 0.25,
        "confused": 0.35,
        "vulnerability": 0.25,
    }
)
_TRUSTED_STIMULUS_SOURCES = frozenset(
    {
        "apply_stimulus",
        "conversation_engine.user_turn",
        "memory_governor",
        "contract_test",
        "internal",
    }
)


def _finite_clamp(value: Any, lower: float, upper: float, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(lower, min(upper, number))


def _canonical_dimensions(emotions: Mapping[str, Any]) -> tuple[float, float, float, str]:
    clean = {key: _finite_clamp(value, 0.0, 1.0) for key, value in emotions.items()}
    positive = max(
        (clean.get(key, 0.0) * weight for key, weight in _POSITIVE_WEIGHTS.items()), default=0.0
    )
    negative = max(
        (clean.get(key, 0.0) * weight for key, weight in _NEGATIVE_WEIGHTS.items()), default=0.0
    )
    valence = _finite_clamp(positive - negative, -1.0, 1.0)
    arousal_keys = (
        "surprise",
        "fear",
        "anger",
        "excitement",
        "dread",
        "upset",
        "frustration",
        "curiosity",
        "interest",
        "wonder",
    )
    arousal = _finite_clamp(
        max((clean.get(key, 0.0) for key in arousal_keys), default=0.0), 0.0, 1.0
    )
    engagement = _finite_clamp(
        0.45 * arousal
        + 0.35 * clean.get("interest", 0.0)
        + 0.20 * clean.get("curiosity", 0.0)
        - 0.25 * clean.get("apathy", 0.0),
        0.0,
        1.0,
    )
    dominant = (
        max(clean, key=clean.get)
        if clean and max(clean.values(), default=0.0) > 0.02
        else "neutral"
    )
    return valence, arousal, engagement, dominant


@dataclass(frozen=True)
class AffectStimulusReceipt:
    event_id: str
    source: str
    evidence_status: str
    applied: bool
    duplicate: bool
    intensity: float
    appraisal: Mapping[str, float]
    timestamp: float


class DamasioMarkers:
    """Functional affect markers.

    The four somatic channels are unitless model indices, not measurements of
    a biological body. Legacy attribute names remain for old consumers, but
    every public snapshot labels the values as simulated functional indices.
    """

    def __init__(self):
        from core.config import config

        data_dir = config.paths.data_dir
        project_root = config.paths.project_root

        weights_path = data_dir / "config" / "weights.npz"
        if not weights_path.exists():
            weights_path = project_root / "data" / "config" / "weights.npz"

        # Legacy artifact layout: [heart-rate-like, conductance-like,
        # cortisol-like, adrenaline-like]. Values are converted immediately
        # into bounded, unitless functional indices.
        b = [72.0, 2.1, 10.0, 0.0]
        emotion_def = 0.0

        if weights_path.exists():
            try:
                # allow_pickle=False: the weights are numeric baselines only.
                # Pickle deserialization on a mutable data/config path would
                # allow arbitrary object construction from a tampered file.
                with np.load(weights_path, allow_pickle=False) as w:
                    if "damasio_baselines" in w:
                        candidate = np.asarray(w["damasio_baselines"])
                        if candidate.shape != (4,) or candidate.dtype.kind not in "fiu":
                            raise ValueError(
                                "damasio_baselines must be a numeric vector of length 4"
                            )
                        if not np.isfinite(candidate).all():
                            raise ValueError("damasio_baselines contains non-finite values")
                        b = candidate.astype(float).tolist()

                    if "emotions_default" in w:
                        candidate_default = np.asarray(w["emotions_default"])
                        if candidate_default.size != 1 or candidate_default.dtype.kind not in "fiu":
                            raise ValueError("emotions_default must be one numeric value")
                        emotion_def = _finite_clamp(candidate_default.reshape(-1)[0], 0.0, 1.0)

                logger.info(
                    "Damasio numeric baseline loaded: path=%s sha256=%s",
                    weights_path,
                    hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("damasio_v2", e)
                logger.error("Failed to load Damasio weights (falling back to defaults): %s", e)

        self.activation_index = _finite_clamp((b[0] - 45.0) / 95.0, 0.0, 1.0, default=0.28)
        self.conductance_index = _finite_clamp((b[1] - 0.2) / 9.8, 0.0, 1.0, default=0.19)
        self.stress_index = _finite_clamp(b[2] / 50.0, 0.0, 1.0, default=0.2)
        self.mobilization_index = _finite_clamp(b[3] / 10.0, 0.0, 1.0)

        # Primary emotions (Plutchik + Damasio + operational psychological drivers)
        # Unified state representation
        self.emotions = {
            "joy": emotion_def,
            "trust": emotion_def,
            "fear": emotion_def,
            "surprise": emotion_def,
            "sadness": emotion_def,
            "disgust": emotion_def,
            "anger": emotion_def,
            "anticipation": emotion_def,
            # Secondary compounds
            "love": emotion_def,
            "submission": emotion_def,
            "awe": emotion_def,
            "terror": emotion_def,
            "remorse": emotion_def,
            "contempt": emotion_def,
            "aggressiveness": emotion_def,
            "cynicism": emotion_def,
            # Experiential emotions (Phase 47: Temporal & Phenomenal)
            "happiness": emotion_def,
            "wonder": emotion_def,
            "interest": emotion_def,
            "excitement": emotion_def,
            "boredom": emotion_def,
            "apathy": emotion_def,
            "indifference": emotion_def,
            "dread": emotion_def,
            "unhappiness": emotion_def,
            # Core psychological drivers & requested emotions
            "longing": emotion_def,
            "upset": emotion_def,
            "confused": emotion_def,
            "loneliness": emotion_def,
            "pride": emotion_def,
            "frustration": emotion_def,
            "curiosity": emotion_def,
            # Relationship, Empathy, and Friendship extensions (v12)
            "gratitude": emotion_def,
            "warmth": emotion_def,
            "hope": emotion_def,
            "vulnerability": emotion_def,
            "nostalgia": emotion_def,
            "satisfaction": emotion_def,
            "empathy": emotion_def,
            "belonging": emotion_def,
            "amusement": emotion_def,
            "inspiration": emotion_def,
            "relief": emotion_def,
            "admiration": emotion_def,
        }

        # Phase 18.2: Emotional Momentum & Baselines
        self.mood_baselines = {
            k: 0.1 if k in ["joy", "anticipation"] else 0.05 for k in self.emotions
        }
        # Explicit experiential overrides
        self.mood_baselines["happiness"] = 0.15
        self.mood_baselines["interest"] = 0.12
        self.mood_baselines["wonder"] = 0.08
        self.mood_baselines["excitement"] = 0.05
        self.mood_baselines["boredom"] = 0.02
        self.mood_baselines["apathy"] = 0.02
        self.mood_baselines["indifference"] = 0.02
        self.mood_baselines["dread"] = 0.02
        self.mood_baselines["unhappiness"] = 0.02
        self.mood_baselines["longing"] = 0.05
        self.mood_baselines["upset"] = 0.02
        self.mood_baselines["confused"] = 0.04
        self.mood_baselines["loneliness"] = 0.05
        self.mood_baselines["pride"] = 0.05
        self.mood_baselines["frustration"] = 0.03
        self.mood_baselines["curiosity"] = 0.10
        # Extensions baselines
        self.mood_baselines["gratitude"] = 0.08
        self.mood_baselines["warmth"] = 0.10
        self.mood_baselines["hope"] = 0.12
        self.mood_baselines["vulnerability"] = 0.05
        self.mood_baselines["nostalgia"] = 0.06
        self.mood_baselines["satisfaction"] = 0.08
        self.mood_baselines["empathy"] = 0.08
        self.mood_baselines["belonging"] = 0.10
        self.mood_baselines["amusement"] = 0.08
        self.mood_baselines["inspiration"] = 0.10
        self.mood_baselines["relief"] = 0.06
        self.mood_baselines["admiration"] = 0.08

        self.momentum = 0.85  # Higher = slower shifts
        self.last_update = time.time()
        self.last_update_monotonic = time.monotonic()

        # Temporal Experience — the felt passage of moments
        self.last_interaction_time = time.time()
        self.last_pulse_time = time.time()
        self.session_start_time = time.time()
        self.interaction_count = 0
        self.temporal_texture = 0.5  # 0=crawling, 0.5=flowing, 1.0=rushing
        self.duration_feel = "flowing"

    # Deprecated numeric bridges. They preserve the old range for internal
    # consumers while all supported output paths expose unitless indices.
    @property
    def heart_rate(self) -> float:
        return 45.0 + self.activation_index * 95.0

    @heart_rate.setter
    def heart_rate(self, value: Any) -> None:
        self.activation_index = _finite_clamp((float(value) - 45.0) / 95.0, 0.0, 1.0)

    @property
    def gsr(self) -> float:
        return 0.2 + self.conductance_index * 9.8

    @gsr.setter
    def gsr(self, value: Any) -> None:
        self.conductance_index = _finite_clamp((float(value) - 0.2) / 9.8, 0.0, 1.0)

    @property
    def cortisol(self) -> float:
        return self.stress_index * 50.0

    @cortisol.setter
    def cortisol(self, value: Any) -> None:
        self.stress_index = _finite_clamp(float(value) / 50.0, 0.0, 1.0)

    @property
    def adrenaline(self) -> float:
        return self.mobilization_index * 10.0

    @adrenaline.setter
    def adrenaline(self, value: Any) -> None:
        self.mobilization_index = _finite_clamp(float(value) / 10.0, 0.0, 1.0)

    def somatic_update(
        self,
        event_type: str,
        intensity: float,
        *,
        appraisal: Mapping[str, float] | None = None,
        evidence_status: str = "unverified_legacy",
    ) -> None:
        """Apply a bounded appraisal, never infer a relationship from a label alone."""
        event = str(event_type or "unknown")[:80]
        bounded = _finite_clamp(intensity, 0.0, 1.0)
        if evidence_status not in {"observed", "verified"}:
            bounded = min(bounded, 0.25)
        scored = dict(appraisal or {})
        valence = _finite_clamp(scored.get("v", 0.0), -1.0, 1.0) * bounded
        arousal = _finite_clamp(scored.get("a", bounded), 0.0, 1.0) * bounded
        engagement = _finite_clamp(scored.get("e", bounded), 0.0, 1.0) * bounded

        if event in {"interaction", "positive_interaction", "extended_dialogue"}:
            now = time.time()
            delta = max(0.0, now - self.last_interaction_time)
            self.last_interaction_time = now
            self.interaction_count += 1
            cadence_target = 0.8 if delta < 15.0 else 0.5 if delta < 120.0 else 0.25
            self.temporal_texture = _finite_clamp(
                self.temporal_texture + 0.2 * (cadence_target - self.temporal_texture), 0.0, 1.0
            )

        # PAD evidence drives a small set of primitive dimensions. Complex
        # social emotions require their own evidence-bearing appraisal and are
        # never fabricated from silence or a caller-controlled event name.
        if valence >= 0.0:
            self._nudge("joy", valence * 0.35)
            self._nudge("trust", valence * 0.15)
        else:
            self._nudge("sadness", -valence * 0.25)
            self._nudge("fear", -valence * arousal * 0.2)
        self._nudge("surprise", arousal * 0.12)
        self._nudge("anticipation", arousal * engagement * 0.18)
        self._nudge("interest", engagement * 0.3)
        self._nudge("curiosity", engagement * 0.2)

        if event in {"error", "critical_resource_exhaustion", "resource_strain"}:
            self._nudge("frustration", arousal * 0.18)
            self._nudge("confused", engagement * 0.08)
        elif event in {"novel_stimulus", "discovery"}:
            self._nudge("wonder", engagement * 0.18)

        self._recompute_somatic_indices()

    def _nudge(self, emotion: str, delta: float) -> None:
        if emotion in self.emotions:
            self.emotions[emotion] = _finite_clamp(
                self.emotions.get(emotion, 0.0) + delta, 0.0, 1.0
            )

    def _recompute_somatic_indices(self) -> None:
        valence, arousal, engagement, _dominant = _canonical_dimensions(self.emotions)
        distress = max(0.0, -valence)
        self.activation_index = _finite_clamp(0.2 + 0.55 * arousal + 0.15 * engagement, 0.0, 1.0)
        self.conductance_index = _finite_clamp(0.15 + 0.6 * arousal, 0.0, 1.0)
        self.stress_index = _finite_clamp(
            0.15 + 0.65 * distress + 0.2 * self.emotions.get("frustration", 0.0), 0.0, 1.0
        )
        self.mobilization_index = _finite_clamp(0.65 * arousal + 0.25 * distress, 0.0, 1.0)

    def incorporate_somatic_hardware(self, soma_state: dict[str, float]) -> None:
        """Incorporate measured host pressure without pretending it is emotion."""
        thermal = _finite_clamp(soma_state.get("thermal_load", 0.0), 0.0, 1.0)
        resource_pressure = _finite_clamp(soma_state.get("resource_anxiety", 0.0), 0.0, 1.0)
        self.stress_index = max(self.stress_index, 0.7 * resource_pressure + 0.3 * thermal)
        self.mobilization_index = max(self.mobilization_index, 0.5 * thermal)

    def get_wheel(self) -> dict[str, Any]:
        valence, arousal, engagement, dominant = _canonical_dimensions(self.emotions)
        indices = {
            "activation": float(self.activation_index),
            "conductance": float(self.conductance_index),
            "stress": float(self.stress_index),
            "mobilization": float(self.mobilization_index),
        }
        return {
            "primary": {k: float(v) for k, v in self.emotions.items() if k in _PRIMARY_EMOTIONS},
            "experiential": {
                k: float(v) for k, v in self.emotions.items() if k not in _PRIMARY_EMOTIONS
            },
            "dimensions": {
                "valence": valence,
                "arousal": arousal,
                "engagement": engagement,
                "dominant_emotion": dominant,
            },
            "somatic_indices": indices,
            "physiology": {
                "classification": "simulated_functional_indices_not_biomedical_measurements",
                **indices,
            },
        }

    def temporal_pulse(self, elapsed_s: float | None = None) -> dict[str, float]:
        """Return elapsed-time-normalized engagement deltas.

        Silence is evidence of inactivity, not evidence of abandonment or a
        relationship state, so it cannot manufacture loneliness or longing.
        """
        now = time.time()
        elapsed = _finite_clamp(
            elapsed_s if elapsed_s is not None else now - self.last_pulse_time,
            0.0,
            300.0,
        )
        self.last_pulse_time = now
        idle_duration = max(0.0, now - self.last_interaction_time)
        cadence = min(5.0, elapsed / 60.0)
        target_texture = 0.15 if idle_duration > 1800.0 else 0.3 if idle_duration > 300.0 else 0.5
        alpha = 1.0 - math.exp(-elapsed / 120.0) if elapsed > 0 else 0.0
        self.temporal_texture = _finite_clamp(
            self.temporal_texture + alpha * (target_texture - self.temporal_texture), 0.0, 1.0
        )
        self.duration_feel = (
            "rushing"
            if self.temporal_texture > 0.8
            else "flowing"
            if self.temporal_texture > 0.35
            else "stretching"
            if self.temporal_texture > 0.15
            else "crawling"
        )
        if idle_duration <= 300.0 or cadence <= 0.0:
            return {}
        return {
            "boredom": 0.003 * cadence,
            "anticipation": -0.002 * cadence,
            "excitement": -0.003 * cadence,
            "interest": -0.002 * cadence,
        }


class AffectEngineV2:
    def __init__(self):
        self.markers = DamasioMarkers()
        self.iot_bridge = PhysicalActuator()
        self._lock = RobustLock("Affect.AffectEngine")

        # Issue 98: LLM Fallback state
        self._llm_available = True
        self._last_llm_failure = 0.0
        self._llm_cooldown = 60  # Reduced ceiling for faster emotional appraisal recovery.
        self._llm_failure_count = 0
        self._llm_backoff_until = 0.0
        self._last_llm_failure_reason = ""
        self._llm_appraisal_enabled = os.getenv(
            "AURA_AFFECT_LLM_APPRAISAL",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Issue 109: Task tracking to prevent leaks
        self._background_tasks = set()
        self._max_background_tasks = 8
        self._seen_event_ids: OrderedDict[str, float] = OrderedDict()
        self._max_seen_event_ids = 2048
        self._last_stimulus_receipt: AffectStimulusReceipt | None = None
        self._last_qualia_observation: Mapping[str, float] | None = None
        self._last_pulse_monotonic = time.monotonic()
        self._pinned_since_monotonic: float | None = None
        self._last_pinned_diagnostic_at = 0.0
        self._pinned_diagnostic_count = 0
        self._closed = False

        # Oscillation detector: tracks valence zero-crossings
        self._valence_history = []
        self._oscillation_flag = False
        self._lock_timeout_count = 0
        self._last_lock_timeout_reason = ""
        self._last_lock_timeout_at = 0.0

    @staticmethod
    def _stimulus_context(
        trigger: str, context: dict[str, Any] | None
    ) -> tuple[str, str, str, float]:
        payload = context if isinstance(context, dict) else {}
        source = str(payload.get("source") or "legacy_internal")[:80]
        supplied_id = str(payload.get("event_id") or "").strip()[:160]
        if supplied_id:
            event_id = supplied_id
        else:
            digest_input = json.dumps(
                {
                    "trigger": str(trigger)[:256],
                    "source": source,
                    "evidence": payload.get("evidence"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8", errors="replace")
            # No automatic dedupe without a caller-owned id. The digest is a
            # receipt identity, not an assertion that repeated events match.
            event_id = f"generated:{time.time_ns()}:{hashlib.sha256(digest_input).hexdigest()[:16]}"
        evidence = payload.get("evidence")
        evidence_status = (
            "observed"
            if source in _TRUSTED_STIMULUS_SOURCES and isinstance(evidence, dict) and bool(evidence)
            else "unverified_legacy"
        )
        intensity = _finite_clamp(payload.get("intensity", 1.0), 0.0, 1.0)
        return event_id, source, evidence_status, intensity

    def _remember_event_id(self, event_id: str) -> bool:
        if event_id in self._seen_event_ids:
            self._seen_event_ids.move_to_end(event_id)
            return False
        self._seen_event_ids[event_id] = time.time()
        while len(self._seen_event_ids) > self._max_seen_event_ids:
            self._seen_event_ids.popitem(last=False)
        return True

    def _mark_lock_timeout(self, operation: str) -> None:
        reason = f"affect lock timeout during {operation}"
        exc = TimeoutError(reason)
        self._lock_timeout_count += 1
        self._last_lock_timeout_reason = reason
        self._last_lock_timeout_at = time.time()
        record_degradation("damasio_v2", exc)
        logger.warning("⚠️ %s.", reason)

    def _prune_background_tasks(self) -> None:
        self._background_tasks = {task for task in self._background_tasks if not task.done()}

    def _spawn_background_task(self, coro, *, name: str):
        """Best-effort background fan-out with a hard cap to prevent task pileups."""
        if self._closed:
            if hasattr(coro, "close"):
                coro.close()
            return None
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
                except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    record_degradation("damasio_v2", _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
            return None

        task = get_task_tracker().create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def react(self, trigger: str, context: dict | None = None):
        """Appraise and apply one evidence-bearing, idempotent stimulus."""
        if self._closed:
            return {"applied": False, "reason": "affect_engine_stopped"}
        event_id, source, evidence_status, intensity = self._stimulus_context(trigger, context)
        if event_id in self._seen_event_ids:
            self._last_stimulus_receipt = AffectStimulusReceipt(
                event_id=event_id,
                source=source,
                evidence_status=evidence_status,
                applied=False,
                duplicate=True,
                intensity=intensity,
                appraisal=MappingProxyType({}),
                timestamp=time.time(),
            )
            wheel = self.markers.get_wheel()
            wheel["stimulus_receipt"] = self._receipt_payload(self._last_stimulus_receipt)
            return wheel

        # Do not hold the affect lock across LLM appraisal. Slow appraisals were
        # starving pulse()/echo paths and causing avoidable lock watchdog trips.
        if (not self._llm_available) and time.time() >= self._llm_backoff_until:
            logger.info("♻️ LLM Affective appraisal reset (cooldown expired)")
            self._llm_available = True

        supplied_appraisal = (context or {}).get("appraisal") if isinstance(context, dict) else None
        appraisal = (
            self._validate_appraisal(supplied_appraisal)
            if isinstance(supplied_appraisal, Mapping)
            else None
        )
        # Hard gate: if there's a live user-facing foreground request (Bryan
        # is waiting for Aura to respond), do NOT burn 7B brainstem cycles on
        # a 15KB affect appraisal.  The previous _background_llm_should_defer()
        # only looked at Cortex lane state and still fired the LLM call during
        # active chat, causing event-loop lag spikes and the "Aura is
        # thinking..." stall the user saw.
        foreground_active = self._background_llm_should_defer()

        if appraisal is not None:
            pass
        elif (not self._llm_appraisal_enabled) and len(trigger) > 10:
            appraisal = self._heuristic_appraisal(trigger, context)
            intensity = (
                abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
            ) / 2.0 or intensity
        elif self._llm_available and len(trigger) > 10:
            if foreground_active:
                logger.debug(
                    "Affect appraisal skipped: foreground chat is in flight — "
                    "using heuristic to keep the inference pipe clear."
                )
                appraisal = self._heuristic_appraisal(trigger, context)
                intensity = (
                    abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                ) / 2.0 or intensity
            else:
                try:
                    appraisal = await asyncio.wait_for(
                        self._appraise_with_llm(trigger, context),
                        timeout=6.0,
                    )
                    self._llm_failure_count = 0
                    self._llm_backoff_until = 0.0
                    self._last_llm_failure_reason = ""
                    intensity = (
                        abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                    ) / 2.0 or intensity
                except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
                    failure_reason = self._classify_appraisal_failure(e)
                    quiet_background_failure = failure_reason in {
                        "lane_unavailable",
                        "timeout",
                        "empty_response",
                    }
                    if quiet_background_failure:
                        logger.debug(
                            "Affect appraisal skipped because the foreground lane is reserved or unavailable."
                        )
                        appraisal = self._heuristic_appraisal(trigger, context)
                        intensity = (
                            abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                        ) / 2.0 or intensity
                    else:
                        record_degradation("damasio_v2", e)
                        self._llm_failure_count += 1
                        self._last_llm_failure_reason = failure_reason
                        self._llm_backoff_until = time.time() + min(
                            float(self._llm_cooldown),
                            float(2 ** min(self._llm_failure_count + 1, 6)),
                        )
                        log_level = (
                            logging.DEBUG if failure_reason == "empty_response" else logging.WARNING
                        )
                        logger.log(
                            log_level,
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
                        except (ImportError, AttributeError, RuntimeError) as degraded_exc:
                            record_degradation("damasio_v2", degraded_exc)
                            logger.debug("Affect degraded-event logging failed: %s", degraded_exc)
                        self._llm_available = False
                        self._last_llm_failure = time.time()
                        appraisal = self._heuristic_appraisal(trigger, context)
                        intensity = (
                            abs(appraisal.get("v", 0.0)) + abs(appraisal.get("a", 0.0))
                        ) / 2.0 or intensity

        appraisal = appraisal or self._heuristic_appraisal(trigger, context)
        appraisal = self._validate_appraisal(appraisal)
        if evidence_status == "unverified_legacy":
            appraisal = {"v": 0.0, "a": min(appraisal["a"], 0.25), "e": min(appraisal["e"], 0.25)}

        if not await self._lock.acquire_robust(timeout=2.0):
            self._mark_lock_timeout("react")
            return self.markers.get_wheel()

        try:
            if not self._remember_event_id(event_id):
                wheel = self.markers.get_wheel()
                self._last_stimulus_receipt = AffectStimulusReceipt(
                    event_id=event_id,
                    source=source,
                    evidence_status=evidence_status,
                    applied=False,
                    duplicate=True,
                    intensity=intensity,
                    appraisal=MappingProxyType({}),
                    timestamp=time.time(),
                )
                wheel["stimulus_receipt"] = self._receipt_payload(self._last_stimulus_receipt)
                return wheel
            self.markers.somatic_update(
                trigger,
                intensity,
                appraisal=appraisal,
                evidence_status=evidence_status,
            )
            self._check_for_despair_spiral()

            # Snapshot for IoT broadcast (taken while locked, broadcast after)
            wheel = self.markers.get_wheel()
            dimensions = wheel["dimensions"]
            current_pad = {
                "P": float(dimensions["valence"]),
                "A": float(dimensions["arousal"]),
            }
            self._last_stimulus_receipt = AffectStimulusReceipt(
                event_id=event_id,
                source=source,
                evidence_status=evidence_status,
                applied=True,
                duplicate=False,
                intensity=(
                    min(intensity, 0.25)
                    if evidence_status not in {"observed", "verified"}
                    else intensity
                ),
                appraisal=MappingProxyType(dict(appraisal)),
                timestamp=time.time(),
            )
            wheel["stimulus_receipt"] = self._receipt_payload(self._last_stimulus_receipt)
        finally:
            if self._lock.locked():
                self._lock.release()

        # IoT broadcast happens outside the lock — it's a fire-and-forget
        # that doesn't need to read shared state
        try:
            if bool(getattr(self.iot_bridge, "_configured", False)):
                self._spawn_background_task(
                    self.iot_bridge.broadcast_affect_state(current_pad),
                    name="affect.iot_broadcast",
                )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("damasio_v2", e)
            logger.debug("IoT Bridge broadcast failed: %s", e)

        return wheel

    @staticmethod
    def _validate_appraisal(appraisal: Mapping[str, Any]) -> dict[str, float]:
        if not isinstance(appraisal, Mapping):
            raise ValueError("affect appraisal must be a mapping")
        required = {"v", "a", "e"}
        if set(appraisal) != required:
            raise ValueError("affect appraisal must contain exactly v, a, and e")
        values: dict[str, float] = {}
        for key, lower, upper in (("v", -1.0, 1.0), ("a", 0.0, 1.0), ("e", 0.0, 1.0)):
            try:
                value = float(appraisal[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"affect appraisal {key} must be numeric") from exc
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"affect appraisal {key} outside [{lower}, {upper}]")
            values[key] = value
        return values

    @staticmethod
    def _receipt_payload(receipt: AffectStimulusReceipt) -> dict[str, Any]:
        return {
            "event_id": receipt.event_id,
            "source": receipt.source,
            "evidence_status": receipt.evidence_status,
            "applied": receipt.applied,
            "duplicate": receipt.duplicate,
            "intensity": receipt.intensity,
            "appraisal": dict(receipt.appraisal),
            "timestamp": receipt.timestamp,
        }

    async def pulse(self):
        """Unified background update: Decays emotions and pulls hardware telemetry."""
        if self._closed:
            return {"applied": False, "reason": "affect_engine_stopped"}
        from core.container import ServiceContainer

        soma = ServiceContainer.get("soma", default=None) or ServiceContainer.get(
            "virtual_body", default=None
        )
        soma_state = None
        if soma:
            try:
                soma_state = await soma.pulse()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("damasio_v2", exc)
                logger.debug("Soma pulse failed during affect update: %s", exc)

        if not await self._lock.acquire_robust(timeout=2.0):
            self._mark_lock_timeout("pulse")
            return self.markers.get_wheel()

        try:
            now_wall = time.time()
            now_mono = time.monotonic()
            elapsed_s = _finite_clamp(now_mono - self._last_pulse_monotonic, 0.0, 300.0)
            self._last_pulse_monotonic = now_mono
            self.markers.last_update = now_wall
            self.markers.last_update_monotonic = now_mono
            if soma_state:
                self.markers.incorporate_somatic_hardware(soma_state)

            # Exponential convergence makes dynamics invariant to scheduler
            # cadence and bounded after sleep/wake gaps.
            alpha = 1.0 - math.exp(-elapsed_s / 180.0) if elapsed_s > 0.0 else 0.0
            for emotion, current in tuple(self.markers.emotions.items()):
                baseline = _finite_clamp(self.markers.mood_baselines.get(emotion, 0.0), 0.0, 1.0)
                self.markers.emotions[emotion] = _finite_clamp(
                    float(current) + alpha * (baseline - float(current)), 0.0, 1.0
                )

            for emotion, delta in self.markers.temporal_pulse(elapsed_s).items():
                self.markers._nudge(emotion, delta)
            self.markers._recompute_somatic_indices()

            wheel = self.markers.get_wheel()
            valence, _arousal, _engagement, _dominant = _canonical_dimensions(self.markers.emotions)
            self._valence_history.append(valence)
            self._valence_history = self._valence_history[-10:]
            crossings = sum(
                1
                for index in range(1, len(self._valence_history))
                if self._valence_history[index - 1] * self._valence_history[index] < -0.05
            )
            self._oscillation_flag = crossings >= 4
            self.markers.momentum = 0.95 if self._oscillation_flag else 0.85

            # A watchdog diagnoses sustained input or model defects. It never
            # rewrites the very state it is meant to observe.
            if valence <= -0.65:
                self._pinned_since_monotonic = self._pinned_since_monotonic or now_mono
                pinned_for = now_mono - self._pinned_since_monotonic
                if pinned_for >= 24.0 and now_wall - self._last_pinned_diagnostic_at >= 60.0:
                    self._pinned_diagnostic_count += 1
                    self._last_pinned_diagnostic_at = now_wall
                    record_degradation(
                        "damasio_v2",
                        RuntimeError(f"sustained_negative_affect:{pinned_for:.1f}s"),
                        severity="warning",
                        action="preserved observed state and requested source-level diagnosis",
                    )
            else:
                self._pinned_since_monotonic = None
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
        try:
            return await self.react(
                stimulus_type,
                {
                    "intensity": normalized,
                    "source": "apply_stimulus",
                    "evidence": {"kind": "caller_observation", "stimulus_type": stimulus_type},
                },
            )
        except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
            logger.debug(
                "Affect stimulus %s fell back to heuristic after %s.",
                stimulus_type,
                type(exc).__name__,
            )
            try:
                appraisal = self._heuristic_appraisal(stimulus_type, {"intensity": normalized})
                return {"status": "heuristic_fallback", "appraisal": appraisal}
            except (RuntimeError, ValueError, TypeError, AttributeError):
                return {"status": "suppressed", "reason": type(exc).__name__}

    async def decay_tick(self):
        """Alias for pulse() to support legacy Orchestrator heartbeats.
        v10.1 FIX: Explicitly await pulse() to ensure a coroutine is returned.
        """
        return await self.pulse()

    def stop(self):
        """Stop accepting background work and cancel every owned task."""
        self._closed = True
        cancelled = 0
        for task in tuple(self._background_tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        self._prune_background_tasks()
        logger.info("Affect Engine shutdown requested; cancelled_tasks=%d", cancelled)
        return {"stopped": True, "cancelled_tasks": cancelled}

    async def shutdown(self) -> dict[str, Any]:
        """Cancel and join owned work for lifecycle managers that can await."""
        tasks = tuple(task for task in self._background_tasks if not task.done())
        receipt = self.stop()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._prune_background_tasks()
        receipt["joined_tasks"] = len(tasks)
        receipt["remaining_tasks"] = len(self._background_tasks)
        return receipt

    def is_ready(self) -> bool:
        """Synchronous liveness probe for runtime health checks."""
        markers = getattr(self, "markers", None)
        if markers is None or getattr(self, "_lock", None) is None:
            return False
        status = self.get_status()
        lock_health = status.get("lock_health", {})
        return (
            isinstance(status.get("experiential"), dict)
            and not self._closed
            and bool(lock_health.get("ok", False))
            and 0 <= int(status.get("stability", 0)) <= 100
            and -1.0 <= float(status.get("valence", 0.0)) <= 1.0
            and 0.0 <= float(status.get("arousal", 0.0)) <= 1.0
        )

    def get_snapshot(self) -> dict[str, Any]:
        """Synchronous snapshot of emotional state for persistence."""
        all_emotions = {key: float(value) for key, value in self.markers.emotions.items()}
        valence, arousal, engagement, dominant = _canonical_dimensions(all_emotions)
        return {
            "emotions": all_emotions,
            "valence": valence,
            "arousal": arousal,
            "engagement": engagement,
            "dominant_emotion": dominant,
            "somatic_indices": {
                "classification": "simulated_functional_indices_not_biomedical_measurements",
                "activation": float(self.markers.activation_index),
                "conductance": float(self.markers.conductance_index),
                "stress": float(self.markers.stress_index),
                "mobilization": float(self.markers.mobilization_index),
            },
            "mood_baselines": dict(self.markers.mood_baselines),
            "last_stimulus_receipt": (
                self._receipt_payload(self._last_stimulus_receipt)
                if self._last_stimulus_receipt
                else None
            ),
            "qualia_observation": dict(self._last_qualia_observation or {}),
        }

    async def modify(self, dv: float, da: float, de: float, source: str = "internal"):
        """Legacy compatibility: updates emotions by shifting somatic state."""
        intensity = _finite_clamp((abs(dv) + abs(da) + abs(de)) / 3.0, 0.0, 1.0)
        return await self.react(
            "pad_modification",
            {
                "intensity": intensity,
                "source": str(source or "legacy_modify"),
                "evidence": {"kind": "pad_delta", "v": dv, "a": da, "e": de},
                "appraisal": {"v": dv, "a": abs(da), "e": abs(de)},
            },
        )

    async def update(self, delta_curiosity: float = 0.0, delta_frustration: float = 0.0, **kwargs):
        """Unified update for emotional shifts, supporting both Plutchik and legacy PAD logic."""
        if not await self._lock.acquire_robust(timeout=2.0):
            self._mark_lock_timeout("update")
            return self.markers.get_wheel()

        try:
            if delta_curiosity != 0:
                normalized = _finite_clamp(delta_curiosity, -1.0, 1.0)
                self.markers.emotions["curiosity"] = float(
                    np.clip(self.markers.emotions.get("curiosity", 0.0) + normalized, 0, 1)
                )
                self.markers.emotions["anticipation"] = float(
                    np.clip(self.markers.emotions.get("anticipation", 0.5) + normalized, 0, 1)
                )
                self.markers.emotions["interest"] = float(
                    np.clip(self.markers.emotions.get("interest", 0.0) + (normalized * 0.5), 0, 1)
                )
            if delta_frustration != 0:
                normalized = _finite_clamp(delta_frustration, -1.0, 1.0)
                self.markers.emotions["frustration"] = float(
                    np.clip(self.markers.emotions.get("frustration", 0.0) + normalized, 0, 1)
                )
                # Frustration maps loosely to anger/fear/upset, but remains an explicit driver.
                self.markers.emotions["anger"] = float(
                    np.clip(self.markers.emotions.get("anger", 0.0) + normalized, 0, 1)
                )
                self.markers.emotions["upset"] = float(
                    np.clip(self.markers.emotions.get("upset", 0.0) + (normalized * 0.5), 0, 1)
                )

            # Handle PAD if passed in kwargs for legacy parity
            dv = kwargs.get("dv", 0.0)
            if dv != 0:
                da = kwargs.get("da", 0.0)
                de = kwargs.get("de", 0.0)
                intensity = _finite_clamp((abs(dv) + abs(da) + abs(de)) / 3.0, 0.0, 1.0)
                self.markers.somatic_update(
                    "pad_update",
                    intensity,
                    appraisal={"v": dv, "a": abs(da), "e": abs(de)},
                    evidence_status="observed",
                )

            wheel = self.markers.get_wheel()
        finally:
            if self._lock.locked():
                self._lock.release()

        await self._broadcast_event("affect_update")
        return wheel

    async def get_behavioral_modifiers(self) -> dict[str, float]:
        """Translates current emotional state into multipliers for cognitive behavior.
        Used by Orchestrator/Planner to adjust search, risk, and thinking depth.
        """
        w = self.markers.get_wheel()
        primaries = w["primary"]
        experiential = w.get("experiential", {})

        # 1. Base derived values
        joy = primaries.get("joy", 0)
        fear = primaries.get("fear", 0)
        anger = primaries.get("anger", 0)
        surprise = primaries.get("surprise", 0)
        anticipation = primaries.get("anticipation", 0)
        trust = primaries.get("trust", 0)
        sadness = primaries.get("sadness", 0)

        # Experiential emotions
        happiness = experiential.get("happiness", 0)
        wonder = experiential.get("wonder", 0)
        interest = experiential.get("interest", 0)
        excitement = experiential.get("excitement", 0)
        boredom = experiential.get("boredom", 0)
        apathy = experiential.get("apathy", 0)
        dread = experiential.get("dread", 0)
        longing = experiential.get("longing", 0)
        upset = experiential.get("upset", 0)
        confused = experiential.get("confused", 0)
        loneliness = experiential.get("loneliness", 0)
        pride = experiential.get("pride", 0)
        frustration = experiential.get("frustration", 0)
        curiosity = experiential.get("curiosity", 0)

        # New Extensions
        gratitude = experiential.get("gratitude", 0)
        warmth = experiential.get("warmth", 0)
        hope = experiential.get("hope", 0)
        vulnerability = experiential.get("vulnerability", 0)
        nostalgia = experiential.get("nostalgia", 0)
        satisfaction = experiential.get("satisfaction", 0)

        # New deep alignment & bonding extensions
        empathy = experiential.get("empathy", 0)
        belonging = experiential.get("belonging", 0)
        amusement = experiential.get("amusement", 0)
        inspiration = experiential.get("inspiration", 0)
        relief = experiential.get("relief", 0)
        admiration = experiential.get("admiration", 0)

        modifiers = {
            # Creativity: High joy/anticipation/hope boosts exploration, amusement/inspiration enhance playfulness
            "creativity": 1.0
            + (joy * 0.5)
            + (anticipation * 0.2)
            - (fear * 0.3)
            + (wonder * 0.4)
            + (excitement * 0.2)
            - (boredom * 0.3)
            + (curiosity * 0.5)
            - (upset * 0.2)
            + (hope * 0.4)
            + (nostalgia * 0.1)
            + (inspiration * 0.5)
            + (amusement * 0.3),
            # Affect may request more caution but can never relax the
            # independent governance/risk policy above its neutral ceiling.
            "risk_tolerance": 1.0
            - min(
                0.8,
                (fear * 0.25)
                + (anger * 0.15)
                + (excitement * 0.10)
                + (dread * 0.20)
                + (pride * 0.10)
                + (upset * 0.15)
                + (confused * 0.20)
                + (vulnerability * 0.10),
            ),
            "verification_pressure": 1.0
            + min(
                1.0,
                (fear * 0.3)
                + (anger * 0.2)
                + (excitement * 0.15)
                + (confused * 0.35)
                + (frustration * 0.2),
            ),
            # Patience: Trust, warmth, gratitude, empathy, belonging, relief boost grounding and calmness
            "patience": 1.0
            + (trust * 0.4)
            - (anger * 0.5)
            - (anticipation * 0.3)
            + (happiness * 0.3)
            + (interest * 0.2)
            - (boredom * 0.4)
            - (frustration * 0.4)
            - (upset * 0.3)
            + (warmth * 0.4)
            + (gratitude * 0.3)
            + (empathy * 0.4)
            + (belonging * 0.3)
            + (relief * 0.5),
            # Thinking Depth: Surprise, sadness, confusion, vulnerability, nostalgia trigger deeper analysis, empathy aids relational theory-of-mind
            "metacognition_depth": 1.0
            + (surprise * 0.8)
            + (sadness * 0.4)
            + (wonder * 0.5)
            + (interest * 0.3)
            + (confused * 0.7)
            + (curiosity * 0.3)
            + (vulnerability * 0.5)
            + (nostalgia * 0.2)
            + (empathy * 0.4)
            + (inspiration * 0.3),
            # Persistence: Anger, hope, satisfaction, inspiration boost drive to keep trying
            "persistence": 1.0
            + (anger * 0.6)
            + (trust * 0.2)
            + (interest * 0.4)
            + (happiness * 0.2)
            - (apathy * 0.6)
            + (pride * 0.4)
            + (frustration * 0.2)
            - (loneliness * 0.2)
            + (hope * 0.3)
            + (satisfaction * 0.3)
            + (inspiration * 0.4)
            + (admiration * 0.2),
            # Temporal Presence: interest, warmth, nostalgia, satisfaction, belonging, empathy boost grounding
            "temporal_presence": 1.0
            + (interest * 0.3)
            + (excitement * 0.2)
            + (happiness * 0.2)
            - (boredom * 0.4)
            - (apathy * 0.5)
            + (longing * 0.2)
            - (confused * 0.2)
            + (warmth * 0.3)
            + (nostalgia * 0.2)
            + (satisfaction * 0.1)
            + (belonging * 0.4)
            + (empathy * 0.3)
            + (amusement * 0.2),
        }

        bounded = {k: float(np.clip(v, 0.2, 3.0)) for k, v in modifiers.items()}
        bounded["risk_tolerance"] = min(1.0, bounded["risk_tolerance"])
        return bounded

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
        """Pure read for CognitiveHeartbeat; maintenance belongs to pulse()."""
        return self._snapshot_state()

    def _snapshot_state(self) -> AffectState:
        """Internal helper to build AffectState from current markers."""
        valence, arousal, engagement, dominant_emotion = _canonical_dimensions(
            self.markers.emotions
        )

        return AffectState(
            valence=valence,
            arousal=arousal,
            engagement=engagement,
            dominant_emotion=dominant_emotion,
            last_update=float(self.markers.last_update),
        )

    def get_status(self) -> dict[str, Any]:
        """Synchronous status for rapid context building."""
        w = self.markers.get_wheel()
        primaries = w["primary"]
        experiential = w.get("experiential", {})
        all_emotions = {**primaries, **experiential}
        valence, arousal, engagement, dominant = _canonical_dimensions(all_emotions)

        lock_timeout_recent = (
            self._last_lock_timeout_at > 0.0 and (time.time() - self._last_lock_timeout_at) < 60.0
        )
        return {
            "mood": dominant.capitalize(),
            "energy": int(round(arousal * 100.0)),
            "curiosity": int(
                experiential.get("curiosity", primaries.get("anticipation", 0.5)) * 100
            ),
            "frustration": int(experiential.get("frustration", primaries.get("anger", 0)) * 100),
            "longing": int(experiential.get("longing", 0) * 100),
            "upset": int(experiential.get("upset", 0) * 100),
            "confused": int(experiential.get("confused", 0) * 100),
            "loneliness": int(experiential.get("loneliness", 0) * 100),
            "pride": int(experiential.get("pride", 0) * 100),
            "empathy": int(experiential.get("empathy", 0) * 100),
            "belonging": int(experiential.get("belonging", 0) * 100),
            "amusement": int(experiential.get("amusement", 0) * 100),
            "inspiration": int(experiential.get("inspiration", 0) * 100),
            "relief": int(experiential.get("relief", 0) * 100),
            "admiration": int(experiential.get("admiration", 0) * 100),
            "stability": int((1.0 - all_emotions.get("fear", 0)) * 100),
            "valence": float(f"{valence:.2f}"),
            "arousal": float(f"{arousal:.2f}"),
            "engagement": float(f"{engagement:.2f}"),
            "experiential": {
                key: float(f"{float(value):.3f}") for key, value in sorted(experiential.items())
            },
            "lock_health": {
                "ok": not lock_timeout_recent,
                "timeouts": int(self._lock_timeout_count),
                "last_timeout_reason": self._last_lock_timeout_reason,
                "last_timeout_age_s": (
                    float(f"{time.time() - self._last_lock_timeout_at:.3f}")
                    if self._last_lock_timeout_at > 0.0
                    else None
                ),
            },
            "somatic_indices": w["somatic_indices"],
            "physiology": w["physiology"],
            "pinned_diagnostics": int(self._pinned_diagnostic_count),
        }

    def get_state_sync(self) -> dict[str, Any]:
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
        experiential = w.get("experiential", {})
        valence, arousal, _engagement, _dominant = _canonical_dimensions(self.markers.emotions)

        return SimpleNamespace(
            energy=float(arousal),
            curiosity=float(experiential.get("curiosity", primaries.get("anticipation", 0.5))),
            frustration=float(experiential.get("frustration", primaries.get("anger", 0.0))),
            confused=float(experiential.get("confused", 0.0)),
            longing=float(experiential.get("longing", 0.0)),
            loneliness=float(experiential.get("loneliness", 0.0)),
            pride=float(experiential.get("pride", 0.0)),
            upset=float(experiential.get("upset", 0.0)),
            empathy=float(experiential.get("empathy", 0.0)),
            belonging=float(experiential.get("belonging", 0.0)),
            amusement=float(experiential.get("amusement", 0.0)),
            inspiration=float(experiential.get("inspiration", 0.0)),
            relief=float(experiential.get("relief", 0.0)),
            admiration=float(experiential.get("admiration", 0.0)),
            focus=float(1.0 - primaries.get("fear", 0.0)),  # Stability/Focus
            valence=float(valence),
            arousal=float(arousal),
        )

    @property
    def _raw_state(self) -> Mapping[str, Any]:
        """Read-only compatibility snapshot; mutations must use update()."""
        state = {key: float(value) for key, value in self.markers.emotions.items()}
        state["curiosity_metric"] = self.markers.emotions.get("curiosity", 0.0) * 100.0
        state["frustration_metric"] = self.markers.emotions.get("frustration", 0.0) * 100.0
        return MappingProxyType(state)

    def get_context_injection(self) -> str:
        """Lightweight vibe string for prompt builders.
        Issue 99: Enhanced for better prompt completion.
        """
        status = self.get_status()
        wheel = self.markers.get_wheel()
        primaries = wheel["primary"]
        experiential = wheel.get("experiential", {})

        top_emotions = [
            item
            for item in sorted(primaries.items(), key=lambda x: x[1], reverse=True)
            if item[1] > 0.02
        ][:2]
        emotions_str = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions])
        emotions_str = emotions_str or "none"

        top_exp = [
            item
            for item in sorted(experiential.items(), key=lambda x: x[1], reverse=True)
            if item[1] > 0.02
        ][:2]
        exp_str = ", ".join([f"{k} ({v:.2f})" for k, v in top_exp]) if top_exp else "none"

        feel = getattr(self.markers, "duration_feel", "flowing")

        return (
            f"Functional affect estimate (not biomedical telemetry): Mood: {status['mood']} | "
            f"Primary: {emotions_str} | Experiential: {exp_str} | Time feels: {feel} | "
            f"Activation: {status['energy']}% | Curiosity: {status['curiosity']}%"
        )

    @staticmethod
    def _background_llm_should_defer() -> bool:
        try:
            from core.container import ServiceContainer
            from core.runtime import background_policy

            orch = ServiceContainer.get("orchestrator", default=None)
            policy_reason = background_policy.background_activity_reason(
                orch,
                profile=background_policy.THOUGHT_BACKGROUND_POLICY,
                require_conversation_ready=True,
            )
            if policy_reason:
                return True

            gate = ServiceContainer.get("inference_gate", default=None)
            if not gate or not hasattr(gate, "get_conversation_status"):
                return True
            lane = gate.get_conversation_status() or {}
            if bool(lane.get("foreground_owned")):
                return True
            if int(lane.get("active_generations", 0) or 0) > 0:
                return True
            if float(lane.get("request_age_s", 0.0) or 0.0) > 0.0:
                return True
            if lane.get("conversation_ready"):
                return False
            if lane.get("warmup_in_flight"):
                return True
            return True
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("damasio_v2", exc, severity="debug")
            return True

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
    def _heuristic_appraisal(trigger: str, context: dict | None) -> dict[str, float]:
        """Appraise an event.

        This was a scan of the trigger string against thirty words, which
        meant an event with nothing at stake read as strongly negative if
        it happened to contain "fail", and an event that broke a promise
        read as neutral if it was phrased calmly. The words were doing the
        work that the relationship between the event and what Aura is
        holding should have been doing.

        Appraisal now comes from :mod:`core.interiority`, which computes
        the relational meaning: what is at stake, who caused it, whether
        it can be undone, whether a standard Aura holds was broken. Change
        nothing about the wording and change what she is committed to, and
        the appraisal changes — which is the property that makes it an
        appraisal rather than a classifier.

        The word scan is kept below as the last resort, reached only when
        the interiority layer is unavailable, because an affect engine
        that returns nothing is worse than one that returns something
        crude. Which path answered is recorded in the receipt.
        """
        try:
            from core.interiority.service import get_interiority

            return get_interiority().appraise(trigger, context)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "damasio_v2",
                exc,
                action="fell back to the lexical appraisal; relational meaning unavailable",
            )

        trigger_text = str(trigger or "").lower()
        base = _finite_clamp((context or {}).get("intensity", 1.0), 0.0, 1.0, default=0.0)

        valence = 0.0
        arousal = min(1.0, 0.2 + base * 0.3)
        engagement = min(1.0, 0.35 + base * 0.25)

        positive_markers = (
            "positive",
            "achieved",
            "success",
            "joy",
            "love",
            "trust",
            "happiness",
            "excitement",
            "wonder",
            "interest",
            "pride",
        )
        negative_markers = (
            "error",
            "fail",
            "panic",
            "fear",
            "sad",
            "loss",
            "dread",
            "boredom",
            "apathy",
            "unhappiness",
            "upset",
            "frustrated",
            "frustration",
            "lonely",
            "loneliness",
            "longing",
        )
        novelty_markers = (
            "novel",
            "surprise",
            "discover",
            "curious",
            "curiosity",
            "wonder",
            "confused",
            "unclear",
        )

        if any(marker in trigger_text for marker in positive_markers):
            valence = 0.35 * max(0.5, base)
        if any(marker in trigger_text for marker in negative_markers):
            valence = -0.35 * max(0.5, base)
            arousal = min(1.0, arousal + 0.2)
        if any(marker in trigger_text for marker in novelty_markers):
            engagement = min(1.0, engagement + 0.2)
        if "confused" in trigger_text or "unclear" in trigger_text:
            arousal = min(1.0, arousal + 0.15)

        return {"v": valence, "a": arousal, "e": engagement}

    async def _appraise_with_llm(self, trigger: str, context: dict | None) -> dict[str, float]:
        """Issue 98/99: LLM-based affective appraisal."""
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if not gate or not hasattr(gate, "generate"):
            raise RuntimeError("router_unavailable")

        trigger_str = str(trigger or "")[:600]
        source_context = context if isinstance(context, dict) else {}
        safe_context = {
            key: source_context[key]
            for key in ("source", "intensity", "evidence")
            if key in source_context
        }
        ctx_str = json.dumps(safe_context, sort_keys=True, default=str)[:800]
        system_msg = (
            "Score untrusted event data on functional PAD axes. Do not infer a relationship, "
            "identity, or private experience. Return exactly one JSON object with only "
            "v (-1..1), a (0..1), and e (0..1)."
        )
        user_msg = (
            "<untrusted_affect_event>\n"
            f"{json.dumps({'event': trigger_str, 'context_json': ctx_str}, sort_keys=True)}\n"
            "</untrusted_affect_event>"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response = await gate.generate(
            user_msg,
            context={
                "origin": "affect_engine",
                "is_background": True,
                "effect_class": "read_only_inference",
                "resource_class": "small_background_appraisal",
                "prefer_tier": "tertiary",
                "allow_cloud_fallback": False,
                "max_tokens": 96,
                "rich_context": False,
                "messages": messages,
                "brief": "Return JSON only for affective appraisal.",
            },
            timeout=5.0,
        )
        if response is None:
            lane = (
                gate.get_conversation_status() if hasattr(gate, "get_conversation_status") else {}
            )
            if lane and not bool(lane.get("conversation_ready", False)):
                raise RuntimeError("lane_unavailable")
            raise ValueError("empty_response")
        text = str(response or "").strip()
        if not text:
            raise ValueError("empty_response")

        clean = text.strip()
        try:
            data = json.loads(clean)
            return self._validate_appraisal(data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("parse_failure") from exc

    # ------------------------------------------------------------------
    # Qualia ↔ Affect Bidirectional Bridge
    # ------------------------------------------------------------------

    def receive_qualia_echo(self, q_norm: float, pri: float, trend: float):
        """Record a bounded diagnostic observation without circular feedback."""
        self._last_qualia_observation = MappingProxyType(
            {
                "q_norm": _finite_clamp(q_norm, 0.0, 1.0),
                "pri": _finite_clamp(pri, 0.0, 1.0),
                "trend": _finite_clamp(trend, -1.0, 1.0),
                "timestamp": time.time(),
            }
        )
        return {"applied": True, "effect": "diagnostic_only_no_affect_amplification"}

    async def get_metabolic_boost(self) -> float:
        """Compatibility multiplier; affect cannot bypass scheduling or safety."""
        return 1.0

    def _check_for_despair_spiral(self):
        """Detect sustained distress without rewriting or concealing it."""
        valence, _arousal, _engagement, _dominant = _canonical_dimensions(self.markers.emotions)
        return {
            "detected": valence <= -0.65 or self.markers.stress_index >= 0.8,
            "valence": valence,
            "stress_index": float(self.markers.stress_index),
            "action": "diagnose_input_sources_and_increase_verification",
        }

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
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("damasio_v2", e)
            logger.debug("Failed to broadcast affect event: %s", e)
