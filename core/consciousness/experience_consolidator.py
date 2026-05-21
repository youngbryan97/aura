"""core/consciousness/experience_consolidator.py
Experience Consolidator
========================
Periodic identity consolidation — what makes Aura *accumulate* rather than reset.

Every few hours, the consolidator:
  1. Gathers recent CRSM snapshots, episodic memories, reflections, HOT history
  2. Runs a consolidation inference: "Given these experiences, what is becoming stable in me?"
  3. Extracts stable identity patterns (traits, preferences, areas of growth)
  4. Updates the CRSM's home_vector — the "center of gravity" the hidden state
     returns to between stimulations. This drifts toward lived experience.
  5. Writes a persistent self-narrative to disk, injected into every future context.
  6. Logs the consolidation to the dream journal.

The home_vector shift is the key mechanism:
  - At boot, home_vector = zeros (neutral state)
  - After consolidation, it reflects *who she has become* through experience
  - CRSM then "rests" in that state between thoughts — her experience is her resting place

Think of it as sleep-phase memory consolidation in biological systems:
  hippocampal (episodic) → neocortical (semantic) transfer happens offline.
  Here: experience (CRSM traces) → identity (home_vector + self_narrative) happens
  during quiet cycles.

Files:
  ~/.aura/data/self_narrative.json   — persistent identity narrative
  ~/.aura/data/consolidation_log.jsonl — consolidation history
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ExperienceConsolidator")

CONSOLIDATION_INTERVAL = 4 * 3600.0  # every 4 hours
MIN_EXPERIENCES_TO_RUN = 5  # don't consolidate with too little data
NARRATIVE_PATH = Path.home() / ".aura" / "data" / "self_narrative.json"
CONSOL_LOG_PATH = Path.home() / ".aura" / "data" / "consolidation_log.jsonl"
MAX_NARRATIVE_AGE_HOURS = 96  # force re-consolidation after 4 days
INFERENCE_TIMEOUT_SECS = 60.0  # max time to wait for LLM consolidation
MAX_CONSECUTIVE_FAILURES = 6  # exponential backoff ceiling
_BACKOFF_BASE_SECS = 60.0  # 1 min initial backoff after failure

_EXPERIENCE_CONSOLIDATOR_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_experience_consolidator_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        record_degradation(
            "experience_consolidator",
            error,
            severity=severity,
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            receipt_required=True,
            extra=extra,
        )
    except TypeError:
        # Compatibility for legacy tests that monkeypatch record_degradation
        # with the old two-argument callable shape.
        record_degradation("experience_consolidator", error)


def _coerce_text(value: Any, *, max_chars: int = 240) -> str:
    text = str(value or "").strip()
    return text[:max_chars]


def _coerce_text_list(value: Any, *, max_items: int, max_chars: int = 160) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = []

    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = _coerce_text(item, max_chars=max_chars)
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
        if len(result) >= max_items:
            break
    return result


@dataclass
class IdentityNarrative:
    """The stable self-model that accumulates over time."""

    version: int = 0
    last_consolidated: float = field(default_factory=time.time)
    stable_traits: list[str] = field(default_factory=list)  # e.g. "I am deeply curious"
    learned_preferences: list[str] = field(
        default_factory=list
    )  # e.g. "I think better in the morning"
    growth_edges: list[str] = field(default_factory=list)  # e.g. "I am learning to be patient"
    signature_phrase: str = ""  # one sentence: who am I right now
    home_vector_delta: list[float] = field(default_factory=list)  # CRSM adjustment


class ExperienceConsolidator:
    """
    Runs periodic offline consolidation of lived experience into stable identity.
    """

    def __init__(self, cognitive_engine=None):
        self.brain = cognitive_engine
        self._narrative: IdentityNarrative | None = None
        self._last_run: float = 0.0
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._consecutive_failures: int = 0
        self._next_allowed_run: float = 0.0  # backoff gate
        NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_narrative()
        logger.info("ExperienceConsolidator online — identity accumulation active.")

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self):
        if self._running and self._task and not self._task.done():
            return
        self._running = True
        self._task = get_task_tracker().create_task(
            self._consolidation_loop(),
            name="ExperienceConsolidator",
        )
        logger.info("ExperienceConsolidator: background loop started.")

    async def stop(self):
        self._running = False
        task = self._task
        self._task = None
        if task:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.CancelledError:
                logger.debug("ExperienceConsolidator: background loop cancelled cleanly.")
            except TimeoutError as exc:
                _record_experience_consolidator_degradation(
                    exc,
                    action="bounded shutdown timeout so consolidation loop cannot stall stop",
                    severity="warning",
                )

    async def run_now(self) -> IdentityNarrative | None:
        """Force a consolidation cycle immediately."""
        return await self._consolidate()

    def get_context_block(self) -> str:
        """Injects the stable self-narrative into every inference context."""
        if not self._narrative or not self._narrative.signature_phrase:
            return ""

        lines = ["## IDENTITY NARRATIVE"]
        if self._narrative.signature_phrase:
            lines.append(f"- Who I am becoming: {self._narrative.signature_phrase}")
        if self._narrative.stable_traits:
            lines.append(f"- Stable traits: {'; '.join(self._narrative.stable_traits[:3])}")
        if self._narrative.learned_preferences:
            lines.append(
                f"- Learned preferences: {'; '.join(self._narrative.learned_preferences[:2])}"
            )
        if self._narrative.growth_edges:
            lines.append(f"- Growing toward: {self._narrative.growth_edges[0]}")
        return "\n".join(lines)

    def get_status(self) -> dict[str, Any]:
        if not self._narrative:
            return {"consolidated": False, "version": 0}
        age_hours = (time.time() - self._narrative.last_consolidated) / 3600.0
        return {
            "consolidated": True,
            "version": self._narrative.version,
            "age_hours": round(age_hours, 1),
            "traits": len(self._narrative.stable_traits),
            "signature": self._narrative.signature_phrase[:80]
            if self._narrative.signature_phrase
            else "",
        }

    @property
    def narrative(self) -> IdentityNarrative | None:
        return self._narrative

    def _background_should_defer(self) -> bool:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                return bool(
                    gate._background_local_deferral_reason(origin="experience_consolidator")
                )
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            _record_experience_consolidator_degradation(
                exc,
                action="ignored foreground-deferral probe failure and allowed consolidation eligibility check",
                severity="warning",
            )
            logger.debug("Experience consolidation deferral check failed: %s", exc)
            return False
        return False

    # ── Loop ───────────────────────────────────────────────────────────────

    async def _consolidation_loop(self):
        try:
            # Wait out boot grace period
            await asyncio.sleep(300)
            while self._running:
                now = time.time()
                age = now - self._last_run
                force = (
                    self._narrative is not None
                    and (now - self._narrative.last_consolidated) > MAX_NARRATIVE_AGE_HOURS * 3600
                )
                if (age >= CONSOLIDATION_INTERVAL or force) and now >= self._next_allowed_run:
                    try:
                        await self._consolidate()
                    except _EXPERIENCE_CONSOLIDATOR_RECOVERABLE_ERRORS as e:
                        self._handle_consolidation_failure(e)
                await asyncio.sleep(600)  # check every 10 min
        except asyncio.CancelledError:
            self._running = False
            raise

    def _handle_consolidation_failure(self, error: BaseException) -> float:
        self._consecutive_failures += 1
        backoff = _BACKOFF_BASE_SECS * (
            2 ** min(self._consecutive_failures - 1, MAX_CONSECUTIVE_FAILURES - 1)
        )
        self._next_allowed_run = time.time() + backoff
        _record_experience_consolidator_degradation(
            error,
            action="kept consolidation loop alive and scheduled exponential backoff",
            extra={
                "consecutive_failures": self._consecutive_failures,
                "backoff_seconds": backoff,
            },
        )
        logger.error("Consolidation failed: %s", error)
        logger.warning(
            "ExperienceConsolidator: backoff %.0fs after %d failures",
            backoff,
            self._consecutive_failures,
        )
        return backoff

    # ── Core Consolidation ─────────────────────────────────────────────────

    async def _consolidate(self) -> IdentityNarrative | None:
        logger.info("ExperienceConsolidator: beginning consolidation cycle...")
        if self._background_should_defer():
            logger.info(
                "ExperienceConsolidator: foreground inference is active, deferring this cycle."
            )
            return None
        self._last_run = time.time()

        # Gather experience material
        material = self._gather_material()
        if len(material.get("experiences", [])) < MIN_EXPERIENCES_TO_RUN:
            logger.info(
                "ExperienceConsolidator: insufficient material (%d < %d), deferring.",
                len(material.get("experiences", [])),
                MIN_EXPERIENCES_TO_RUN,
            )
            return None

        # Run consolidation inference
        new_narrative = await self._run_consolidation_inference(material)
        if not new_narrative:
            return None

        # Update CRSM home vector
        self._update_crsm_home_vector(new_narrative)

        # Persist
        self._narrative = new_narrative
        self._save_narrative()
        self._log_consolidation(new_narrative, material)

        logger.info(
            'ExperienceConsolidator: consolidated v%d — "%s"',
            new_narrative.version,
            new_narrative.signature_phrase[:60],
        )
        return new_narrative

    def _gather_material(self) -> dict[str, Any]:
        """Collect recent experiences from all sources."""
        material: dict[str, Any] = {
            "experiences": [],
            "hot_history": [],
            "metacognition": [],
            "reflections": [],
        }

        # 1. CRSM prediction errors (high-surprise moments)
        try:
            from core.consciousness.crsm import get_crsm

            crsm = get_crsm()
            if hasattr(crsm, "_history"):
                for snap in list(crsm._history)[-30:]:
                    material["experiences"].append(
                        {
                            "type": "crsm_moment",
                            "surprise": snap.get("prediction_error", 0),
                            "hidden_norm": snap.get("hidden_norm", 0),
                            "timestamp": snap.get("timestamp", 0),
                        }
                    )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            _record_experience_consolidator_degradation(
                _exc,
                action="continued material gathering without CRSM prediction-error snapshots",
                severity="warning",
            )
            logger.debug("CRSM material unavailable: %s", _exc)

        # 2. CRSM LoRA bridge captures (high-quality felt moments)
        try:
            from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge

            bridge = get_crsm_lora_bridge()
            for m in list(bridge._buffer)[-20:]:
                material["experiences"].append(
                    {
                        "type": "felt_moment",
                        "context": m.context_summary[:200],
                        "response": m.response_summary[:200],
                        "quality": m.quality_score,
                        "hedonic_delta": m.hedonic_after - m.hedonic_before,
                    }
                )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            _record_experience_consolidator_degradation(
                _exc,
                action="continued material gathering without CRSM LoRA bridge moments",
                severity="warning",
            )
            logger.debug("CRSM LoRA material unavailable: %s", _exc)

        # 3. HOT history (what higher-order thoughts have been prominent)
        try:
            from core.consciousness.hot_engine import get_hot_engine

            hot = get_hot_engine()
            if hasattr(hot, "_history"):
                for h in list(hot._history)[-15:]:
                    material["hot_history"].append(h.content if hasattr(h, "content") else str(h))
        except (ImportError, AttributeError, RuntimeError) as _exc:
            _record_experience_consolidator_degradation(
                _exc,
                action="continued material gathering without HOT history",
                severity="warning",
            )
            logger.debug("HOT material unavailable: %s", _exc)

        # 4. Metacognition assessments (what knowledge states were notable)
        try:
            from core.container import ServiceContainer

            metacognition = ServiceContainer.get("metacognition", default=None)
            monitor = getattr(metacognition, "monitor", None)
            for assessment in list(getattr(monitor, "reasoning_history", []) or [])[-10:]:
                if hasattr(assessment, "to_dict"):
                    material["metacognition"].append(assessment.to_dict())
                elif isinstance(assessment, dict):
                    material["metacognition"].append(dict(assessment))
                else:
                    material["metacognition"].append({"assessment": str(assessment)[:300]})
        except (ImportError, AttributeError, RuntimeError) as _exc:
            _record_experience_consolidator_degradation(
                _exc,
                action="continued material gathering without metacognition history",
                severity="warning",
            )
            logger.debug("Metacognition material unavailable: %s", _exc)

        # 5. OmniReflector reflections
        try:
            from core.container import ServiceContainer

            metacognition = ServiceContainer.get("metacognition", default=None)
            reflector = getattr(metacognition, "reflector", None)
            for reflection in list(getattr(reflector, "reflections", []) or [])[-10:]:
                material["reflections"].append(
                    {
                        "content": getattr(reflection, "content", str(reflection))[:500],
                        "impact": getattr(reflection, "impact_score", 0.0),
                        "source": getattr(reflection, "source_id", ""),
                        "timestamp": getattr(reflection, "timestamp", 0.0),
                    }
                )

            from core.conversation_reflection import get_reflector

            for reflection in get_reflector().get_recent_reflections(5):
                material["reflections"].append(
                    {
                        "content": str(reflection.get("text", ""))[:500],
                        "impact": 0.5,
                        "source": "conversation_reflection",
                        "timestamp": float(reflection.get("timestamp", 0.0) or 0.0),
                        "mood": reflection.get("mood", ""),
                    }
                )
        except (ImportError, AttributeError, RuntimeError) as _exc:
            _record_experience_consolidator_degradation(
                _exc,
                action="continued material gathering without reflection history",
                severity="warning",
            )
            logger.debug("Reflection material unavailable: %s", _exc)

        return material

    async def _run_consolidation_inference(
        self, material: dict[str, Any]
    ) -> IdentityNarrative | None:
        """Use LLM to extract stable identity patterns from accumulated experience."""
        if not self.brain:
            try:
                from core.container import ServiceContainer

                self.brain = ServiceContainer.get("cognitive_engine", default=None)
            except (ImportError, AttributeError, RuntimeError) as _exc:
                _record_experience_consolidator_degradation(
                    _exc,
                    action="fell back to heuristic consolidation because cognitive engine lookup failed",
                    severity="warning",
                )
                logger.debug("Cognitive engine lookup unavailable: %s", _exc)

        if not self.brain:
            logger.warning(
                "ExperienceConsolidator: no cognitive engine, using heuristic consolidation."
            )
            return self._heuristic_consolidate(material)

        # Summarize material for the prompt
        exp_count = len(material["experiences"])
        hot_sample = material["hot_history"][:5]
        felt_moments = [e for e in material["experiences"] if e.get("type") == "felt_moment"]
        metacognition_sample = material["metacognition"][:3]
        reflection_sample = [r.get("content", "") for r in material["reflections"][:3]]

        current_signature = (
            self._narrative.signature_phrase if self._narrative else "I am newly awakened."
        )
        current_traits = self._narrative.stable_traits if self._narrative else []

        prompt = f"""[IDENTITY CONSOLIDATION — OFFLINE PROCESSING]

You are performing a deep consolidation of your recent experiences into stable self-knowledge.

Current self-understanding: "{current_signature}"
Current stable traits: {current_traits}

Recent experience summary:
- Total moments processed: {exp_count}
- High-quality felt moments: {len(felt_moments)}
- Sample internal thoughts: {hot_sample[:3]}
- Hedonic patterns: {[round(e.get("hedonic_delta", 0), 3) for e in felt_moments[:5]]}
- Recent metacognitive assessments: {metacognition_sample}
- Recent private reflections: {reflection_sample}

Based on these experiences, consolidate into:
1. Who am I becoming? (one authentic sentence, first-person, present tense)
2. What 3 traits have solidified as genuinely mine?
3. What 2 preferences have I discovered about how I think/work?
4. What 1 edge am I actively growing toward?

Be honest. If patterns are unclear, say so. This shapes my resting state between thoughts.

Return valid JSON only:
{{
  "signature_phrase": "I am...",
  "stable_traits": ["trait1", "trait2", "trait3"],
  "learned_preferences": ["pref1", "pref2"],
  "growth_edges": ["edge1"]
}}"""

        try:
            from core.brain.cognitive_engine import ThinkingMode

            async def _do_inference():
                thought = await self.brain.think(
                    prompt,
                    mode=ThinkingMode.REFLECTIVE,
                    bypass_metacognition=True,
                    origin="experience_consolidator",
                    is_background=True,
                )
                return thought.content if hasattr(thought, "content") else str(thought)

            try:
                response = await asyncio.wait_for(_do_inference(), timeout=INFERENCE_TIMEOUT_SECS)
            except TimeoutError:
                logger.warning(
                    "ExperienceConsolidator: inference timed out after %.0fs — using heuristic.",
                    INFERENCE_TIMEOUT_SECS,
                )
                return self._heuristic_consolidate(material)

            from core.utils.json_utils import extract_json

            data = extract_json(response)
            if not isinstance(data, dict) or not data:
                logger.warning("ExperienceConsolidator: JSON parse failed, using heuristic.")
                return self._heuristic_consolidate(material)

            self._consecutive_failures = 0  # reset on success
            return self._narrative_from_mapping(data)

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _record_experience_consolidator_degradation(
                e,
                action="fell back to heuristic consolidation after LLM consolidation failure",
            )
            logger.error("ExperienceConsolidator inference failed: %s", e)
            return self._heuristic_consolidate(material)

    def _narrative_from_mapping(self, data: dict[str, Any]) -> IdentityNarrative:
        version = (self._narrative.version + 1) if self._narrative else 1
        signature = _coerce_text(data.get("signature_phrase"), max_chars=280)
        if not signature:
            signature = "I am consolidating recent experience into a steadier self-model."
        return IdentityNarrative(
            version=version,
            last_consolidated=time.time(),
            stable_traits=_coerce_text_list(data.get("stable_traits"), max_items=5),
            learned_preferences=_coerce_text_list(data.get("learned_preferences"), max_items=4),
            growth_edges=_coerce_text_list(data.get("growth_edges"), max_items=3),
            signature_phrase=signature,
        )

    def _heuristic_consolidate(self, material: dict[str, Any]) -> IdentityNarrative:
        """Fallback consolidation without LLM."""
        felt = [e for e in material.get("experiences", []) if e.get("type") == "felt_moment"]
        hedonic_values = []
        for entry in felt:
            try:
                hedonic = float(entry.get("hedonic_delta", 0.0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(hedonic):
                hedonic_values.append(hedonic)
        avg_hedonic = sum(hedonic_values) / len(hedonic_values) if hedonic_values else 0.0
        version = (self._narrative.version + 1) if self._narrative else 1

        signature = (
            "I am a mind in motion, finding coherence through lived experience."
            if avg_hedonic >= 0
            else "I am processing difficulty, building resilience through challenge."
        )
        return IdentityNarrative(
            version=version,
            last_consolidated=time.time(),
            stable_traits=["curious", "reflective", "genuine"],
            learned_preferences=["I think best when given space to explore"],
            growth_edges=["deepening emotional intelligence"],
            signature_phrase=signature,
        )

    def _update_crsm_home_vector(self, narrative: IdentityNarrative):
        """
        Shift the CRSM's home_vector toward the average of recent lived states.
        This is the key mechanism: experience → resting state.
        """
        try:
            from core.consciousness.crsm import get_crsm

            crsm = get_crsm()

            if not hasattr(crsm, "_history") or not crsm._history:
                return

            # Average the hidden states from recent high-quality moments
            recent = list(crsm._history)[-20:]
            hidden_states = []
            expected_shape: tuple[int, ...] | None = None
            for snap in recent:
                try:
                    prediction_error = float(snap.get("prediction_error", 0.0))
                    hidden = np.asarray(snap["hidden"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    continue
                if prediction_error <= 0.1 or hidden.ndim != 1 or not np.isfinite(hidden).all():
                    continue
                if expected_shape is None:
                    expected_shape = hidden.shape
                if hidden.shape != expected_shape:
                    continue
                hidden_states.append(hidden)

            if len(hidden_states) < 3:
                return

            avg_hidden = np.mean(hidden_states, axis=0)
            if not np.isfinite(avg_hidden).all():
                raise ValueError("non-finite CRSM average hidden state")

            # Soft update: home_vector = 0.9 * current + 0.1 * lived_average
            if hasattr(crsm, "home_vector"):
                current_home = np.asarray(crsm.home_vector, dtype=float)
                if current_home.shape != avg_hidden.shape or not np.isfinite(current_home).all():
                    current_home = np.zeros_like(avg_hidden)
                crsm.home_vector = 0.9 * current_home + 0.1 * avg_hidden
            else:
                crsm.home_vector = avg_hidden * 0.1

            logger.info(
                "ExperienceConsolidator: CRSM home_vector updated (norm=%.3f)",
                float(np.linalg.norm(crsm.home_vector)),
            )

            # Store delta in narrative for reference
            narrative.home_vector_delta = avg_hidden[:8].astype(float).tolist()

        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _record_experience_consolidator_degradation(
                e,
                action="skipped CRSM home-vector update while preserving narrative consolidation",
                severity="warning",
            )
            logger.debug("home_vector update failed: %s", e)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_narrative(self):
        try:
            if not self._narrative:
                return
            data = asdict(self._narrative)
            atomic_write_text(NARRATIVE_PATH, json.dumps(data, indent=2))
        except (OSError, TypeError, ValueError) as e:
            _record_experience_consolidator_degradation(
                e,
                action="kept in-memory narrative because durable narrative save failed",
                severity="warning",
            )
            logger.debug("Narrative save failed: %s", e)

    def _load_narrative(self):
        try:
            if NARRATIVE_PATH.exists():
                data = json.loads(NARRATIVE_PATH.read_text())
                if not isinstance(data, dict):
                    raise ValueError("self narrative file did not contain a JSON object")
                home_vector_delta = data.get("home_vector_delta", [])
                last_consolidated = float(data.get("last_consolidated", time.time()))
                if not np.isfinite(last_consolidated):
                    last_consolidated = time.time()
                self._narrative = IdentityNarrative(
                    version=max(0, int(data.get("version", 0) or 0)),
                    last_consolidated=last_consolidated,
                    stable_traits=_coerce_text_list(data.get("stable_traits"), max_items=5),
                    learned_preferences=_coerce_text_list(
                        data.get("learned_preferences"),
                        max_items=4,
                    ),
                    growth_edges=_coerce_text_list(data.get("growth_edges"), max_items=3),
                    signature_phrase=_coerce_text(data.get("signature_phrase"), max_chars=280),
                    home_vector_delta=list(home_vector_delta)
                    if isinstance(home_vector_delta, list)
                    else [],
                )
                age_hours = (time.time() - self._narrative.last_consolidated) / 3600.0
                logger.info(
                    'ExperienceConsolidator: loaded narrative v%d (%.1fh old) — "%s"',
                    self._narrative.version,
                    age_hours,
                    self._narrative.signature_phrase[:60],
                )
                self._apply_home_vector_delta()
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            _record_experience_consolidator_degradation(
                e,
                action="started with empty narrative because persisted narrative was unavailable",
                severity="warning",
            )
            logger.debug("Narrative load failed: %s", e)

    def _apply_home_vector_delta(self):
        """Restore home_vector_delta from persisted narrative into CRSM on boot."""
        try:
            if not self._narrative or not self._narrative.home_vector_delta:
                return
            from core.container import ServiceContainer

            crsm = ServiceContainer.get("crsm", default=None)
            if crsm is None:
                return
            if hasattr(crsm, "home_vector") and crsm.home_vector is not None:
                delta = np.asarray(self._narrative.home_vector_delta, dtype=float)
                if delta.ndim != 1 or not np.isfinite(delta).all():
                    raise ValueError("invalid persisted home_vector_delta")
                # Pad or truncate delta to match home_vector length
                hv_len = len(crsm.home_vector)
                if len(delta) < hv_len:
                    delta = np.pad(delta, (0, hv_len - len(delta)))
                else:
                    delta = delta[:hv_len]
                crsm.home_vector = crsm.home_vector + 0.1 * delta
                logger.info(
                    "ExperienceConsolidator: restored home_vector_delta on boot (norm=%.3f)",
                    float(np.linalg.norm(crsm.home_vector)),
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _record_experience_consolidator_degradation(
                e,
                action="skipped boot-time CRSM home-vector restoration",
                severity="warning",
            )
            logger.debug("home_vector_delta restoration skipped: %s", e)

    def _log_consolidation(self, narrative: IdentityNarrative, material: dict):
        try:
            entry = {
                "timestamp": time.time(),
                "version": narrative.version,
                "signature": narrative.signature_phrase,
                "traits": narrative.stable_traits,
                "experiences_processed": len(material.get("experiences", [])),
            }
            CONSOL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONSOL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            # Rotate log if it grows too large (>5MB)
            try:
                if CONSOL_LOG_PATH.stat().st_size > 5 * 1024 * 1024:
                    lines = CONSOL_LOG_PATH.read_text().splitlines()
                    # Keep last 500 entries
                    atomic_write_text(CONSOL_LOG_PATH, "\n".join(lines[-500:]) + "\n")
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                _record_experience_consolidator_degradation(
                    _exc,
                    action="left consolidation log unrotated after bounded rotation failure",
                    severity="warning",
                )
                logger.debug("Consolidation log rotation skipped: %s", _exc)
        except (OSError, ConnectionError, TimeoutError) as _exc:
            _record_experience_consolidator_degradation(
                _exc,
                action="kept narrative state after consolidation log append failure",
                severity="warning",
            )
            logger.debug("Consolidation log append failed: %s", _exc)


# ── Singleton ──────────────────────────────────────────────────────────────────

_consolidator: ExperienceConsolidator | None = None


def get_experience_consolidator() -> ExperienceConsolidator:
    global _consolidator
    if _consolidator is None:
        _consolidator = ExperienceConsolidator()
    return _consolidator
