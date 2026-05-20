"""Persona Evolver (Phase 8).

Analyzes interaction memory and adapts personality baselines subtly over time.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from typing import Any

from core.brain.personality_engine import get_personality_engine
from core.config import config
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.PersonaEvolver")

MAX_REFLECTION_CHARS = 20_000
MAX_MEMORY_LINES = 200
MAX_MEMORY_LINE_CHARS = 600
MAX_LLM_RESPONSE_CHARS = 30_000
MIN_REFLECTION_CHARS = 20
TRAIT_DELTA_LIMIT = 0.05
EMOTION_BASE_DELTA_LIMIT = 5.0
EMOTION_VOLATILITY_DELTA_LIMIT = 5.0


def _emit_persona_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "persona_evolver",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("persona_evolver", error)


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_float(value: Any, default: float = 0.0, *, limit: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if limit is not None:
        number = max(-limit, min(limit, number))
    return number


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _extract_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("content", "text", "response", "answer"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("content", "text", "response", "answer"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    return _safe_text(response, max_chars=MAX_LLM_RESPONSE_CHARS)


def _extract_json_object(text: str) -> dict[str, Any]:
    bounded = _safe_text(text, max_chars=MAX_LLM_RESPONSE_CHARS)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", bounded):
        try:
            parsed, _end = decoder.raw_decode(bounded[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise TypeError("persona evolution response root must be an object")
    raise ValueError("persona evolution response did not contain a JSON object")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class PersonaEvolver:
    """Adapts Aura's personality over long-term interactions."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.last_evolution_time = time.time()
        self.evolution_interval = 3600 * 24  # Evaluate daily or when explicitly triggered
        self.min_memories_for_evolution = 10

    async def update_persona(self, reflection: str):
        """Allows for manual or event-driven persona updates based on reflection."""
        # v34 Hardening: Validate reflection before self-mod
        reflection = _safe_text(reflection, max_chars=MAX_REFLECTION_CHARS)
        if not reflection or len(reflection) < MIN_REFLECTION_CHARS:
            logger.debug("Reflection too short for persona evolution.")
            return

        try:
            logger.info("Performing manual persona evolution from reflection.")
            await self.run_evolution_cycle(force=True, custom_reflection=reflection)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _emit_persona_fault(
                e,
                action="left persona unchanged after manual evolution failed",
                severity="degraded",
                stage="update_persona",
            )
            logger.error("Persona evolution failed: %s", e)

    async def run_evolution_cycle(self, force: bool = False, custom_reflection: str | None = None):
        """Analyze memory and apply drift to personality if needed."""
        try:
            from core.safe_mode import runtime_feature_enabled

            if not runtime_feature_enabled(self.orchestrator, "persona_evolution", default=True):
                logger.debug("Persona evolution skipped by runtime mode configuration.")
                return
        except (ImportError, AttributeError, RuntimeError) as exc:
            _emit_persona_fault(
                exc,
                action="continued persona evolution after runtime-mode check failed",
                severity="warning",
                stage="runtime_mode",
            )
            logger.debug("Persona evolution runtime-mode check skipped: %s", exc)

        now = time.time()
        if not force and (now - self.last_evolution_time < self.evolution_interval):
            return

        try:
            personality = get_personality_engine()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _emit_persona_fault(
                exc,
                action="left persona unchanged because personality engine was unavailable",
                severity="degraded",
                stage="get_personality_engine",
            )
            return

        memories = getattr(personality, "interaction_memories", [])
        if not custom_reflection and len(memories) < self.min_memories_for_evolution:
            logger.debug("PersonaEvolver: Not enough new interaction memories to evolve.")
            return

        logger.info("Initiating Persona Evolution Cycle.")

        # Format memories for the LLM
        memory_text = (
            _safe_text(custom_reflection, max_chars=MAX_REFLECTION_CHARS)
            if custom_reflection
            else ""
        )
        if not custom_reflection:
            for m in memories[:MAX_MEMORY_LINES]:
                if not isinstance(m, dict):
                    continue
                msg = _safe_text(m.get("message", ""), max_chars=MAX_MEMORY_LINE_CHARS)
                sent = _safe_text(m.get("sentiment", "neutral"), max_chars=80)
                memory_text += f"- [{sent}] {msg}\n"

        prompt = f"""You are analyzing Aura's recent interactions to adjust her personality.
Based on the following interactions, how should her core traits and emotional baselines drift?
Output JSON ONLY with no markdown formatting.
Small fractional changes (-0.05 to +0.05 for traits, -5.0 to +5.0 for emotion base/volatility).
Only include fields that should change. Example format:
{{
  "traits": {{"agreeableness": 0.02, "extraversion": -0.01}},
  "emotions": {{"frustration": {{"base": -2.0, "volatility": 0.1}}}}
}}

Recent Interactions:
{memory_text}
"""
        try:
            from core.brain.cognitive_engine import ThinkingMode

            if not hasattr(self.orchestrator, "cognitive_engine"):
                logger.warning("No cognitive engine available for evolution.")
                return

            engine = self.orchestrator.cognitive_engine

            # Using deep thinking for self-reflection
            response = await asyncio.wait_for(
                engine.think(prompt, mode=ThinkingMode.DEEP, block_user=True),
                timeout=120.0,
            )

            content = _extract_content(response).strip()
            changes = _extract_json_object(content)

            if changes:
                self._apply_evolution(changes, personality)
                # Clear memories after evolution to prevent over-weighting
                personality.interaction_memories = []
                self.last_evolution_time = time.time()

        except asyncio.CancelledError:
            raise
        except (
            ImportError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            TimeoutError,
        ) as e:
            _emit_persona_fault(
                e,
                action="left persona unchanged after evolution cycle failed validation",
                severity="degraded",
                stage="run_evolution_cycle",
            )
            logger.error("PersonaEvolver cycle failed: %s", e)

    def _apply_evolution(self, changes: dict[str, Any], personality):
        """Merge changes into evolved_persona.json and reload."""
        evolved_path = config.paths.data_dir / "evolved_persona.json"

        evolved_data = {"traits": {}, "emotions": {}}
        if evolved_path.exists():
            try:
                with open(evolved_path, encoding="utf-8") as f:
                    evolved_data = json.load(f)
                if not isinstance(evolved_data, dict):
                    raise TypeError("evolved persona root must be an object")
                evolved_data.setdefault("traits", {})
                evolved_data.setdefault("emotions", {})
                if not isinstance(evolved_data["traits"], dict) or not isinstance(
                    evolved_data["emotions"], dict
                ):
                    raise TypeError("evolved persona traits/emotions must be objects")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                _emit_persona_fault(
                    exc,
                    action="reset invalid evolved persona cache before applying new changes",
                    severity="warning",
                    stage="apply_evolution.load_existing",
                    extra={"path": str(evolved_path)},
                )
                evolved_data = {"traits": {}, "emotions": {}}

        if not isinstance(changes, dict):
            raise TypeError("persona changes must be an object")

        # Merge traits
        new_traits = changes.get("traits", {})
        if not isinstance(new_traits, dict):
            new_traits = {}
        for t, val in new_traits.items():
            trait = _safe_text(t, max_chars=80)
            if not trait:
                continue
            current = _safe_float(
                evolved_data["traits"].get(trait, personality.traits.get(trait, 0.5))
            )
            delta = _safe_float(val, limit=TRAIT_DELTA_LIMIT)
            evolved_data["traits"][trait] = _clamp(current + delta, 0.0, 1.0)

        # Merge emotions
        new_emotions = changes.get("emotions", {})
        if not isinstance(new_emotions, dict):
            new_emotions = {}
        for e, data in new_emotions.items():
            emotion = _safe_text(e, max_chars=80)
            if not emotion or not isinstance(data, dict):
                continue
            if emotion not in evolved_data["emotions"]:
                evolved_data["emotions"][emotion] = {}

            if emotion in personality.emotions:
                current_state = personality.emotions[emotion]
                if "base" in data:
                    c_base = _safe_float(
                        evolved_data["emotions"][emotion].get("base", current_state.base_level)
                    )
                    evolved_data["emotions"][emotion]["base"] = _clamp(
                        c_base + _safe_float(data["base"], limit=EMOTION_BASE_DELTA_LIMIT),
                        0.0,
                        100.0,
                    )
                if "volatility" in data:
                    c_vol = _safe_float(
                        evolved_data["emotions"][emotion].get(
                            "volatility",
                            current_state.volatility,
                        )
                    )
                    evolved_data["emotions"][emotion]["volatility"] = max(
                        0.1,
                        c_vol
                        + _safe_float(
                            data["volatility"],
                            limit=EMOTION_VOLATILITY_DELTA_LIMIT,
                        ),
                    )

        # Save and trigger reload
        payload = json.dumps(evolved_data, indent=2, sort_keys=True, allow_nan=False)
        atomic_write_text(evolved_path, payload)

        logger.info("Persona evolved based on interaction sentiment.")
        personality.reload_persona()
