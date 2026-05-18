"""core/emotional_coloring.py — Affective Grounding for Aura Zenith.

Uses emotionally tagged episodic memory to color cognitive tone.
"""

import inspect
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.EmotionalColoring")

_RECOVERABLE_COLORING_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_MOOD_VALENCE = {
    "joy": 0.75,
    "trust": 0.55,
    "anticipation": 0.35,
    "surprise": 0.15,
    "neutral": 0.0,
    "sadness": -0.45,
    "fear": -0.55,
    "anger": -0.65,
    "disgust": -0.7,
}


@dataclass
class EmotionalTexture:
    """The emotional 'residue' of a topic/experience."""

    net_valence: float  # -1.0 to 1.0
    arousal_boost: float
    tone_hint: str
    relevant_episode_count: int


class EmotionalColoring:
    """Calculating the emotional weight of topics based on history."""

    def __init__(self):
        logger.info("EmotionalColoring initialized.")

    async def get_texture_for_topic(self, topic: str) -> EmotionalTexture:
        """Retrieve emotional residue for a specific topic string."""
        from core.container import ServiceContainer

        memory = ServiceContainer.get("memory", default=None)
        liquid_state = ServiceContainer.get("liquid_state", default=None)

        episodes: list[Any] = []
        if memory:
            try:
                episodes = await _search_memory(memory, topic)
            except _RECOVERABLE_COLORING_ERRORS as exc:
                record_degradation("emotional_coloring", exc)
                logger.debug("Emotional memory search failed for topic %r: %s", topic, exc)

        values = [_episode_affect(episode) for episode in episodes]
        values = [item for item in values if item is not None]

        if values:
            net_v = _clamp(sum(item[0] for item in values) / len(values))
            arousal = _clamp(sum(item[1] for item in values) / len(values), 0.0, 1.0)
            count = len(values)
        else:
            net_v = 0.0
            arousal = 0.0
            count = 0

        # Integrate with current mood if liquid_state available
        if liquid_state:
            try:
                baseline_v = _current_liquid_valence(liquid_state)
                net_v = _clamp((net_v * 0.7) + (baseline_v * 0.3))
            except _RECOVERABLE_COLORING_ERRORS as exc:
                record_degradation("emotional_coloring", exc)
                logger.debug("Liquid-state valence read failed: %s", exc)

        if net_v > 0.5:
            tone = "warm/exploratory"
        elif net_v < -0.5:
            tone = "cautionary/guarded"
        else:
            tone = "analytical/neutral"

        return EmotionalTexture(
            net_valence=net_v,
            arousal_boost=arousal,
            tone_hint=tone,
            relevant_episode_count=count,
        )


async def _search_memory(memory: Any, topic: str) -> list[Any]:
    episodic = getattr(memory, "episodic", None)
    for owner, method_name in ((episodic, "search"), (memory, "search"), (memory, "recall")):
        if owner is None:
            continue
        method = getattr(owner, method_name, None)
        if not callable(method):
            continue
        try:
            result = method(topic, limit=5)
        except TypeError:
            result = method(topic)
        result = await result if inspect.isawaitable(result) else result
        return list(result) if isinstance(result, Iterable) and not isinstance(result, str) else []
    return []


def _episode_affect(episode: Any) -> tuple[float, float] | None:
    valence = _coerce_float(_episode_value(episode, "valence", "emotion_valence", "emotional_valence"))
    if valence is None:
        mood = str(_episode_value(episode, "mood", "emotion", "dominant_emotion") or "").lower()
        valence = _MOOD_VALENCE.get(mood)
    if valence is None:
        affect = _episode_value(episode, "affect", "affect_state")
        valence = _coerce_float(_episode_value(affect, "valence")) if affect is not None else None
    if valence is None:
        return None

    arousal = _coerce_float(_episode_value(episode, "arousal", "salience", "importance"))
    return (_clamp(valence), _clamp(arousal if arousal is not None else abs(valence), 0.0, 1.0))


def _episode_value(episode: Any, *names: str) -> Any:
    if episode is None:
        return None
    for name in names:
        if isinstance(episode, dict) and name in episode:
            return episode[name]
        if hasattr(episode, name):
            return getattr(episode, name)
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _current_liquid_valence(liquid_state: Any) -> float:
    getter = getattr(liquid_state, "get_valence", None)
    if callable(getter):
        return float(getter())
    return float(liquid_state.valence)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# Service Registration
def register_emotional_coloring():
    """Register the emotional coloring service."""
    from core.container import ServiceContainer, ServiceLifetime

    ServiceContainer.register(
        "emotional_coloring",
        factory=lambda: EmotionalColoring(),
        lifetime=ServiceLifetime.SINGLETON,
    )
