from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from core.container import ServiceContainer
from core.kernel.bridge import Phase
from core.runtime.errors import FallbackClassification, record_degradation
from core.service_names import ServiceNames
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.SocialContextPhase")

_SOCIAL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "and",
    "or",
    "but",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "had",
    "was",
    "were",
    "you",
    "your",
    "yours",
    "i",
    "me",
    "my",
    "mine",
    "we",
    "our",
}


def _record_social_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
) -> None:
    try:
        record_degradation(
            "social_context_phase",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra={"stage": stage},
        )
    except TypeError:
        record_degradation(
            "social_context_phase",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


def _safe_text(value: Any, *, max_chars: int = 60_000) -> str:
    if value is None:
        return ""
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return ""
    text = text.replace("\x00", "").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value if value is not None else default)
    except (RuntimeError, TypeError, ValueError):
        parsed = default
    return min(1.0, max(0.0, parsed))


def _ensure_modifiers(state: AuraState) -> dict[str, Any]:
    modifiers = getattr(state.cognition, "modifiers", None)
    if not isinstance(modifiers, dict):
        modifiers = {}
        state.cognition.modifiers = modifiers
    return modifiers


def _service_get(container: Any, name: str, default: Any = None) -> Any:
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(name, default=default)
    return default


class SocialContextPhase(Phase):
    """
    Phase to inject social context from Ava (SocialModelingEngine).
    Ensures that Aura's responses are tailored to the user's communication style.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container or ServiceContainer

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Analyze social context for a user turn and write response-shaping modifiers.
        """
        objective_text = _safe_text(objective)
        if not objective_text:
            return state

        try:
            ava = _service_get(self.container, ServiceNames.AVA, default=None)
        except _SOCIAL_RECOVERABLE_ERRORS as exc:
            ava = None
            _record_social_degradation(
                exc,
                action="continued with local engagement cues after Ava service lookup failed",
                stage="ava_lookup",
            )

        modifiers = _ensure_modifiers(state)
        if state.cognition.current_origin in ("user", "voice", "admin") and ava:
            await self._analyze_user_message(ava, objective_text)

        self._apply_engagement_cues(modifiers, objective_text)
        if ava:
            self._synchronize_ava_context(ava, modifiers)
        self._apply_theory_of_mind_register(modifiers)
        return state

    async def _analyze_user_message(self, ava: Any, objective: str) -> None:
        analyzer = getattr(ava, "analyze_message", None)
        if not callable(analyzer):
            return
        try:
            result = analyzer(objective)
            if inspect.isawaitable(result):
                await result
        except _SOCIAL_RECOVERABLE_ERRORS as exc:
            _record_social_degradation(
                exc,
                action="continued social cue injection after Ava message analysis failed",
                stage="ava_analysis",
            )

    def _apply_engagement_cues(self, modifiers: dict[str, Any], objective: str) -> None:
        words = _WORD_PATTERN.findall(objective.lower())
        user_msg_len = len(words)

        modifiers["mirror_length_target"] = user_msg_len
        if user_msg_len < 4:
            modifiers["interaction_style"] = "proactive_engagement"
            modifiers["desired_brevity"] = "extreme"
            logger.debug("SocialContext: User under-engaging. Pushing proactive engagement.")
        elif user_msg_len > 100:
            modifiers["interaction_style"] = "backchannel_heavy"
            modifiers["desired_brevity"] = "low"
            logger.debug("SocialContext: User over-explaining. Pushing backchannel logic.")
        else:
            modifiers["interaction_style"] = "balanced_flow"
            modifiers["desired_brevity"] = "moderate"

        seen: set[str] = set()
        signal_words = []
        for word in words:
            if word in _STOP_WORDS or len(word) <= 4 or word in seen:
                continue
            seen.add(word)
            signal_words.append(word)
            if len(signal_words) >= 5:
                break
        modifiers["lexical_mirror"] = signal_words

    def _synchronize_ava_context(self, ava: Any, modifiers: dict[str, Any]) -> None:
        context_provider = getattr(ava, "get_context_injection", None)
        if not callable(context_provider):
            return
        try:
            injection = _safe_text(context_provider(), max_chars=4_000)
        except _SOCIAL_RECOVERABLE_ERRORS as exc:
            _record_social_degradation(
                exc,
                action="kept local social cues after Ava context injection failed",
                stage="ava_context",
            )
            return
        if injection and modifiers.get("social_context") != injection:
            modifiers["social_context"] = injection
            logger.debug("Social context synchronized: %s", injection)

    def _apply_theory_of_mind_register(self, modifiers: dict[str, Any]) -> None:
        try:
            tom = _service_get(self.container, "theory_of_mind", default=None)
            if tom is None:
                tom = ServiceContainer.get("theory_of_mind", default=None)
            known_selves = getattr(tom, "known_selves", None) if tom else None
            if not known_selves:
                return
            values = known_selves.values() if hasattr(known_selves, "values") else known_selves
            user_model = next(iter(values), None)
            if user_model is None:
                return
            rapport = _safe_float(getattr(user_model, "rapport", 0.5), default=0.5)
            if rapport > 0.75:
                modifiers["relational_register"] = "intimate"
            elif rapport > 0.4:
                modifiers["relational_register"] = "warm"
            else:
                modifiers["relational_register"] = "cordial"
            modifiers["rapport_level"] = rapport
        except _SOCIAL_RECOVERABLE_ERRORS as exc:
            _record_social_degradation(
                exc,
                action="kept social cues without theory-of-mind rapport register",
                stage="theory_of_mind",
            )
