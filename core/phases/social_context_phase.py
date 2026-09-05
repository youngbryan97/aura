from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from core.container import ServiceContainer
from core.runtime.service_access import optional_service
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
        if state.cognition.current_origin in ("user", "voice", "admin"):
            await self._infer_communicative_stance(state, modifiers, objective_text)
        return state

    async def _infer_communicative_stance(
        self, state: AuraState, modifiers: dict[str, Any], message: str
    ) -> None:
        """Classify *how* the user's message is meant (sincere / joke / sarcasm /
        guess / hypothesis / role-play / mistaken / deceptive) and write it as a
        response-shaping modifier. Grounded against known facts and the recent turn
        history so a confident falsehood is flagged, not swallowed. Live on every
        user turn; deterministic spine, no model call in the hot path."""
        try:
            from core.social.stance_inference import get_stance_inference

            known_facts = await self._gather_known_facts(message)
            recent = self._recent_user_messages(state)
            assessment = get_stance_inference().assess(
                message, known_facts=known_facts, recent_messages=recent
            )
            modifiers["communicative_stance"] = assessment.primary.value
            modifiers["stance_confidence"] = round(assessment.confidence, 3)
            modifiers["take_literally"] = assessment.take_literally
            if assessment.factual_conflict:
                modifiers["stance_factual_conflict"] = True
            if assessment.primary.value in {"deceptive", "mistaken"} and assessment.factual_conflict:
                # Hand a concrete cue to the honesty/deception guards downstream.
                modifiers["flagged_false_claim"] = assessment.rationale
            if not assessment.take_literally:
                logger.info(
                    "🎭 SocialContext: user message read as %s (conf %.2f): %s",
                    assessment.primary.value, assessment.confidence, assessment.rationale,
                )
        except _SOCIAL_RECOVERABLE_ERRORS as exc:
            _record_social_degradation(
                exc,
                action="kept social cues without communicative-stance inference",
                stage="stance_inference",
            )

    async def _gather_known_facts(self, message: str) -> list[str]:
        """Pull a few grounded facts relevant to the message for belief-conflict checks."""
        facade = _service_get(self.container, ServiceNames.MEMORY_FACADE, default=None)
        if facade is None:
            facade = optional_service("memory_facade")
        if facade is None or not hasattr(facade, "search"):
            return []
        try:
            results = await facade.search(message, limit=4)
        except _SOCIAL_RECOVERABLE_ERRORS:
            return []
        facts: list[str] = []
        for item in list(results or [])[:4]:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
            else:
                content = str(item or "").strip()
            if content:
                facts.append(content[:200])
        return facts

    @staticmethod
    def _recent_user_messages(state: AuraState, limit: int = 4) -> list[str]:
        history = getattr(getattr(state, "conversation", None), "history", None)
        if not isinstance(history, (list, tuple)):
            history = getattr(getattr(state, "memory", None), "short_term", None)
        out: list[str] = []
        for item in list(history or [])[-12:]:
            try:
                role = item.get("role") if isinstance(item, dict) else getattr(item, "role", "")
                content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            except (AttributeError, TypeError):
                continue
            if role in ("user", "human") and content:
                out.append(str(content)[:300])
        return out[-limit:]

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
            estimator = _service_get(self.container, "other_agent_model", default=None)
            if estimator is None:
                estimator = optional_service("other_agent_model")
            agent_id = str(getattr(estimator, "active_agent_id", "") or "")
            if estimator and agent_id and hasattr(estimator, "cognitive_snapshot"):
                snapshot = estimator.cognitive_snapshot(agent_id)
                recommendation = snapshot.get("recommendation")
                recommendation = recommendation if isinstance(recommendation, dict) else {}
                modifiers["social_state_confidence"] = _safe_float(
                    snapshot.get("confidence"),
                    default=0.0,
                )
                modifiers["social_state_is_hypothesis"] = True
                if recommendation.get("slow_down"):
                    modifiers["relational_register"] = "repair"
                elif recommendation.get("offer_reassurance"):
                    modifiers["relational_register"] = "careful"
                else:
                    modifiers["relational_register"] = "neutral"
                return

            # No exact active estimator means no person-specific social claim.
            return
        except _SOCIAL_RECOVERABLE_ERRORS as exc:
            _record_social_degradation(
                exc,
                action="kept local social cues without calibrated exact-agent register",
                stage="theory_of_mind",
            )


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
        name="SocialContextPhase",
        version="1.0",
        module=__name__,
        purpose=(
            "Establish who is being spoken to and how, and express it as "
            "modifiers the rest of the tick can read."
        ),
        reads=("cognition.working_memory", "identity.relationships"),
        writes=("cognition.modifiers",),
        preconditions=("state carries a cognition block",),
        branches=(
            BranchSpec(
                "known_interlocutor",
                "the speaker resolves to a known relationship",
                "apply that relationship's social modifiers",
            ),
            BranchSpec(
                "unknown_interlocutor",
                "no relationship resolves",
                "apply default social modifiers",
            ),
        ),
        calibration_source=(
            "writes measured by tools/observe_phase_writes.py"
            "; reads reach state through this phase's delegate rather than appearing in this module, so they are declared from the delegate's behaviour and not checkable by scanning this file alone"
        ),
    )
)
