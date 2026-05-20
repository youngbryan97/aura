from __future__ import annotations

import logging
from typing import Any

from core.container import ServiceContainer
from core.kernel.bridge import Phase
from core.runtime.errors import FallbackClassification, record_degradation
from core.state.aura_state import AuraState

logger = logging.getLogger("Aura.BondingPhase")

_BONDING_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
)
_USER_FACING_ORIGINS = frozenset({"user", "voice", "admin"})
_PERSONALITY_GROWTH_KEYS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


def _record_bonding_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    stage: str = "",
) -> None:
    extra = {"stage": stage} if stage else None
    try:
        record_degradation(
            "bonding_phase",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=extra,
        )
    except TypeError:
        record_degradation(
            "bonding_phase",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action or "captured bonding phase fault",
        )


def _safe_text(value: Any, default: str = "", *, max_chars: int = 4_000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "").strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _bounded_float(
    value: Any,
    default: float,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number:
        return default
    return max(lower, min(upper, number))


class BondingPhase(Phase):
    """
    Phase to handle long-term personality evolution and user bonding.
    Adjusts Aura's traits based on interaction history and depth.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container or ServiceContainer

    async def execute(
        self,
        state: AuraState,
        objective: str | None = None,
        **kwargs,
    ) -> AuraState:
        """
        1. Evaluate interaction depth from current tick.
        2. Increment bonding_level.
        3. Evolve personality_growth offsets.
        """
        origin = _safe_text(getattr(getattr(state, "cognition", None), "current_origin", "system"))
        if origin not in _USER_FACING_ORIGINS:
            return state

        try:
            cognition = getattr(state, "cognition", None)
            identity = getattr(state, "identity", None)
            if cognition is None or identity is None:
                raise AttributeError("AuraState must expose cognition and identity")

            objective_text = _safe_text(objective, max_chars=4_000)
            msg_len = len(objective_text.split())
            modifiers = getattr(cognition, "modifiers", None)
            if not isinstance(modifiers, dict):
                modifiers = {}
                cognition.modifiers = modifiers
            subtext = _safe_text(modifiers.get("user_subtext", ""), max_chars=1_000)

            multiplier = 1.0
            if msg_len > 50:
                multiplier += 0.5
            if len(subtext) > 10:
                multiplier += 0.5

            rapport = 0.5
            try:
                tom = ServiceContainer.get("theory_of_mind", default=None)
                if tom and tom.known_selves:
                    user_model = next(iter(tom.known_selves.values()))
                    rapport = _bounded_float(getattr(user_model, "rapport", 0.5), 0.5)
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_bonding_fault(
                    exc,
                    action="used neutral rapport and continued bonding update",
                    severity="warning",
                    stage="theory_of_mind",
                )
                logger.debug("Theory-of-mind rapport unavailable: %s", exc)

            rapport_multiplier = 0.5 + rapport
            increment = 0.0001 * multiplier * rapport_multiplier
            current_bonding = _bounded_float(getattr(identity, "bonding_level", 0.0), 0.0)
            identity.bonding_level = min(1.0, current_bonding + increment)

            growth = getattr(identity, "personality_growth", None)
            if not isinstance(growth, dict):
                growth = {}
                identity.personality_growth = growth
            for key in _PERSONALITY_GROWTH_KEYS:
                growth[key] = _bounded_float(growth.get(key, 0.0), 0.0, lower=-1.0, upper=1.0)

            bonding = identity.bonding_level
            if bonding > 0.3:
                growth["openness"] = min(0.1, growth["openness"] + 0.0005)
                growth["agreeableness"] = min(0.05, growth["agreeableness"] + 0.0002)

            if bonding > 0.7:
                growth["extraversion"] = min(0.15, growth["extraversion"] + 0.001)
                growth["agreeableness"] = min(0.15, growth["agreeableness"] + 0.0005)
                growth["neuroticism"] = max(-0.1, growth["neuroticism"] - 0.0005)

            modifiers["bonding_phase"] = {
                "increment": round(increment, 7),
                "rapport": round(rapport, 3),
                "bonding_level": round(identity.bonding_level, 5),
            }

            logger.debug(
                "Bonding Update: Level=%s, Growth=%s",
                f"{identity.bonding_level:.4f}",
                growth,
            )

        except _BONDING_RECOVERABLE_ERRORS as exc:
            _record_bonding_fault(
                exc,
                action="returned prior AuraState after bonding update failed",
                severity="degraded",
                stage="execute",
            )
            logger.warning("BondingPhase failed: %s", exc)

        return state
