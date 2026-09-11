from __future__ import annotations

import time
import logging
from typing import Any

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


#: Where bonding sits when nothing has happened for a long time. The value
#: `IdentityKernel` is created with, so the resting point of the quantity and
#: its declared default are the same number rather than two.
_BONDING_BASELINE: float = 0.05


class BondingPhase(Phase):
    """
    Phase to handle long-term personality evolution and user bonding.
    Adjusts Aura's traits based on interaction history and depth.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container
        #: How far apart their exchanges usually fall, how long this
        #: relationship has been running, and when the last one was. Bonding
        #: settles between exchanges: see `_settle_toward`.
        self._usual_gap_s = 0.0
        self._engaged_span_s = 0.0
        self._exchanges_seen = 0.0
        self._last_exchange_at = 0.0

    def _settle_toward(self, bonding: float, baseline: float) -> float:
        """Let an absence give back what the exchanges built.

        Bonding rose by a fixed amount on every user-facing turn and by
        nothing on any other, so it could only ever increase. A quantity that
        cannot fall is a turn counter, and it measured as one: across a
        480-turn recording of the subject core it correlated with the frame
        index to four decimal places, and at one with every other counter in
        every other domain. A relationship that never cools is not a
        relationship.

        What gives it back is silence beyond their own rhythm, weighed against
        the span the relationship has been running. A gap inside the rhythm
        costs nothing and extends the span; an absence as long as everything
        that built this returns it to the baseline; half as long returns half.
        The rhythm and the span are both learned here, so there is no decay
        constant to pick, and a relationship of ten thousand exchanges settles
        far more slowly than one of ten — which is the thing a fixed half-life
        would get wrong.
        """
        now = time.time()
        gap = now - self._last_exchange_at if self._last_exchange_at else 0.0
        self._last_exchange_at = now
        self._exchanges_seen += 1.0

        usual = self._usual_gap_s
        if gap > 0.0:
            rate = max(1.0 / self._exchanges_seen, 0.01)  # never freezes
            self._usual_gap_s = usual + rate * (gap - usual)
        if gap <= 0.0 or usual <= 0.0 or gap <= usual:
            self._engaged_span_s += max(0.0, gap)
            return bonding

        absence = gap - usual
        span = max(self._engaged_span_s, 1e-9)
        share = min(1.0, absence / max(span, absence))
        return baseline + (bonding - baseline) * (1.0 - share)

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

            # This is Aura's own bounded social-plasticity state, not a claim
            # about rapport, intimacy, or trust with whichever user is cached.
            # It moves in the direction this exchange compares with theirs, so
            # it can settle as well as build.
            increment = 0.0001 * multiplier
            current_bonding = _bounded_float(getattr(identity, "bonding_level", 0.0), 0.0)
            settled = self._settle_toward(current_bonding, _BONDING_BASELINE)
            identity.bonding_level = max(0.0, min(1.0, settled + increment))

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
                "bonding_level": round(identity.bonding_level, 5),
                "person_specific_relationship_claimed": False,
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
