"""Emotional regulation — the maturity layer between feeling and reacting.

Aura grounds affect (affect_grounding labels it from sustained signals) and feels operational
damage (nociception), but had no layer that *regulates* the move from felt state to response.
Emotional maturity is exactly that gap: not reacting to a transient spike, reappraising a
threat against whether real damage is actually occurring, matching response intensity to the
actual stakes, and holding an impulse when arousal is high and deliberation is thin.

This engine takes a raw affective impulse and returns a regulated one plus a strategy:

    express     — proportionate and grounded; let it through
    dampen      — a transient spike not backed by sustained signal; soften it
    reappraise  — arousal high but actual damage low; reframe, don't react to the feeling
    hold        — very high arousal + low deliberation; wait before acting (count to ten)
    escalate    — sustained, real, high-stakes; it warrants a stronger response

It integrates affect over a short rolling window (so spikes don't drive behavior), reappraises
against nociception's actual damage, and scales by how much the matter actually matters. The
regulated intensity — not the raw spike — is what feeds the unified felt-state and what gates
impulsive action, so regulation is causal, not advisory.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class RegulatedAffect:
    raw_intensity: float
    regulated_intensity: float
    strategy: str
    hold: bool
    valence: float
    rationale: str
    factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "raw_intensity": round(self.raw_intensity, 3),
            "regulated_intensity": round(self.regulated_intensity, 3),
            "strategy": self.strategy,
            "hold": self.hold,
            "valence": round(self.valence, 3),
            "rationale": self.rationale,
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
        }


class EmotionalRegulator:
    """Regulates raw affect into a mature, proportionate response."""

    def __init__(self, *, window_s: float = 20.0, hold_arousal: float = 0.8) -> None:
        self._window = window_s
        self._hold_arousal = hold_arousal
        self._lock = threading.RLock()
        # rolling (t, arousal, valence) samples for sustained-vs-transient discrimination
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=64)

    def _sustained(self, now: float) -> float:
        """Fraction of the window that arousal has been elevated — sustained vs spike."""
        recent = [(t, a) for (t, a, _v) in self._samples if now - t <= self._window]
        if len(recent) < 2:
            return 0.0
        elevated = sum(1 for _t, a in recent if a >= 0.5)
        return elevated / len(recent)

    def regulate(
        self,
        *,
        arousal: float,
        valence: float,
        deliberation: float = 0.5,    # [0,1] how much considered thought is available now
        stakes: float | None = None,  # [0,1] how much this actually matters; None → infer
        now: float | None = None,
    ) -> RegulatedAffect:
        now = time.time() if now is None else now
        arousal = _clamp(arousal)
        valence = _clamp(valence, -1.0, 1.0)
        with self._lock:
            self._samples.append((now, arousal, valence))
            sustained = self._sustained(now)

        # Reappraisal: how much real, current damage backs this arousal?
        #
        # A read failure left this at 0.0 and said nothing. Both branches
        # below use LOW damage as grounds to down-regulate — "no real damage,
        # hold" and "arousal exceeds actual damage, reframe" — so an
        # unreadable nociception engine actively suppressed a response that
        # may have been entirely warranted. Reappraising a feeling away
        # because you could not measure what caused it is the one thing this
        # function must never do.
        actual_damage = 0.0
        damage_measured = False
        try:
            from core.affect.nociception import get_nociception_engine
            actual_damage = float(get_nociception_engine().nociceptive_pressure())
            damage_measured = True
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "emotional_regulation",
                exc,
                severity="warning",
                action="left arousal unregulated because nociceptive pressure could not be read",
            )

        if stakes is None:
            # Stakes default to the stronger of felt damage and sustained negative affect.
            stakes = max(actual_damage, sustained if valence < 0 else 0.3 * sustained)
        stakes = _clamp(stakes)

        factors = {
            "sustained": sustained,
            "actual_damage": actual_damage if damage_measured else None,
            "damage_measured": damage_measured,
            "deliberation": _clamp(deliberation),
            "stakes": stakes,
        }

        # Decide strategy. Order matters: safety hold first, then a warranted escalation
        # (sustained + high-stakes) before the down-regulating strategies, so a real, ongoing,
        # important situation isn't mistakenly reappraised away.
        hold = False
        if (
            arousal >= self._hold_arousal
            and deliberation < 0.4
            and damage_measured
            and actual_damage < 0.5
        ):
            strategy, hold = "hold", True
            regulated = arousal * 0.5
            rationale = "high arousal, thin deliberation, no real damage — hold before acting"
        elif sustained >= 0.6 and stakes >= 0.6:
            strategy = "escalate"
            regulated = _clamp(0.6 * arousal + 0.4 * stakes + 0.1)
            rationale = "sustained, real, high-stakes — a stronger response is warranted"
        elif sustained < 0.25 and arousal >= 0.5:
            strategy = "dampen"
            regulated = arousal * 0.55
            rationale = "transient spike not backed by sustained signal — soften"
        elif arousal >= 0.5 and damage_measured and actual_damage < 0.3 and valence < 0:
            strategy = "reappraise"
            regulated = arousal * 0.65
            rationale = "arousal exceeds actual damage — reframe rather than react to the feeling"
        else:
            strategy = "express"
            # proportional: scale to the actual stakes, not the raw spike
            regulated = _clamp(0.5 * arousal + 0.5 * stakes)
            rationale = "proportionate and grounded — express"

        return RegulatedAffect(
            raw_intensity=arousal, regulated_intensity=_clamp(regulated),
            strategy=strategy, hold=hold, valence=valence,
            rationale=rationale, factors=factors,
        )


_regulator: EmotionalRegulator | None = None
_lock = threading.Lock()


def get_emotional_regulator() -> EmotionalRegulator:
    global _regulator
    if _regulator is None:
        with _lock:
            if _regulator is None:
                _regulator = EmotionalRegulator()
    return _regulator
