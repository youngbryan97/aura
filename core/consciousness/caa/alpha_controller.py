from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from .mode_collapse_detector import CollapseSeverity, CollapseSignal


@dataclass(frozen=True)
class AlphaState:
    current_alpha: float
    target_alpha: float
    readiness_level: str
    reason: str
    collapse_events: int = 0
    health_score: float = 1.0
    cross_entropy: float | None = None
    dampening: float = 1.0
    #: Steering magnitude over the noise 4-bit weights already put into the
    #: same residual stream. Below 1.0 the injection is smaller than the noise
    #: it competes with; at the live surface alpha of 0.35 it is about 0.056.
    quantization_snr: float | None = None

    @property
    def below_quantization_floor(self) -> bool | None:
        if self.quantization_snr is None:
            return None
        return self.quantization_snr < 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["below_quantization_floor"] = self.below_quantization_floor
        return payload


logger = logging.getLogger("Aura.CAA.AlphaController")


def quantization_snr(alpha: float, residual_norm: float) -> float | None:
    """Steering magnitude over 4-bit quantisation noise, or None if unmeasured."""
    try:
        from core.consciousness.caa.quantization_floor import assess_steering_precision

        return assess_steering_precision(alpha, residual_norm).snr
    except (ImportError, ArithmeticError, TypeError, ValueError) as exc:
        logger.warning("Quantisation floor unavailable: %s", exc)
        return None


class AlphaController:
    """Adaptive steering strength with conservative collapse backoff."""

    def __init__(
        self,
        *,
        base_alpha: float = 5.0,
        min_alpha: float | None = None,
        max_alpha: float = 8.5,
        residual_norm: float = 70.0,
    ) -> None:
        self._base_alpha = float(base_alpha)
        self._min_alpha = float(0.25 if min_alpha is None else min_alpha)
        self._max_alpha = float(max_alpha)
        self._residual_norm = float(residual_norm)
        # 4-bit weights put noise worth ~8.8% of the activation norm into the
        # same residual stream the steering vector is added to, which at the
        # live surface α of 0.35 makes the injection roughly EIGHTEEN TIMES
        # smaller than the noise it competes with (measured; see
        # core/consciousness/caa/quantization_floor.py).
        #
        # That is deliberately NOT used to clamp α. Steering below the floor is
        # weak, not harmful, and a consistent bias summed over 64 blocks and
        # hundreds of tokens is not a zero-mean perturbation — the live A/B is
        # what decides whether it works. Raising the floor to SNR=1 would pin α
        # at ~6.3, above this controller's own base, and would be a behavioural
        # change justified by nothing measured.
        #
        # What it DOES end is reporting "steering applied" with no idea at what
        # strength. Every state now carries the ratio.
        self._quantization_snr: float | None = None
        self._state = AlphaState(
            current_alpha=float(base_alpha),
            target_alpha=float(base_alpha),
            readiness_level="bootstrap",
            reason="bootstrap baseline",
            quantization_snr=quantization_snr(float(base_alpha), self._residual_norm),
        )

    @staticmethod
    def _health_dampening(
        generation_health: float | None,
        cross_entropy: float | None,
    ) -> tuple[float, float, float | None, str]:
        health = 1.0 if generation_health is None else max(0.0, min(1.0, float(generation_health)))
        entropy = None if cross_entropy is None else max(0.0, float(cross_entropy))
        dampening = 1.0
        reasons: list[str] = []
        if health < 0.55:
            dampening = min(dampening, max(0.18, 0.30 + health))
            reasons.append("low_generation_health")
        if entropy is not None and entropy > 5.5:
            entropy_backoff = max(0.15, 1.0 - min(0.85, (entropy - 5.5) / 6.0))
            dampening = min(dampening, entropy_backoff)
            reasons.append("high_cross_entropy")
        return dampening, health, entropy, "+".join(reasons)

    def update(
        self,
        *,
        readiness_level: str,
        exact_match_ratio: float = 0.0,
        extracted_ratio: float = 0.0,
        collapse_signal: CollapseSignal | None = None,
        generation_health: float | None = None,
        cross_entropy: float | None = None,
    ) -> AlphaState:
        target = self._base_alpha
        reason = "bootstrap baseline"
        if readiness_level == "production":
            target = min(self._max_alpha, 8.0 if exact_match_ratio >= 0.99 and extracted_ratio >= 0.99 else 7.0)
            reason = "production vectors validated"
        elif readiness_level == "validated":
            target = min(self._max_alpha, 6.5)
            reason = "activation vectors validated"
        elif readiness_level == "mixed":
            target = min(self._max_alpha, 5.5)
            reason = "mixed exact and nearest activation vectors"
        current = self._state.current_alpha
        collapse_events = self._state.collapse_events
        if collapse_signal is not None and collapse_signal.severity != CollapseSeverity.NONE:
            collapse_events += 1
            if collapse_signal.severity == CollapseSeverity.CRITICAL:
                current = max(self._min_alpha, min(current, current * 0.6))
                target = min(target, current)
                reason = "critical collapse backoff"
            elif collapse_signal.severity == CollapseSeverity.WARNING:
                current = max(self._min_alpha, min(current, current * 0.8))
                target = min(target, current)
                reason = "warning collapse backoff"
            else:
                target = min(target, max(self._min_alpha, current))
                reason = "watch collapse hold"
        else:
            current = current + (target - current) * 0.35
        dampening, health, entropy, health_reason = self._health_dampening(generation_health, cross_entropy)
        if dampening < 1.0:
            target *= dampening
            current = min(current, target)
            reason = f"{reason}; health dampening:{health_reason}"
        current = max(self._min_alpha, min(self._max_alpha, current))
        self._state = AlphaState(
            current_alpha=round(float(current), 4),
            target_alpha=round(float(target), 4),
            readiness_level=str(readiness_level),
            reason=reason,
            collapse_events=collapse_events,
            health_score=round(float(health), 4),
            cross_entropy=round(float(entropy), 4) if entropy is not None else None,
            dampening=round(float(dampening), 4),
            quantization_snr=quantization_snr(float(current), self._residual_norm),
        )
        return self._state

    @property
    def state(self) -> AlphaState:
        return self._state
