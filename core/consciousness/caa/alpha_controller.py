from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .mode_collapse_detector import CollapseSeverity, CollapseSignal


@dataclass(frozen=True)
class AlphaState:
    current_alpha: float
    target_alpha: float
    readiness_level: str
    reason: str
    collapse_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlphaController:
    """Adaptive steering strength with conservative collapse backoff."""

    def __init__(self, *, base_alpha: float = 5.0, min_alpha: float = 3.0, max_alpha: float = 8.5) -> None:
        self._base_alpha = float(base_alpha)
        self._min_alpha = float(min_alpha)
        self._max_alpha = float(max_alpha)
        self._state = AlphaState(
            current_alpha=float(base_alpha),
            target_alpha=float(base_alpha),
            readiness_level="bootstrap",
            reason="bootstrap baseline",
        )

    def update(
        self,
        *,
        readiness_level: str,
        exact_match_ratio: float = 0.0,
        extracted_ratio: float = 0.0,
        collapse_signal: CollapseSignal | None = None,
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
        current = max(self._min_alpha, min(self._max_alpha, current))
        self._state = AlphaState(
            current_alpha=round(float(current), 4),
            target_alpha=round(float(target), 4),
            readiness_level=str(readiness_level),
            reason=reason,
            collapse_events=collapse_events,
        )
        return self._state

    @property
    def state(self) -> AlphaState:
        return self._state
