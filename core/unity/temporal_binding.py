from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Optional

from .unity_state import BoundContent, TemporalWindow


@dataclass
class TemporalEvent:
    content_id: str
    source: str
    modality: str
    summary: str
    salience: float
    timestamp: float


class TemporalBindingField:
    """Maintains a rolling temporal horizon instead of a single frozen tick."""

    def __init__(self, window_s: float = 4.0, decay_half_life_s: float = 1.2, max_events: int = 512):
        self.window_s = max(0.5, float(window_s))
        self.decay_half_life_s = max(0.1, float(decay_half_life_s))
        self.max_events = max(32, int(max_events))
        self._events: list[TemporalEvent] = []

    def ingest(self, event: TemporalEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def _prune(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_s
        self._events = [event for event in self._events if event.timestamp >= cutoff]

    def _ingest_contents(self, contents: Iterable[BoundContent]) -> None:
        for item in contents:
            self.ingest(
                TemporalEvent(
                    content_id=item.content_id,
                    source=item.source,
                    modality=item.modality,
                    summary=item.summary,
                    salience=float(item.salience or 0.0),
                    timestamp=float(item.timestamp or time.time()),
                )
            )

    @staticmethod
    def _weighted_overlap(current_ids: list[str], previous_ids: Optional[list[str]]) -> float:
        if not current_ids or not previous_ids:
            return 0.0
        current = set(current_ids)
        previous = set(previous_ids)
        overlap = len(current & previous)
        baseline = max(1, len(current | previous))
        return max(0.0, min(1.0, overlap / baseline))

    def bind_now(
        self,
        tick_id: str,
        contents: list[BoundContent],
        *,
        previous_temporal: TemporalWindow | None = None,
        previous_content_ids: Optional[list[str]] = None,
        now_ts: Optional[float] = None,
    ) -> TemporalWindow:
        now_ts = float(now_ts or time.time())
        self._ingest_contents(contents)
        self._prune(now_ts)

        if not self._events:
            return TemporalWindow(
                tick_id=tick_id,
                opened_at=now_ts,
                closed_at=now_ts,
                subjective_center_t=now_ts,
                duration_s=0.0,
                continuity_from_previous=0.0,
                drift_from_previous=1.0 if previous_temporal else 0.0,
                phase_lag={},
            )

        weighted_events: list[tuple[TemporalEvent, float]] = []
        total_weight = 0.0
        for event in self._events:
            age = max(0.0, now_ts - event.timestamp)
            decay = math.exp(-math.log(2.0) * age / self.decay_half_life_s)
            weight = max(0.01, float(event.salience or 0.0)) * decay
            weighted_events.append((event, weight))
            total_weight += weight

        weighted_center = sum(event.timestamp * weight for event, weight in weighted_events) / max(total_weight, 1e-6)
        event_ids = [event.content_id for event, _weight in sorted(weighted_events, key=lambda pair: pair[1], reverse=True)[:6]]
        current_ids = [item.content_id for item in list(contents or [])[:6]]
        current_overlap = self._weighted_overlap(current_ids, previous_content_ids)
        horizon_overlap = self._weighted_overlap(event_ids, previous_content_ids)

        phase_lag: dict[str, float] = {}
        for event, _weight in weighted_events:
            lag = max(0.0, now_ts - event.timestamp)
            current = phase_lag.get(event.source)
            if current is None or lag < current:
                phase_lag[event.source] = lag

        max_lag = max(phase_lag.values(), default=0.0)
        lag_penalty = min(1.0, max_lag / max(self.window_s, 1e-6))
        stability_bonus = max(0.0, 1.0 - lag_penalty)
        continuity = max(
            0.0,
            min(
                1.0,
                (current_overlap * 0.65) + (horizon_overlap * 0.15) + (stability_bonus * 0.2),
            ),
        )
        drift = max(0.0, min(1.0, 1.0 - continuity))

        opened_at = min(event.timestamp for event, _weight in weighted_events)
        duration_s = max(0.0, now_ts - opened_at)
        if previous_temporal and previous_temporal.closed_at:
            duration_s = max(duration_s, now_ts - min(opened_at, previous_temporal.opened_at))

        return TemporalWindow(
            tick_id=tick_id,
            opened_at=opened_at,
            closed_at=now_ts,
            subjective_center_t=weighted_center,
            duration_s=min(self.window_s, duration_s),
            continuity_from_previous=round(continuity, 4),
            drift_from_previous=round(drift, 4),
            phase_lag={key: round(value, 4) for key, value in phase_lag.items()},
        )
