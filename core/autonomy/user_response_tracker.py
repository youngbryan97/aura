"""core/autonomy/user_response_tracker.py
==========================================
Tracks user response patterns for proactive presence backoff.

Learns when the user typically responds vs. ignores autonomous messages,
and adjusts the proactive presence throttle accordingly. This prevents
Aura from repeatedly messaging a user who isn't engaging.

Signals:
  - response_rate: Rolling window response rate (0-1)
  - avg_response_time_s: Average time to respond
  - engagement_score: Composite score of user engagement
  - should_backoff: Whether proactive presence should throttle
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger("Aura.Autonomy.UserResponseTracker")


@dataclass
class ProactiveEvent:
    """Record of a proactive message sent to the user."""
    sent_at: float = field(default_factory=time.time)
    source: str = ""
    responded: bool = False
    response_time_s: Optional[float] = None


class UserResponseTracker:
    """Tracks user response patterns to proactive messages.

    Maintains a rolling window of proactive events and computes
    response metrics for adaptive backoff.
    """

    # Backoff thresholds
    LOW_RESPONSE_RATE = 0.2  # Below this = definitely back off
    MEDIUM_RESPONSE_RATE = 0.5  # Below this = reduce frequency

    # Window for considering a message "responded to"
    RESPONSE_WINDOW_S = 600.0  # 10 minutes

    # Maximum events to track
    MAX_EVENTS = 50

    def __init__(self) -> None:
        self._events: Deque[ProactiveEvent] = deque(maxlen=self.MAX_EVENTS)
        self._pending_event: Optional[ProactiveEvent] = None
        self._total_sent: int = 0
        self._total_responded: int = 0
        self._backoff_multiplier: float = 1.0
        self._last_backoff_adjustment: float = 0.0

    def record_proactive_sent(self, source: str = "proactive_presence") -> None:
        """Record that a proactive message was sent to the user."""
        # Close out any pending event as unresponded
        if self._pending_event is not None:
            elapsed = time.time() - self._pending_event.sent_at
            if elapsed > self.RESPONSE_WINDOW_S:
                self._pending_event.responded = False
                self._events.append(self._pending_event)

        event = ProactiveEvent(source=source)
        self._pending_event = event
        self._total_sent += 1

    def record_user_response(self) -> None:
        """Record that the user responded (to the most recent proactive message)."""
        if self._pending_event is not None:
            elapsed = time.time() - self._pending_event.sent_at
            if elapsed <= self.RESPONSE_WINDOW_S:
                self._pending_event.responded = True
                self._pending_event.response_time_s = elapsed
                self._total_responded += 1
            self._events.append(self._pending_event)
            self._pending_event = None

    @property
    def response_rate(self) -> float:
        """Rolling window response rate (0-1)."""
        events = list(self._events)
        if not events:
            return 1.0  # Assume responsive until proven otherwise
        responded = sum(1 for e in events if e.responded)
        return responded / len(events)

    @property
    def avg_response_time_s(self) -> float:
        """Average response time in seconds (for events that got responses)."""
        times = [
            e.response_time_s
            for e in self._events
            if e.responded and e.response_time_s is not None
        ]
        if not times:
            return 0.0
        return sum(times) / len(times)

    @property
    def engagement_score(self) -> float:
        """Composite engagement score (0-1).

        Higher = more engaged, lower = should back off more.
        Factors: response rate, response speed, recency of engagement.
        """
        rate = self.response_rate
        avg_time = self.avg_response_time_s

        # Speed factor: faster responses = higher engagement
        speed_factor = 1.0
        if avg_time > 0:
            # < 30s = fully engaged, > 300s = barely engaged
            speed_factor = max(0.1, min(1.0, 1.0 - (avg_time - 30) / 270))

        # Recency factor: recent responses weight more
        recency = 1.0
        if self._events:
            recent = [e for e in self._events if e.responded]
            if recent:
                last_response = max(e.sent_at for e in recent)
                age = time.time() - last_response
                recency = max(0.1, min(1.0, 1.0 - age / 7200))  # 2hr decay

        return rate * 0.5 + speed_factor * 0.3 + recency * 0.2

    @property
    def should_backoff(self) -> bool:
        """Whether proactive presence should throttle output."""
        if len(self._events) < 3:
            return False  # Not enough data
        return self.response_rate < self.MEDIUM_RESPONSE_RATE

    def get_backoff_multiplier(self) -> float:
        """Get the recommended throttle multiplier for proactive intervals.

        Returns:
            Multiplier >= 1.0 (higher = more throttled).
            1.0 = no backoff
            2.0 = double intervals
            5.0 = max backoff (5x intervals)
        """
        if len(self._events) < 3:
            return 1.0

        rate = self.response_rate

        if rate >= self.MEDIUM_RESPONSE_RATE:
            self._backoff_multiplier = max(1.0, self._backoff_multiplier * 0.9)
        elif rate >= self.LOW_RESPONSE_RATE:
            self._backoff_multiplier = min(3.0, self._backoff_multiplier * 1.2)
        else:
            self._backoff_multiplier = min(5.0, self._backoff_multiplier * 1.5)

        return self._backoff_multiplier

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_sent": self._total_sent,
            "total_responded": self._total_responded,
            "response_rate": round(self.response_rate, 3),
            "avg_response_time_s": round(self.avg_response_time_s, 1),
            "engagement_score": round(self.engagement_score, 3),
            "should_backoff": self.should_backoff,
            "backoff_multiplier": round(self.get_backoff_multiplier(), 2),
            "tracked_events": len(self._events),
        }


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: Optional[UserResponseTracker] = None


def get_user_response_tracker() -> UserResponseTracker:
    """Get the singleton UserResponseTracker instance."""
    global _instance
    if _instance is None:
        _instance = UserResponseTracker()
    return _instance
