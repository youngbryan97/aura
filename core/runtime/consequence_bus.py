"""core/runtime/consequence_bus.py — Action Consequence Broadcast System.

Every consequential action publishes a before/after state delta so that
welfare, memory, body, will, and learning subsystems can update from real
outcomes rather than predictions alone.

Design:
  - Synchronous publish (no async tax for the critical path)
  - Subscribers register with a topic filter
  - Each event carries predicted + actual welfare deltas
  - Thread-safe via simple lock (consequences are rare, <100/s)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ConsequenceBus")

_SUBSCRIBER_DELIVERY_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
)


@dataclass(frozen=True)
class ConsequenceEvent:
    """Immutable record of an action's predicted and actual consequences."""
    event_id: str
    timestamp: float
    source: str                      # which subsystem produced this action
    domain: str                      # action domain (tool_execution, memory_write, etc.)
    action_content: str              # short description of the action
    predicted_welfare_delta: dict[str, float] = field(default_factory=dict)
    predicted_body_cost: dict[str, float] = field(default_factory=dict)
    predicted_memory_risk: float = 0.0
    predicted_integrity_risk: float = 0.0
    actual_outcome: str = ""         # success / failure / partial / timeout
    actual_welfare_delta: dict[str, float] = field(default_factory=dict)
    actual_body_cost: dict[str, float] = field(default_factory=dict)
    recovery_required: float = 0.0   # 0-1 how much recovery is needed
    will_receipt_id: str = ""        # provenance link
    error: str = ""                  # error message if failed
    #: True when this event REPORTS a measurement rather than records an
    #: action's consequence.
    #:
    #: CP126 594d43b7: subsystems were publishing here specifically to appear
    #: in the system-Φ stream — one docstring said so outright ("join the
    #: ghost line's system-Φ event stream as a real subsystem"). Φ measures
    #: how much the organs actually cause one another; an event published in
    #: order to be counted raises cross-subsystem influence and subsystem
    #: diversity using the measurement's own publication. The instrument was
    #: reading its own reflection.
    #:
    #: Measurement events stay on the bus — they are useful to subscribers —
    #: and are excluded from integration accounting, which reports how many
    #: it excluded so the exclusion is visible rather than silent.
    measurement_only: bool = False


# Type for subscriber callbacks
ConsequenceSubscriber = Callable[[ConsequenceEvent], None]


class ConsequenceBus:
    """Central broadcast bus for action consequences.

    Subscribers register for specific domains or '*' for all events.
    Events are delivered synchronously in registration order.
    """

    _instance: ConsequenceBus | None = None
    _lock_cls = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: dict[str, list[ConsequenceSubscriber]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_count = 0
        self._history: list[ConsequenceEvent] = []
        self._max_history = 500

    @classmethod
    def get(cls) -> ConsequenceBus:
        """Singleton accessor."""
        if cls._instance is None:
            with cls._lock_cls:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def subscribe(self, domain: str, callback: ConsequenceSubscriber) -> None:
        """Register a callback for events in the given domain (or '*' for all)."""
        with self._lock:
            self._subscribers[domain].append(callback)

    def publish(self, event: ConsequenceEvent) -> None:
        """Broadcast a consequence event to all matching subscribers."""
        with self._lock:
            self._event_count += 1
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Collect matching subscribers
            targets = list(self._subscribers.get("*", []))
            targets.extend(self._subscribers.get(event.domain, []))

        # Deliver outside the lock
        for callback in targets:
            try:
                callback(event)
            except _SUBSCRIBER_DELIVERY_ERRORS as exc:
                record_degradation(
                    "consequence_bus",
                    exc,
                    action="continued consequence broadcast after subscriber delivery failed",
                )
                logger.warning("ConsequenceBus subscriber error: %s", exc)

        # Automatically route to the central AuraEventBus
        try:
            from core.event_bus import get_event_bus, EventPriority
            import dataclasses
            event_dict = dataclasses.asdict(event)
            get_event_bus().publish_threadsafe(
                topic="aura/events/consequences",
                data=event_dict,
                priority=EventPriority.AUTONOMIC,
            )
        except _SUBSCRIBER_DELIVERY_ERRORS as eb_err:
            record_degradation("consequence_bus.event_bridge", eb_err)
            logger.debug("Failed to publish consequence to central EventBus: %s", eb_err)

    def publish_action(
        self,
        *,
        source: str,
        domain: str,
        action_content: str,
        predicted_welfare_delta: dict[str, float] | None = None,
        predicted_body_cost: dict[str, float] | None = None,
        predicted_memory_risk: float = 0.0,
        predicted_integrity_risk: float = 0.0,
        actual_outcome: str = "",
        actual_welfare_delta: dict[str, float] | None = None,
        actual_body_cost: dict[str, float] | None = None,
        recovery_required: float = 0.0,
        will_receipt_id: str = "",
        error: str = "",
    ) -> ConsequenceEvent:
        """Convenience method to create and publish an event."""
        import hashlib
        event_id = hashlib.sha256(
            f"{time.time():.9f}:{source}:{domain}:{action_content[:100]}".encode()
        ).hexdigest()[:16]

        event = ConsequenceEvent(
            event_id=event_id,
            timestamp=time.time(),
            source=source,
            domain=domain,
            action_content=action_content[:200],
            predicted_welfare_delta=dict(predicted_welfare_delta or {}),
            predicted_body_cost=dict(predicted_body_cost or {}),
            predicted_memory_risk=predicted_memory_risk,
            predicted_integrity_risk=predicted_integrity_risk,
            actual_outcome=actual_outcome,
            actual_welfare_delta=dict(actual_welfare_delta or {}),
            actual_body_cost=dict(actual_body_cost or {}),
            recovery_required=recovery_required,
            will_receipt_id=will_receipt_id,
            error=error,
        )
        self.publish(event)
        return event

    @property
    def event_count(self) -> int:
        return self._event_count

    def recent_events(self, n: int = 20) -> list[ConsequenceEvent]:
        """Return the last n events."""
        with self._lock:
            return list(self._history[-n:])

    def events_for_domain(self, domain: str, n: int = 50) -> list[ConsequenceEvent]:
        """Return recent events for a specific domain."""
        with self._lock:
            return [e for e in self._history if e.domain == domain][-n:]

    def failure_rate(self, window: int = 50) -> float:
        """Fraction of recent events that were failures."""
        with self._lock:
            recent = self._history[-window:]
        if not recent:
            return 0.0
        failures = sum(1 for e in recent if e.actual_outcome == "failure")
        return failures / len(recent)
