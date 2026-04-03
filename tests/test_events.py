################################################################################

"""
Unit tests for core.events.InputBus.
"""
import pytest
import threading
import time
from core.events import InputBus, Event, EventType


@pytest.fixture
def event_bus():
    """Provide a fresh InputBus for each test."""
    return InputBus()

class TestPublishAndNext:
    def test_publish_and_next(self, event_bus):
        """Published events can be dequeued via next()."""
        evt = Event(type=EventType.SYSTEM, payload={"msg": "hello"})
        event_bus.publish(evt)
        result = event_bus.next(timeout=1.0)
        assert result is not None
        assert result.payload["msg"] == "hello"

    def test_next_returns_none_on_empty(self, event_bus):
        """next() returns None when queue is empty."""
        result = event_bus.next(timeout=0.05)
        assert result is None

    def test_queue_overflow_drops_silently(self):
        """When queue is full, publish drops events without raising."""
        bus = InputBus(maxsize=2)
        bus.publish(Event(type=EventType.SYSTEM, payload={"i": 1}))
        bus.publish(Event(type=EventType.SYSTEM, payload={"i": 2}))
        # This should not raise or block
        bus.publish(Event(type=EventType.SYSTEM, payload={"i": 3}), block=False)


class TestTypedSubscriptions:
    def test_subscribe_receives_events(self, event_bus):
        """Subscribers are called when matching events are published."""
        received = []
        event_bus.subscribe(EventType.ERROR, lambda e: received.append(e))
        event_bus.emit(EventType.ERROR, {"error": "test_error"})
        assert len(received) == 1
        assert received[0].payload["error"] == "test_error"

    def test_subscribe_filters_by_type(self, event_bus):
        """Subscribers only receive events of their subscribed type."""
        error_events = []
        system_events = []
        event_bus.subscribe(EventType.ERROR, lambda e: error_events.append(e))
        event_bus.subscribe(EventType.SYSTEM, lambda e: system_events.append(e))

        event_bus.emit(EventType.ERROR, {"msg": "err"})
        event_bus.emit(EventType.SYSTEM, {"msg": "sys"})

        assert len(error_events) == 1
        assert len(system_events) == 1

    def test_subscriber_error_does_not_crash_bus(self, event_bus):
        """A failing subscriber doesn't prevent other subscribers from running."""
        results = []

        def bad_subscriber(e):
            raise ValueError("subscriber exploded")

        def good_subscriber(e):
            results.append(e.payload)

        event_bus.subscribe(EventType.SYSTEM, bad_subscriber)
        event_bus.subscribe(EventType.SYSTEM, good_subscriber)

        event_bus.emit(EventType.SYSTEM, {"test": True})
        assert len(results) == 1
        assert results[0]["test"] is True

    def test_emit_convenience(self, event_bus):
        """emit() creates and publishes an event in one call."""
        collected = []
        event_bus.subscribe(EventType.HEALTH_ALERT, lambda e: collected.append(e))
        event_bus.emit(EventType.HEALTH_ALERT, {"cpu": 95.0}, source="monitor")

        assert len(collected) == 1
        assert collected[0].source == "monitor"
        assert collected[0].payload["cpu"] == 95.0

    def test_subscriber_count(self, event_bus):
        """subscriber_count reflects total subscriptions."""
        assert event_bus.subscriber_count == 0
        event_bus.subscribe(EventType.ERROR, lambda e: None)
        event_bus.subscribe(EventType.SYSTEM, lambda e: None)
        assert event_bus.subscriber_count == 2


class TestThreadSafety:
    def test_concurrent_publish(self, event_bus):
        """Concurrent publishers don't corrupt the queue."""
        def publisher(bus, n):
            for i in range(n):
                bus.emit(EventType.SYSTEM, {"thread": threading.current_thread().name, "i": i})

        threads = [threading.Thread(target=publisher, args=(event_bus, 20)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        count = 0
        while event_bus.next(timeout=0.01) is not None:
            count += 1
        assert count == 100  # 5 threads × 20 events


##
