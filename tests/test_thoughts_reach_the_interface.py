"""Seventy-two modules emitted into a channel with no reader.

`ThoughtEmitter` broadcasts to listeners that call `register`, and nothing in
the codebase calls it. The interface's neural feed is fed from the event bus
instead, so every one of those emits was invisible — including a pursuit
narrating its choices while it worked.
"""

from __future__ import annotations

import core.thought_stream as thought_stream
from core.thought_stream import ThoughtEmitter


class _Bus:
    def __init__(self):
        self.published = []

    def publish_threadsafe(self, topic, data, priority=None):
        self.published.append((topic, data))


def test_an_emitted_thought_is_published_where_the_interface_reads(monkeypatch):
    bus = _Bus()
    monkeypatch.setattr("core.event_bus.get_event_bus", lambda: bus)

    ThoughtEmitter().emit("Browsing", "Question 4 -> I agree", category="ToolExecution")

    assert bus.published, "the thought never reached the bus the interface subscribes to"
    topic, payload = bus.published[-1]
    assert topic == "thoughts", "the bridge forwards the topic the UI renders as a thought"
    assert payload["content"] == "Question 4 -> I agree"
    assert payload["title"] == "Browsing"


def test_a_broken_bus_never_breaks_the_work(monkeypatch):
    def explode():
        raise RuntimeError("bus down")

    monkeypatch.setattr("core.event_bus.get_event_bus", explode)
    recorded = []
    monkeypatch.setattr(
        thought_stream, "record_degradation", lambda *a, **k: recorded.append(a)
    )

    ThoughtEmitter().emit("Browsing", "still working")

    assert recorded, "a thought that cannot be shown must degrade, not vanish silently"
