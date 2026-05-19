import asyncio

import pytest


def test_conversation_lane_resilient_helper_contains_legacy_override_failure(monkeypatch):
    from interface.routes import system as system_routes

    def broken_legacy_override():
        failure = RuntimeError("legacy lane collector exploded")
        raise failure

    monkeypatch.setattr(system_routes, "_collect_conversation_lane_status", broken_legacy_override)

    lane = system_routes._collect_conversation_lane_status_resilient()

    assert lane["state"] == "degraded"
    assert lane["conversation_ready"] is False
    assert "legacy lane collector exploded" in lane["last_failure_reason"]


@pytest.mark.asyncio
async def test_telemetry_stream_emits_idle_heartbeat_and_unsubscribes(monkeypatch):
    from interface.routes import system as system_routes

    queue: asyncio.Queue = asyncio.Queue()
    unsubscribed = []

    class _Request:
        def __init__(self):
            self.checks = 0

        async def is_disconnected(self):
            self.checks += 1
            return self.checks > 2

    class _Bus:
        async def subscribe(self):
            return queue

        async def unsubscribe(self, subscribed_queue):
            unsubscribed.append(subscribed_queue)

    monkeypatch.setattr(system_routes.config.security, "internal_only_mode", False)
    monkeypatch.setattr(system_routes, "_SSE_IDLE_HEARTBEAT_S", 0.001)
    monkeypatch.setattr(system_routes, "broadcast_bus", _Bus())

    response = await system_routes.telemetry_stream(_Request())
    iterator = response.body_iterator
    first_event = await anext(iterator)
    heartbeat_event = await anext(iterator)
    await iterator.aclose()

    assert "event: telemetry" in first_event
    assert "event: heartbeat" in heartbeat_event
    assert unsubscribed == [queue]
