"""Draft telemetry follows delivery ownership; private channels remain separate."""
from types import SimpleNamespace

from core.runtime.chat_delivery_progress import (
    bind_chat_delivery_progress,
    current_chat_delivery_identity,
)


def test_identity_is_scoped_and_does_not_export_credentials():
    def admission(key):
        return SimpleNamespace(record=SimpleNamespace(
            identity=SimpleNamespace(idempotency_key=key, principal_digest="private"),
            turn_id="turn", generation=2), owner_token="secret")

    assert current_chat_delivery_identity() is None
    with bind_chat_delivery_progress(None, admission("first")):
        assert current_chat_delivery_identity() == {
            "idempotency_key": "first", "turn_id": "turn", "generation": 2}
        with bind_chat_delivery_progress(None, admission("second")):
            assert current_chat_delivery_identity()["idempotency_key"] == "second"
        assert current_chat_delivery_identity()["idempotency_key"] == "first"
    assert current_chat_delivery_identity() is None


def test_state_machine_does_not_broadcast_unowned_drafts():
    from core.cognitive.state_machine import StateMachine

    events = []
    machine = object.__new__(StateMachine)
    machine.orchestrator = SimpleNamespace(_publish_telemetry=events.append)
    machine._emit_telemetry({"type": "chat_stream_start"})
    machine._emit_telemetry({"type": "chat_stream_chunk", "chunk": "private"})
    machine._emit_telemetry({"type": "chat_stream_end"})
    assert events == []
    machine._emit_telemetry({"type": "activity", "label": "working"})
    assert len(events) == 1


def test_state_machine_binds_public_drafts_without_mutating_payload():
    from core.cognitive.state_machine import StateMachine

    events = []
    machine = object.__new__(StateMachine)
    machine.orchestrator = SimpleNamespace(_publish_telemetry=events.append)
    admission = SimpleNamespace(record=SimpleNamespace(
        identity=SimpleNamespace(idempotency_key="owned"), turn_id="turn", generation=1))
    payload = {"type": "chat_stream_chunk", "chunk": "answer"}
    with bind_chat_delivery_progress(None, admission):
        machine._emit_telemetry(payload)
    assert events == [{**payload, "idempotency_key": "owned", "turn_id": "turn", "generation": 1}]
    assert "idempotency_key" not in payload
