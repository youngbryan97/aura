from __future__ import annotations

import threading

import pytest

from core.brain.inference_gate import InferenceGate


class _ReadyClient:
    def get_lane_status(self):
        return {
            "state": "ready",
            "conversation_ready": True,
            "readiness_blockers": [],
            "last_ready_at": 1.0,
            "last_progress_at": 1.0,
            "last_visible_readiness_at": 1.0,
        }


def test_chat_dependencies_block_public_readiness_after_cortex_is_ready():
    gate = InferenceGate()
    gate._mlx_client = _ReadyClient()
    gate.set_chat_dependencies_ready(False)

    cortex = gate.get_cortex_readiness_status()
    public = gate.get_conversation_status()

    assert cortex["conversation_ready"] is True
    assert public["conversation_ready"] is False
    assert public["chat_dependencies_ready"] is False
    assert "chat_dependencies_warming" in public["readiness_blockers"]
    assert public["last_failure_reason"] == "chat_dependencies_warming"


def test_chat_dependencies_release_same_resident_lane_without_reloading_model():
    gate = InferenceGate()
    client = _ReadyClient()
    gate._mlx_client = client

    gate.set_chat_dependencies_ready(True)
    public = gate.get_conversation_status()

    assert gate._mlx_client is client
    assert public["conversation_ready"] is True
    assert public["chat_dependencies_ready"] is True
    assert "chat_dependencies_warming" not in public["readiness_blockers"]


@pytest.mark.parametrize("ready", [True, False])
def test_dependency_publication_does_not_wait_for_model_operation_owner(ready):
    gate = InferenceGate()
    gate._mlx_client = _ReadyClient()
    published = threading.Event()

    def publish():
        gate.set_chat_dependencies_ready(ready, blocker="chat_dependencies_failed")
        published.set()

    with gate._foreground_ready_lock:
        publisher = threading.Thread(target=publish)
        publisher.start()
        completed_while_owned = published.wait(1.0)
    publisher.join(timeout=2.0)

    assert not publisher.is_alive()
    assert completed_while_owned
    public = gate.get_conversation_status()
    assert public["chat_dependencies_ready"] is ready
    assert public["conversation_ready"] is ready
    assert ("chat_dependencies_failed" in public["readiness_blockers"]) is not ready
