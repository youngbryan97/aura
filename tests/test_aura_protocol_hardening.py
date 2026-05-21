from __future__ import annotations

import importlib
import math
import sys

import pytest

from core.consciousness.aura_protocol import AuraMessage, AuraProtocolClient, AuraProtocolServer


def test_message_validation_rejects_non_finite_values():
    assert not AuraMessage(intent="status", urgency=math.nan).validate()
    assert not AuraMessage(intent="status", affect_vector=[math.inf]).validate()
    assert not AuraMessage(intent="status", semantic_embedding=[float("nan")]).validate()


def test_message_serialization_rejects_oversized_intent():
    msg = AuraMessage(intent="x" * 9000, source_identity="Aura-Test")

    assert not msg.validate()

    with pytest.raises(ValueError):
        msg.to_json()


@pytest.mark.asyncio
async def test_process_message_rejects_malformed_payload():
    server = AuraProtocolServer()

    await server._process_message(b"{not json")

    assert server.get_status()["messages_rejected"] == 1


@pytest.mark.asyncio
async def test_handler_failure_does_not_block_later_handlers():
    server = AuraProtocolServer()
    observed: list[str] = []

    async def bad_handler(msg):
        if msg.message_id:
            raise RuntimeError("handler offline")

    async def good_handler(msg):
        observed.append(msg.message_id)

    async def inject_noop(msg):
        observed.append(f"injected:{msg.message_id}")

    server.register_handler(bad_handler)
    server.register_handler(good_handler)
    server._inject_into_workspace = inject_noop

    msg = AuraMessage(intent="coordinate", source_identity="Aura-Beta")
    await server._process_message(msg.to_json().encode("utf-8"))

    assert observed == [msg.message_id, f"injected:{msg.message_id}"]
    assert server.get_status()["messages_received"] == 1


@pytest.mark.asyncio
async def test_client_invalid_message_fails_without_connecting():
    client = AuraProtocolClient()

    result = await client.send(AuraMessage(intent="", semantic_embedding=[]))

    assert result is False
    assert client.get_status()["messages_failed"] == 1
    assert client.get_status()["connected"] is False


def test_consciousness_package_does_not_eager_import_system():
    sys.modules.pop("core.consciousness.system", None)

    import core.consciousness as consciousness

    importlib.reload(consciousness)

    assert "core.consciousness.system" not in sys.modules
