from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.orchestrator.meta_cognition_shard import MetaCognitionShard


@pytest.mark.asyncio
async def test_meta_cognition_audit_loop_defers_when_background_policy_blocks(monkeypatch):
    orchestrator = MagicMock()
    orchestrator.status = SimpleNamespace(healthy=True)
    shard = MetaCognitionShard(orchestrator)
    shard.is_running = True
    shard.perform_audit = AsyncMock()

    monkeypatch.setattr(
        "core.orchestrator.meta_cognition_shard.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.24",
    )

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds):
        sleep_calls["count"] += 1
        shard.is_running = False

    monkeypatch.setattr("core.orchestrator.meta_cognition_shard.asyncio.sleep", _fake_sleep)

    await shard._audit_loop()

    shard.perform_audit.assert_not_awaited()
    assert sleep_calls["count"] == 1
