"""A dependency still initializing must get its chance to finish admission."""

import pytest


@pytest.mark.asyncio
async def test_dependency_warmup_retries_without_downgrading(monkeypatch):
    from interface.routes import chat

    class Gate:
        calls = 0

        async def ensure_foreground_ready(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("chat_dependencies_warming")
            return {"conversation_ready": True, "state": "ready"}

    async def yield_once(_delay):
        return None

    monkeypatch.setattr(chat.asyncio, "sleep", yield_once)
    gate = Gate()
    _, reason, hard_failure, lane = await chat._admit_to_foreground_lane(
        _remaining_foreground_budget=lambda **kwargs: 30.0,
        gate=gate,
        lane={"state": "ready", "conversation_ready": False},
    )
    assert gate.calls == 2
    assert reason == ""
    assert hard_failure is False
    assert lane["conversation_ready"] is True
