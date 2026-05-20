from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.runtime.errors import get_degradation_tracker
from core.skills import native_chat
from core.skills.native_chat import NativeChatSkill, _schedule_background_task


class _Brain:
    async def think(self, prompt, *, context, mode):
        self.prompt = prompt
        self.context = context
        self.mode = mode
        return SimpleNamespace(content="A clear response.")


class _EmptyBrain:
    async def think(self, prompt, *, context, mode):
        return SimpleNamespace(content="")


class _BrokenMemory:
    def remember(self, *args, **kwargs):
        raise RuntimeError("memory database unavailable")


async def _simple_context(message, context, intent):
    return {"prompt_segment": "system context", "intent": intent, **context}


@pytest.fixture(autouse=True)
def isolated_native_chat_state(monkeypatch):
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    monkeypatch.setattr(native_chat, "emitter", None)
    yield
    ServiceContainer.clear()
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_background_scheduler_falls_back_to_active_loop(monkeypatch):
    import core.utils.task_tracker as task_tracker_module

    ran = {"value": False}

    async def marker():
        ran["value"] = True

    def tracker_down():
        raise RuntimeError("tracker down")

    monkeypatch.setattr(task_tracker_module, "get_task_tracker", tracker_down)

    assert _schedule_background_task(marker(), name="native_chat.test") is True
    await asyncio.sleep(0)

    assert ran["value"] is True
    last = get_degradation_tracker().recent(subsystem="native_chat")[-1]
    assert (
        last.action
        == "task tracker could not schedule native_chat.test; falling back to the active event loop"
    )


@pytest.mark.asyncio
async def test_native_chat_returns_response_when_memory_write_fails():
    skill = NativeChatSkill(_Brain())
    skill._build_rich_context = _simple_context

    result = await skill.execute(
        {"params": {"message": "hello"}},
        {"memory": _BrokenMemory()},
    )

    assert result["ok"] is True
    assert result["response"] == "A clear response."
    assert result["degraded"] is True
    assert set(result["degraded_stages"]) == {"memory_aura_write", "memory_user_write"}


@pytest.mark.asyncio
async def test_native_chat_fails_closed_on_empty_brain_response():
    skill = NativeChatSkill(_EmptyBrain())
    skill._build_rich_context = _simple_context

    result = await skill.execute({"params": {"message": "hello"}}, {})

    assert result["ok"] is False
    assert result["stage"] == "cognitive_generation"
    last = get_degradation_tracker().recent(subsystem="native_chat")[-1]
    assert last.severity == "critical"
    assert "no fabricated response" in last.action
