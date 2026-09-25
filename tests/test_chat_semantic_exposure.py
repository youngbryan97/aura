"""The admitted chat turn records exposure without putting it on the reply path."""

import asyncio

import pytest

from core.cognition import semantic_runtime
from core.runtime import foreground_guard
from interface.routes import chat


@pytest.mark.asyncio
async def test_chat_observation_schedules_real_semantic_intake(monkeypatch):
    calls = []
    tasks = []

    class Tracker:
        def bounded_track(self, coro, *, name, owner):
            assert name == "ChatSemanticExposure"
            assert owner == "interface.routes.chat"
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

    monkeypatch.setattr(chat, "get_task_tracker", lambda: Tracker())
    monkeypatch.setattr(semantic_runtime, "record_chat_usage",
                        lambda message, **kwargs: calls.append((message, kwargs)))
    monkeypatch.setattr(foreground_guard, "notify_user_spoke",
                        lambda message: calls.append(("foreground", message)))

    chat._observe_admitted_chat_turn("Soooo unusual", "session", "turn",
                                     is_benchmark=False)
    await asyncio.gather(*tasks)
    assert ("Soooo unusual", {"session_id": "session", "turn_id": "turn"}) in calls
    assert ("foreground", "Soooo unusual") in calls

    calls.clear()
    chat._observe_admitted_chat_turn("benchmark text", "session", "turn:2",
                                     is_benchmark=True)
    assert calls == []
