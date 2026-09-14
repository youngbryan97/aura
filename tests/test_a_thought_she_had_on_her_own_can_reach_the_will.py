"""The thoughts topic reaches the neural intent router, for trusted sources only.

The router gates on trusted internal sources, a whitelist of action schemas and
the Will, and nothing ever called it. These pin the listener in front of it:
which thoughts it hands on, which it ignores, and that a router failure is
recorded rather than lost.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.agency.thought_to_action import ThoughtRouter


class _Router:
    def __init__(self, raises: BaseException | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.raises = raises

    async def route(self, source: str, text: str) -> Any:
        if self.raises is not None:
            raise self.raises
        self.calls.append((source, text))
        return None


def _handle(router: _Router, message: Any) -> tuple[bool, ThoughtRouter]:
    listener = ThoughtRouter(router=router)
    return asyncio.run(listener.handle(message)), listener


def test_a_goal_from_volition_reaches_the_router() -> None:
    router = _Router()
    reached, listener = _handle(router, {"title": "Volition", "content": "Goal: search tide tables", "source": "volition_engine"})
    assert reached and router.calls == [("volition_engine", "Goal: search tide tables")]
    assert listener.as_dict() == {"routed": 1, "ignored": 0, "failed": 0}


def test_a_drive_intention_reaches_the_router() -> None:
    router = _Router()
    reached, _ = _handle(router, {"content": "look up how kilns are fired", "source": "Drive_Engine"})
    assert reached and router.calls == [("drive_engine", "look up how kilns are fired")]


def test_a_thought_that_does_not_say_where_it_came_from_is_not_routed() -> None:
    router = _Router()
    reached, listener = _handle(router, {"title": "Volition", "content": "Goal: search tide tables"})
    assert not reached and router.calls == [] and listener.ignored == 1


def test_a_source_the_router_does_not_trust_is_not_routed() -> None:
    router = _Router()
    reached, _ = _handle(router, {"content": "search anything", "source": "user_message"})
    assert not reached and router.calls == []


def test_an_empty_thought_is_not_routed() -> None:
    router = _Router()
    reached, _ = _handle(router, {"content": "   ", "source": "volition_engine"})
    assert not reached and router.calls == []


def test_something_that_is_not_a_message_is_not_routed() -> None:
    router = _Router()
    reached, _ = _handle(router, "Goal: search tide tables")
    assert not reached and router.calls == []


def test_a_router_that_raises_is_recorded_and_counted() -> None:
    router = _Router(raises=RuntimeError("will unavailable"))
    reached, listener = _handle(router, {"content": "Goal: search tide tables", "source": "volition_engine"})
    assert not reached
    assert listener.as_dict() == {"routed": 0, "ignored": 0, "failed": 1}


def test_starting_and_stopping_the_listener_leaves_no_task_behind() -> None:
    async def run() -> None:
        listener = ThoughtRouter(router=_Router())
        await listener.start()
        assert listener._task is not None
        await listener.stop()
        assert listener._task is None

    asyncio.run(run())


def test_a_thought_arriving_in_the_bus_envelope_reaches_the_router(monkeypatch) -> None:
    import core.event_bus as event_bus

    router = _Router()

    class _Bus:
        async def subscribe(self, topic: str) -> asyncio.Queue:
            queue: asyncio.Queue = asyncio.Queue()
            # The shape _publish_local puts on a subscriber's queue.
            queue.put_nowait((1, 1, {"topic": topic, "data": {"content": "Goal: search tide tables", "source": "volition_engine"}}))
            return queue

    monkeypatch.setattr(event_bus, "get_event_bus", lambda: _Bus())

    async def run() -> None:
        listener = ThoughtRouter(router=router)
        await listener.start()
        for _ in range(200):
            if router.calls:
                break
            await asyncio.sleep(0)
        await listener.stop()

    asyncio.run(run())
    assert router.calls == [("volition_engine", "Goal: search tide tables")]


def test_the_activator_does_not_start_a_foreground_only_runtime() -> None:
    from core.agency.thought_to_action import activate_thought_router

    result = asyncio.run(activate_thought_router(foreground_only=True))
    assert result.ok and "foreground-only" in result.detail
