"""The steady mind answers a repeated call from memory only in the same state.

Her feelings reach her language organ's forward pass through the steering hooks,
not through the call. Keyed on the call alone, an arm of a paired trial that
moved only her feelings was handed the answer the other arm got, which erases
the one channel the reports ground is about.
"""

from __future__ import annotations

import asyncio

import pytest

from core.consciousness import steering_channel
from core.subject import steady_mind

pytestmark = pytest.mark.unit


class _Router:
    def __init__(self) -> None:
        self.calls = 0

    async def think(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return f"answer {self.calls}"


def test_same_call_same_state_is_kept_and_another_state_is_asked_again(monkeypatch) -> None:
    steady_mind.forget_for_test()
    state = {"now": [0.5] * 15}
    monkeypatch.setattr(steering_channel, "steering_now", lambda: list(state["now"]))
    router = _Router()
    mind = steady_mind.SteadyMind(router)
    try:
        first = asyncio.run(mind.think("how do you feel"))
        again = asyncio.run(mind.think("how do you feel"))
        assert first == again and router.calls == 1
        state["now"] = [0.9] + [0.5] * 14
        moved = asyncio.run(mind.think("how do you feel"))
        assert router.calls == 2 and moved != first
    finally:
        steady_mind.forget_for_test()
