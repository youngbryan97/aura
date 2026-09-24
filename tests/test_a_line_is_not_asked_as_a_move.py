"""A question about a line to take is not sent under the role for one move.

LIVE 2026-09-24: asked how she would go about the game, under the role that
says to name one move and give one sentence of why, she answered "I am going
to play **up**", which was read, correctly, as a move and not a line.
"""

from __future__ import annotations

import asyncio

from core.agency import her_reasoning


class _Router:
    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    async def think(self, *, messages, **_kwargs):
        self.sent.append(list(messages))
        return "a line"


def _asked(monkeypatch, role):
    router = _Router()
    monkeypatch.setattr(her_reasoning, "_router", lambda: router)
    monkeypatch.setattr(her_reasoning, "time_this_question_needs", lambda _p, _t, floor: floor)
    produce = her_reasoning.generator(origin="agency_settling_on_an_approach", role=role)
    asyncio.run(produce("how will you play this game", 0.3))
    return router.sent[0]


def test_a_line_question_carries_no_move_choosing_role(monkeypatch):
    sent = _asked(monkeypatch, None)
    assert [message["role"] for message in sent] == ["user"]


def test_a_move_question_keeps_its_role(monkeypatch):
    sent = _asked(monkeypatch, her_reasoning.CHOOSING_ROLE)
    assert sent[0] == {"role": "system", "content": her_reasoning.CHOOSING_ROLE}


def test_the_plan_reasoning_asks_without_it():
    import inspect

    source = inspect.getsource(her_reasoning.reasoning_for_a_plan)
    assert "role=None" in source
