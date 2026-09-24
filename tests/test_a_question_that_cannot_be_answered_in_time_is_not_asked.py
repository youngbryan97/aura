"""Her voice is told when an answer is wanted, and does not start one it cannot finish.

LIVE 2026-09-23, a game answering in two and a half seconds a move: every
question was given the eight seconds ten moves take, the 27B needs about
thirty-five to read one and answer it, and each was started, cancelled at
eight, and left the GPU busy for nothing — the cortex did not finish one
generation that boot.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.agency import her_reasoning


def _a_voice(monkeypatch, needs_s: float) -> list[str]:
    asked: list[str] = []

    async def produce(prompt: str, temperature: float) -> str:
        asked.append(prompt)
        return "left, because it keeps the corner"

    monkeypatch.setattr(her_reasoning, "generator", lambda **_kw: produce)
    monkeypatch.setattr(
        her_reasoning, "time_this_question_needs", lambda _p, _t, floor: max(floor, needs_s)
    )
    return asked


def test_a_question_it_cannot_answer_in_time_is_not_put_to_it(monkeypatch):
    asked = _a_voice(monkeypatch, needs_s=35.0)
    think = her_reasoning.quick_reasoning()

    async def ask() -> str:
        with her_reasoning.answering_by(time.monotonic() + 8.0):
            return await think("which way", ["the board"])

    with pytest.raises(TimeoutError, match="needs about 35s"):
        asyncio.run(ask())
    assert asked == []


def test_a_question_it_can_answer_in_time_is_asked(monkeypatch):
    asked = _a_voice(monkeypatch, needs_s=3.0)
    think = her_reasoning.quick_reasoning()

    async def ask() -> str:
        with her_reasoning.answering_by(time.monotonic() + 8.0):
            return await think("which way", ["the board"])

    assert asyncio.run(ask()).startswith("left")
    assert len(asked) == 1


def test_with_nobody_waiting_the_question_is_asked(monkeypatch):
    asked = _a_voice(monkeypatch, needs_s=100.0)
    think = her_reasoning.quick_reasoning()
    assert asyncio.run(think("how to play", ["the board"])).startswith("left")
    assert len(asked) == 1


def test_the_run_tells_her_voice_when(monkeypatch):
    from core.skills.screen_pursuit_bearings import _within_the_run

    seen: list[float | None] = []

    async def think(objective, evidence):
        seen.append(her_reasoning._ANSWER_BY.get())
        return "up"

    bounded = _within_the_run(think, time.monotonic() + 600.0, 0.8)
    began = time.monotonic()
    assert asyncio.run(bounded("which way", [])) == "up"
    assert seen[0] is not None
    assert seen[0] - began == pytest.approx(8.0, abs=0.5)
    assert her_reasoning._ANSWER_BY.get() is None
