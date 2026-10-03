"""A pursuit is not silent while her model is reading or writing for it.

LIVE 2026-10-01: her forecast for a personality test decoded for 599 seconds,
the pursuit reported nothing for the whole of it, and the executor's
600-second silence ceiling stopped the run before the first question. Only
real reading and writing counts: a generation that stops moving is still
silence.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills import sovereign_browser_one_question as u
from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


class _Client:
    def __init__(self) -> None:
        self.at = 0.0

    def get_lane_status(self) -> dict[str, float]:
        return {"last_token_progress_at": self.at, "last_prefill_progress_at": 0.0}


def _run_with(monkeypatch, client: _Client, *, moves: bool) -> list[str]:
    heard: list[str] = []
    monkeypatch.setattr("core.brain.llm.mlx_client.clients_snapshot", lambda: [("cortex", client)])
    monkeypatch.setattr(u, "_HER_MODEL_SPEAKS_UP_EVERY_S", 0.01)

    async def _go() -> None:
        u.SAYING_IT_MOVES.set(heard.append)
        async with u.while_she_writes("her model is writing"):
            for _ in range(8):
                await asyncio.sleep(0.02)
                if moves:
                    client.at += 1.0

    asyncio.run(_go())
    return heard


def test_writing_is_reported_as_progress(monkeypatch):
    assert _run_with(monkeypatch, _Client(), moves=True), "her model wrote and nothing said so"


def test_a_generation_that_stops_moving_is_still_silence(monkeypatch):
    assert _run_with(monkeypatch, _Client(), moves=False) == []


def test_outside_a_pursuit_nothing_is_watched(monkeypatch):
    called: list[Any] = []
    monkeypatch.setattr("core.brain.llm.mlx_client.clients_snapshot", lambda: called.append(1) or [])

    async def _go() -> None:
        # Outside one, said here: an async test before this one on a shared
        # loop can leave a pursuit's progress hook in the context this copies.
        u.SAYING_IT_MOVES.set(None)
        async with u.while_she_writes("x"):
            await asyncio.sleep(0.01)

    asyncio.run(_go())
    assert called == []


def test_a_forecast_is_given_the_room_a_line_can_show(monkeypatch):
    skill = SovereignBrowserSkill()
    handed: dict[str, Any] = {}

    async def _asked(prompt, mind="", *, shaped=True, most_tokens=None, worked_out_here=True):
        handed["most_tokens"] = most_tokens
        return "I expect INTJ.", "Cortex"

    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    said = asyncio.run(
        skill._what_she_expects_it_to_say("take it", {"url": "u", "title": "t", "text": "x", "elements": []}, "")
    )
    assert said == "I expect INTJ."
    assert handed["most_tokens"] == SovereignBrowserSkill.REASON_MAX_TOKENS
