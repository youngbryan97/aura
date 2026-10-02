"""A question is measured among every question on its page, once, whatever round answers it.

Grid statements are placed by how they stand beside the rest of the grid, and
two-ended questions in proportion to the widest lean on the page. Measuring each
round's batch on its own gave a statement different neighbours from one round to
the next: LIVE 2026-10-02, "I study how to hold on to my money" read +0.99
against the whole page and was answered Disagree among the last four left open.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.self import where_i_stand
from core.skills.sovereign_browser_one_question import SCREENS_MEASURED, measure_the_screen

pytestmark = pytest.mark.unit


def _question(group: str) -> tuple[str, list[dict[str, Any]]]:
    return group, [{"group": group, "selector": f"#{group}{n}", "asks": f"{group} [1] [2] [3]"} for n in range(3)]


class _Skill:
    def __init__(self) -> None:
        self.measured_among: list[list[str]] = []

    def _a_grid_of_statements(self, questions):
        self.measured_among.append([group for group, _options in questions])
        return {}

    def _measure_where_she_stands(self, options):
        group = options[0]["group"]
        lean = where_i_stand.Lean(toward=0.0, first=0.0, second=0.0, measured=True, gap=float(len(group)))
        return 1, lean, "a", "b"


def test_a_batch_is_measured_among_the_whole_screen_and_only_once():
    skill = _Skill()
    page = [_question(g) for g in ("q1", "q2", "q3", "q4")]

    async def two_rounds():
        SCREENS_MEASURED.set({})
        first = await measure_the_screen(skill, page[:2], among=page)
        second = await measure_the_screen(skill, page[2:], among=page)
        return first, second

    first, second = asyncio.run(two_rounds())
    assert [item["group"] for item in first] == ["q1", "q2"]
    assert [item["group"] for item in second] == ["q3", "q4"]
    assert skill.measured_among == [["q1", "q2", "q3", "q4"]], (
        "each round must read the one measurement of the whole screen"
    )


def test_a_question_left_out_of_among_is_still_measured():
    skill = _Skill()
    page = [_question(g) for g in ("q1", "q2")]
    measured = asyncio.run(measure_the_screen(skill, page, among=[]))
    assert [item["group"] for item in measured] == ["q1", "q2"]


def test_each_text_is_embedded_once_in_a_measurement(monkeypatch):
    calls: list[str] = []

    class _Embedder:
        def embed(self, text: str):
            calls.append(text)
            return [1.0, float(len(text))]

    monkeypatch.setattr(where_i_stand, "_the_embedder", lambda: _Embedder())
    record = [where_i_stand.Piece(said="truth"), where_i_stand.Piece(said="care")]
    with where_i_stand.one_measurement():
        where_i_stand.where_she_stands("makes lists", "relies on memory", record)
        where_i_stand.where_she_stands("sceptical", "wants to believe", record)
    assert calls.count("truth") == 1 and calls.count("care") == 1
    # Outside a measurement nothing is kept.
    where_i_stand.where_she_stands("makes lists", "relies on memory", record)
    assert calls.count("truth") == 2
