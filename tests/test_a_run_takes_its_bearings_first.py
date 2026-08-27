"""What a run belongs to is worked out before anything depends on the answer.

The block that gave a run its window sat inside a guard that required one. So
with nothing named the run never acquired an anchor, never brought anything
forward, and sent every key to whatever happened to be in front — for want of
a first cycle it could never have. Measured live 2026-08-26: thirty-five moves
into a terminal with the game open one window back.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.skills import screen_pursuit
from core.skills.screen_pursuit import _take_the_run_its_bearings

SOURCE = Path("core/skills/screen_pursuit.py").read_text()
BODY = SOURCE[SOURCE.index("async def observe") :]


def test_the_bearings_are_taken_before_the_window_is_needed():
    takes = BODY.index("_take_the_run_its_bearings(")
    needs = BODY.index('mine = target_app or anchor["app"]')
    assert takes < needs


def test_nothing_about_taking_them_depends_on_already_having_them():
    where = BODY.index("_take_the_run_its_bearings(")
    guard = BODY[BODY.rindex("if ", 0, where) : where]
    assert 'if not anchor["page"] or not anchor["app"]' in guard


@pytest.mark.asyncio
async def test_a_run_about_a_page_belongs_to_the_application_holding_it(monkeypatch):
    async def page():
        return {"url": "https://play2048.co/", "title": "2048", "app": "Google Chrome"}

    monkeypatch.setattr(screen_pursuit, "current_page_identity", page)
    anchor = {"page": "", "app": ""}
    await _take_the_run_its_bearings(anchor, open_page="2048")
    assert anchor["app"] == "Google Chrome"
    assert "play2048" in anchor["page"]


@pytest.mark.asyncio
async def test_a_run_about_something_else_belongs_to_what_is_in_front(monkeypatch):
    async def nothing():
        return {"url": "", "title": "", "app": ""}

    async def front():
        return "Notes"

    monkeypatch.setattr(screen_pursuit, "current_page_identity", nothing)
    monkeypatch.setattr(screen_pursuit, "_frontmost", front)
    anchor = {"page": "", "app": ""}
    await _take_the_run_its_bearings(anchor)
    assert anchor["app"] == "Notes"


@pytest.mark.asyncio
async def test_a_caller_that_named_them_keeps_what_it_named(monkeypatch):
    async def page():
        return {"url": "https://elsewhere.example", "title": "x", "app": "Safari"}

    monkeypatch.setattr(screen_pursuit, "current_page_identity", page)
    anchor = {"page": "docs.example", "app": "Numbers"}
    await _take_the_run_its_bearings(anchor, expect_page="docs.example")
    assert anchor == {"page": "docs.example", "app": "Numbers"}
