"""The time is a fact about the world, shown beside the page's address.

The Cartoon Network demo asks her to count from the last digit of the current
time. Her page decisions saw the address, the title and the page, and no clock,
so a request that turns on the time could not be followed from any page that
does not print one.
"""
from __future__ import annotations

import time

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


def test_her_page_decision_is_shown_the_time():
    rendered = SovereignBrowserSkill._render_observation(
        {"url": "https://example.test/list", "title": "Games", "text": "Game 1\nGame 2", "elements": []}
    )
    lines = rendered.splitlines()
    now = next(line for line in lines if line.startswith("Now: "))
    assert time.strftime("%Y") in now and time.strftime("%B") in now
    assert lines.index(now) == lines.index("Title: Games") + 1


def test_a_pursuit_shows_when_it_was_asked_for():
    """LIVE 2026-10-03 05:46-05:56: the game was worked out from the clock again
    on every page, 8 then 11, and she left the right page as the wrong one."""
    import contextvars

    from core.skills.sovereign_browser_one_question import ASKED_AT

    def _render() -> str:
        ASKED_AT.set(time.mktime((2026, 10, 3, 5, 46, 38, 0, 0, -1)))
        return SovereignBrowserSkill._render_observation(
            {"url": "https://example.test/list", "title": "Games", "text": "Game 1", "elements": []}
        )

    lines = contextvars.copy_context().run(_render).splitlines()
    asked = next(line for line in lines if line.startswith("Asked at: "))
    assert "05:46" in asked
    assert lines.index(asked) == lines.index("Title: Games") + 2


def test_outside_a_pursuit_there_is_no_asked_at_and_the_text_keeps_its_heading():
    import contextvars

    from core.skills.sovereign_browser_one_question import ASKED_AT

    def _render() -> str:
        ASKED_AT.set(None)
        return SovereignBrowserSkill._render_observation(
            {"url": "https://example.test/list", "title": "Games", "text": "Game 1", "elements": []}
        )

    lines = contextvars.copy_context().run(_render).splitlines()
    assert not any(line.startswith("Asked at: ") for line in lines)
    assert lines[lines.index("PAGE TEXT:") + 1] == "Game 1"
