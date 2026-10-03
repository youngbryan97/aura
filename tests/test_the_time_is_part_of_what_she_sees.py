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
