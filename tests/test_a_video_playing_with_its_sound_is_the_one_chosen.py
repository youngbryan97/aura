"""A video playing with its sound on is the one somebody chose, whether or not its page states its length.

LIVE 7 November 2026: asked to find a cooking video and watch it, she chose a short one. Its page stated no length and
its video took 0.269 of the view, so it was not taken as what the page is for, and no watching began.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit


class _Page:
    def __init__(self, plays):
        self.plays = plays

    async def evaluate(self, script):
        from core.skills.watching_along import WHAT_PLAYS

        return self.plays if script == WHAT_PLAYS else {}


class _Browser:
    def __init__(self, plays):
        self.page = _Page(plays)


def _offered(plays):
    from core.skills.sovereign_browser_on_the_page import WATCHING, with_what_plays

    observation = asyncio.run(with_what_plays(_Browser(plays), {"url": "https://videos.example/s/1", "elements": []}))
    return next(e for e in observation["elements"] if e.get("selector") == WATCHING)


SHORT = {"x": 380, "y": 60, "w": 270, "h": 480, "vw": 1024, "vh": 768, "duration": 31.0}


def test_a_video_playing_with_its_sound_is_what_the_page_is_for():
    from core.skills.sovereign_browser_on_the_page import watched_first

    offered = _offered({**SHORT, "paused": False, "muted": False})
    assert offered["its_own"] is True
    assert watched_first({"url": "https://videos.example/s/1", "elements": [offered]},
                         "Find us a cooking video, and let's watch it together.", take=False) is not None


def test_a_muted_preview_is_not():
    from core.skills.sovereign_browser_on_the_page import watched_first

    offered = _offered({**SHORT, "paused": False, "muted": True})
    assert offered["its_own"] is False
    assert watched_first({"url": "https://videos.example/results", "elements": [offered]},
                         "Find us a cooking video, and let's watch it together.", take=False) is None
