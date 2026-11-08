"""A page she opened by a link she chose plays what she chose, playing yet or not.

LIVE 8 November 2026, rehearsal 11: the video she clicked opened paused at 0.429 of the view, was not taken as hers,
and its page was read for a minute before she pressed its Play.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from core.skills import sovereign_browser, sovereign_browser_on_the_page as on_the_page


class _Page:
    def __init__(self, plays):
        self.plays = plays

    async def evaluate(self, script, *_args):
        if script is on_the_page.WHAT_PLAYS:
            return self.plays
        return {}


def _watching(browser, url):
    seen = asyncio.run(on_the_page.with_what_plays(browser, {"url": url, "elements": []}))
    return next(e for e in seen["elements"] if e.get("selector") == on_the_page.WATCHING)


PAUSED = {"w": 640, "h": 360, "vw": 1280, "vh": 420, "paused": True, "muted": True, "duration": 231}


def test_the_video_on_the_page_her_click_opened_is_the_one_she_chose():
    browser = SimpleNamespace(page=_Page(PAUSED), _chose_to_open=["https://www.youtube.com/watch?v=soxxPyaAT1k&pp=ygUL"])
    assert _watching(browser, "https://www.youtube.com/watch?v=soxxPyaAT1k")["chosen"] is True


def test_a_page_she_did_not_click_through_to_is_not_her_choice():
    browser = SimpleNamespace(page=_Page(PAUSED), _chose_to_open=["https://www.youtube.com/watch?v=OTHER"])
    assert _watching(browser, "https://www.youtube.com/watch?v=soxxPyaAT1k")["chosen"] is False
    assert _watching(SimpleNamespace(page=_Page(PAUSED)), "https://www.youtube.com/watch?v=soxxPyaAT1k")["chosen"] is False


def test_the_loop_keeps_where_her_clicks_lead():
    assert "browser._chose_to_open = [" in inspect.getsource(sovereign_browser)
