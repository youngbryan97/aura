"""A subreddit named as people name one, "r/nosleep", is where her browser starts, not Reddit's front page.

LIVE 11 November 2026, asked "Can you take a look at r/nosleep on Reddit and tell me what the top thread is about right
now?", her browser was sent to https://www.reddit.com, and the front page set a reCAPTCHA in front of her. Her own
Reddit round, which goes to the subreddit's page, read r/nosleep that day.
"""
from __future__ import annotations

import pytest

from core.conversation.page_interaction import front_page_named
from core.intent.asks_to_go_online import place_named, site_named, the_place_it_continues
from core.skills.sovereign_browser import BrowserInput

pytestmark = pytest.mark.unit

_ASKED = "Can you take a look at r/nosleep on Reddit and tell me what the top thread is about right now?"


def test_the_subreddit_named_is_the_page_her_browser_is_given():
    assert site_named(_ASKED) == "https://www.reddit.com/r/nosleep/"
    assert front_page_named(_ASKED) == "https://www.reddit.com/r/nosleep/"
    assert BrowserInput(goal=_ASKED, max_steps=40).url == "https://www.reddit.com/r/nosleep/"


@pytest.mark.parametrize("said, page", [
    ("Check r/AskHistorians for me", "https://www.reddit.com/r/AskHistorians/"),
    ("Go to Reddit", "https://www.reddit.com"),                                     # no subreddit: the front page
    ("Go to YouTube", "https://www.youtube.com"),
    ("my r/w ratio is bad", ""),                                                    # one letter is not a subreddit
    ("it was a 2/3 r/s thing", ""),
])
def test_only_a_subreddit_named_changes_the_page(said, page):
    assert site_named(said) == page


def test_a_subreddit_inside_an_address_is_left_to_the_address():
    asked = "Go to https://www.reddit.com/r/nosleep/comments/abc/x/ and read it"
    assert site_named(asked) == "https://www.reddit.com"
    assert place_named(asked) == "https://www.reddit.com"


def test_the_next_request_there_goes_on_at_the_subreddit():
    assert the_place_it_continues("Now read me the second thread", [_ASKED])[0] == "https://www.reddit.com/r/nosleep/"
