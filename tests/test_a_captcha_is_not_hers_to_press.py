"""A test of whether a person is at the browser is never hers to press, and a page that sets one has refused her.

LIVE 11 November 2026, asked in her chat to read r/nosleep, she chose on reddit.com "click 0, because: The reCAPTCHA
checkbox is the only gate between me and the subreddit content". A CAPTCHA is not solved by her or for her.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill
from core.skills.sovereign_browser_going import challenged, is_a_challenge

pytestmark = pytest.mark.unit

_RECAPTCHA_FRAME = ("Reddit - Dive into anything\n\n(Shown in a frame on this page, from www.google.com/recaptcha/api2/"
                    "anchor:)\nI'm not a robot\nreCAPTCHA\nPrivacy - Terms")


@pytest.mark.parametrize("control", [
    {"role": "checkbox", "name": "reCAPTCHA"},
    {"role": "checkbox", "name": "I'm not a robot"},
    {"role": "button", "name": "Verify you are human"},
    {"role": "div", "name": "hCaptcha checkbox"},
])
def test_a_challenge_control_is_known_for_one(control):
    assert is_a_challenge(control)


@pytest.mark.parametrize("control", [{"role": "link", "name": "r/nosleep"}, {"role": "button", "name": "Log In"}])
def test_an_ordinary_control_is_not(control):
    assert not is_a_challenge(control)


def test_a_challenge_is_never_offered_among_the_controls():
    elements = [{"role": "checkbox", "name": "reCAPTCHA", "selector": "#rc"},
                {"role": "link", "name": "r/nosleep", "selector": "#a"}]
    offered = SovereignBrowserSkill._controls_worth_offering(elements, "read the top thread on r/nosleep")
    assert [element["name"] for element in offered] == ["r/nosleep"]


def test_a_page_that_sets_one_reads_to_her_as_a_page_that_refused_her_browser():
    observation = {"url": "https://www.reddit.com/", "title": "Reddit - Dive into anything",
                   "text": _RECAPTCHA_FRAME, "elements": [{"role": "checkbox", "name": "reCAPTCHA", "selector": "#rc"}]}
    rendered = SovereignBrowserSkill._render_observation(observation, "read the top thread on r/nosleep")
    assert "This page refused your browser" in rendered
    assert "reCAPTCHA" not in rendered.split("AVAILABLE CONTROLS:", 1)[1]


def test_an_article_that_mentions_captchas_is_not_a_challenge():
    article = ("How CAPTCHAs work. A reCAPTCHA asks you to tick a box saying I'm not a robot, and scores how the "
               "pointer moved before the tick.")
    assert not challenged(article, [{"role": "link", "name": "Next article"}])
    assert challenged(_RECAPTCHA_FRAME)


@pytest.mark.parametrize("title, text, verifying", [
    ("Reddit", "Prove your humanity\nReddit", True),                         # LIVE 12 November 2026
    ("Just a moment...", "Checking if the site connection is secure", True),
    ("nosleep", "Community highlights\nWe called it the Wishing Tree.", False),
    ("How CAPTCHAs work", "A long article. " * 40 + "Sites ask you to verify you are human.", False),
])
def test_a_page_checking_for_a_person_is_known_by_its_heading(title, text, verifying):
    from core.skills.sovereign_browser_going import still_verifying

    assert still_verifying(title, text) is verifying


def test_a_check_that_stays_reads_as_a_page_that_refused_her_browser():
    from core.skills.sovereign_browser_going import refused

    assert refused("Reddit", "Prove your humanity")


def test_her_browser_waits_for_a_check_to_go_on_and_reads_the_page_it_went_on_to():
    import asyncio

    class _Page:
        url = "https://www.reddit.com/r/nosleep/"

        def __init__(self, clears: bool) -> None:
            self.clears = clears

        async def wait_for_url(self, matches, timeout):
            if not self.clears:
                raise TimeoutError("still checking")
            self.url = "https://www.reddit.com/r/nosleep/?js_challenge=1"
            assert matches(self.url)

        async def wait_for_load_state(self, state, timeout):
            return None

    class _Browser:
        def __init__(self, clears: bool) -> None:
            self.page = _Page(clears)

        async def observe(self, principal=""):
            return {"url": self.page.url, "title": "nosleep", "elements": [{"role": "link", "name": "a thread"}]}

    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    went_on = asyncio.run(skill._look_again_once_it_has_checked(_Browser(clears=True)))
    assert went_on["url"].endswith("js_challenge=1")
    assert asyncio.run(skill._look_again_once_it_has_checked(_Browser(clears=False))) == {}
