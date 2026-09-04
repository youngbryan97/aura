"""Driving an application is not driving whatever a browser is showing.

The run anchors itself on its first cycle so drift is always detectable. It
took the browser's current page whether or not the task had anything to do
with one — and then every cycle checked that the page was still in front and
brought the BROWSER forward to restore it, over the window it was driving.

LIVE 2026-09-04: "this run belongs to '2048 Game' on 'https://x.com/home'",
and readings of a four by four board that came back full of a timeline. The
same care already stood two lines below, deciding which application the run
belongs to, and had never been applied to the page.
"""

from __future__ import annotations

import inspect

from core.skills import screen_pursuit

SOURCE = inspect.getsource(screen_pursuit.pursue_on_screen)


def test_the_page_anchor_asks_whether_the_run_is_about_a_page():
    at = SOURCE.index('if not anchor["page"]:')
    nearby = SOURCE[at : at + 1400]
    assert "about_a_page = bool(open_page or expect_page)" in nearby
    assert "if about_a_page else" in nearby


def test_a_caller_that_named_a_page_still_gets_one():
    at = SOURCE.index('if not anchor["page"]:')
    nearby = SOURCE[at : at + 1400]
    assert "expect_page\n" in nearby or "expect_page " in nearby


def test_the_application_anchor_asks_the_same_question():
    at = SOURCE.index('if not anchor["app"]:')
    nearby = SOURCE[at : at + 700]
    assert "about_a_page else" in nearby


def test_it_is_asked_once_and_used_for_both():
    """One question, two answers that must agree."""
    asked = SOURCE.count("about_a_page = bool(open_page or expect_page)")
    assert asked == 1
