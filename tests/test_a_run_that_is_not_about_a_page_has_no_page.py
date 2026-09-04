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


# ── and the same question, asked when the run takes its bearings ─────────


def test_taking_its_bearings_does_not_call_an_open_browser_a_page_task():
    """The test included whether any page was open anywhere, which is true
    whenever a browser is running."""
    bearings = inspect.getsource(screen_pursuit._take_the_run_its_bearings)
    assert "about_a_page = bool(open_page or expect_page)" in bearings
    assert "page.get(\"url\"))" not in bearings.split("about_a_page = ")[1][:80]


def test_a_run_that_names_an_application_belongs_to_it():
    bearings = inspect.getsource(screen_pursuit._take_the_run_its_bearings)
    at = bearings.index('anchor["app"] = (')
    assert "str(target_app or \"\").strip() or holder" in bearings[at : at + 200]


def test_the_page_is_only_taken_when_the_run_is_about_one():
    bearings = inspect.getsource(screen_pursuit._take_the_run_its_bearings)
    assert 'if not anchor["page"] and about_a_page:' in bearings


def test_the_pursuit_tells_it_which_application_it_is_driving():
    at = SOURCE.index("await _take_the_run_its_bearings(")
    assert "target_app=target_app" in SOURCE[at : at + 300]
