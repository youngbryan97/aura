"""A program that does not exist yet shows nothing on a screen.

LIVE, 2026-08-20. "build me a small web app: a single HTML page that tracks
how long I've been sitting… tell me where you put it" was routed to the
governed desktop task lane and answered:

    "os_automation failed: OS automation refused to act because the objective
    has no complete observable acceptance contract. Completed 0/1 steps. I am
    not claiming the desktop action finished."

It never could have had one. Nothing appears on a screen when a file is
written, and build_app — which builds exactly this — was ranked first by
capability selection and never asked.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import (
    asks_to_build_software,
    looks_like_desktop_objective,
)

THE_REQUEST = (
    "build me a small web app: a single HTML page that tracks how long I've "
    "been sitting. A start button, a stop button, a running timer, and it "
    "turns amber after 30 minutes and red after 50. Keep it one self-contained "
    "file, no CDNs. Tell me where you put it."
)


def test_the_live_request_is_not_desktop_work() -> None:
    assert asks_to_build_software(THE_REQUEST) is True
    assert looks_like_desktop_objective(THE_REQUEST) is False


@pytest.mark.parametrize(
    "request_text",
    [
        "write me a python script that renames files by date",
        "make a little dashboard showing my disk usage",
        "create an html page with a countdown timer",
        "knock up a small tool that converts csv to json",
    ],
)
def test_other_ways_of_asking_for_a_program(request_text: str) -> None:
    assert asks_to_build_software(request_text) is True
    assert looks_like_desktop_objective(request_text) is False


@pytest.mark.parametrize(
    "request_text",
    [
        "open safari and go to my bank",
        "open the app and build a new project in it",
        "take a screenshot of my screen",
        "click the submit button on that form",
    ],
)
def test_driving_the_screen_is_still_desktop_work(request_text: str) -> None:
    assert asks_to_build_software(request_text) is False


@pytest.mark.parametrize("request_text", ["what is 2 + 2", "how are you?", ""])
def test_conversation_is_neither(request_text: str) -> None:
    assert asks_to_build_software(request_text) is False
    assert looks_like_desktop_objective(request_text) is False


def test_the_capability_that_should_have_been_asked_ranks_first() -> None:
    from core.intent.capability_selection import select_capabilities
    from core.skills.discovery import build_skill_catalog

    class _Meta:
        def __init__(self, declaration):
            self.description = declaration.description
            self.effect_scope = declaration.effect_scope
            self.enabled = True
            self.name = declaration.name
            self.module_path = declaration.module_path
            self.class_name = declaration.class_name
            self.skill_class = None
            self.instance = None

    skills = {d.name: _Meta(d) for d in build_skill_catalog().accepted}
    offered = select_capabilities(
        THE_REQUEST,
        skills,
        ceiling="read_write_artifacts",
        admissible_scopes=frozenset(
            {"status", "pure_compute", "read_only", "sandboxed_compute", "read_write_artifacts"}
        ),
    )
    assert offered and offered[0] == "build_app"
