"""Her reading of a page she has just reached is said, not kept to herself.

LIVE 2026-10-03 00:46: "Going to https://openpsychometrics.org" appeared, her
model read the page for fifty seconds and decided for fifty more, nothing was
said in between, and the person closed the app one second before her first
click.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def test_what_she_makes_of_a_page_is_her_reading_of_it():
    from core.skills.sovereign_browser_what_it_did import what_she_makes_of_the_page

    said = what_she_makes_of_the_page(
        {
            "here": "The front page of a site that lists personality tests.",
            "to_progress": "Find the Open Extended Jungian Type Scales and open it",
            "done_when": "not said here",
        }
    )
    assert said == (
        "The front page of a site that lists personality tests. "
        "Find the Open Extended Jungian Type Scales and open it."
    )
    assert what_she_makes_of_the_page({}) == ""


def test_a_new_understanding_is_said_out_loud(monkeypatch):
    from core.skills import sovereign_browser as sb

    said: list[tuple[str, dict[str, Any]]] = []

    class _Her:
        async def _understand_page(self, *_a: Any) -> dict[str, str]:
            return {"here": "A results page", "to_progress": "Read the type"}

        def _recall_about(self, *_a: Any) -> str:
            return ""

        def _remember_the_place(self, *_a: Any) -> None:
            return None

        @staticmethod
        def _controls_worth_offering(elements: list, _goal: str) -> list:
            return elements

        @staticmethod
        def _say_out_loud(line: str, parts: dict[str, Any]) -> None:
            said.append((line, parts))

    asyncio.run(
        sb._understand_the_page_again(
            goal="take the test", mind="", observation={"url": "https://a.example/r"},
            self=_Her(), shape="", surprised=False, understanding=None,
        )
    )
    assert said == [
        ("A results page. Read the type.", {"label": "What I see", "said": "A results page. Read the type."})
    ]


def test_a_page_already_understood_is_not_said_again():
    from core.skills import sovereign_browser as sb

    class _Her:
        async def _understand_page(self, *_a: Any) -> dict[str, str]:
            raise AssertionError("not read again")

        @staticmethod
        def _say_out_loud(*_a: Any) -> None:
            raise AssertionError("not said again")

    surprised, kept = asyncio.run(
        sb._understand_the_page_again(
            goal="g", mind="", observation={}, self=_Her(), shape="",
            surprised=False, understanding={"here": "known"},
        )
    )
    assert kept == {"here": "known"} and surprised is False


def test_a_place_in_her_list_is_given_as_the_control_it_names():
    """LIVE 2026-10-03 03:31: "Click the 'Open Jungian Type Scales' link (control [0])"."""
    from core.skills.sovereign_browser_what_it_did import in_words_the_watcher_can_follow

    offered = [{"name": "Open Jungian Type Scales"}, {"name": "Next"}]
    assert in_words_the_watcher_can_follow(
        "Click the 'Open Jungian Type Scales' link (control [0]) to open the test.", offered
    ) == "Click the 'Open Jungian Type Scales' link to open the test."
    assert in_words_the_watcher_can_follow("Then press [1].", offered) == 'Then press "Next".'
    assert in_words_the_watcher_can_follow("Then press it (index [1]).", offered) == (
        'Then press it ("Next").'
    )
    # A place that names nothing she was shown is left as she said it.
    assert in_words_the_watcher_can_follow("press [7]", offered) == "press [7]"
    assert in_words_the_watcher_can_follow("nothing listed here", offered) == "nothing listed here"


def test_a_reading_that_failed_is_not_said_again():
    """LIVE 2026-10-03 04:02: the questionnaire's reading failed, the intro
    page's came back, and it was said as "What I see" with its "Start"
    named by the questionnaire's controls."""
    from core.skills import sovereign_browser as sb

    said: list[str] = []
    before = {"here": "The intro page", "to_progress": "Press Start"}

    class _Her:
        async def _understand_page(self, _goal: Any, _obs: Any, prior: Any, *_a: Any) -> dict:
            return dict(prior or {})

        def _recall_about(self, *_a: Any) -> str:
            return ""

        def _remember_the_place(self, *_a: Any) -> None:
            return None

        @staticmethod
        def _say_out_loud(line: str, _parts: Any) -> None:
            said.append(line)

    surprised, kept = asyncio.run(
        sb._understand_the_page_again(
            goal="g", mind="", observation={"url": "https://a.example/1"}, self=_Her(),
            shape="", surprised=True, understanding=before,
        )
    )
    assert kept == before and said == []
