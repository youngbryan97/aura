"""A noun in a story is not a thing to go and look at.

The heuristic tool rules match bare substrings, so "A clock strikes 6 times in
5 seconds. How many seconds will it take to strike 12 times?" ranked the
REALTIME CLOCK as a tool candidate. detect_intent already rejected it; the
ranker did not, because the exclusion guarded only one of the paths that can
add a candidate and semantic retrieval had already added it upstream.

The distinguishing property is not which noun appears — that set is unbounded,
and every new tool adds to it — but where the DATA comes from. A word problem
carries its own quantities and asks for a derived one: it needs arithmetic, not
a sensor.
"""

from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine


@pytest.fixture(scope="module")
def engine() -> CapabilityEngine:
    return CapabilityEngine()


def test_a_clock_word_problem_does_not_reach_for_the_clock(engine) -> None:
    prompt = (
        "A clock strikes 6 times in 5 seconds. How many seconds will it take "
        "to strike 12 times?"
    )

    assert "clock" not in engine._rank_tool_candidates(objective=prompt, max_tools=3)


def test_asking_the_actual_time_still_reaches_for_the_clock(engine) -> None:
    """The exclusion must not cost her the capability itself."""
    assert "clock" in engine._rank_tool_candidates(objective="what time is it?", max_tools=3)


def test_a_screen_request_still_routes_to_screen_tools(engine) -> None:
    candidates = engine._rank_tool_candidates(objective="read my screen", max_tools=3)

    assert any(c in candidates for c in ("computer_use", "desktop_task", "os_manipulation"))


def test_a_counting_question_about_the_machine_is_not_a_word_problem(engine) -> None:
    """It has a number and asks 'how many', but the data is on the disk."""
    prompt = "how many files are in my downloads folder?"

    assert engine._is_self_contained_word_problem(prompt) is False


@pytest.mark.parametrize(
    "prompt",
    [
        "A train leaves at 3 and travels 60 miles. How long until it arrives?",
        "If 4 machines make 4 widgets in 5 minutes, how many minutes for 100 widgets?",
        "A file is 3 MB and the disk holds 500 MB. How many files fit?",
    ],
)
def test_self_contained_problems_are_recognised(engine, prompt: str) -> None:
    assert engine._is_self_contained_word_problem(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    ["what time is it?", "read my screen", "open my notes", "hey", ""],
)
def test_requests_about_the_world_are_not_word_problems(engine, prompt: str) -> None:
    assert engine._is_self_contained_word_problem(prompt) is False


def test_one_number_is_not_a_problem(engine) -> None:
    """A single number can be a reference ('open tab 2'), not a given quantity."""
    assert engine._is_self_contained_word_problem("what is tab 2 showing?") is False
