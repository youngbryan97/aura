"""A capability that nothing can route to is one that was never built.

MEASURED live 2026-08-18. "go take it for real:
https://www.16personalities.com/free-personality-test — work through the whole
thing, answer every question as yourself" was planned as `web_search`. The page
was fetched, synthesis produced nothing usable for "take the test", and the turn
ended in "I couldn't get to an answer I'd stand behind on that one."

The interaction capability existed by then. Nothing could reach it. The planner
offered exactly two readings of a URL — search FOR it, or search WITH it — so
every phrasing of "do this on that page" collapsed into "find that page".

The distinction it was missing is the one BrowserAuthority already draws one
layer down: a read needs no lease, while a click "changes state on the far side
and needs a lease". Retrieval phrasings still go to search, which is the right
tool for them. What this recovers is the case search cannot serve at all — a
page whose next screen depends on what you did to the current one.
"""

from __future__ import annotations

import pytest

from core.agency.autonomous_task_engine import AutonomousTaskEngine


class TestActingOnAPageIsRecognised:
    @pytest.mark.parametrize(
        "goal,expected",
        [
            (
                "go take it for real: https://www.16personalities.com/free-personality-test "
                "- work through the whole thing",
                "https://www.16personalities.com/free-personality-test",
            ),
            ("fill out the form at https://example.com/signup", "https://example.com/signup"),
            ("complete the checkout at https://shop.example.com/cart", "https://shop.example.com/cart"),
            ("sign up at https://example.com/join please", "https://example.com/join"),
        ],
    )
    def test_a_page_to_work_through_is_found(self, goal, expected):
        assert AutonomousTaskEngine._page_interaction_target(goal) == expected

    @pytest.mark.parametrize(
        "goal",
        [
            "read https://example.com/article and summarise it",
            "what does https://example.com/page say?",
            "search for personality tests",
            "tell me about https://example.com/docs",
            "just the url https://example.com",
        ],
    )
    def test_reading_still_belongs_to_search(self, goal):
        """The branch must not swallow retrieval, which search does better."""
        assert AutonomousTaskEngine._page_interaction_target(goal) == ""

    def test_trailing_punctuation_is_not_part_of_the_page(self):
        found = AutonomousTaskEngine._page_interaction_target(
            "complete the survey at https://example.com/survey."
        )
        assert found == "https://example.com/survey"

    def test_a_verb_without_a_page_routes_nowhere_new(self):
        assert AutonomousTaskEngine._page_interaction_target("take the test") == ""


def test_the_plan_opens_the_page_in_pursue_mode():
    """The plan has to carry the goal, not a query distilled from it.

    Once a goal becomes a search string the fact that it named a page to work
    through is gone, which is why this branch runs before both search branches.
    """
    import inspect

    source = inspect.getsource(AutonomousTaskEngine)
    interaction = source.index("_page_interaction_target(goal)")
    search_branch = source.index("# 4. Web Search / Search Web")
    assert interaction < search_branch, "acting on a page must be considered before searching"
    assert '"mode": "pursue"' in source
