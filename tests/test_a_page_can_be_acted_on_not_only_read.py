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


class TestTheThreeLinksInTheChain:
    """One request, three classifiers, and it had to pass all of them.

    Fixing the planner alone changed nothing live, because the turn never
    reached the planner: the contract had already called it a search, and the
    desktop-objective test had already called it not-an-action. A capability is
    reachable only when every gate between the sentence and the skill agrees.
    """

    ACT = (
        "ok. go take it for real: https://www.16personalities.com/free-personality-test "
        "- work through the whole thing, answer every question as yourself"
    )
    READ = "read https://example.com/article and summarise it"

    def test_the_classifier_separates_working_from_reading(self):
        from core.conversation.page_interaction import asks_to_act_on_a_page

        assert asks_to_act_on_a_page(self.ACT) is True
        assert asks_to_act_on_a_page(self.READ) is False

    def test_a_page_to_work_is_not_a_search(self):
        """`has_url` alone used to force a search, which cannot serve it."""
        from interface.routes.chat import _resolve_chat_response_contract

        assert getattr(_resolve_chat_response_contract(self.ACT), "requires_search", None) is False

    def test_reading_a_page_is_still_a_search(self):
        from interface.routes.chat import _resolve_chat_response_contract

        assert getattr(_resolve_chat_response_contract(self.READ), "requires_search", None) is True

    def test_a_page_to_work_reaches_the_execution_lane(self):
        from core.runtime.desktop_objective_intent import looks_like_desktop_objective

        assert looks_like_desktop_objective(self.ACT) is True
        assert looks_like_desktop_objective(self.READ) is False

    def test_ordinary_conversation_is_untouched(self):
        from core.runtime.desktop_objective_intent import looks_like_desktop_objective

        assert looks_like_desktop_objective("how are you feeling?") is False


class TestTheDesktopLaneDelegatesPagesToTheBrowser:
    """The GUI body cannot work a web page, and stopped pretending to.

    This lane acts through coordinates, apps and keystrokes; its whole action
    vocabulary is about a screen. Asked to work THROUGH a page, the step
    deriver returned `read_screen_text`, nothing executed, and the turn fell
    back to generation — which answered "The website you provided does not
    exist, and the URL is invalid" about a page it had never opened, and
    offered to simulate the test from memory instead.
    """

    def test_a_gui_objective_stays_with_the_gui_lane(self):
        import asyncio

        from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill

        result = asyncio.run(
            DesktopTaskSkill()._delegate_page_objective(
                DesktopTaskParams(objective="open the notes app and write a note"), {}
            )
        )
        assert result is None

    def test_a_page_objective_is_handed_to_the_browser_whole(self, monkeypatch):
        """The objective crosses intact, through the governed executor.

        Calling the skill object directly reached the browser on nobody's
        authority and the will refused it: "denied_by_default: network_call
        requires validated scoped authority". The grant is what makes the
        lease, the receipt and the origin check mean anything.
        """
        import asyncio

        from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill

        seen = {}

        class _Engine:
            async def execute(self, skill, params, context=None):
                seen.update({"skill": skill, **params})
                return {
                    "ok": True,
                    "completed": True,
                    "final_url": params["url"],
                    "result_text": "Your personality type is Architect (INTJ-A)",
                    "steps": [
                        {
                            "asked": "You regularly make new friends.",
                            "chose": ["I disagree"],
                            "why": "that is truer of me",
                            "ok": True,
                        }
                    ],
                }

        import core.container as container

        monkeypatch.setattr(
            container.ServiceContainer, "get",
            staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
        )
        objective = "take the test at https://example.com/quiz and tell me the result"
        result = asyncio.run(
            DesktopTaskSkill()._delegate_page_objective(DesktopTaskParams(objective=objective), {})
        )

        assert seen["skill"] == "sovereign_browser", "must go through the governed executor"
        assert seen["mode"] == "pursue"
        assert seen["url"] == "https://example.com/quiz"
        assert seen["goal"] == objective, "the goal must cross whole, not as a query"
        assert result["ok"] is True
        assert "Architect" in result["result_text"]
        # The names this lane's task-level contract checks. Returning them
        # under any other name fails the objective with "expectation
        # incomplete: steps_requested; steps_completed".
        assert "steps_requested" in result and "steps_completed" in result

    def test_her_narration_survives_as_the_record(self, monkeypatch):
        """A step count is what the machine did; the answer is what she chose."""
        import asyncio

        from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill

        class _Engine:
            async def execute(self, skill, params, context=None):
                return {
                    "ok": True,
                    "completed": True,
                    "final_url": params["url"],
                    "result_text": "done",
                    "steps": [
                        {"asked": "You regularly make new friends.", "chose": ["I disagree"], "why": "solitude suits me", "ok": True},
                        {"asked": "", "chose": [], "why": "advancing", "ok": True},
                    ],
                }

        import core.container as container

        monkeypatch.setattr(
            container.ServiceContainer, "get",
            staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
        )
        result = asyncio.run(
            DesktopTaskSkill()._delegate_page_objective(
                DesktopTaskParams(objective="complete https://example.com/quiz"), {}
            )
        )
        assert result["narration"] == [
            {"asked": "You regularly make new friends.", "chose": ["I disagree"], "why": "solitude suits me"}
        ]
