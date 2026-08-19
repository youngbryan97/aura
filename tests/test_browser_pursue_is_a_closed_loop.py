"""`interact` runs a script; `pursue` runs a loop.

A scripted action list presumes every selector is known before the first click.
That is an open loop, and it cannot carry a flow whose next screen depends on
the answer given to the last one — which is most of the web, and all of any
multi-page form.

`pursue` closes it: observe, decide, act, observe again. It adds perception and
choice and takes no authority of its own — execution is delegated to
`_handle_interact` unchanged, so the lease, the ActionExecutor receipt, the
origin check and the effect verification apply exactly as they do to a scripted
interaction.

Nothing in the loop knows what kind of page it is looking at. A loop that
recognises page types is a collection of special cases wearing a general name.
"""

from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill



def _page(checked=False, url="https://example.com/q1"):
    return {
        "url": url,
        "title": "Test",
        "text": "Question 1 of 60: You regularly make new friends.",
        "elements": [
            {"role": "radio", "name": "I agree", "selector": "#a", "checked": checked, "value": "2"},
            {"role": "radio", "name": "I disagree", "selector": "#b", "checked": False, "value": "-2"},
            {"role": "button", "name": "Next", "selector": "#next"},
        ],
    }


class _Browser:
    """Enough browser to drive the loop, and a record of what it was told."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.observed = 0

    async def observe(self, **_kwargs):
        page = self._pages[min(self.observed, len(self._pages) - 1)]
        self.observed += 1
        return page


def _skill_with(decisions, executed):
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def _decide(goal, observation, history, understanding=None):
        return decisions.pop(0) if decisions else {"done": True, "actions": []}

    async def _understand(goal, observation, prior, mind, recalled=""):
        return {"here": "a test page", "done_when": "the stub says so"}

    async def _mind():
        return ""

    async def _interact(browser, url, actions, *, action_context=None):
        executed.append(list(actions))
        return {"ok": True}

    skill._decide_next_actions = _decide
    skill._understand_page = _understand
    skill._assembled_mind = _mind
    skill._handle_interact = _interact
    return skill


class TestDecisionParsing:
    def test_json_inside_a_fence_is_found(self):
        parsed = SovereignBrowserSkill._parse_decision(
            'Sure.\n```json\n{"actions": [{"index": 1, "type": "click"}], "done": false}\n```'
        )
        assert parsed["actions"] == [{"index": 1, "type": "click"}]

    def test_json_after_a_preamble_is_found(self):
        parsed = SovereignBrowserSkill._parse_decision(
            'I think this one. {"actions": [{"index": 0, "type": "click"}], "done": false}'
        )
        assert parsed["actions"][0]["index"] == 0

    @pytest.mark.parametrize("raw", ["", "no json at all", "{not json}"])
    def test_unusable_output_is_an_error_not_a_crash(self, raw):
        assert "error" in SovereignBrowserSkill._parse_decision(raw)

    def test_a_missing_actions_list_is_normalised(self):
        assert SovereignBrowserSkill._parse_decision('{"done": true}')["actions"] == []


class TestProgress:
    def test_the_signature_changes_when_a_control_is_selected(self):
        before = SovereignBrowserSkill._observation_signature(_page(checked=False))
        after = SovereignBrowserSkill._observation_signature(_page(checked=True))
        assert before != after

    def test_the_signature_changes_when_the_page_does(self):
        a = SovereignBrowserSkill._observation_signature(_page(url="https://example.com/q1"))
        b = SovereignBrowserSkill._observation_signature(_page(url="https://example.com/q2"))
        assert a != b

    @pytest.mark.asyncio
    async def test_a_wall_stops_the_loop(self):
        """An unchanging page twice over is a wall, not a slow render."""
        executed: list = []
        skill = _skill_with([{"actions": [{"index": 0, "type": "click"}]}] * 10, executed)
        result = await skill._handle_pursue(_Browser([_page()]), None, "answer it", 10)
        assert result["rounds"] <= SovereignBrowserSkill.PURSUE_STALL_LIMIT + 1
        assert any(step.get("error") == "no_progress" for step in result["steps"])


@pytest.mark.asyncio
class TestTheLoopTakesNoAuthorityOfItsOwn:
    async def test_execution_is_delegated_with_resolved_selectors(self):
        executed: list = []
        skill = _skill_with([{"actions": [{"index": 2, "type": "click"}], "why": "move on"}], executed)
        await skill._handle_pursue(_Browser([_page(), _page(url="https://example.com/q2")]), None, "go", 2)
        assert executed, "the loop must not execute anything itself"
        assert executed[0][0].selector == "#next"

    async def test_an_index_off_the_page_is_refused(self):
        """The model names a slot; the loop still checks it exists."""
        executed: list = []
        skill = _skill_with([{"actions": [{"index": 99, "type": "click"}], "why": "?"}], executed)
        result = await skill._handle_pursue(_Browser([_page()]), None, "go", 3)
        assert not executed
        assert result["steps"][-1]["error"] == "no_executable_action"

    async def test_narration_is_kept_for_every_round(self):
        executed: list = []
        skill = _skill_with(
            [{"actions": [{"index": 1, "type": "click"}], "why": "that one is more like me"}],
            executed,
        )
        result = await skill._handle_pursue(
            _Browser([_page(), _page(url="https://example.com/q2")]), None, "go", 1
        )
        assert result["steps"][0]["why"] == "that one is more like me"
        assert result["steps"][0]["chose"] == ["I disagree"]


class TestObservationRendering:
    def test_the_question_and_the_controls_both_reach_the_decision(self):
        rendered = SovereignBrowserSkill._render_observation(_page())
        assert "You regularly make new friends" in rendered
        assert "[0] radio" in rendered and "I agree" in rendered
        assert "[2] button" in rendered

    def test_selected_state_is_visible_so_it_is_not_re_clicked(self):
        rendered = SovereignBrowserSkill._render_observation(_page(checked=True))
        assert "already selected" in rendered


class TestPursueIsFirstClassInTheTransaction:
    """A new mode that no surrounding machinery knows about can never succeed.

    `_verify_browser_effect` branched on `browse` and `interact` and fell to
    `effect_verified = False` for anything else, so every pursuit — however
    well it went — was unverifiable. `_execution_timeout` budgeted it like a
    single interaction, which would kill a working run partway through a form
    and report a timeout rather than a refusal. And the loop returned
    `final_url` while the verifier reads `observed_url`, so a completed pursuit
    presented no evidence it had been anywhere.
    """

    def test_a_real_pursuit_verifies(self):
        verdict = SovereignBrowserSkill._verify_browser_effect(
            {
                "result": {"ok": True, "rounds": 7, "observed_url": "https://example.com/q"},
                "params": {"mode": "pursue"},
            }
        )
        assert verdict["effect_verified"] is True

    def test_a_pursuit_that_did_nothing_does_not_verify(self):
        verdict = SovereignBrowserSkill._verify_browser_effect(
            {"result": {"ok": False, "rounds": 0, "observed_url": ""}, "params": {"mode": "pursue"}}
        )
        assert verdict["effect_verified"] is False

    def test_the_declared_action_check_is_not_applied_to_a_pursuit(self):
        """A pursuit declares no actions in advance; that is the point of it."""
        verdict = SovereignBrowserSkill._verify_browser_effect(
            {
                "result": {"ok": True, "rounds": 3, "observed_url": "https://example.com/q"},
                "params": {"mode": "pursue", "actions": None},
            }
        )
        assert verdict["effect_verified"] is True

    def test_a_pursuit_gets_more_time_than_one_interaction(self):
        skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
        assert skill._execution_timeout("pursue") > skill._execution_timeout("interact")


class TestTheRequestSizesItsOwnBudget:
    """A working pursuit was cancelled and called "Operation took too long".

    The engine asks any skill that can size its own budget and keeps the
    declared number otherwise. The hook exists because a flat per-skill timeout
    "cannot describe 'make a folder' and 'read three articles and write a
    synthesis' at once" — desktop_task's 180s sat inside its own measured
    spread until a successful 93.5s run was cancelled and reported as
    "Completed 0/0 steps".

    A search and a sixty-question pursuit are that same pair, and the browser
    was not answering, so a run with the page open and the loop turning was
    killed at the flat budget.
    """

    def test_a_pursuit_asks_for_much_more_than_a_search(self):
        skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
        assert skill.timeout_for({"mode": "pursue"}) > 5 * skill.timeout_for({"mode": "search"})

    def test_the_outer_budget_exceeds_the_inner_wait(self):
        """The inner timeout should fire first; it can say which round it was on."""
        skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
        for mode in ("search", "browse", "interact", "pursue"):
            assert skill.timeout_for({"mode": mode}) > skill._execution_timeout(mode)

    def test_it_reads_an_object_as_well_as_a_mapping(self):
        from types import SimpleNamespace

        skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
        assert skill.timeout_for(SimpleNamespace(mode="pursue")) == skill.timeout_for(
            {"mode": "pursue"}
        )


class TestSheActsFromAnUnderstandingNotAStepCount:
    """A step-picker asks "which control advances the goal" from nothing, forever.

    That is not how anyone uses a website. A person arrives with an aim, works
    out what the place IS — a sixty-item survey, six to a screen, a seven-point
    scale, a Next button — and acts fluently from that, revising only when the
    page does something unexpected. Without it the loop has no answer to "why
    this control and not that one", no notion of which controls are merely
    present, and no way to know it is finished except a budget.
    """

    def test_the_understanding_reaches_the_decision(self):
        rendered = SovereignBrowserSkill._render_understanding(
            {
                "here": "a 60-item personality questionnaire",
                "to_progress": "answer every item on screen, then press Next",
                "relevant": "the seven agree/disagree radios",
                "present_but_not_needed": "the login and language links",
                "done_when": "a personality type is shown",
            }
        )
        assert "60-item personality questionnaire" in rendered
        assert "then press Next" in rendered
        assert "login and language links" in rendered
        assert "a personality type is shown" in rendered

    def test_nothing_understood_renders_nothing(self):
        assert SovereignBrowserSkill._render_understanding(None) == ""
        assert SovereignBrowserSkill._render_understanding({}) == ""

    @pytest.mark.asyncio
    async def test_it_is_revised_on_surprise_not_rebuilt_each_round(self):
        """Re-deriving the page every round is a rebuild wearing another name."""
        executed: list = []
        skill = _skill_with(
            [
                {"actions": [{"index": 2, "type": "click"}], "why": "advance", "expect": "next screen"},
                {"actions": [{"index": 2, "type": "click"}], "why": "advance", "expect": "next screen"},
            ],
            executed,
        )
        calls: list = []

        async def _counting_understand(goal, observation, prior, mind, recalled=""):
            calls.append(prior)
            return {"here": "a page"}

        skill._understand_page = _counting_understand
        pages = [_page(), _page(url="https://example.com/q2"), _page(url="https://example.com/q3")]
        await skill._handle_pursue(_Browser(pages), None, "go", 2)
        assert len(calls) == 1, "the page moved as expected, so nothing needed re-deriving"

    @pytest.mark.asyncio
    async def test_a_violated_expectation_is_recorded_on_the_round(self):
        """Noticing the page did not do what she said is the useful signal."""
        executed: list = []
        skill = _skill_with(
            [{"actions": [{"index": 0, "type": "click"}], "why": "answer", "expect": "it selects"}],
            executed,
        )
        result = await skill._handle_pursue(_Browser([_page()]), None, "go", 1)
        assert result["steps"][0]["expected"] == "it selects"
        assert result["steps"][0]["moved"] is False


class TestWhatSheLearnsTransfersAndPersists:
    """Knowing "this site is a questionnaire" helps exactly once.

    Knowing "a page with repeated radio groups and something that advances is a
    multi-page form: answer what is visible, then advance" helps on every
    survey, application and signup wizard she meets afterwards. So what is
    written is structural and carries no site text — the moment it does, it
    stops transferring.

    The world model persists across restarts, so a pursuit begins by asking
    what she worked out last time. Written knowledge that is never read back is
    a diary, not learning.
    """

    def test_pages_of_the_same_kind_share_a_shape(self):
        survey_a = {"elements": [{"role": "radio", "name": "I agree"}] * 14 + [{"role": "button", "name": "Next"}]}
        survey_b = {"elements": [{"role": "radio", "name": "Strongly disagree"}] * 21 + [{"role": "button", "name": "Continue"}]}
        assert SovereignBrowserSkill._page_shape(survey_a) == SovereignBrowserSkill._page_shape(survey_b)

    def test_different_kinds_of_page_do_not(self):
        survey = {"elements": [{"role": "radio", "name": "x"}] * 14 + [{"role": "button", "name": "Next"}]}
        login = {"elements": [{"role": "text", "name": "Email"}, {"role": "button", "name": "Sign in"}]}
        assert SovereignBrowserSkill._page_shape(survey) != SovereignBrowserSkill._page_shape(login)

    def test_the_shape_carries_no_site_text(self):
        """A fingerprint containing the site stops transferring immediately."""
        shape = SovereignBrowserSkill._page_shape(
            {
                "url": "https://16personalities.com/free-personality-test",
                "title": "16Personalities",
                "text": "Question 1 of 60: You regularly make new friends.",
                "elements": [{"role": "radio", "name": "I agree"}] * 14,
            }
        )
        for leak in ("16personalities", "friends", "Question"):
            assert leak.lower() not in shape.lower()

    def test_something_that_advances_is_part_of_the_shape(self):
        with_next = {"elements": [{"role": "radio", "name": "x"}] * 8 + [{"role": "button", "name": "Next"}]}
        without = {"elements": [{"role": "radio", "name": "x"}] * 8 + [{"role": "button", "name": "Help"}]}
        assert "advances" in SovereignBrowserSkill._page_shape(with_next)
        assert "advances" not in SovereignBrowserSkill._page_shape(without)

    def test_recall_is_quiet_when_she_knows_nothing(self):
        assert SovereignBrowserSkill._recall_about("https://example.com", "radio:many") == ""

    def test_recall_finds_both_the_place_and_the_kind(self, monkeypatch):
        from types import SimpleNamespace

        import core.container as container

        beliefs = {
            "a": SimpleNamespace(claim="example.com is a survey", tags=["web", "page_model"]),
            "b": SimpleNamespace(
                claim="a page shaped radio:many|advances is a multi-page form",
                tags=["web", "page_model", "radio:many|advances"],
            ),
            "c": SimpleNamespace(claim="unrelated fact", tags=["other"]),
        }
        monkeypatch.setattr(
            container.ServiceContainer,
            "get",
            staticmethod(
                lambda name, default=None: SimpleNamespace(beliefs=beliefs)
                if name == "world_model"
                else default
            ),
        )
        recalled = SovereignBrowserSkill._recall_about(
            "https://example.com/x", "radio:many|advances"
        )
        assert "example.com is a survey" in recalled
        assert "multi-page form" in recalled
        assert "unrelated fact" not in recalled


class TestAPartialBatchIsProgress:
    """One round, several answers, the page advanced — reported as 0/1 steps.

    `interact` verifies all-or-nothing, which is right for a scripted sequence:
    you declared five actions and five must happen. A pursuit is not that. It
    answers what is on screen, and a live form re-renders the moment the last
    visible item is answered — so selectors chosen a second ago stop resolving
    and the round is marked `browser_interaction_incomplete` for having worked.
    """

    @pytest.mark.asyncio
    async def test_a_round_where_some_actions_landed_continues(self):
        executed: list = []
        skill = _skill_with(
            [
                {"actions": [{"index": 0, "type": "click"}], "why": "answer", "expect": "advances"},
                {"done": True, "actions": []},
            ],
            executed,
        )

        async def _partial(browser, url, actions, *, action_context=None):
            return {
                "ok": False,
                "error": "browser_interaction_incomplete",
                "action_report": [{"action": "click", "ok": True}, {"action": "click", "ok": False}],
            }

        skill._handle_interact = _partial
        result = await skill._handle_pursue(
            _Browser([_page(), _page(url="https://example.com/q2")]), None, "go", 2
        )
        assert result["steps"][0].get("error") is None
        assert result["steps"][0]["landed"] == 1

    @pytest.mark.asyncio
    async def test_a_round_where_nothing_landed_still_stops(self):
        executed: list = []
        skill = _skill_with(
            [{"actions": [{"index": 0, "type": "click"}], "why": "answer", "expect": "advances"}],
            executed,
        )

        async def _nothing(browser, url, actions, *, action_context=None):
            return {
                "ok": False,
                "error": "browser_interaction_incomplete",
                "action_report": [{"action": "click", "ok": False}],
            }

        skill._handle_interact = _nothing
        result = await skill._handle_pursue(_Browser([_page()]), None, "go", 2)
        assert result["steps"][-1]["error"] == "browser_interaction_incomplete"


class TestDoneBeforeActingIsNotDone:
    """Asked to work a sixty-item questionnaire, she declared it complete on the
    first look, having answered nothing — one round, zero actions, reported as
    a success.

    A goal that requires acting on a page cannot be finished before a single
    action has landed, and accepting the claim makes the loop a very expensive
    way to open a URL. Once something HAS landed she is trusted: she can see
    the result page and knows what finished looks like better than any rule.
    """

    @pytest.mark.asyncio
    async def test_an_immediate_done_does_not_complete_the_task(self):
        executed: list = []
        skill = _skill_with([{"done": True, "actions": [], "why": "looks finished"}], executed)
        result = await skill._handle_pursue(_Browser([_page()]), None, "answer it", 3)
        assert result["completed"] is False
        assert not executed

    @pytest.mark.asyncio
    async def test_she_is_asked_to_look_again_rather_than_stopped(self):
        executed: list = []
        skill = _skill_with(
            [
                {"done": True, "actions": [], "why": "looks finished"},
                {"actions": [{"index": 0, "type": "click"}], "why": "actually, answer this", "expect": "selects"},
            ],
            executed,
        )
        await skill._handle_pursue(
            _Browser([_page(), _page(url="https://example.com/q2"), _page(url="https://example.com/q3")]),
            None,
            "answer it",
            3,
        )
        assert executed, "the second look should have produced an action"

    @pytest.mark.asyncio
    async def test_done_after_work_is_accepted(self):
        executed: list = []
        skill = _skill_with(
            [
                {"actions": [{"index": 0, "type": "click"}], "why": "answer", "expect": "selects"},
                {"done": True, "actions": [], "why": "the result page is showing"},
            ],
            executed,
        )
        result = await skill._handle_pursue(
            _Browser([_page(), _page(url="https://example.com/done"), _page(url="https://example.com/done")]),
            None,
            "answer it",
            3,
        )
        assert result["completed"] is True
