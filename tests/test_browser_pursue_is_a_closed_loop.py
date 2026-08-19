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

    async def _decide(goal, observation, history):
        return decisions.pop(0) if decisions else {"done": True, "actions": []}

    async def _interact(browser, url, actions, *, action_context=None):
        executed.append(list(actions))
        return {"ok": True}

    skill._decide_next_actions = _decide
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
