"""What she says before she starts on a page, and what she makes of it at the end.

LIVE 26 Sep, asked: "Before you start, tell me what type you think it will give
you and why ... When you get your result, tell me whether it matches what you
predicted." Three faults stood between that and an answer:

- her reply, written before any page opened, was kept only as a body the task
  might type, so the person never saw it;
- the page loop was never told what she had said, so at the result she had
  nothing to hold it against;
- the round where she judged the goal met, reading the finished page, was
  recorded and left out of the reply.
"""

from __future__ import annotations

import asyncio

import pytest

from core.skills import sovereign_browser_understanding as understanding
from core.skills.sovereign_browser import SovereignBrowserSkill
from tests.answers_thought_through import answered_and_thought

SAID = "I expect INTJ: I plan, and I would rather think alone."


def _decision_prompt(monkeypatch, page, said_before=SAID, about_her=None) -> str:
    seen: list[str] = []

    class Router:
        async def think(self, prompt, **_kw):
            seen.append(prompt)
            return '{"actions": [{"index": 0, "type": "click"}], "why": "on", "done": false}'

    monkeypatch.setattr(understanding, "optional_service", lambda name, default=None: Router())
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def mind():
        return "her mind"

    skill._assembled_mind = mind
    asyncio.run(
        skill._decide_next_actions(
            "take the test", page, [], None, said_before=said_before, about_her=about_her
        )
    )
    return seen[0]


def test_a_page_of_mechanics_is_decided_with_what_she_said_in_view(monkeypatch):
    results = {"url": "u", "title": "t", "text": "Your type: INTJ", "elements": [{"role": "button", "name": "Retake", "selector": "#r"}]}
    assert SAID in _decision_prompt(monkeypatch, results)


def test_a_question_about_her_is_never_answered_with_her_forecast_in_view(monkeypatch):
    item = {
        "url": "u",
        "title": "t",
        "text": "",
        "elements": [
            {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "selector": f"#q{q}v{v}",
             "asks": "quiet [1] [2] [3] [4] [5] talkative"}
            for q in range(2)
            for v in range(1, 6)
        ],
    }
    # A decision about her is one the caller marks as such; the page's shape
    # no longer makes the whole-page decision one.
    assert SAID not in _decision_prompt(monkeypatch, item, about_her=True)


def _delegated(monkeypatch, steps, context):
    from core.skills.desktop_task import DesktopTaskParams, DesktopTaskSkill

    sent: list[dict] = []

    class _Engine:
        async def execute(self, skill, params, context=None):
            sent.append({"params": dict(params), "context": dict(context or {})})
            return {"ok": True, "completed": True, "final_url": params["url"], "result_text": "Your type: INTJ", "steps": steps}

    import core.container as container

    monkeypatch.setattr(
        container.ServiceContainer, "get",
        staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
    )
    result = asyncio.run(
        DesktopTaskSkill()._delegate_page_objective(
            DesktopTaskParams(objective="take the test at https://example.com/quiz"), context
        )
    )
    return result, sent


def test_the_page_loop_is_told_what_she_said_before(monkeypatch):
    _result, sent = _delegated(monkeypatch, [], {"cognitive_reply": SAID})
    assert sent[0]["context"]["cognitive_reply"] == SAID


def test_what_she_said_is_not_read_as_what_the_action_touches(monkeypatch):
    """LIVE 27 Sep 03:51: her reply named her "strongest drive", the permission
    model read "drive" in the arguments as Google Drive, and blocked the test."""
    said = SAID + " My values overrode my strongest drive 75 times in 500."
    _result, sent = _delegated(monkeypatch, [], {"cognitive_reply": said})
    assert "drive" not in str(sent[0]["params"]).lower()

    from core.capabilities.permission_model import PermissionRiskModel

    decision = PermissionRiskModel().check_permission(
        "sovereign_browser", str(sent[0]["params"]), {}, effect_scope="external_io", execution_risk="medium"
    )
    assert "cloud_write" not in str(getattr(decision, "reason", ""))


def test_the_browser_takes_her_words_from_the_context(monkeypatch):
    seen: dict[str, str] = {}

    async def pursue(self, browser, url, goal, max_steps, *, action_context=None, said_before=""):
        seen["said_before"] = said_before
        return {"ok": True, "steps": []}

    async def create(self, preference, *, visible=False):
        return object()

    async def close(self, browser):
        return None

    monkeypatch.setattr(SovereignBrowserSkill, "_handle_pursue", pursue)
    monkeypatch.setattr(SovereignBrowserSkill, "_create_browser", create)
    monkeypatch.setattr(SovereignBrowserSkill, "_safe_close", close, raising=False)
    from core.skills.sovereign_browser import BrowserInput

    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    params = BrowserInput(mode="pursue", url="https://example.org", goal="take the test")
    asyncio.run(skill._execute_browser(params, action_context={"cognitive_reply": SAID}))
    assert seen["said_before"] == SAID


def test_what_she_concluded_at_the_finished_page_is_kept(monkeypatch):
    steps = [
        {"asked": "You plan ahead.", "chose": ["5"], "why": "I do", "ok": True, "landed": True},
        {"why": "It says INTJ, as I said it would; I think it is right about the planning.", "done": True},
    ]
    result, _sent = _delegated(monkeypatch, steps, {})
    assert result["concluded"].startswith("It says INTJ, as I said it would")

    from interface.routes.chat_desktop_objective import _pursuit_account

    account = _pursuit_account(result)
    # Her judgement comes FIRST, and this used to require it last.
    #
    # The account is what she did and what she made of it, and the verdict is
    # the part that was asked for — the working is what supports it. Put last it
    # was the part that got cut: LIVE 2026-09-29, she read her result, held it
    # against what she had predicted, said so out loud, and the reply ended
    # mid-sentence sixteen items earlier with nothing about the outcome in it.
    assert account[0] == result["concluded"], "what she was asked for comes first"
    # And, since 1 Oct, alone: each answer reached the person as she made it,
    # and the rounds and the page's tail listed under her verdict were the
    # part of the reply Bryan called "a little ugly when they come in".
    assert account == [result["concluded"]]


def test_her_reply_is_said_before_a_page_is_worked_and_not_before_other_work(monkeypatch):
    from core.agency.narrator import Narrator
    from interface.routes.chat_desktop_objective import _say_before_working_a_page

    said: list[str] = []
    monkeypatch.setattr(Narrator, "say_everywhere", staticmethod(lambda line, *_a, **_k: said.append(line)))
    _say_before_working_a_page("Take the personality test on openpsychometrics.org", SAID)
    assert said == [SAID]
    said.clear()
    _say_before_working_a_page("open Notes and write a paragraph about yourself", SAID)
    assert said == [], "a reply the task may type as its body is not said first"


def test_a_pursuit_is_given_the_time_her_decisions_take(monkeypatch):
    """Forty rounds at a flat forty-five seconds gave a thirty-two-item test
    half an hour, at a decode rate that needed longer."""
    from core.brain.llm import thinking_reserve

    monkeypatch.setattr(thinking_reserve, "reserve_tokens", lambda model="": 0)
    monkeypatch.setattr(thinking_reserve, "seconds_to_decode", lambda tokens, model="", typical=False: 0.0)
    unmeasured = SovereignBrowserSkill.timeout_for({"mode": "pursue"})
    monkeypatch.setattr(thinking_reserve, "seconds_to_decode", lambda tokens, model="", typical=False: tokens / 7.0)
    measured = SovereignBrowserSkill.timeout_for({"mode": "pursue"})
    # The widest round is one decision about the page or a screen of reasons,
    # whichever is longer; a reason says what was measured and pays no
    # private-channel reserve (1 Oct: eight decisions with the reserve each
    # sized one round at 26,681 seconds).
    widest = max(
        SovereignBrowserSkill.DECISION_MAX_TOKENS,
        SovereignBrowserSkill.REASON_MAX_TOKENS * SovereignBrowserSkill.PURSUE_PARALLEL_ITEMS,
    ) / 7.0
    assert measured - unmeasured == pytest.approx(SovereignBrowserSkill.PURSUE_DEFAULT_STEPS * widest)
    assert SovereignBrowserSkill.timeout_for({"mode": "search"}) < unmeasured


def test_page_work_is_given_what_the_browser_says_it_costs():
    """The desktop task's flat 180 seconds would have cut the test off three minutes in."""
    from core.skills.desktop_task import DesktopTaskSkill

    page = DesktopTaskSkill.timeout_for({"objective": "Take the personality test on openpsychometrics.org"})
    assert page >= SovereignBrowserSkill.timeout_for({"mode": "pursue"})
    assert DesktopTaskSkill.timeout_for({"objective": "open Notes and write hello"}) == DesktopTaskSkill.timeout_seconds


def test_each_question_decided_is_reported_as_progress(monkeypatch):
    from core.runtime.still_getting_somewhere import a_place_to_report_it, when_it_last_got_somewhere

    class Router:
        async def think(self, prompt, **_kw):
            return '{"actions": [{"index": 0, "type": "click"}], "why": "so", "done": false}'

    monkeypatch.setattr(understanding, "optional_service", lambda name, default=None: Router())
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def mind():
        return "her mind"

    skill._assembled_mind = mind
    page = {
        "url": "u", "title": "t", "text": "",
        "elements": [
            {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "selector": f"#q{q}v{v}",
             "asks": "quiet [1] [2] [3] talkative"}
            for q in range(3)
            for v in range(1, 4)
        ],
    }
    from core.self import where_i_stand

    # Every item measured first, then each one thought about as it is answered.
    monkeypatch.setattr(where_i_stand, "themes_among", lambda names: [list(range(len(names)))])
    heard: list[str] = []
    asyncio.run(answered_and_thought(skill, "take the test", page, [], None, on_progress=heard.append))
    assert heard == ["a question measured"] * 3 + ["a question thought about"] * 3

    async def in_a_run():
        slot = a_place_to_report_it()
        from core.runtime.still_getting_somewhere import it_got_somewhere

        it_got_somewhere("a question decided")
        return when_it_last_got_somewhere(slot)

    assert asyncio.run(in_a_run()) < 1.0
