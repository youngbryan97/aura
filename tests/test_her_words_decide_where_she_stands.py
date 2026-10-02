"""Where her own sentence puts her, read by entailment, is the answer that is clicked.

LIVE 2026-10-02: "I constantly overextend myself" answered Disagree under a sentence
saying she does; "procrastinates" at 5 of 5 under "I don't procrastinate". The
record measures relatedness, and her words were written to a place given in
advance. With an entailment reader present, she thinks first with no place in
front of her, and the place is read from what she said. Without one, nothing
changes. No model is loaded here; the reader is replaced.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from core.self import how_her_words_stand as reading
from core.self.where_i_stand import Lean
from core.skills.sovereign_browser import SovereignBrowserSkill as S
from core.skills.sovereign_browser_one_question import (
    _thinking_for_one_answer,
    _where_her_words_place_her,
)

pytestmark = pytest.mark.unit


def _row(left: str = "gets work done right away", right: str = "procrastinates"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, 6)) + f" {right}"
    return [
        {"group": "Q21", "role": "radio", "name": "Q21", "value": str(n), "asks": asks,
         "selector": f"#Q21V{n}"}
        for n in range(1, 6)
    ]


def _item(options, index: int = 4, facing: float = 0.0, toward: float = 1.0):
    return {
        "group": "Q21",
        "options": options,
        "index": index,
        "count": len(options),
        "lean": Lean(toward=toward, first=0.0, second=0.0, measured=True, facing=facing),
        "first": "gets work done right away",
        "second": "procrastinates",
    }


def test_without_a_reader_nothing_is_read(monkeypatch):
    monkeypatch.setattr(reading, "_reader", lambda: None)
    assert not reading.can_read_her_words()
    assert reading.where_her_words_put_her("I do it at once.", "now", "later") is None
    assert _where_her_words_place_her(S, _item(_row()), "I do it at once.") is None


def test_a_sentence_that_denies_one_side_is_placed_on_the_other(monkeypatch):
    # Entailment less contradiction: the first side borne out, the second denied.
    monkeypatch.setattr(reading, "_how_far_it_bears_out", lambda words, hyps: [0.9, -0.85])
    toward = reading.where_her_words_put_her("I don't procrastinate.", "gets work done", "procrastinates")
    assert toward == pytest.approx(-0.875)
    assert _where_her_words_place_her(S, _item(_row()), "I don't procrastinate.") == 0


def test_a_hedged_sentence_lands_near_the_middle(monkeypatch):
    monkeypatch.setattr(reading, "_how_far_it_bears_out", lambda words, hyps: [0.2, 0.1])
    assert _where_her_words_place_her(S, _item(_row()), "Some of both.") == 2


def test_a_statement_she_affirms_goes_to_the_end_that_means_yes(monkeypatch):
    monkeypatch.setattr(reading, "_how_far_it_bears_out", lambda words, hyps: [0.8])
    monkeypatch.setattr(
        S, "_a_statement_on_a_named_scale", staticmethod(lambda options: ("I overextend myself.", "Disagree", "Agree"))
    )
    options = _row()
    assert _where_her_words_place_her(S, _item(options, index=0, facing=1.0), "I do.") == 4
    # A run printed from yes to no puts the same answer at the other end.
    assert _where_her_words_place_her(S, _item(options, index=4, facing=-1.0), "I do.") == 0


def test_the_click_follows_her_words(monkeypatch):
    monkeypatch.setattr(reading, "_how_far_it_bears_out", lambda words, hyps: [0.9, -0.85])
    options = _row()
    item = _item(options, index=4)
    resolved: dict[str, Any] = {"selector": "#Q21V5"}
    decision: dict[str, Any] = {"answered": [], "noticed": [], "resolved_actions": [resolved], "why": ""}

    async def thinking(goal, theme, mind, **_k):
        return {"Q21": "I don't procrastinate; when something matters I take it at once."}

    skill = SimpleNamespace(
        _her_thinking_about=thinking,
        _the_two_sides=S._the_two_sides,
        _a_statement_on_a_named_scale=S._a_statement_on_a_named_scale,
        _the_choice_disagrees_with_its_reason=S._the_choice_disagrees_with_its_reason,
        _an_answer_in_words=S._an_answer_in_words,
    )
    think = _thinking_for_one_answer(skill, "take it", [item], "", item, resolved, decision)
    words, parts, selector = asyncio.run(think())
    assert selector == "#Q21V1"
    assert resolved["selector"] == "#Q21V1"
    assert "1 of 5" in parts["chose"]


def test_the_move_clicks_where_her_words_put_her(monkeypatch):
    clicked: list[str] = []
    skill = S()

    async def _interact(_browser, _url, actions, **_k):
        clicked.extend(action.selector for action in actions)
        return {"ok": True, "action_report": [{"ok": True}]}

    async def _held(*_a, **_k):
        return None

    async def _said():
        return "words", {"said": "words"}, "#Q21V1"

    monkeypatch.setattr(skill, "_handle_interact", _interact)
    monkeypatch.setattr(skill, "_say_out_loud", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    from core.skills.sovereign_browser import BrowserAction

    asyncio.run(skill._make_each_move(None, [(BrowserAction(type="click", selector="#Q21V5"), _said)]))
    assert clicked == ["#Q21V1"]


def test_she_is_not_told_the_record_s_place_when_her_words_decide(monkeypatch):
    seen: dict[str, str] = {}
    monkeypatch.setattr(reading, "can_read_her_words", lambda: True)

    async def _asked(self_, prompt, mind, **_k):
        seen["prompt"] = prompt
        return '{"each": {"Q21": "I take it at once."}}', S._HER_OWN_LANE

    monkeypatch.setattr(S, "_asked_of_her", _asked)
    item = _item(_row(), index=4)
    asyncio.run(S()._her_thinking_about("take it", [item], "mind", about=item))
    assert "5 of 5" not in seen["prompt"]
    assert "procrastinates" in seen["prompt"]
