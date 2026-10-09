"""Her model's answer to a choice is held to the choices on offer, so it is as long as the name of the move.

LIVE 2026-10-08 a move was asked for with room for ninety-six tokens; the move
came first and the rest explained it, about six and a half seconds of writing on
the 27B. A game wanted its move in eight seconds, and it was made without her.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.a_choice_the_decoder_enforces import (  # noqa: E402
    enforce_one_of,
    the_ways_through,
)

pytestmark = pytest.mark.unit

END = 0


class _Letters:
    """A tokenizer whose tokens are letters, so what is allowed can be read off by eye."""

    eos_token_id = END

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def _allowed(held, tokens: list[int]) -> set[int]:
    out = held(mx.array(tokens), mx.zeros((1, 200)))
    return {i for i, v in enumerate(out.tolist()[0]) if v != float("-inf")}


def test_only_a_start_of_a_choice_and_then_only_its_end():
    held = enforce_one_of(_Letters(), ["up", "down", 'click "next"'])
    prompt = [5, 6, 7]
    assert _allowed(held, prompt) == {ord("u"), ord("d"), ord("c")}
    assert _allowed(held, [*prompt, ord("u")]) == {ord("p")}
    assert _allowed(held, [*prompt, ord("u"), ord("p")]) == {END}


def test_a_choice_that_begins_another_may_end_or_go_on():
    held = enforce_one_of(_Letters(), ["go", "go on"])
    assert _allowed(held, [9]) == {ord("g")}
    assert _allowed(held, [9, ord("g"), ord("o")]) == {END, ord(" ")}


def test_held_only_after_a_private_channel_closes():
    held = enforce_one_of(_Letters(), ["up"], after_token=99)
    assert len(_allowed(held, [5])) == 200
    assert len(_allowed(held, [5, 42])) == 200
    assert _allowed(held, [5, 42, 99]) == {ord("u")}


def test_a_token_it_did_not_allow_ends_the_hold():
    held = enforce_one_of(_Letters(), ["up"])
    _allowed(held, [5])
    assert len(_allowed(held, [5, ord("x")])) == 200
    assert held.state["lost"]


def test_nothing_to_hold_to_is_said_by_returning_nothing():
    assert enforce_one_of(_Letters(), ["", "  "]) is None
    following, whole = the_ways_through(_Letters(), ["up", "up", " up  "])
    assert whole == {(ord("u"), ord("p"))} and following[()] == {ord("u")}


def _asked_through_quick_reasoning(monkeypatch, choices: tuple[str, ...]) -> dict:
    import asyncio

    from core.agency import her_reasoning
    from core.agency.deliberate_action import CHOOSING_FROM

    seen: dict = {}

    class _Router:
        async def think(self, **asked):
            seen.update(asked)
            return "up"

    monkeypatch.setattr(her_reasoning, "_router", lambda: _Router())
    token = CHOOSING_FROM.set(choices)
    try:
        assert asyncio.run(her_reasoning.quick_reasoning()("Choose the next move", ["what is on screen"])) == "up"
    finally:
        CHOOSING_FROM.reset(token)
    return seen


def test_a_choice_among_moves_is_asked_with_room_for_the_name_of_one(monkeypatch):
    seen = _asked_through_quick_reasoning(monkeypatch, ("up", 'click "next"'))
    assert seen["choose_from"] == ["up", 'click "next"'] and seen["max_tokens"] == len('click "next"') + 1


def test_outside_a_choice_the_question_is_asked_as_before(monkeypatch):
    from core.agency.her_reasoning import CHOICE_TOKENS

    seen = _asked_through_quick_reasoning(monkeypatch, ())
    assert "choose_from" not in seen and seen["max_tokens"] == CHOICE_TOKENS


def test_a_move_question_says_what_stays_the_same_first():
    """Two moves in a row share the goal and what she knows at the front, so her model reads them once."""
    from core.agency.deliberate_action import ActionOption, _objective, _situation_evidence

    known = ["Its rules: the arrows move the paddle; miss three balls and it is over."]
    first = [_objective("win it", []), *_situation_evidence("win it", "SCORE 10", [ActionOption(name="up")], [], [], known)]
    then = [_objective("win it", []), *_situation_evidence("win it", "SCORE 20", [ActionOption(name="down")], [], [], known)]
    shared = next(i for i, (a, b) in enumerate(zip(first, then, strict=True)) if a != b)
    assert first[:shared] == [_objective("win it", []), "Goal: win it", *known]
    assert first[-1] == "The available moves are: up." and first[shared].startswith("What is visible now")
