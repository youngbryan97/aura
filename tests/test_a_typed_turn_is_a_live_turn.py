"""A typed turn is a live turn from its reservation, so her background work stands aside before it reaches her model.

LIVE 20 November 2026, 12:18: a typed turn took the foreground reservation at 12:18:54 and prepared until 12:19:10 with
nobody counted as waiting for her model, because a typed turn was counted only once it waited at the gate. Her goal
pass's generation running then went on to its end, the pass began a 5,270-token generation at 12:19:10, and the turn
asked for it to stop at 12:19:17; it stopped at 12:19:29, mid-prefill. A voice turn was already counted from its first
word.
"""
from __future__ import annotations

import ast
import inspect

import pytest

pytestmark = pytest.mark.unit


def test_a_live_typed_turn_is_a_person_waiting_and_stops_counting_when_it_ends(monkeypatch):
    import core.brain.llm_health_router_gate_owners as owners

    asked_to_stop = []
    monkeypatch.setattr(owners, "_background_stands_aside", asked_to_stop.append)
    assert owners.a_person_is_waiting_for_her_model() is False
    owners.a_live_turn("typed:abc", True)
    try:
        assert owners.a_person_is_waiting_for_her_model() is True
        assert asked_to_stop == ["a live turn began: typed:abc"], "what holds her model is asked to stop at once"
    finally:
        owners.a_live_turn("typed:abc", False)
    assert owners.a_person_is_waiting_for_her_model() is False


def _turn_source() -> str:
    from interface.routes import chat

    return inspect.getsource(chat._api_chat_turn)


def test_the_chat_turn_is_live_from_its_reservation_and_not_after_its_release():
    source = _turn_source()
    began = source.index("a_live_turn(typed_turn_key, True)")
    assert source.index('logger.info("Foreground chat reservation acquired")') < began
    assert began < source.index("memory_admission_response = await _foreground_memory_admission_response(")
    tree = ast.parse(inspect.cleandoc("\n" + source) if source.startswith(" ") else source)
    finals = [node for node in ast.walk(tree) if isinstance(node, ast.Try) and node.finalbody]
    ended = [node for node in finals if "a_live_turn(typed_turn_key, False)" in ast.unparse(node.finalbody)]
    assert ended, "the turn stops counting in a finally, so a turn that fails or is cancelled stops too"
    assert "_foreground_chat_lock.release(foreground_lock_token)" in ast.unparse(ended[0].finalbody)
