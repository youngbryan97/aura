"""A turn typed into her chat reaches her model of the person who typed it.

Her chat answers typed turns without running a phase, and the conversational dynamics phase was the only thing that
told her model of a person what they did and asked. On 13 November 2026 it had learned nothing from anything Bryan
typed. The chat's durable post-reply step now hands each typed exchange over once (core/social/watching_people.py).
"""
from __future__ import annotations

import pytest

from core.consciousness import theory_of_mind
from core.conversational import dynamics
from core.social import watching_people


@pytest.fixture
def watched(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(theory_of_mind, "watch_the_turn", lambda who, **k: seen.append({"who": who, **k}))
    monkeypatch.setattr(dynamics, "_engine_instance", None)
    monkeypatch.setattr(watching_people, "_HER_LAST", {})
    return seen


def test_what_they_did_and_what_her_last_move_was_reach_her_model(watched):
    assert watching_people.watch_a_typed_exchange("bryan", "How do black holes form?", "From collapsing stars.")
    assert watching_people.watch_a_typed_exchange("bryan", "Wait, why do they collapse?", "Gravity wins.")

    first, second = watched
    assert first["who"] == "bryan" and first["their_act"] == "question" and first["her_act"] == ""
    # The second question faced her first answer, and her model is told so.
    assert second["their_act"] == "question" and second["her_act"] == "statement"


def test_nobody_named_reaches_nothing(watched):
    assert watching_people.watch_a_typed_exchange("", "hello", "Hi.") is False
    assert watched == []


def test_the_chat_hands_each_typed_exchange_over_from_its_post_reply_step():
    from pathlib import Path

    source = Path("interface/routes/chat_preflight.py").read_text(encoding="utf-8")
    step = source[source.index('if not bool(payload.get("experience_recorded")):'):]
    step = step[: step.index('_mark_chat_turn_memory_log_stage(operation_id, "experience")')]
    assert "watch_a_typed_exchange(user_id, user_message, aura_response)" in step
