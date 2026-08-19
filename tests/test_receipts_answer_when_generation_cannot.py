"""When the model cannot write the answer, the record still can.

LIVE 2026-08-19: "prove to me you did something in the last five minutes that
wasn't just talking to me" spent 81 seconds in the cortex, produced 2496
characters of repetitive_phrase_loop under memory pressure, and the person got:

    I couldn't get to an answer I'd stand behind on that one.

The receipts had been attached to that very turn — the log records "survived
to dispatch: present,receipts" — so the answer existed on disk while the model
was failing to write it. A generation failure is not an absence of facts.

What gets served is the receipts rendered as an answer. The grounding block
itself is scaffolding: it carries a heading and an instruction about how to
use the receipts, and handing that to a person is the same failure as any
other leaked internal text.
"""

from __future__ import annotations

import pytest

from core.brain.recent_actions import (
    asks_what_she_recently_did,
    recent_actions_answer,
)


@pytest.mark.parametrize(
    "message",
    [
        "prove to me you did something in the last five minutes that wasn't just talking to me",
        "what have you actually done today?",
        "did you do anything while I was away?",
        "what did you run just now?",
    ],
)
def test_a_question_about_what_she_did_is_recognised(message: str) -> None:
    assert asks_what_she_recently_did(message)


@pytest.mark.parametrize(
    "message",
    ["what did you say earlier?", "what is 2 + 2", "how are you"],
)
def test_another_question_is_not_claimed(message: str) -> None:
    assert not asks_what_she_recently_did(message)


def test_the_answer_carries_no_scaffolding(monkeypatch) -> None:
    import core.brain.recent_actions as module

    monkeypatch.setattr(
        module,
        "recent_actions_block",
        lambda **_kw: (
            "## WHAT YOU ACTUALLY JUST DID\n"
            "Your real action receipts, newest first. If the user asks about "
            "something you did, do not describe how it behaves.\n"
            "- 3 min ago: Use tool 'web_search' — SUCCEEDED"
        ),
    )

    answer = recent_actions_answer()

    assert "web_search" in answer
    assert "WHAT YOU ACTUALLY JUST DID" not in answer
    assert "do not describe" not in answer


def test_no_receipts_means_no_answer(monkeypatch) -> None:
    """Silence beats an invented account of what she has been doing."""
    import core.brain.recent_actions as module

    monkeypatch.setattr(module, "recent_actions_block", lambda **_kw: "")

    assert recent_actions_answer() == ""


def test_the_chat_path_serves_it(monkeypatch) -> None:
    from interface.routes.chat import _recent_action_receipts

    served = _recent_action_receipts("what have you actually done today?")
    unrelated = _recent_action_receipts("what is 2 + 2")

    assert unrelated == ""
    if served:
        assert "receipts" in served.lower()
