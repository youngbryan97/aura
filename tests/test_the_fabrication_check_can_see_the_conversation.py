"""The check that decides whether a recall is invented can see what was said.

`fabricated_shared_history` catches a real failure — a reply that supplies a
shared past nothing supports — and it decides by looking for content that
appears nowhere in the conversation. Its own code says so: "NO CONTEXT MEANS
NO VERDICT... an empty vocabulary makes everything novel."

Inside the worker it had an empty vocabulary. It read the conversation from
`user_surface_recent_messages`, and nothing in the tree ever put that key in a
job — the client assembles the payload field by field and this one is not
among the fields. So the check ran against no conversation at all, and a
correct recall reads as an invention.

The cost is not one rejected draft. A rejected draft goes to the repair path,
which disables the prompt cache, so a recall question pays a full re-prefill
on top of being answered twice.

LIVE, 2026-09-07: "What did I just ask you?" was rejected as
`fabricated_shared_history` one turn after the question it was recalling.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker import _recent_user_turns
from core.dialogue.shared_history import has_fabricated_shared_history


def _job(*turns: tuple[str, str]) -> dict:
    return {
        "messages": [{"role": "system", "content": "who you are"}]
        + [{"role": role, "content": text} for role, text in turns]
    }


def test_the_worker_reads_the_conversation_off_the_transcript() -> None:
    job = _job(
        ("user", "Hey. What are you actually made of?"),
        ("assistant", "A layered architecture across three lanes."),
        ("user", "What did I just ask you?"),
    )
    assert _recent_user_turns(job) == ["Hey. What are you actually made of?"]


def test_the_turn_being_answered_is_not_counted_twice() -> None:
    """It reaches the check separately, as the prompt."""

    job = _job(("user", "only this one"))
    assert _recent_user_turns(job) == []


def test_a_caller_that_states_the_history_is_believed() -> None:
    assert _recent_user_turns({"user_surface_recent_messages": ["stated"]}) == ["stated"]


def test_a_job_with_no_transcript_says_nothing() -> None:
    """No context, no verdict — the check's own rule."""

    assert _recent_user_turns({}) == []
    assert _recent_user_turns({"messages": "not a list"}) == []


#: A recall question whose own words carry enough content for the check to
#: reach a verdict. Below that it declines to judge, which is its other
#: guard and is not the one that was failing.
_ASKED = "I'm rewriting the retry logic in the payment service this week."
_RECALL = "Remind me what I told you about my current project."


def test_a_correct_recall_stops_reading_as_an_invention() -> None:
    correct = (
        "You told me you are rewriting the retry logic in the payment service "
        "this week."
    )
    assert has_fabricated_shared_history(correct, _RECALL, [])
    assert not has_fabricated_shared_history(correct, _RECALL, [_ASKED])


def test_an_invented_past_is_still_caught_with_the_history_present() -> None:
    """Supplying the conversation must not turn the check off."""

    invented = "You told me you were moving house in March and hated the new landlord."
    assert has_fabricated_shared_history(invented, _RECALL, [_ASKED])


def test_the_two_halves_meet_at_the_worker() -> None:
    """What the worker reads off the transcript is what clears the recall."""

    job = _job(
        ("user", _ASKED),
        ("assistant", "Noted."),
        ("user", _RECALL),
    )
    correct = (
        "You told me you are rewriting the retry logic in the payment service "
        "this week."
    )
    assert not has_fabricated_shared_history(correct, _RECALL, _recent_user_turns(job))


@pytest.mark.parametrize("depth", [1, 5, 20])
def test_the_window_is_bounded(depth: int) -> None:
    from core.brain.llm.mlx_worker import _RECENT_TURNS_FOR_GROUNDING

    job = _job(*[("user", f"turn {index}") for index in range(depth)])
    assert len(_recent_user_turns(job)) <= _RECENT_TURNS_FOR_GROUNDING
