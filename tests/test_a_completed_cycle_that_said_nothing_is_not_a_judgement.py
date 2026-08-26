"""Refusing has to say what actually stopped the turn.

LIVE 2026-08-26: three identical questions refused in a row and the fourth
answered normally, while the runtime was still warming its caches and
foreground turns averaged sixteen seconds. Each refusal said:

    I couldn't put together an answer I'd stand behind for that one, and I
    won't hand you a thinner substitute and call it mine.

She had not weighed an answer and found it wanting. She ran out of time. A
person can act on "I ran out of time, ask me again" and can do nothing with a
statement about her standards.
"""
from __future__ import annotations

import inspect

from interface.routes.chat import _why_there_is_no_answer


def test_a_timeout_is_called_a_timeout():
    said = _why_there_is_no_answer({"last_failure_reason": "foreground_timeout"})
    assert "ran out of time" in said
    assert "stand behind" not in said


def test_a_mind_still_coming_up_says_so_and_names_what_it_waits_on():
    said = _why_there_is_no_answer(
        {"warmup_in_flight": True, "readiness_blockers": ["model_not_loaded"]}
    )
    assert "still loading" in said
    assert "Give me a moment" in said


def test_an_empty_cycle_says_nothing_came_back_rather_than_claiming_a_judgement():
    said = _why_there_is_no_answer({"last_failure_reason": ""})
    assert "Nothing came back" in said
    assert "stand behind" not in said


def test_a_named_reason_is_passed_through_in_plain_words():
    said = _why_there_is_no_answer({"last_failure_reason": "cortex_recovery_pending"})
    assert "cortex recovery pending" in said


def test_the_old_fixed_line_is_no_longer_served():
    """It survives only where the record of the defect is written, which is
    what those comments are for."""
    from interface.routes import chat

    source = inspect.getsource(chat)
    # Nothing assigns it, returns it, or hands it to a caller. It appears only
    # inside prose that records why it is gone.
    for line in source.splitlines():
        if "stand behind" not in line:
            continue
        stripped = line.strip()
        assert not stripped.startswith(("return", "reply", "failure_reply", "yield")), (
            f"the fixed refusal is still served: {stripped[:80]}"
        )
        assert '= "' not in stripped and "= '" not in stripped, (
            f"the fixed refusal is still assigned: {stripped[:80]}"
        )


def test_an_empty_cycle_is_tried_once_more_before_refusing():
    """Nothing was wrong with the question and nothing was wrong with her.
    One more attempt is cheaper than telling somebody their answer does not
    exist when it exists a second later."""
    from interface.routes import chat

    source = inspect.getsource(chat)
    where = source.index("canonical_empty_reply")
    block = source[where - 700 : where + 400]
    assert "_attempt_protected_foreground_reply" in block
    assert "reply_text = second_try" in block


def test_a_lane_this_turn_just_marked_failed_is_not_called_warming():
    """`conversation_ready` is False on this lane because the turn itself
    marked it failed. Reading that as "she is still coming up" told people to
    wait for something that had already finished — measured live, on a lane
    whose own health said ready with no blockers at all."""
    said = _why_there_is_no_answer(
        {
            "state": "failed",
            "conversation_ready": False,
            "readiness_blockers": [],
            "last_failure_reason": "",
        }
    )
    assert "still coming up" not in said
    assert "Nothing came back" in said


def test_a_real_warm_up_still_says_so():
    said = _why_there_is_no_answer(
        {"warmup_in_flight": True, "readiness_blockers": ["worker_not_alive"]}
    )
    assert "starting up" in said


def test_an_internal_token_never_reaches_the_person():
    """A token is not a reason. LIVE 2026-08-26: "I am still
    worker_not_alive." Anything without a plain reading is left out rather
    than shown raw — the sentence around it already says she is not ready."""
    from interface.routes.chat import _in_plain_words

    assert _in_plain_words("worker_not_alive") == "My mind is still starting up."
    assert _in_plain_words("some_internal_thing_42") == ""
    said = _why_there_is_no_answer(
        {"warmup_in_flight": True, "readiness_blockers": ["some_internal_thing_42"]}
    )
    assert "some_internal_thing_42" not in said
    assert "_" not in said
