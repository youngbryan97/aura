"""A person who gets no answer is a thing the health system hears about.

LIVE, 2026-09-21: the same question refused three times with
`canonical_chat_no_reply` while the runtime reported itself HEALTHY
throughout. `_mark_conversation_lane_state` builds the dict that goes back
in the response — it writes `state`, `conversation_ready` and
`last_failure_reason` into a fresh copy and returns it. Nothing durable
sees it, so nothing counted these turns and nothing escalated on a run of
them.

The receipt was already emitted; what was missing was the degradation, and
with it the error budget, which measures distinct degradation classes per
hour and is the thing that notices a bad hour.
"""

from __future__ import annotations

import inspect

from interface.routes import chat_refusals


def test_the_empty_canonical_reply_records_a_degradation():
    source = inspect.getsource(chat_refusals._refuse_an_empty_canonical_reply)
    assert "record_degradation(" in source, (
        "a turn that ends with no answer must reach the degradation sink"
    )
    assert "chat.canonical_reply" in source


def test_the_empty_benchmark_reply_records_one_too():
    source = inspect.getsource(chat_refusals._refuse_an_empty_benchmark_reply)
    assert "record_degradation(" in source
    assert "chat.canonical_reply" in source


def test_the_record_names_what_was_asked():
    """A count of refusals with no question in it cannot be acted on."""
    source = inspect.getsource(chat_refusals._refuse_an_empty_canonical_reply)
    start = source.index("record_degradation(")
    call = source[start : start + 600]
    assert "_semantic_user_message" in call, (
        "the degradation must carry the question, not just the fact"
    )
    assert "severity=\"warning\"" in call
    assert "action=" in call


def test_marking_the_lane_is_not_itself_a_record():
    """The thing that looked like a record and was not.

    If this ever starts writing somewhere durable, the two can be merged;
    until then the refusal sites own the recording.
    """
    from interface.routes import chat_lane_state

    source = inspect.getsource(chat_lane_state._mark_conversation_lane_state)
    assert "record_degradation" not in source
    assert "return lane" in source
