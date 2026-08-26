"""Asked what she just did, she has to answer with what she just did.

LIVE 2026-08-26: minutes after playing a game to a 256 tile, "what did you
just do?" was answered with a wallpaper she had set in an earlier session,
read out as raw receipt fields.

Two faults in one answer. The sort meant to put the newest first read a field
the receipt does not have, so every entry sorted as zero and nothing was
ordered at all — and the comment above that sort records fixing exactly this
once before. And what came back was a log line where a person had asked a
question.
"""
from __future__ import annotations

import inspect

from core.introspection.self_evidence import _said_plainly, resolve_past_actions


def test_the_sort_reads_a_field_the_receipt_has():
    source = inspect.getsource(resolve_past_actions)
    assert '"at": getattr(row, "created_at", None)' in source
    assert '"at": getattr(row, "timestamp", None),' not in source


def test_an_action_is_read_out_the_way_a_person_would_say_it():
    said = _said_plainly(
        {
            "action": "pursue_on_screen",
            "evidence": "pursuit_reached_goal;moves=25;outcome=goal_reached",
            "cause": "Find 2048 online, play it, and get to a 256 tile.",
        }
    )
    assert "got there" in said
    assert "25 moves" in said
    assert "You had asked: Find 2048 online" in said
    # None of the filing.
    assert "outcome=" not in said
    assert "pursuit_reached_goal" not in said


def test_an_outcome_with_no_plain_word_is_said_as_it_is_recorded():
    """Nothing is invented. An outcome nobody has a plain word for is passed
    through rather than dressed up."""
    said = _said_plainly({"action": "open_url", "evidence": "outcome=redirected", "cause": ""})
    assert "redirected" in said


def test_a_failure_is_not_softened():
    for token, plain in (
        ("out_of_time", "ran out of time"),
        ("needs_person", "stopped and left it to you"),
        ("cannot_decide", "could not decide"),
    ):
        said = _said_plainly({"action": "pursue_on_screen", "evidence": f"outcome={token}"})
        assert plain in said


def test_the_numbers_come_from_the_receipt():
    said = _said_plainly({"action": "x", "evidence": "moves=1;outcome=goal_reached"})
    assert "1 move" in said and "1 moves" not in said


def test_work_that_never_happened_is_not_reported_as_done():
    """A goal already true when she arrived is a real outcome and a different
    one. Reading it out as "got there" claims work that never happened."""
    assert _said_plainly({"action": "pursue_on_screen", "evidence": "moves=0;outcome=goal_reached"}) == ""
    assert _said_plainly({"action": "x", "evidence": "steps=0;outcome=goal_reached"}) == ""
    assert _said_plainly({"action": "x", "evidence": "moves=3;outcome=goal_reached"}) != ""


def test_the_same_thing_twice_is_one_thing():
    """A store holding a retry and its original holds two receipts for one
    action, and reading both out claims she did it twice."""
    from core.introspection.self_evidence import EvidenceBundle, Reading, ReadingState, render_past_actions

    same = {"action": "pursue_on_screen", "evidence": "moves=4;outcome=goal_reached", "cause": "play"}
    rendered = render_past_actions(
        EvidenceBundle(
            demand="past_actions",
            readings=(Reading(
                channel="tool_receipts",
                state=ReadingState.READ,
                value=[dict(same), dict(same), dict(same)],
                unit="actions",
                provenance="test",
            ),),
        )
    )
    assert rendered.count("got there") == 1


def test_a_cause_is_shortened_at_a_word_not_through_one():
    from core.introspection.self_evidence import _whole_words

    said = _whole_words("Find 2048 online, play it, and tell me here when you have it", 30)
    assert said.endswith("…")
    assert not said.endswith("wh…")
    assert _whole_words("short", 30) == "short"
