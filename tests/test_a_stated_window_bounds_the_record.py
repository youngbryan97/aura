"""The stretch of time a question names is a constraint like any other.

LIVE, 2026-08-27: "of everything I've thrown at you in the last hour or so,
what did you actually do well, and where were you leaning on something outside
yourself?" came back with an activity log spanning days — 2048, a sliding
puzzle, notes written to a Desktop — because the record was read by COUNT and
the window in the sentence was never read at all.

Reading it in one place means the activity record, the receipt record, and
anything else that reports history bound themselves the same way rather than
each growing its own vocabulary of "today" and "this morning".
"""

from __future__ import annotations

import pytest

from core.language.stated_window import describe_window, seconds_named

_HOUR = 3600.0


@pytest.mark.parametrize(
    ("asked", "seconds"),
    [
        ("in the last hour or so, what did you do?", _HOUR),
        ("over the past twenty minutes", 20 * 60),
        ("in the last 10 minutes", 10 * 60),
        ("what did you do today?", 24 * _HOUR),
        ("in the last couple of days", 48 * _HOUR),
        ("what have you been up to this week?", 7 * 24 * _HOUR),
    ],
)
def test_the_window_is_read(asked: str, seconds: float) -> None:
    assert seconds_named(asked) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "asked",
    [
        "what have you been up to?",
        "what did you do?",
        "tell me about the migration",
        "",
    ],
)
def test_no_window_named_is_not_a_window(asked: str) -> None:
    """None is different from a long window: a caller keeps its own default
    rather than being handed a made-up bound."""
    assert seconds_named(asked) is None


def test_the_window_reads_back_the_way_a_person_says_it() -> None:
    assert describe_window(_HOUR) == "the last hour"
    assert describe_window(20 * 60) == "the last 20 minutes"
    assert describe_window(48 * _HOUR) == "the last 2 days"
    assert describe_window(None) == ""
    assert describe_window(0) == ""


def test_the_activity_record_takes_a_bound() -> None:
    """Bounding it is the whole point; the signature has to carry one."""
    import inspect

    from core.self.recent_activity import read_recent_activity

    assert "since_seconds" in inspect.signature(read_recent_activity).parameters


def test_nothing_in_the_window_is_an_answer() -> None:
    """Bounding the record turned a wrong answer into no answer at all.

    The turn then fell through to the model, which is how a question with a
    definite answer ends up guessed at.
    """
    import inspect

    from core.introspection import self_evidence

    source = inspect.getsource(self_evidence.past_actions_answer)
    assert "Nothing in " in source
    assert "_window_named(message)" in source
