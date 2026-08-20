"""What she has been doing, read from her own record of intending it.

LIVE, 2026-08-20. "what's actually been the most interesting thing you've
worked on lately, and why that one rather than something else?" was answered
with her interest in modelling consciousness — true about her, and not an
answer — while four thousand recorded intentions sat on disk, two thousand of
them completed with a drive, an outcome and a timestamp.

queued_work answers the forward half and refuses the backward half outright:
its matcher returns False on a past-tense question.
"""

from __future__ import annotations

import pytest

from core.self.recent_activity import (
    ActivityWindow,
    describe_recent_activity,
    looks_like_a_question_about_recent_activity as asks,
)


@pytest.mark.parametrize(
    "prompt",
    [
        "what's actually been the most interesting thing you've worked on lately?",
        "what have you been doing?",
        "what have you been up to?",
        "what did you do today?",
        "been busy?",
        "how was your night?",
        "what have you been working on lately?",
        "what did you get done while I was away?",
        "keeping yourself busy?",
    ],
)
def test_the_question(prompt: str) -> None:
    assert asks(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "what do you do?",
        "anything planned?",
        "what will you be doing next?",
        "what did I ask you earlier today?",
        "what is 2 + 2",
        "what has Bryan been working on?",
        "",
    ],
)
def test_not_the_question(prompt: str) -> None:
    assert asks(prompt) is False


def test_nothing_recorded_says_nothing() -> None:
    """Silence, not an invented summary of a day that has no record."""
    assert describe_recent_activity(ActivityWindow()) == ""


def test_the_span_comes_from_the_record() -> None:
    window = ActivityWindow(completed=4, failed=1, span_seconds=7200.0, for_them=3, her_own=2)
    described = describe_recent_activity(window)
    assert "4 pieces of work finished" in described
    assert "2.0 hours" in described
    assert "1 that failed" in described
    assert "Of 5 recorded" in described


def test_a_serialised_payload_is_never_read_back() -> None:
    """An outcome that is a dict describes a call, not the work."""
    from core.self.recent_activity import _readable

    assert _readable('{"ok": true, "url": "https://x"}') == ""
    assert _readable("[1, 2, 3]") == ""
    assert _readable("short") == ""
    assert _readable("research curiosity topic: who wrote Solaris") == (
        "research curiosity topic: who wrote Solaris"
    )


def test_the_live_record_reads_without_writing() -> None:
    """Read-only: a chat turn must never write the record it describes."""
    from core.self.recent_activity import read_recent_activity

    window = read_recent_activity(limit=5)
    assert isinstance(window, ActivityWindow)
