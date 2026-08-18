"""A restart ends a conversation, whatever the clock says.

LIVE 2026-08-17: "what was the first thing I said to you in this conversation?"
answered "opening-marker-zulu" — the opening turn of a session two restarts
earlier. The boots were minutes apart, so the 45-minute silence rule saw one
unbroken conversation spanning three separate runs of the process.

She was not there in between. Turns from before this process started belong to
a different conversation by the plainest reading of the word, and answering
from them misattributes what the person said in a session that has ended.
"""

from __future__ import annotations

from core.conversation.grounded_recall import (
    _CONVERSATION_GAP_S,
    _process_start_time,
    _within_current_conversation,
)


def test_turns_from_before_this_process_are_excluded() -> None:
    boot = _process_start_time()
    assert boot is not None

    kept = _within_current_conversation(
        [
            {"content": "from a previous run", "timestamp": boot - 600},
            {"content": "this run first", "timestamp": boot + 1},
            {"content": "this run second", "timestamp": boot + 2},
        ]
    )

    assert [entry["content"] for entry in kept] == ["this run first", "this run second"]


def test_a_short_gap_within_one_run_stays_one_conversation() -> None:
    """The restart rule must not fragment a live session."""
    boot = _process_start_time()

    kept = _within_current_conversation(
        [
            {"content": "a", "timestamp": boot + 1},
            {"content": "b", "timestamp": boot + 30},
            {"content": "c", "timestamp": boot + 60},
        ]
    )

    assert len(kept) == 3


def test_a_long_silence_within_one_run_still_splits() -> None:
    """The time-gap rule is kept, not replaced."""
    boot = _process_start_time()

    kept = _within_current_conversation(
        [
            {"content": "old", "timestamp": boot + 1},
            {"content": "new", "timestamp": boot + 2 + _CONVERSATION_GAP_S},
        ]
    )

    assert [entry["content"] for entry in kept] == ["new"]


def test_history_with_no_timestamps_is_returned_whole() -> None:
    """An unknown boundary must not silently discard history."""
    entries = [{"content": "a"}, {"content": "b"}]

    assert _within_current_conversation(entries) == entries


def test_empty_history_is_safe() -> None:
    assert _within_current_conversation([]) == []
    assert _within_current_conversation(None) == []
