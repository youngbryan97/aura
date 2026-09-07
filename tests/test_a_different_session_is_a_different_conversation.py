"""Two sessions in one process are two conversations.

Working memory is one list per process, and the boundary of "this
conversation" was a time gap and a process boot. Two sessions minutes apart in
one run therefore read as one conversation.

LIVE 2026-09-07: in a fresh session, "what was the very first thing I said to
you in this conversation?" answered "my favourite colour is teal, remember
that" — accurately, and from a different session entirely. The transcript
grounding block that fed the model was reading every session's turns and
labelling them "WHAT WAS ACTUALLY SAID IN THIS CONVERSATION".

The boot rule this joins was written for the same reason and is quoted in the
source: turns from before this process started belong to a different
conversation by the plainest reading of the word.
"""

from __future__ import annotations

import time

from core.conversation.grounded_recall import _within_current_conversation, _user_turns
from core.conversation.session_scope import conversation_session_scope


def _turn(content: str, session: str = "", *, ago: float = 0.0) -> dict:
    # `origin` because role alone does not say a person typed it: several
    # writers append role="user" for the runtime talking to itself.
    entry = {
        "role": "user",
        "content": content,
        "timestamp": time.time() - ago,
        "origin": "user",
    }
    if session:
        entry["session_id"] = session
    return entry


def test_another_session_is_not_this_conversation() -> None:
    history = [
        _turn("my favourite colour is teal", "session-a", ago=60),
        _turn("what did I say first?", "session-b", ago=1),
    ]
    with conversation_session_scope("session-b"):
        kept = [entry["content"] for entry in _within_current_conversation(history)]
    assert "my favourite colour is teal" not in kept
    assert "what did I say first?" in kept


def test_this_session_is_kept_whole() -> None:
    history = [
        _turn("first thing", "session-b", ago=60),
        _turn("second thing", "session-b", ago=1),
    ]
    with conversation_session_scope("session-b"):
        kept = [entry["content"] for entry in _within_current_conversation(history)]
    assert kept == ["first thing", "second thing"]


def test_an_unstamped_entry_keeps_the_behaviour_it_had() -> None:
    """Only where both sides are stamped. Older entries must not vanish."""
    history = [
        _turn("older unstamped turn", ago=60),
        _turn("current turn", "session-b", ago=1),
    ]
    with conversation_session_scope("session-b"):
        kept = [entry["content"] for entry in _within_current_conversation(history)]
    assert "older unstamped turn" in kept


def test_no_session_in_scope_changes_nothing() -> None:
    history = [
        _turn("one", "session-a", ago=60),
        _turn("two", "session-b", ago=1),
    ]
    kept = [entry["content"] for entry in _within_current_conversation(history)]
    assert kept == ["one", "two"]


def test_the_recall_reader_sees_the_same_boundary() -> None:
    history = [
        _turn("teal is my favourite", "session-a", ago=60),
        _turn("hello there", "session-b", ago=2),
    ]
    with conversation_session_scope("session-b"):
        turns = _user_turns("", history=history)
    assert turns == ["hello there"]
