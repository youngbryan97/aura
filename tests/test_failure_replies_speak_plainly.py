"""When Aura can't answer, she must say so like a person, not a subsystem.

Bryan reported this from a real chat: small talk answered with
"I could not produce a reliable full-mind reply for that turn, so I failed
closed instead of sending an ungrounded answer." The 2026-07-25 endurance
probe served the same sentence on every retention turn.

"full-mind", "failed closed", "ungrounded", "output contract" are words for
the engineering log. To the person waiting they read as a machine describing
its own internals instead of talking to them — which is most of why a runtime
that is merely busy comes across as broken.

This is a ratchet over the user-visible failure strings in the chat route: a
new one written in engineering vocabulary fails here.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
#: Where the words a person is shown on a failed turn are written. Since 18
#: November they are the runtime's notices, not sentences in her voice.
_SOURCES = (
    _ROOT / "interface" / "routes" / "chat.py",
    _ROOT / "interface" / "routes" / "chat_refusals.py",
    _ROOT / "interface" / "routes" / "chat_lane_state.py",
    _ROOT / "interface" / "routes" / "chat_no_reply.py",
    _ROOT / "core" / "conversation" / "runtime_notice.py",
)

# Vocabulary that belongs in the log, never in a sentence aimed at a person.
_JARGON = (
    "full-mind",
    "failed closed",
    "fail closed",
    "ungrounded answer",
    "output contract",
    "bounded contract",
    "response_path",
    "lane_status",
    "not proven",
    "unverified answer",
)

# A user-visible failure string is long enough to be prose and reads as an
# apology/refusal rather than a log line.
_USER_FACING_HINTS = (
    "I could not", "I couldn't", "I can't", "I was unable",
    "No reply formed", "Her model", "Her draft", "This turn", "no reply was started",
)


def _user_facing_failure_strings() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in _SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value
            if len(text) < 30:
                continue
            if not any(hint in text for hint in _USER_FACING_HINTS):
                continue
            found.append((f"{path.name}:{node.lineno}", text))
    return found


def test_no_user_facing_failure_reply_speaks_in_jargon():
    offenders: list[str] = []
    for where, text in _user_facing_failure_strings():
        lowered = text.lower()
        hits = [word for word in _JARGON if word in lowered]
        if hits:
            offenders.append(f"{where} uses {hits}: {text[:90]!r}")
    assert not offenders, (
        "these replies are shown to a person and read like a subsystem "
        "describing itself:\n  " + "\n  ".join(offenders)
    )


def test_the_detector_finds_real_prose():
    """A guard that matches nothing is not a guard."""
    strings = _user_facing_failure_strings()
    assert len(strings) >= 3, (
        "the scan should be finding the route's user-facing failure replies; "
        f"found only {len(strings)}"
    )


def test_a_failure_reply_tells_the_person_what_to_do():
    """Refusal without a next step leaves the user stuck."""
    replies = [t for _, t in _user_facing_failure_strings()]
    actionable = [
        t for t in replies
        if re.search(r"try again|ask (me )?again|in a moment|shortly|retry", t, re.I)
    ]
    assert actionable, (
        "at least the main fail-closed replies must say what the person can do next"
    )
