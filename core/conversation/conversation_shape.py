"""How long we have been talking, and what about — measured, not remembered.

LIVE 2026-08-18, asked "how long have we been talking?":

    About an hour. Time flies when we're geeking out on distributed systems
    and mycelial networks, doesn't it? What's next? More Solaris analysis?

None of that had been discussed. The conversation was minutes old and had been
about a clipboard, a file, and a prime number. The duration was invented and so
were the topics, warmly and in detail.

Nothing was wrong with the model. A question about the SHAPE of the
conversation — how long, how many, what about — reached no reading at all: the
transcript observable answers "what did I ask you first", and these are
different questions, so they arrived with nothing to answer from. Asked for a
number it was never given, a language model supplies one.

The transcript carries a timestamp on every entry, so all three facts are
arithmetic.
"""

from __future__ import annotations

import re
import time

__all__ = [
    "CONVERSATION_SHAPE_HEADER",
    "asks_about_conversation_shape",
    "conversation_shape_block",
]

CONVERSATION_SHAPE_HEADER = "## THE SHAPE OF THIS CONVERSATION"

_ASKS_SHAPE_RE = re.compile(
    r"\bhow\s+long\s+(?:have\s+)?(?:we|you\s+and\s+i)\s+(?:been\s+)?(?:talk|chat|going|at\s+it)"
    r"|\bhow\s+long\s+(?:has\s+)?(?:this|our)\s+(?:conversation|chat|session)\b"
    r"|\bhow\s+many\s+(?:messages?|questions?|things?|times?)\s+(?:have\s+)?i\b"
    r"|\bwhat\s+(?:have|did)\s+we\s+(?:been\s+)?(?:talk|talked|discuss|discussed|cover|covered)\b"
    r"|\bwhat\s+have\s+we\s+been\s+(?:talking|discussing)\s+about\b"
    r"|\bhow\s+far\s+(?:into|through)\s+(?:this|our)\s+(?:conversation|chat)\b"
    r"|\bwhat\s+(?:else\s+)?did\s+we\s+(?:cover|discuss|talk\s+about)\b",
    re.IGNORECASE,
)

#: Uptime is not conversation length, and neither is a question about how long
#: some future thing will take.
_NOT_ABOUT_THIS_CONVERSATION_RE = re.compile(
    r"\bhow\s+long\s+(?:have\s+)?you\s+been\s+(?:running|up|online|awake|alive)\b"
    r"|\bhow\s+long\s+(?:will|would|does|did)\s+(?:it|that|this)\s+take\b",
    re.IGNORECASE,
)


def asks_about_conversation_shape(prompt: str) -> bool:
    """True when the turn asks how long, how many, or what about."""
    text = str(prompt or "")
    if not text.strip():
        return False
    if _NOT_ABOUT_THIS_CONVERSATION_RE.search(text):
        return False
    return bool(_ASKS_SHAPE_RE.search(text))


def _entries() -> list:
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        return list(UnifiedTranscript.get_instance().entries_for_conversation() or [])
    except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError):
        return []


def _spoken(seconds: float) -> str:
    """The duration as a person would say it, without rounding it away."""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} minute{'s' if round(minutes) != 1 else ''}"
    hours = minutes / 60
    return f"{hours:.1f} hours"


def conversation_shape_block(prompt: str) -> str:
    """The measured shape, or "" when the turn is not asking for it."""
    if not asks_about_conversation_shape(prompt):
        return ""
    entries = _entries()
    if not entries:
        # A named absence. "I cannot see this conversation's record" is true;
        # a remembered hour is not.
        return (
            "No transcript is available for this conversation, so its length "
            "and topics cannot be read."
        )

    stamps = [
        float(getattr(entry, "timestamp", 0.0) or 0.0)
        for entry in entries
        if float(getattr(entry, "timestamp", 0.0) or 0.0) > 0
    ]
    user_turns = [
        " ".join(str(getattr(entry, "content", "") or "").split())
        for entry in entries
        if str(getattr(entry, "role", "")) == "user"
    ]
    user_turns = [turn for turn in user_turns if turn]

    lines = [f"{len(user_turns)} message(s) from the person, {len(entries)} entries total."]
    if stamps:
        elapsed = max(stamps[-1], time.time()) - min(stamps)
        lines.append(f"First entry was {_spoken(elapsed)} ago.")
    if user_turns:
        lines.append("What they actually said, earliest first:")
        # Both ends, because "what have we covered" wants the whole arc and a
        # window of the recent turns answers a different question.
        shown = user_turns if len(user_turns) <= 12 else user_turns[:6] + ["..."] + user_turns[-6:]
        lines.extend(f"- {turn[:220]}" for turn in shown)
    return "\n".join(lines)
