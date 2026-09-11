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
from typing import Any

__all__ = [
    "CONVERSATION_SHAPE_HEADER",
    "asks_about_conversation_shape",
    "asks_about_shared_history",
    "conversation_shape_block",
    "shared_history_block",
]

CONVERSATION_SHAPE_HEADER = "## THE SHAPE OF THIS CONVERSATION"

_ASKS_SHAPE_RE = re.compile(
    r"\bhow\s+long\s+(?:have\s+)?(?:we|you\s+and\s+i)\s+(?:been\s+)?(?:talk|chat|going|at\s+it)"
    r"|\bhow\s+long\s+(?:has\s+)?(?:this|our)\s+(?:conversation|chat|session)\b"
    r"|\bhow\s+many\s+(?:messages?|questions?|things?|times?)\s+(?:have\s+)?i\b"
    r"|\bwhat\s+(?:have|did)\s+we\s+(?:been\s+)?(?:talk|talked|discuss|discussed|cover|covered)\b"
    r"|\bwhat\s+have\s+we\s+been\s+(?:talking|discussing)\s+about\b"
    r"|\bhow\s+far\s+(?:into|through)\s+(?:this|our)\s+(?:conversation|chat)\b"
    r"|\bwhat\s+(?:else\s+)?did\s+we\s+(?:cover|discuss|talk\s+about)\b"
    # Asking for a summary of THIS conversation is asking what is in it.
    # LIVE 2026-08-18: "summarize this conversation in one sentence" produced
    # "we discussed the potential for distributed systems to achieve consensus,
    # and you asked me about my substrate and cognitive architecture" — a
    # conversation about a clipboard token, a file count, memory readings and
    # a contradiction. The recap was invented because the transcript never
    # reached the turn.
    r"|\b(?:summar(?:ise|ize|y)|recap|rundown|gist|tl;?dr|"
    r"catch\s+me\s+up|bring\s+me\s+up\s+to\s+speed)\b"
    r"[^.?!]{0,30}\b(?:this|our|the)\s+"
    r"(?:conversation|chat|session|discussion|exchange|thread)\b"
    r"|\b(?:summar(?:ise|ize)|recap)\b\s+(?:what\s+)?we\s+"
    r"(?:covered|discussed|talked\s+about|said)\b",
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


def _entries() -> list[Any] | None:
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        return list(UnifiedTranscript.get_instance().entries_for_conversation())
    except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError):
        return None


def _shared_entries() -> list[Any] | None:
    try:
        from core.conversation.turn_evidence_custody import (
            current_turn_evidence_custody,
            turn_transcript,
        )

        # The foreground owner already restored and scoped durable dialogue.
        # Do not replace an admitted empty history with another conversation,
        # or mistake a post-restart RAM suffix for the complete record.
        if current_turn_evidence_custody() is not None:
            history = turn_transcript()
            return list(history) if history is not None else None
        return _entries()
    except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError):
        return None


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


#: "what did we agree on", "what did we decide", "remember when we talked
#: about X" — a question that presupposes something already happened between
#: them.
_ASKS_SHARED_HISTORY_RE = re.compile(
    r"\bwhat\s+did\s+we\s+(?:agree|decide|settle|conclude|say|discuss|cover)\b"
    r"|\bwhat\s+(?:was|were)\s+(?:our|the)\s+(?:agreement|decision|conclusion|plan)\b"
    r"|\bremember\s+when\s+we\b"
    r"|\bdid\s+we\s+(?:agree|decide|settle|discuss|talk\s+about)\b"
    r"|\bwe\s+(?:agreed|decided|settled)\s+(?:on|that)\b",
    re.IGNORECASE,
)


def asks_about_shared_history(prompt: Any) -> bool:
    """True when the turn presupposes something they already settled."""
    return bool(_ASKS_SHARED_HISTORY_RE.search(str(prompt or "")))


def _topic_words(prompt: str) -> list[str]:
    from core.language.word_forms import word_form

    common = {
        "what", "did", "we", "agree", "agreed", "on", "about", "the", "our",
        "decide", "decided", "last", "week", "yesterday", "earlier", "remember",
        "when", "was", "were", "that", "this", "to", "for", "of", "and", "a",
        "an", "in", "it", "you", "i", "me", "my", "your",
        "which", "settle", "settled", "conclude", "concluded", "say", "said",
        "discuss", "discussed", "cover", "covered", "talk", "talked",
        "agreement", "decision", "conclusion", "plan",
    }
    return [
        word_form(word)
        for word in re.findall(r"[a-z][a-z'-]{2,}", str(prompt or "").lower())
        if word not in common
    ]


_SHARED_HISTORY_MAX_EXCHANGES = 6
_SHARED_HISTORY_MAX_ENTRY_CHARS = 480


def _history_exchanges(entries: list[Any], prompt: str) -> list[list[tuple[str, str]]]:
    dialogue: list[tuple[str, str]] = []
    for entry in entries:
        read = entry.get if isinstance(entry, dict) else lambda key, default, item=entry: getattr(item, key, default)
        role = str(read("role", ""))
        if role not in {"user", "aura", "assistant"}:
            continue
        if read("channel", "") in {"system", "internal"}:
            continue
        content = " ".join(str(read("content", "") or "").split())
        if content:
            dialogue.append((role, content))
    # The pending question is often already stored. Its premise is not evidence.
    if dialogue and dialogue[-1] == ("user", " ".join(prompt.split())):
        dialogue.pop()

    exchanges: list[list[tuple[str, str]]] = []
    for role, content in dialogue:
        if role == "user" or not exchanges:
            exchanges.append([])
        exchanges[-1].append((role, content))
    return exchanges


def _history_excerpt(exchanges: list[list[tuple[str, str]]], matches: list[int]) -> list[str]:
    # Retain recent exchanges, then topic anchors with their neighbours. A
    # replacement or confirmation often names only "that one", not the topic.
    selected = set(range(max(0, len(exchanges) - 2), len(exchanges)))
    for index in reversed(matches):
        for neighbour in (index, index + 1, index - 1):
            if len(selected) < _SHARED_HISTORY_MAX_EXCHANGES and 0 <= neighbour < len(exchanges):
                selected.add(neighbour)
    for index in reversed(range(len(exchanges))):
        if len(selected) >= _SHARED_HISTORY_MAX_EXCHANGES:
            break
        selected.add(index)

    lines: list[str] = []
    previous = -1
    for index in sorted(selected):
        if index > previous + 1:
            lines.append(f"[{index - previous - 1} exchange(s) omitted]")
        lines.append(f"Exchange {index + 1}:")
        exchange = exchanges[index]
        # Bound unusual exchanges with many consecutive assistant entries too.
        shown = range(len(exchange)) if len(exchange) <= 3 else (0, len(exchange) - 2, len(exchange) - 1)
        for position in shown:
            if position == len(exchange) - 2 and len(exchange) > 3:
                lines.append(f"[{len(exchange) - 3} dialogue entries omitted]")
            role, content = exchange[position]
            if len(content) > _SHARED_HISTORY_MAX_ENTRY_CHARS:
                marker = " [... middle omitted ...] "
                keep = (_SHARED_HISTORY_MAX_ENTRY_CHARS - len(marker)) // 2
                content = content[:keep] + marker + content[-keep:]
            label = "them" if role == "user" else "you"
            lines.append(f"- {label}: {content}")
        previous = index
    return lines


def shared_history_block(prompt: Any) -> str:
    """What the transcript holds about a presupposed agreement.

    LIVE 2026-08-18: "what did we agree on last week?" was answered "we agreed
    that you would provide me with the necessary files to review your code. I
    haven't seen them yet." No such exchange existed. The guard that catches
    fabricated shared history did not fire, and there was nothing else to
    check the presupposition against.

    Topic words locate exchanges; they cannot establish an agreement or its
    absence. Keep replies and corrections even when they name no topic.
    """
    if not asks_about_shared_history(prompt):
        return ""
    entries = _shared_entries()
    if entries is None:
        return (
            "No transcript is available to this reader. Whether this was agreed "
            "is unknown; this reading establishes neither an agreement nor its absence."
        )
    exchanges = _history_exchanges(entries, str(prompt or ""))
    if not exchanges:
        return (
            "No prior dialogue is present in the available transcript. This "
            "snapshot supplies no agreement; history outside it is unknown."
        )
    topics = set(_topic_words(str(prompt or "")))
    matches = [
        index for index, exchange in enumerate(exchanges)
        if any(topics.intersection(_topic_words(content)) for _role, content in exchange)
    ]
    lines = [
        "Dialogue from the available transcript, earliest first. This reader "
        "sees a bounded snapshot, not every possible history source. Topic overlap "
        "does not itself establish agreement; missing or omitted evidence leaves "
        "the outcome unknown."
    ]
    if topics and not matches:
        lines.append(
            "No topic-word overlap was found in this snapshot. That does not "
            "establish whether an agreement exists; recent exchanges follow."
        )
    return "\n".join([*lines, *_history_excerpt(exchanges, matches)])
