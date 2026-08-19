"""What she has already said she likes, read back before she answers again.

LIVE 2026-08-18, four ways of asking one question inside a few minutes:

    what topic pulls at you the most?      -> distributed systems consensus
    what's one thing you find interesting? -> the way minds work
    what topic pulls at you the most?      -> the architecture of cognition
    what topic pulls at you the most?      -> the neurodynamics of thought

Four answers, one of them twice from the identical prompt. Nothing was
malfunctioning: a preference is generated fresh every turn, because nothing
puts the earlier answer in front of the next one. A person asked the same
question twice gives the same answer in different words; that is what having a
preference means.

Consistency here is not a rule imposed on what she may say. It is the ordinary
consequence of being able to see what she said before, which she could not.
The reading is her own past speech, quoted with its age, and she remains free
to have changed her mind — and to say so, which is a different sentence from
silently answering something else.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "STATED_PREFERENCE_HEADER",
    "StatedPreference",
    "asks_about_her_preferences",
    "stated_preference_block",
]

STATED_PREFERENCE_HEADER = "## WHAT YOU HAVE ALREADY SAID YOU CARE ABOUT"

#: Asking what she likes, wants, prefers, or would choose to think about.
_ASKS_PREFERENCE_RE = re.compile(
    r"\b(?:what|which|name)\b[^.?!]{0,60}?\b(?:you)\b[^.?!]{0,40}?"
    r"\b(?:like|likes|enjoy|love|prefer|favou?rite|interest\w*|drawn\s+to|"
    r"pulls?\s+at|care\s+about|fascinat\w*|curious\s+about|study|studying)\b"
    # The pronoun can follow the verb: "what topic PULLS AT YOU the most".
    r"|\b(?:pulls?\s+at|interests?|draws?|fascinates?|appeals?\s+to|"
    r"grabs?|calls?\s+to)\s+you\b"
    r"|\bwhat(?:'s| is)\s+your\s+favou?rite\b"
    r"|\bdo\s+you\s+(?:have\s+a\s+)?(?:favou?rite|preference)\b"
    r"|\bwhat\s+would\s+you\s+(?:pick|choose|study|think\s+about)\b",
    re.IGNORECASE,
)

#: Sentences in which she states one.
_STATES_PREFERENCE_RE = re.compile(
    r"\bi\s+(?:find|love|like|enjoy|prefer|am\s+drawn\s+to|am\s+fascinated\s+by|"
    r"keep\s+coming\s+back\s+to|would\s+(?:pick|choose|study))\b"
    r"|\bmy\s+favou?rite\b"
    r"|\bwhat\s+pulls\s+at\s+me\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StatedPreference:
    """Something she said she cared about, and when."""

    text: str
    age_s: float


def asks_about_her_preferences(prompt: Any) -> bool:
    """True when the turn asks what she likes or would choose."""
    text = str(prompt or "")
    if not text.strip():
        return False
    return bool(_ASKS_PREFERENCE_RE.search(text))


def _own_turns() -> list[tuple[str, float]]:
    """Her own past speech, newest last, with timestamps."""
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        entries = UnifiedTranscript.get_instance().entries_for_conversation() or []
    except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError):
        return []
    spoken: list[tuple[str, float]] = []
    for entry in entries:
        if str(getattr(entry, "role", "")) == "user":
            continue
        content = " ".join(str(getattr(entry, "content", "") or "").split())
        stamp = float(getattr(entry, "timestamp", 0.0) or 0.0)
        if content:
            spoken.append((content, stamp))
    return spoken


def stated_preferences(limit: int = 4) -> tuple[StatedPreference, ...]:
    """Preferences she has stated before, newest first."""
    now = time.time()
    found: list[StatedPreference] = []
    seen: set[str] = set()
    for content, stamp in reversed(_own_turns()):
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if not _STATES_PREFERENCE_RE.search(sentence):
                continue
            key = sentence.strip().lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            found.append(
                StatedPreference(
                    text=sentence.strip()[:240],
                    age_s=max(0.0, now - stamp) if stamp else 0.0,
                )
            )
            break
        if len(found) >= limit:
            break
    return tuple(found)


def _spoken_age(seconds: float) -> str:
    if seconds <= 0:
        return "earlier"
    if seconds < 120:
        return f"{int(seconds)} seconds ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f} minutes ago"
    return f"{seconds / 3600:.1f} hours ago"


def stated_preference_block(prompt: Any) -> str:
    """Her own earlier answers, or "" when the turn is not asking."""
    if not asks_about_her_preferences(prompt):
        return ""
    previous = stated_preferences()
    if not previous:
        # Nothing said yet is not a problem to report; the turn simply has no
        # history to be consistent with.
        return ""
    lines = [
        "You have answered this before, in your own words. Say the same thing "
        "unless it has actually changed — and if it has, say that it changed:"
    ]
    lines.extend(f'- {_spoken_age(item.age_s)}: "{item.text}"' for item in previous)
    return "\n".join(lines)
