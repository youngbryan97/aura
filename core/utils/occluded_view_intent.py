"""Is this person asking about what her own window is COVERING?

One definition, because two layers need the answer.

The conversational floor answers it from the window layout — naming each
window, how much of it is visible, and saying plainly that it cannot read
what is covered. The desktop-objective router uses it to decline: a screen
CAPTURE reads what is visible, so sending "ignore your own window, what else
is on the screen?" down the capture lane returns a raw OCR dump of whichever
window happened to be readable, when the question was about the arrangement.

Kept in core/utils deliberately. core/runtime may not import cognition, and a
second copy of this predicate over there is how the two answers drift apart —
the same lesson as core/utils/own_source_intent.py.
"""
from __future__ import annotations

import re
from typing import Any

#: Asking what is behind, under, or covered by something…
# What is behind, under or covered by her or a window, said of her or the window. A word of place on its own is not
# the question: LIVE 2026-10-20 "find a random video from Allen Pan, under 15 minutes ... tell me what you think it'll
# be about" was taken for it, and the answer before the video and every remark during it became a list of windows.
OCCLUDED_VIEW_RE = re.compile(
    r"\b(?:(?:behind|underneath|under|beneath|beside|covered\s+by|hidden\s+(?:by|behind)|obscured\s+by)"
    r"\s+(?:\w+\s+){0,2}?(?:you|yours|it|window|windows|app|ui|interface)"
    # …or asking her to look PAST her own window, which is the same question.
    r"|(?:ignor\w*|exclud\w*|apart\s+from|aside\s+from|other\s+than|besides|"
    r"without|not\s+counting|leaving\s+out|except)\s+"
    r"(?:your|the|her|its|that)?\s*(?:own\s+)?(?:window|app|ui|interface|self)"
    r"|\bwhat\s+else\s+(?:\S+\s+){0,3}?(?:screen|display|desktop|monitor)"
    r")\b",
    re.IGNORECASE,
)

#: …about the screen, and not about something else entirely.
SCREEN_SUBJECT_RE = re.compile(
    r"\b(?:screen|window|display|desktop|monitor|you|your\s+window|it)\b",
    re.IGNORECASE,
)

SEEING_VERB_RE = re.compile(
    r"\b(?:see|seeing|look|looking|view|show|read|tell\s+me\s+what)\b",
    re.IGNORECASE,
)


def asks_about_occluded_view(user_message: Any) -> bool:
    """True when the question is about what is behind or beside her window."""

    raw = str(user_message or "")
    if not raw.strip():
        return False
    if not OCCLUDED_VIEW_RE.search(raw):
        return False
    if not (SEEING_VERB_RE.search(raw) or "what" in raw.lower()):
        return False
    return bool(SCREEN_SUBJECT_RE.search(raw))


__all__ = [
    "OCCLUDED_VIEW_RE",
    "SCREEN_SUBJECT_RE",
    "SEEING_VERB_RE",
    "asks_about_occluded_view",
]
