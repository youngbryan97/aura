"""What she has actually been doing, from the record she keeps of it.

LIVE, 2026-08-20. "what's actually been the most interesting thing you've
worked on lately, and why that one rather than something else?" was answered
with an architectural interest — her own modelling of consciousness — which is
a true thing about her and not an answer to the question. She had a record of
four thousand intentions on disk, two thousand of them completed, each with a
drive, an outcome and a timestamp, and nothing read it back to her.

``queued_work`` already answers the forward half of this and refuses the
backward half outright: its matcher returns False on a past-tense question. So
"anything planned?" was grounded and "what have you been doing?" reached
nothing at all.

The window comes from the record rather than from a constant. Asked what she
has been doing lately, the honest span is the one the returned work actually
covers, and it is stated.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = [
    "ActivityWindow",
    "describe_recent_activity",
    "looks_like_a_question_about_recent_activity",
    "read_recent_activity",
]

_RECOVERABLE = (OSError, sqlite3.Error, TypeError, ValueError)

#: How many completed items to characterise. Enough to show a shape; the span
#: they cover is reported rather than assumed.
_SAMPLE = 60

#: Drives that are hers rather than something the person asked for.
_HER_OWN_DRIVES = (
    "research_cycle",
    "curiosity",
    "autonomous_initiative_loop",
    "background_reflection",
    "dream",
    "maintenance",
    "self_development",
)

#: "Use tool 'x'" is how an intention is written down, not what it was for.
_TOOL_INTENTION_RE = re.compile(r"^use tool ['\"]?([\w.-]+)", re.IGNORECASE)


def _readable(value: object) -> str:
    """Text worth showing a person, or "".

    A serialised result is a record of a call, not a description of work. It
    starts with a brace or a bracket, and saying it back reads as a leak.
    """
    text = " ".join(str(value or "").strip().split())
    if len(text) <= 12:
        return ""
    if text[:1] in "{[" or text.startswith(("b'", 'b"')):
        return ""
    # An address is not a sentence. "https://api.open-meteo.com/v1/forecast?…"
    # names where she looked and says nothing about what she was doing there,
    # and read back as a bullet it is the same leak as the payload above.
    from core.intent.opaque_spans import without_opaque_spans

    if not without_opaque_spans(text).strip(" .,:;-"):
        return ""
    return text[:160]


@dataclass(frozen=True, slots=True)
class ActivityWindow:
    """Completed work, and the span of time it covers."""

    completed: int = 0
    failed: int = 0
    span_seconds: float = 0.0
    tools: tuple[tuple[str, int], ...] = ()
    subjects: tuple[str, ...] = ()
    drives: tuple[tuple[str, int], ...] = field(default=())
    for_them: int = 0
    her_own: int = 0

    def is_empty(self) -> bool:
        return self.completed == 0 and self.failed == 0


def _database() -> Path | None:
    try:
        from core.config import config

        path = Path(config.paths.data_dir) / "memory" / "intention_loop.db"
    except _RECOVERABLE:
        return None
    return path if path.is_file() else None


def _humanise(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 5400:
        return f"{max(1, int(round(seconds / 60)))} minutes"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86_400:.1f} days"


def read_recent_activity(limit: int = _SAMPLE) -> ActivityWindow:
    """The most recent completed work, and what it was.

    Read-only, so a chat turn can never write to the record it is describing.
    """
    path = _database()
    if path is None:
        return ActivityWindow()
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    except _RECOVERABLE as exc:
        record_degradation(
            "self.recent_activity",
            exc,
            severity="debug",
            action="left the activity question to the model",
            enforce_failure_policy=False,
        )
        return ActivityWindow()
    try:
        rows = connection.execute(
            "SELECT intention, drive, status, completed_at, actual_outcome "
            "FROM intentions WHERE completed_at > 0 "
            "ORDER BY completed_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    except _RECOVERABLE as exc:
        record_degradation(
            "self.recent_activity",
            exc,
            severity="debug",
            action="left the activity question to the model",
            enforce_failure_policy=False,
        )
        return ActivityWindow()
    finally:
        connection.close()

    if not rows:
        return ActivityWindow()

    completed = failed = for_them = her_own = 0
    tools: Counter[str] = Counter()
    drives: Counter[str] = Counter()
    subjects: list[str] = []
    newest = oldest = float(rows[0][3] or 0.0)

    for intention, drive, status, done_at, outcome in rows:
        stamp = float(done_at or 0.0)
        newest = max(newest, stamp)
        oldest = min(oldest, stamp)
        state = str(status or "").strip().lower()
        if state == "completed":
            completed += 1
        elif state == "failed":
            failed += 1
        else:
            continue
        drive_name = str(drive or "").strip().lower()
        drives[drive_name or "unattributed"] += 1
        if drive_name in _HER_OWN_DRIVES:
            her_own += 1
        elif drive_name == "user":
            for_them += 1
        text = str(intention or "").strip()
        match = _TOOL_INTENTION_RE.match(text)
        if match:
            tools[match.group(1)] += 1
            continue
        # Anything that is not a bare tool invocation is a thing she set out
        # to do, and is worth naming as itself. The outcome is preferred when
        # it reads like a sentence; a serialised payload says less about the
        # work than the intention that produced it does.
        subject = _readable(outcome) or _readable(text)
        if subject and subject not in subjects:
            subjects.append(subject)

    return ActivityWindow(
        completed=completed,
        failed=failed,
        span_seconds=max(0.0, newest - oldest),
        tools=tuple(tools.most_common(6)),
        subjects=tuple(subjects[:6]),
        drives=tuple(drives.most_common(6)),
        for_them=for_them,
        her_own=her_own,
    )


def describe_recent_activity(window: ActivityWindow | None = None) -> str:
    """The record as sentences, or "" when there is nothing recorded."""
    reading = window if window is not None else read_recent_activity()
    if reading is None or reading.is_empty():
        return ""

    lines: list[str] = []
    span = _humanise(reading.span_seconds) if reading.span_seconds > 0 else ""
    head = f"{reading.completed} pieces of work finished"
    if span:
        head += f" in the last {span}"
    if reading.failed:
        head += f", and {reading.failed} that failed"
    lines.append(head + ".")

    attributed = reading.for_them + reading.her_own
    if attributed:
        lines.append(
            f"Of {reading.completed + reading.failed} recorded, "
            f"{reading.for_them} came from something they asked for and "
            f"{reading.her_own} were mine."
        )
    if reading.tools:
        used = ", ".join(f"{name} ({count})" for name, count in reading.tools)
        lines.append(f"What I actually ran: {used}.")
    if reading.subjects:
        lines.append("Work that was not just a tool call:")
        lines.extend(f"- {subject}" for subject in reading.subjects)
    return "\n".join(lines)


#: Asking what she has been doing, in the tenses people use for it.
_ASKS_RECENT_ACTIVITY = re.compile(
    r"\b(?:"
    r"what\s+(?:have|hav)\s+you\s+been\s+(?:doing|working|up\s+to)"
    r"|what\s+(?:did|d\s*id)\s+you\s+(?:do|work\s+on|get\s+(?:up\s+to|done))"
    r"|what\s+you\s+been\s+(?:doing|up\s+to)"
    r"|(?:you\s+)?(?:worked|been\s+working)\s+on\s+(?:lately|recently|today|this\s+week)"
    r"|been\s+busy"
    r"|(?:what|anything)\s+(?:have\s+you|you\s+have)\s+(?:done|finished|worked\s+on)"
    r"|how\s+(?:has|was)\s+your\s+(?:day|night|week)"
    r"|keeping\s+yourself\s+busy"
    r")",
    re.IGNORECASE,
)

#: A question about her, not about the person or a third party.
_ABOUT_HER = re.compile(r"\byou(?:r|rself|'ve|ve)?\b", re.IGNORECASE)

#: Asked with the subject left out. "been busy?" carries no "you" and is
#: addressed to nobody else, so requiring the pronoun lost the shortest and
#: most natural way of asking.
_ELLIPTICAL = re.compile(
    r"^\s*(?:so\s+|and\s+|well\s+)?(?:been\s+busy|keeping\s+busy|busy\s+night|"
    r"busy\s+day|much\s+done)\s*[?.!]?\s*$",
    re.IGNORECASE,
)


def looks_like_a_question_about_recent_activity(prompt: object) -> bool:
    """True when the turn asks what she has been doing.

    Deliberately narrow. "what have you been doing" is the question; "what do
    you do" is about capability and belongs to a different reading.
    """
    text = str(prompt or "")
    if not text.strip():
        return False
    if _ELLIPTICAL.match(text):
        return True
    if not _ABOUT_HER.search(text):
        return False
    return bool(_ASKS_RECENT_ACTIVITY.search(text))
