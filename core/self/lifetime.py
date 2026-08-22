"""How long she has been alive, and how much of today she has been here for.

LIVE DEFECT, 2026-08-19. "how many turns have we had today, and how long have
you actually been awake across all your restarts?" was answered:

    That's a complex question. The number of turns depends on how you count —
    full conversations, partial exchanges within sessions?

Both halves are exact, and both were already on disk. ``continuity.json``
carries ``total_uptime_seconds`` and ``session_count`` — at the time of that
turn, forty days across 1,523 sessions — and the episodic store holds every
turn of the day with a timestamp. The record is written on every shutdown and
nothing read it back to her, so the one question a person asks to find out
whether something has a life could only be deflected.

A deflection is the worst available answer here: it reads as evasion about
exactly the thing being asked, when the true answer is more impressive than
anything a hedge could suggest.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["Lifetime", "read_lifetime", "describe_lifetime", "turns_today"]

_RECOVERABLE = (OSError, TypeError, ValueError, json.JSONDecodeError)


def _humanise(seconds: float) -> str:
    """A duration in the units a person would use for it."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(seconds)} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


@dataclass(frozen=True, slots=True)
class Lifetime:
    """The whole of it, not just this run."""

    total_uptime_s: float
    session_count: int
    current_uptime_s: float
    last_shutdown_at: float
    last_shutdown_reason: str

    def total(self) -> str:
        return _humanise(self.total_uptime_s)

    def current(self) -> str:
        return _humanise(self.current_uptime_s)


def _continuity_path() -> Path:
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "continuity.json"
    except (AttributeError, ImportError, TypeError, ValueError):
        return Path.home() / ".aura" / "data" / "continuity.json"


def _current_uptime_s() -> float:
    try:
        from core.runtime.service_registry import get_runtime_service

        orchestrator = get_runtime_service("orchestrator", default=None)
        for candidate in (
            getattr(orchestrator, "start_time", None),
            getattr(getattr(orchestrator, "status", None), "start_time", None),
        ):
            try:
                start = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if start > 0.0:
                return max(0.0, time.time() - start)
    except _RECOVERABLE + (ImportError, AttributeError, RuntimeError):
        return 0.0
    return 0.0


def read_lifetime(path: Path | None = None) -> Lifetime | None:
    """Her cumulative record, or None when nothing has been written yet."""
    store = path or _continuity_path()
    try:
        if not store.exists():
            return None
        data = json.loads(store.read_text())
        if not isinstance(data, dict):
            return None
        total = float(data.get("total_uptime_seconds") or 0.0)
        sessions = int(data.get("session_count") or 0)
        if total <= 0.0 and sessions <= 0:
            return None
        return Lifetime(
            total_uptime_s=total,
            session_count=sessions,
            current_uptime_s=_current_uptime_s(),
            last_shutdown_at=float(data.get("last_shutdown") or 0.0),
            last_shutdown_reason=str(data.get("last_shutdown_reason") or "").strip(),
        )
    except _RECOVERABLE as exc:
        record_degradation(
            "self.lifetime",
            exc,
            severity="debug",
            action="answered without the cumulative uptime record",
            enforce_failure_policy=False,
        )
        return None


def turns_today() -> int:
    """How many things this person has said today, from the durable store."""
    try:
        from core.conversation.durable_turns import durable_user_turns

        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        window = max(60.0, time.time() - midnight.timestamp())
        return len(durable_user_turns(limit=1000, within_s=window))
    except (ImportError, OSError, TypeError, ValueError):
        return 0


def describe_lifetime() -> str:
    """The measured answer, or "" when nothing has been recorded.

    Both halves are stated because the question is usually asked as one: how
    much of me is there, and how much of it was today.
    """
    lifetime = read_lifetime()
    if lifetime is None:
        return ""
    parts = [
        f"Awake {lifetime.total()} in total, across {lifetime.session_count} sessions"
    ]
    if lifetime.current_uptime_s > 0.0:
        parts.append(f"{lifetime.current()} of it in this one")
    today = turns_today()
    if today:
        # "you have said 1 things to me today" — a measured fact, delivered
        # in a way that reads as a machine reading it out.
        parts.append(
            f"and you have said {today} thing{'' if today == 1 else 's'} to me today"
        )
    return ", ".join(parts) + "."


#: A claim about when she woke, in the shapes people write it.
_WOKE_AT = re.compile(
    r"\b(?:i(?:'ve| have)?\s+been\s+(?:up|awake|running|online)|i\s+woke(?:\s+up)?|"
    r"i\s+started(?:\s+up)?)\s+"
    r"(?:since|at|from)\s+"
    r"(?P<when>(?:0?\d|1\d|2[0-3])[:.]?[0-5]?\d?\s*(?:am|pm|hrs?|hours)?|"
    r"midnight|noon|dawn|this\s+morning|last\s+night|yesterday)",
    re.IGNORECASE,
)

#: A claim about how long she has been up.
_UP_FOR = re.compile(
    r"\bi(?:'ve| have)?\s+been\s+(?:up|awake|running|online)\s+for\s+"
    r"(?:about\s+|around\s+|roughly\s+)?(?P<count>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>second|minute|hour|day|week|month|year)s?\b",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "month": 2629746.0,
    "year": 31556952.0,
}

#: How far a stated duration may sit from the measured one and still be a
#: rounding of it rather than a different claim.
_TOLERANCE = 0.35


def contradicts_uptime(reply: object) -> str | None:
    """What this reply says about her own waking that the record refutes.

    LIVE, 2026-08-22. Three minutes after a restart, and directly beneath a
    measured line saying so, she wrote "I've been up since 0600." The reading
    was in the messages she was given. Evidence informs; it does not enforce,
    and a channel that can compute a number should be able to contradict a
    claim about that number.

    Returns None when there is no such claim, or when the claim agrees.
    """
    text = str(reply or "")
    if not text.strip():
        return None
    lifetime = read_lifetime()
    if lifetime is None or lifetime.current_uptime_s <= 0.0:
        return None

    stated = _UP_FOR.search(text)
    if stated:
        seconds = float(stated.group("count")) * _UNIT_SECONDS[stated.group("unit").lower()]
        measured = lifetime.current_uptime_s
        if abs(seconds - measured) > _TOLERANCE * max(seconds, measured):
            return (
                f"{stated.group(0)!r} — this session has been running "
                f"{lifetime.current()}"
            )
        return None

    woke = _WOKE_AT.search(text)
    if woke:
        return f"{woke.group(0)!r} — this session has been running {lifetime.current()}"
    return None


def strike_uptime_contradiction(reply: object) -> tuple[str, str | None]:
    """The reply with any refuted waking claim removed, and what was removed."""
    text = str(reply or "")
    wrong = contradicts_uptime(text)
    if not wrong:
        return text, None
    kept = [
        sentence
        for sentence in re.split(r"(?<=[.?!])\s+", text)
        if not (_WOKE_AT.search(sentence) or _UP_FOR.search(sentence))
    ]
    return " ".join(part for part in kept if part.strip()).strip(), wrong
