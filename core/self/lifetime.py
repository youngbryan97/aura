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
        parts.append(f"and you have said {today} things to me today")
    return ", ".join(parts) + "."
