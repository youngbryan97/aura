"""Reminders that actually exist after she says they do.

LIVE 2026-08-19: "remind me in 20 minutes to check the oven" was answered "I've
set a reminder for 20 minutes to check the oven." No tool ran and no reminder
existed. The person stops thinking about the oven, which is the whole cost.

The pieces were nearly all here: an intention loop that records what was
attempted, a commitment engine for promises she makes, and a Scheduler class
holding one-shot timers in a list. What was missing is the part that makes a
reminder a reminder — it has to survive a restart, and something has to notice
when it comes due.

So this is a durable store, written through the governed write gateway, read
back by whoever asks what is queued. Nothing here notifies on its own; a
reminder is DUE and the surfaces that already report queued work say so. That
is the honest shape while there is no push channel, and it is the difference
between a record and a claim.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "Reminder",
    "RequestedReminder",
    "requested_reminder",
    "reminder_answer",
    "add_reminder",
    "due_reminders",
    "complete_reminder",
    "pending_reminders",
    "reminders_path",
]

_RECOVERABLE = (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError)


@dataclass(frozen=True, slots=True)
class Reminder:
    """One thing to be reminded about, and when."""

    id: str
    text: str
    due_at: float
    created_at: float
    completed: bool = False

    @property
    def is_due(self) -> bool:
        return not self.completed and time.time() >= self.due_at

    def seconds_remaining(self) -> float:
        return max(0.0, self.due_at - time.time())


def reminders_path() -> Path:
    """Where reminders live. Beside the rest of her durable state."""
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "reminders.json"
    except _RECOVERABLE:
        return Path.home() / ".aura" / "data" / "reminders.json"


def _load() -> list[Reminder]:
    path = reminders_path()
    try:
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8") or "[]")
    except _RECOVERABLE as exc:
        record_degradation(
            "reminders", exc, severity="warning", action="read no reminders from disk"
        )
        return []
    found: list[Reminder] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            found.append(
                Reminder(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    due_at=float(item["due_at"]),
                    created_at=float(item.get("created_at") or 0.0),
                    completed=bool(item.get("completed")),
                )
            )
        except (KeyError, TypeError, ValueError):
            # One malformed row must not lose the others.
            continue
    return found


def _save(reminders: list[Reminder]) -> bool:
    path = reminders_path()
    payload = json.dumps([asdict(item) for item in reminders], indent=2)
    try:
        # Through the gateway, not around it. Every consequential write in this
        # runtime is governed there, and a reminder store that wrote directly
        # would be exactly the kind of ungoverned side channel the gateway
        # exists to prevent — noticed by the governance ratchet the moment it
        # was added.
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="reminders")
        gateway.write_text(path, payload, encoding="utf-8", source="reminders")
        return True
    except _RECOVERABLE as exc:
        record_degradation(
            "reminders",
            exc,
            severity="warning",
            action="did not persist a reminder; refused to report it as set",
        )
        return False


def add_reminder(text: str, delay_s: float) -> Reminder | None:
    """Record a reminder, or None when it could not be stored.

    None is the point: a reminder that was not written down is not a reminder,
    and the caller must say so rather than describing one.
    """
    body = " ".join(str(text or "").split())
    if not body:
        return None
    try:
        seconds = float(delay_s)
    except (TypeError, ValueError):
        return None
    if seconds < 0 or seconds != seconds:
        return None
    now = time.time()
    reminder = Reminder(
        id=uuid.uuid4().hex[:12], text=body[:400], due_at=now + seconds, created_at=now
    )
    existing = _load()
    existing.append(reminder)
    if not _save(existing):
        return None
    return reminder


def pending_reminders() -> list[Reminder]:
    """Everything still outstanding, soonest first."""
    return sorted(
        (item for item in _load() if not item.completed), key=lambda item: item.due_at
    )


def due_reminders() -> list[Reminder]:
    """The ones whose time has come."""
    return [item for item in pending_reminders() if item.is_due]


def complete_reminder(reminder_id: str) -> bool:
    """Mark one done. False when it was not there to complete."""
    wanted = str(reminder_id or "").strip()
    if not wanted:
        return False
    existing = _load()
    found = False
    updated: list[Reminder] = []
    for item in existing:
        if item.id == wanted and not item.completed:
            updated.append(
                Reminder(
                    id=item.id,
                    text=item.text,
                    due_at=item.due_at,
                    created_at=item.created_at,
                    completed=True,
                )
            )
            found = True
        else:
            updated.append(item)
    if not found:
        return False
    return _save(updated)


def spoken_delay(seconds: float) -> str:
    """The delay as a person would say it."""
    if seconds < 90:
        return f"{int(round(seconds))} seconds"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    return f"{minutes / 60:.1f} hours"


#: "in 20 minutes", "in an hour and a half", "in 30s"
_DELAY_RE = re.compile(
    # "in 20 minutes" and "for 5 mins" are the same request.
    r"\b(?:in|for|after)\s+(?P<count>a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|forty|"
    r"forty-five|sixty|half|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)

_WORD_COUNTS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "sixty": 60, "half": 0.5,
}

_UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}

#: "remind me ...", "set a reminder ...", "give me a nudge ..."
_ASKS_REMINDER_RE = re.compile(
    r"\bremind\s+me\b|\bset\s+(?:a\s+)?(?:reminder|timer|alarm)\b"
    r"|\bgive\s+me\s+a\s+(?:nudge|reminder)\b|\bnudge\s+me\b",
    re.IGNORECASE,
)

#: What to be reminded ABOUT, after the delay clause is removed.
_SUBJECT_RE = re.compile(
    r"\bto\s+(?P<subject>.+)$|\babout\s+(?P<subject2>.+)$"
    r"|\bthat\s+(?P<subject3>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RequestedReminder:
    """A reminder the person asked for, parsed but not yet stored."""

    text: str
    delay_s: float


def requested_reminder(user_message: Any) -> RequestedReminder | None:
    """What was asked for, or None when this is not a reminder request."""
    raw = " ".join(str(user_message or "").split())
    if not raw or not _ASKS_REMINDER_RE.search(raw):
        return None
    match = _DELAY_RE.search(raw)
    if match is None:
        # A reminder with no time is a request this cannot honour, and
        # inventing a delay would be worse than asking for one.
        return None
    count_text = match.group("count").lower()
    try:
        count = float(count_text) if count_text not in _WORD_COUNTS else float(
            _WORD_COUNTS[count_text]
        )
    except ValueError:
        return None
    unit = match.group("unit").lower().rstrip(".")
    seconds = count * _UNIT_SECONDS.get(unit, _UNIT_SECONDS.get(unit.rstrip("s"), 0))
    if seconds <= 0:
        return None
    without_delay = (raw[: match.start()] + " " + raw[match.end() :]).strip()
    subject_match = _SUBJECT_RE.search(without_delay)
    subject = ""
    if subject_match:
        subject = next(
            (g for g in subject_match.groups() if g), ""
        ).strip(" .!?")
    return RequestedReminder(text=subject or without_delay[:200], delay_s=seconds)


def reminder_answer(user_message: Any) -> str:
    """Create the reminder and say so truthfully, or "" if not asked.

    The failure branch matters more than the success one: when the store
    cannot be written, this says the reminder was NOT set. Reporting one that
    does not exist is the defect this whole module was built for.
    """
    asked = requested_reminder(user_message)
    if asked is None:
        return ""
    stored = add_reminder(asked.text, asked.delay_s)
    if stored is None:
        return (
            "I could not store that reminder, so it is not set. Nothing will "
            "come back to you about it."
        )
    return (
        f"Reminder set: {stored.text} — in {spoken_delay(asked.delay_s)}. "
        f"It is recorded as {stored.id} and I will report it as due when you "
        "ask what is queued."
    )
