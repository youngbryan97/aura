"""The stretch of time a message says it is asking about.

LIVE, 2026-08-27: "of everything I've thrown at you in the last hour or so,
what did you actually do well?" came back with an activity log spanning several
days — 2048, a sliding puzzle, notes written to a Desktop — because the record
was read by COUNT and the window in the sentence was never read at all.

A stated window is a constraint like any other. Reading it in one place means
the activity record, the memory readers and anything else that reports history
bound themselves the same way, rather than each growing its own vocabulary of
"today" and "this morning".
"""

from __future__ import annotations

import re

__all__ = ["seconds_named", "describe_window"]

_MINUTE = 60.0
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR

#: Counts a person spells out.
_SPELLED = {
    "a": 1, "an": 1, "one": 1, "couple": 2, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "sixty": 60, "ninety": 90,
}

_UNITS = {
    "second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0,
    "minute": _MINUTE, "minutes": _MINUTE, "min": _MINUTE, "mins": _MINUTE,
    "hour": _HOUR, "hours": _HOUR, "hr": _HOUR, "hrs": _HOUR,
    "day": _DAY, "days": _DAY,
    "week": 7 * _DAY, "weeks": 7 * _DAY,
}

#: "in the last hour", "over the past twenty minutes", "in the last day or so".
_SPAN_RE = re.compile(
    r"\b(?:in|over|during|within|for)?\s*(?:the\s+)?"
    r"(?:last|past|previous|recent)\s+"
    r"(?:(\d{1,4})|([a-z]+(?:-[a-z]+)?))?\s*(?:of\s+)?"
    r"(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs|"
    r"day|days|week|weeks)\b",
    re.IGNORECASE,
)

#: Named stretches with an obvious length.
_NAMED = (
    (re.compile(r"\bjust\s+now\b", re.IGNORECASE), 5 * _MINUTE),
    (re.compile(r"\bthis\s+morning\b", re.IGNORECASE), 6 * _HOUR),
    (re.compile(r"\bthis\s+afternoon\b", re.IGNORECASE), 6 * _HOUR),
    (re.compile(r"\bthis\s+evening\b", re.IGNORECASE), 6 * _HOUR),
    (re.compile(r"\btonight\b", re.IGNORECASE), 8 * _HOUR),
    (re.compile(r"\btoday\b", re.IGNORECASE), _DAY),
    (re.compile(r"\bthis\s+session\b", re.IGNORECASE), 12 * _HOUR),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), 7 * _DAY),
)


def seconds_named(text: object) -> float | None:
    """How far back this message is asking, in seconds, or None.

    None means the message named no window, which is different from naming a
    long one: a caller that reads history should keep its own default rather
    than being handed a made-up bound.
    """
    body = str(text or "")
    if not body.strip():
        return None
    match = _SPAN_RE.search(body)
    if match:
        digits, spelled, unit = match.groups()
        size = _UNITS.get(unit.lower())
        if size:
            if digits:
                count = float(digits)
            elif spelled:
                count = float(_SPELLED.get(spelled.lower(), 0) or 0)
            else:
                count = 1.0
            if count > 0:
                return count * size
    for pattern, span in _NAMED:
        if pattern.search(body):
            return span
    return None


def describe_window(seconds: float | None) -> str:
    """The window as a person would say it, or "" when there is none."""
    if not seconds or seconds <= 0:
        return ""
    if seconds < _HOUR:
        return f"the last {int(round(seconds / _MINUTE))} minutes"
    if seconds < _DAY:
        hours = seconds / _HOUR
        return "the last hour" if abs(hours - 1) < 0.01 else f"the last {int(round(hours))} hours"
    days = seconds / _DAY
    return "the last day" if abs(days - 1) < 0.01 else f"the last {int(round(days))} days"
