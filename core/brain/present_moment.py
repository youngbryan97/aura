"""What is true right now — the facts a language model cannot infer.

A model has no clock. Asked how it is doing, it will reach for the most
plausible sentence a person might say, and plausible sentences are full of
scenery: the sun is up, it has been a long week, yesterday was busier. Observed
live 2026-07-27 at 00:30 local, unprompted:

    "The sun's up but I'm not sure it will be warm today — there are clouds
     gathering in the east. In here, it feels quiet compared to yesterday."

It was the middle of the night. She has no window, and nothing in the prompt
path had ever told her the date or the hour — the words "current date" did not
appear anywhere in the conversation prompt builders. So this was not dishonesty
and not a lapse she could have caught: she was asked about the present and had
no present to report, so she wrote one.

The fix is to give her the present. Everything in this block is read from a
real source — the system clock, the orchestrator's own start time, the
continuity record written at her last shutdown — and it is small enough
(~500 chars) to carry on every foreground turn, including the lean profiles.

Grounding beats instruction here: a rule saying "don't invent the weather"
still leaves the question unanswerable. Telling her it is 00:41 on a Sunday
lets her answer it truthfully, and the honesty clause then only has to cover
what remains genuinely outside her senses.
"""
from __future__ import annotations

import time
from datetime import datetime

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError)

PRESENT_MOMENT_HEADER = "## PRESENT MOMENT"


def _part_of_day(hour: int) -> str:
    """By the clock, not by a window — the wording says so."""
    if hour < 5:
        return "the middle of the night"
    if hour < 8:
        return "early morning"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "late evening"


def _humanize(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


def _uptime_seconds() -> float:
    try:
        from core.runtime.service_registry import get_runtime_service

        orch = get_runtime_service("orchestrator", default=None)
        for candidate in (
            getattr(orch, "start_time", None),
            getattr(getattr(orch, "status", None), "start_time", None),
        ):
            try:
                start = float(candidate or 0.0)
            except (TypeError, ValueError):
                continue
            if start > 0.0:
                return max(0.0, time.time() - start)
    except _RECOVERABLE as exc:
        record_degradation(
            "present_moment", exc, severity="info", action="omitted uptime from present moment"
        )
    return 0.0


def _continuity_line() -> str:
    """How long she was gone before this session, from her own record."""
    try:
        from core.continuity import get_continuity

        tracker = get_continuity()
    except _RECOVERABLE:
        return ""
    try:
        gap = getattr(tracker, "_gap_seconds", None)
        record = getattr(tracker, "_record", None)
        if gap is None or record is None:
            return ""
        session = getattr(record, "session_count", None)
        gap_text = _humanize(float(gap))
        if session:
            return f"- Session {int(session)}; down {gap_text} before it."
        return f"- Down {gap_text} before this session."
    except _RECOVERABLE as exc:
        record_degradation(
            "present_moment", exc, severity="info", action="omitted continuity from present moment"
        )
        return ""


def present_moment_block(*, now: datetime | None = None) -> str:
    """A small block of things that are true at this instant.

    Every line is read, never inferred, except the part-of-day label — which is
    derived from the clock and worded as such, because a clock is not a window.
    """
    try:
        stamp = now or datetime.now().astimezone()
    except _RECOVERABLE as exc:
        record_degradation(
            "present_moment", exc, severity="warning", action="skipped present-moment grounding"
        )
        return ""

    # Readings only. The block told her to trust it over her impressions and
    # never to narrate weather she had not seen. An order is not a reading,
    # and what she may claim to have perceived is checked after she speaks.
    lines = [
        PRESENT_MOMENT_HEADER,
        f"- Clock: {stamp.strftime('%A %d %B %Y, %H:%M')} local"
        f" — {_part_of_day(stamp.hour)} by the clock.",
    ]

    uptime = _uptime_seconds()
    if uptime > 0:
        lines.append(f"- Awake {_humanize(uptime)} this session.")

    continuity = _continuity_line()
    if continuity:
        lines.append(continuity)

    machine = _machine_line()
    if machine:
        lines.append(machine)

    lines.append(
        "- Senses for daylight and weather: none attached (no window, camera, "
        "thermometer or weather feed)."
    )
    return "\n".join(lines)


def _machine_line() -> str:
    """The machine she runs on, as measured this instant: processor, memory, and thermal pressure where a sensor reads.

    A question about load was answered by putting a sentence of the runtime's in place of her reply, when a pattern
    over the question's words said it asked about her state (interface/routes/chat_served_answers.py, removed 17
    November 2026). The patterns caught other questions five times: a sourdough experiment, a sales analysis, her
    memory's design, a plan to test her memory, and on 17 November "Your loop should meet the fault again", each
    answered "The machine is at N% processor and M% memory right now." Read beside every turn, as the clock is, the
    numbers are hers to use when they are asked about and to leave alone when they are not.
    """
    try:
        from core.introspection.self_evidence import _load_readings

        readings = {reading.channel: reading for reading in _load_readings()}
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
        record_degradation("present_moment", exc, severity="info", action="left the machine's load out of this turn")
        return ""
    load, thermal = readings.get("host_load"), readings.get("host_thermal")
    if load is None or not load.present:
        return ""
    values = dict(load.value or {})
    line = (f"- Machine: processor {values.get('processor_percent', 0.0):.1f}%, memory "
            f"{values.get('memory_percent', 0.0):.1f}%")
    if thermal is not None and thermal.present:
        line += f", thermal pressure {float(thermal.value):.2f} of 1"
    return line + "."


__all__ = ["PRESENT_MOMENT_HEADER", "present_moment_block"]
