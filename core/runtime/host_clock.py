"""Small, dependency-free reads of the host wall clock.

The wall clock is operating-system state. Reading it must not initialize the
desktop automation stack or depend on Accessibility, Apple Events, or a GUI
process being responsive.
"""

from __future__ import annotations

from datetime import datetime


def read_host_clock_text(*, now: datetime | None = None) -> str:
    """Return a bounded, timezone-bearing rendering of the host wall clock."""
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    text = current.strftime("%a %b %d %Y %I:%M:%S %p %Z")
    # ``%d`` is portable but zero-padded. Remove only that presentation detail.
    text = text.replace(f" {current.day:02d} ", f" {current.day} ", 1).strip()
    if not text:
        raise RuntimeError("host system clock returned no value")
    return text[:240]
