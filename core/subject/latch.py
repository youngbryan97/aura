"""Whether a run is still measuring her, or her under the shutdown latch.

A process that has asked to shut down is not the system it was. 235 places in
the tree read the latch: a new SQLite connection is refused, the spiking
advisor stops answering, the soma's decay loop ends. In one process on seed 79,
two of four moments rolled two turns from their own snapshots moved under it,
by 0.054 and 0.0087, the same on both passes. Seed 73's first campaign lost a
model-lane lease at 17:00 on 26 October and went on measuring under the latch
for seven hours, and only a git call at the very end noticed.

So a subject tool asks here before it records a stage, a row or a report. A
latched process records nothing more and stops; what it recorded before the
latch stands, and a resume starts from it in a fresh process.
"""

from __future__ import annotations


def latch_reason() -> str | None:
    """Why the process asked to shut down, or None while it has not."""
    from core.runtime.shutdown_coordinator import shutdown_request_snapshot

    state = shutdown_request_snapshot()
    if not state.get("requested"):
        return None
    return str(state.get("first_reason") or "no reason was given")


def refuse_if_latched(what: str) -> None:
    """Stop the run rather than record `what`, when it was measured under the latch."""
    reason = latch_reason()
    if reason is None:
        return
    raise SystemExit(
        f"refusing to record {what}: the shutdown latch was set ({reason}) before it was "
        "measured to the end, and a latched process is not the system the run measures. "
        "What was recorded before stands; resume from it in a fresh process."
    )
