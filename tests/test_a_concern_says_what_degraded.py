"""A runtime concern says what degraded (third day run, 6 November 2026).

"intentional_retrieval: 10 degradations in the last 30m" opened an incident twice. The degradations were recorded at
debug, which her log does not keep, so nothing said what they were.
"""
from __future__ import annotations

from core.runtime import integrity_audit
from core.runtime.errors import DegradationRecord, DegradationTracker


def test_the_concern_line_carries_the_newest_degradation():
    tracker = DegradationTracker(records=[
        DegradationRecord("intentional_retrieval", "debug", "TimeoutError", "older", "kept going", 1.0),
        DegradationRecord("intentional_retrieval", "debug", "KeyError", "'goal_id'", "skipped one intention", 2.0),
    ])
    said = integrity_audit._latest_degradation_of(tracker, "intentional_retrieval")
    assert said == " (latest: KeyError: 'goal_id'; skipped one intention)"
    assert integrity_audit._latest_degradation_of(tracker, "nothing_here") == ""
