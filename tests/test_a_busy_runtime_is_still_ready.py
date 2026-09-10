"""A server answering a request is ready, never un-ready.

`/api/readyz` returned 503 for the whole of every turn. Measured live
2026-09-07: `state=ready active_generations=1
blockers=['active_generation_in_flight']` and the endpoint answering 503 while
the runtime was answering perfectly.

`boot_status` already made the distinction — `conversation_lane_is_serving`
exists for exactly this, with the Kubernetes rule written in its docstring —
and `readyz` re-derived readiness from the raw `conversation_ready` flag
instead. Two readiness deciders, and the endpoint used the one that reads busy
as broken. Anything watching readiness — the desktop shell, a supervisor, a
probe — concluded the runtime was unhealthy whenever somebody was talking to
it.
"""

from __future__ import annotations

from pathlib import Path

from core.health.conversation_lane import (
    conversation_lane_is_available,
    conversation_lane_is_busy,
    conversation_lane_is_serving,
)

_SERVING = {
    "state": "ready",
    "active_generations": 1,
    "readiness_blockers": ["active_generation_in_flight"],
    "last_failure_reason": "active_generation_in_flight",
    "conversation_ready": False,
}
_WARMING = {
    "state": "recovering",
    "active_generations": 0,
    "warmup_in_flight": True,
    "readiness_blockers": ["warmup"],
    "conversation_ready": False,
}


def test_a_lane_answering_a_turn_counts_as_serving() -> None:
    assert conversation_lane_is_serving(_SERVING) is True
    assert conversation_lane_is_busy(_SERVING) is True
    assert conversation_lane_is_available(_SERVING) is True


def test_a_lane_still_warming_up_does_not() -> None:
    """Habituating busy to ready must not swallow a lane that cannot serve."""
    assert conversation_lane_is_serving(_WARMING) is False


def test_readyz_asks_the_helper_rather_than_the_raw_flag() -> None:
    """The defect was a second decider. There must be one."""
    source = Path("interface/routes/system.py").read_text()
    start = source.index('@router.get("/readyz"')
    end = source.index('@router.get(', start + 10)
    route = source[start:end]
    assert "conversation_lane_is_serving" in route, (
        "readyz no longer consults the serving helper; busy will read as broken"
    )
    assert "conversation_can_serve" in route
    assert 'readiness.get("conversation_ready") is True\n            and' not in route, (
        "readyz is back to re-deriving readiness from the raw flag"
    )


def test_readyz_reports_busy_by_name() -> None:
    """A caller that wants to know a turn is in flight must not infer it."""
    source = Path("interface/routes/system.py").read_text()
    start = source.index('@router.get("/readyz"')
    end = source.index('@router.get(', start + 10)
    assert '"conversation_busy"' in source[start:end]
