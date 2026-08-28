"""A turn finishing its own answer is not competing with itself.

The completion pass probes the conversation lane fresh, which is right: a lane
can go unhealthy mid-turn. Probing fresh is also what switches on the checks
asking whether anything is generating — so the one caller whose whole job is
to finish a generation already in progress was the only caller measured
against that generation, and it lost to itself every time.

Live on 2026-08-28: a 614-character answer stopped mid-sentence, the
completion was refused with
"continuation_admission_denied:conversation_generation_already_active", and
the turn ended on an apology.
"""

from __future__ import annotations

from interface.routes.chat import _desktop_secondary_model_repair_allowed

_BUSY_BUT_HEALTHY = {
    "state": "ready",
    "conversation_ready": True,
    "warmup_in_flight": False,
    "active_generations": 1,
    "foreground_owned": True,
    "foreground_guard_active_count": 2,
}

_NOT_READY = {
    "state": "recovering",
    "conversation_ready": False,
    "warmup_in_flight": False,
    "active_generations": 0,
}


def test_a_continuation_is_not_blocked_by_its_own_generation(monkeypatch) -> None:
    import interface.routes.chat as chat

    monkeypatch.setattr(
        chat._chat_preflight,
        "_collect_conversation_lane_status",
        lambda: dict(_BUSY_BUT_HEALTHY),
    )
    allowed, reason = _desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_completion_retry",
        lane_snapshot=None,
        continuing_this_turn=True,
    )
    assert allowed is True, reason


def test_a_fresh_request_still_waits_its_turn(monkeypatch) -> None:
    """The guard still does the job it was written for."""

    import interface.routes.chat as chat

    monkeypatch.setattr(
        chat._chat_preflight,
        "_collect_conversation_lane_status",
        lambda: dict(_BUSY_BUT_HEALTHY),
    )
    allowed, reason = _desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_repair_retry",
        lane_snapshot=None,
        continuing_this_turn=False,
    )
    assert allowed is False
    assert reason == "conversation_generation_already_active"


def test_a_continuation_still_needs_a_lane_that_works(monkeypatch) -> None:
    """Only the "is somebody else busy" checks are waived, not readiness."""

    import interface.routes.chat as chat

    monkeypatch.setattr(
        chat._chat_preflight,
        "_collect_conversation_lane_status",
        lambda: dict(_NOT_READY),
    )
    allowed, reason = _desktop_secondary_model_repair_allowed(
        reason="cognitive_engine_completion_retry",
        lane_snapshot=None,
        continuing_this_turn=True,
    )
    assert allowed is False
    assert "not_ready" in reason
