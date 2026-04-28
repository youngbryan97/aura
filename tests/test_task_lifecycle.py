"""Tests for the formal 11-state task lifecycle and migration."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from core.goals.lifecycle import (
    ACTIVE_STATES,
    ALLOWED_TRANSITIONS,
    GoalState,
    IDLE_STATES,
    IllegalTransitionError,
    MissingEvidenceError,
    STATE_POLICIES,
    TERMINAL_STATES,
    TaskLifecycleManager,
    TransitionRequest,
    apply_transition,
    coerce_state,
    is_active,
    is_idle,
    is_terminal,
    migrate_legacy_status_db,
    reachable_states,
    validate_transition,
)


# ---------------------------------------------------------------------------
# enum + policy invariants
# ---------------------------------------------------------------------------
def test_eleven_states_defined():
    assert len(GoalState) == 11
    assert {s.value for s in GoalState} == {
        "proposed",
        "accepted",
        "planned",
        "in_progress",
        "blocked",
        "waiting_for_user",
        "testing",
        "completed",
        "failed",
        "abandoned",
        "deferred",
    }


def test_terminal_states_have_no_outgoing_transitions():
    for state in TERMINAL_STATES:
        assert ALLOWED_TRANSITIONS[state] == frozenset(), state


def test_active_and_idle_classifications_are_disjoint_and_cover_non_terminal():
    non_terminal = set(GoalState) - TERMINAL_STATES
    assert ACTIVE_STATES.isdisjoint(IDLE_STATES)
    assert (ACTIVE_STATES | IDLE_STATES) == non_terminal


def test_every_non_terminal_state_has_at_least_one_outgoing_transition():
    for state in GoalState:
        if state in TERMINAL_STATES:
            continue
        assert ALLOWED_TRANSITIONS[state], f"{state.value} is a sink"


def test_policy_table_covers_all_states():
    assert set(STATE_POLICIES.keys()) == set(GoalState)


def test_classifier_helpers():
    assert is_terminal(GoalState.COMPLETED)
    assert not is_terminal(GoalState.IN_PROGRESS)
    assert is_active(GoalState.TESTING)
    assert is_idle(GoalState.BLOCKED)
    assert not is_idle(GoalState.IN_PROGRESS)


# ---------------------------------------------------------------------------
# coercion (legacy status migration in memory)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("queued", GoalState.ACCEPTED),
        ("paused", GoalState.DEFERRED),
        ("active", GoalState.IN_PROGRESS),
        ("completed", GoalState.COMPLETED),
        ("succeeded", GoalState.COMPLETED),
        ("rejected", GoalState.ABANDONED),
        ("waiting_for_user", GoalState.WAITING_FOR_USER),
        ("testing", GoalState.TESTING),
        ("DEFERRED", GoalState.DEFERRED),
    ],
)
def test_coerce_legacy_strings(legacy, expected):
    assert coerce_state(legacy) is expected


def test_coerce_unknown_raises():
    with pytest.raises(ValueError):
        coerce_state("not_a_state")


# ---------------------------------------------------------------------------
# pure transition validator
# ---------------------------------------------------------------------------
def _req(from_s: GoalState, to_s: GoalState, **kwargs):
    return TransitionRequest(
        goal_id="g-1",
        from_state=from_s,
        to_state=to_s,
        actor=kwargs.pop("actor", "alice"),
        evidence=kwargs.pop("evidence", {}),
        deadline=kwargs.pop("deadline", None),
        reason=kwargs.pop("reason", ""),
        metadata=kwargs.pop("metadata", {}),
    )


def test_legal_transition_passes():
    req = _req(GoalState.PROPOSED, GoalState.ACCEPTED, evidence={"acceptance_reason": "ok"})
    validate_transition(req)


def test_illegal_transition_raises():
    with pytest.raises(IllegalTransitionError):
        validate_transition(_req(GoalState.PROPOSED, GoalState.IN_PROGRESS))


def test_terminal_state_blocks_all_outgoing():
    for terminal in TERMINAL_STATES:
        for target in GoalState:
            if target == terminal:
                continue
            with pytest.raises(IllegalTransitionError):
                validate_transition(_req(terminal, target))


def test_self_transition_is_allowed():
    # E.g. progress update from IN_PROGRESS to IN_PROGRESS.
    validate_transition(_req(GoalState.IN_PROGRESS, GoalState.IN_PROGRESS))


def test_missing_exit_evidence_raises():
    # PROPOSED requires acceptance_reason to leave for any target.
    with pytest.raises(MissingEvidenceError) as exc:
        validate_transition(_req(GoalState.PROPOSED, GoalState.ACCEPTED, evidence={}))
    assert "acceptance_reason" in exc.value.missing


def test_target_requiring_deadline_blocks_without_one():
    # PLANNED requires a deadline.
    with pytest.raises(IllegalTransitionError):
        validate_transition(
            _req(
                GoalState.ACCEPTED,
                GoalState.PLANNED,
                evidence={"plan_id": "p1"},
                deadline=None,
            )
        )


def test_target_requiring_deadline_passes_with_one():
    validate_transition(
        _req(
            GoalState.ACCEPTED,
            GoalState.PLANNED,
            evidence={"plan_id": "p1"},
            deadline=time.time() + 60,
        )
    )


def test_target_requiring_owner_blocks_without_actor():
    with pytest.raises(IllegalTransitionError):
        validate_transition(_req(GoalState.PROPOSED, GoalState.ACCEPTED,
                                 actor="", evidence={"acceptance_reason": "ok"}))


def test_apply_transition_returns_rollback_path():
    result = apply_transition(
        _req(
            GoalState.PLANNED,
            GoalState.IN_PROGRESS,
            deadline=time.time() + 60,
        )
    )
    # IN_PROGRESS rolls back to PLANNED.
    assert result.rollback_to is GoalState.PLANNED


# ---------------------------------------------------------------------------
# DB-bound TaskLifecycleManager
# ---------------------------------------------------------------------------
@pytest.fixture
def manager(tmp_path: Path) -> TaskLifecycleManager:
    return TaskLifecycleManager(tmp_path / "lifecycle.db")


def test_create_then_transition_persists(manager: TaskLifecycleManager):
    manager.create(goal_id="g1", name="Test goal", state=GoalState.PROPOSED, actor="alice")
    assert manager.get_state("g1") is GoalState.PROPOSED

    manager.transition(
        TransitionRequest(
            goal_id="g1",
            from_state=GoalState.PROPOSED,
            to_state=GoalState.ACCEPTED,
            actor="alice",
            evidence={"acceptance_reason": "looks reasonable"},
        )
    )
    assert manager.get_state("g1") is GoalState.ACCEPTED


def test_transition_rejects_stale_from_state(manager: TaskLifecycleManager):
    manager.create(goal_id="g1", state=GoalState.PROPOSED, actor="a")
    with pytest.raises(IllegalTransitionError):
        manager.transition(
            TransitionRequest(
                goal_id="g1",
                from_state=GoalState.IN_PROGRESS,  # wrong; persisted is PROPOSED
                to_state=GoalState.COMPLETED,
                actor="a",
                evidence={},
            )
        )


def test_full_happy_path(manager: TaskLifecycleManager):
    manager.create(goal_id="g1", state=GoalState.PROPOSED, actor="alice")
    deadline = time.time() + 3600

    sequence = [
        (GoalState.PROPOSED, GoalState.ACCEPTED, {"acceptance_reason": "ok"}, None),
        (GoalState.ACCEPTED, GoalState.PLANNED, {"plan_id": "p1"}, deadline),
        (GoalState.PLANNED, GoalState.IN_PROGRESS, {}, deadline),
        (
            GoalState.IN_PROGRESS,
            GoalState.TESTING,
            {"progress_summary": "done coding"},
            deadline,
        ),
        (
            GoalState.TESTING,
            GoalState.COMPLETED,
            {"verification_result": "passed"},
            None,
        ),
    ]
    for from_s, to_s, ev, dl in sequence:
        manager.transition(
            TransitionRequest(
                goal_id="g1",
                from_state=from_s,
                to_state=to_s,
                actor="alice",
                evidence=ev,
                deadline=dl,
            )
        )

    assert manager.get_state("g1") is GoalState.COMPLETED
    history = manager.history("g1")
    assert [(h["from"], h["to"]) for h in history] == [
        ("proposed", "accepted"),
        ("accepted", "planned"),
        ("planned", "in_progress"),
        ("in_progress", "testing"),
        ("testing", "completed"),
    ]


def test_recovery_path_in_progress_to_blocked_to_in_progress(manager: TaskLifecycleManager):
    manager.create(goal_id="g1", state=GoalState.PROPOSED, actor="a")
    deadline = time.time() + 3600
    moves = [
        (GoalState.PROPOSED, GoalState.ACCEPTED, {"acceptance_reason": "ok"}, None),
        (GoalState.ACCEPTED, GoalState.PLANNED, {"plan_id": "p1"}, deadline),
        (GoalState.PLANNED, GoalState.IN_PROGRESS, {}, deadline),
        (
            GoalState.IN_PROGRESS,
            GoalState.BLOCKED,
            {"progress_summary": "waiting on dep"},
            None,
        ),
        (GoalState.BLOCKED, GoalState.IN_PROGRESS, {"blocker": "dep landed"}, deadline),
    ]
    for from_s, to_s, ev, dl in moves:
        manager.transition(
            TransitionRequest(
                goal_id="g1",
                from_state=from_s,
                to_state=to_s,
                actor="a",
                evidence=ev,
                deadline=dl,
            )
        )
    assert manager.get_state("g1") is GoalState.IN_PROGRESS


def test_history_records_evidence_and_rollback(manager: TaskLifecycleManager):
    manager.create(goal_id="g1", state=GoalState.PROPOSED, actor="a")
    manager.transition(
        TransitionRequest(
            goal_id="g1",
            from_state=GoalState.PROPOSED,
            to_state=GoalState.ACCEPTED,
            actor="a",
            evidence={"acceptance_reason": "useful"},
            reason="passed review",
        )
    )
    [entry] = manager.history("g1")
    assert entry["evidence"] == {"acceptance_reason": "useful"}
    assert entry["reason"] == "passed review"
    # ACCEPTED's rollback target is PROPOSED.
    assert entry["rollback_to"] == "proposed"


def test_unknown_goal_id_raises(manager: TaskLifecycleManager):
    with pytest.raises(IllegalTransitionError):
        manager.transition(
            TransitionRequest(
                goal_id="missing",
                from_state=GoalState.PROPOSED,
                to_state=GoalState.ACCEPTED,
                actor="a",
                evidence={"acceptance_reason": "x"},
            )
        )


def test_creating_in_terminal_state_is_rejected(manager: TaskLifecycleManager):
    with pytest.raises(IllegalTransitionError):
        manager.create(goal_id="g1", state=GoalState.COMPLETED, actor="a")


# ---------------------------------------------------------------------------
# legacy DB migration
# ---------------------------------------------------------------------------
def test_migrate_rewrites_legacy_paused_to_deferred(tmp_path: Path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE goals (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.executemany(
        "INSERT INTO goals(id, status, metadata_json) VALUES (?, ?, ?);",
        [
            ("g1", "paused", "{}"),
            ("g2", "queued", "{}"),
            ("g3", "completed", "{}"),  # already canonical
            ("g4", "running", '{"trace": "alpha"}'),
            ("g5", "weird_old_state", "{}"),
        ],
    )
    conn.commit()
    conn.close()

    stats = migrate_legacy_status_db(db)
    assert stats["scanned"] == 5
    assert stats["rewritten"] == 3  # paused, queued, running
    assert stats["skipped"] == 1  # completed
    assert stats["unrecognized"] == 1  # weird_old_state

    conn = sqlite3.connect(str(db))
    rows = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT id, status, metadata_json FROM goals").fetchall()}
    conn.close()
    assert rows["g1"][0] == "deferred"
    assert json.loads(rows["g1"][1])["legacy_status"] == "paused"
    assert rows["g2"][0] == "accepted"
    assert rows["g3"][0] == "completed"  # untouched
    assert rows["g4"][0] == "in_progress"
    assert json.loads(rows["g4"][1])["trace"] == "alpha"
    assert rows["g5"][0] == "weird_old_state"  # unrecognized -> left alone


def test_migration_is_idempotent(tmp_path: Path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE goals (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.execute("INSERT INTO goals(id, status) VALUES ('g1', 'paused');")
    conn.commit()
    conn.close()

    first = migrate_legacy_status_db(db)
    second = migrate_legacy_status_db(db)
    assert first["rewritten"] == 1
    assert second["rewritten"] == 0
    assert second["skipped"] == 1


def test_migration_on_missing_db_returns_zero(tmp_path: Path):
    stats = migrate_legacy_status_db(tmp_path / "does_not_exist.db")
    assert stats == {"scanned": 0, "rewritten": 0, "skipped": 0, "unrecognized": 0}
