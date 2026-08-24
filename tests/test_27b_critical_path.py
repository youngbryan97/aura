"""One receipt for the whole remaining path, and it never guesses green.

Four tools answer parts of this. Reading them one at a time is how a blocker in
the fourth gets missed while the first three look green, so this reads all of
them and emits no command while any of them refuses.
"""
from __future__ import annotations

import pytest

from tools import report_27b_critical_path as critical


@pytest.fixture
def receipt():
    return critical.build()


# ── Blockers are never inferred green ───────────────────────────────────


def test_a_tool_that_cannot_run_is_a_blocker_not_a_pass(monkeypatch):
    def _boom():
        raise RuntimeError("readiness exploded")

    monkeypatch.setattr(critical.readiness, "build", _boom)
    built = critical.build()
    assert built["ready_to_launch"] is False
    assert any(b["kind"] == "tool_failed" for b in built["blockers"])
    assert built["launch_command"] is None


def test_a_launch_blocker_propagates(monkeypatch):
    monkeypatch.setattr(
        critical.readiness,
        "build",
        lambda: {"blockers": [{"kind": "source_drifted", "detail": "x"}]},
    )
    built = critical.build()
    assert built["ready_to_launch"] is False
    assert any(b["source"] == "launch_readiness" for b in built["blockers"])


def test_an_uncovered_capability_blocks_the_launch(monkeypatch):
    monkeypatch.setattr(
        critical.queue_tool,
        "build",
        lambda: {"uncovered_capabilities": ["mystery"], "by_disposition": {}},
    )
    built = critical.build()
    assert built["ready_to_launch"] is False
    assert any(b["kind"] == "capability_uncovered" for b in built["blockers"])


def test_a_missing_measured_reference_blocks(monkeypatch):
    monkeypatch.setattr(
        critical.execution, "build", lambda: {"blocked": "no reference installed"}
    )
    built = critical.build()
    assert built["ready_to_launch"] is False
    assert any(b["kind"] == "no_measured_reference" for b in built["blockers"])


# ── Commands and blockers are mutually exclusive ────────────────────────


def test_no_command_is_emitted_while_anything_blocks(receipt):
    if receipt["blockers"]:
        assert receipt["launch_command"] is None
        assert receipt["promotion_command"] is None


def test_both_commands_appear_only_when_nothing_blocks(monkeypatch):
    monkeypatch.setattr(
        critical.readiness,
        "build",
        lambda: {"blockers": [], "launch_command": "run-the-campaign"},
    )
    monkeypatch.setattr(
        critical.execution,
        "build",
        lambda: {"measured_reference": {"decode_calls": 364, "elapsed_seconds": 1.0}},
    )
    monkeypatch.setattr(
        critical.queue_tool,
        "build",
        lambda: {"uncovered_capabilities": [], "by_disposition": {}},
    )
    built = critical.build()
    assert built["ready_to_launch"] is True
    assert built["launch_command"] == "run-the-campaign"
    assert built["promotion_command"]


def test_launch_and_promotion_are_different_commands(monkeypatch):
    monkeypatch.setattr(
        critical.readiness,
        "build",
        lambda: {"blockers": [], "launch_command": "run-the-campaign"},
    )
    monkeypatch.setattr(
        critical.execution, "build", lambda: {"measured_reference": {}}
    )
    monkeypatch.setattr(
        critical.queue_tool,
        "build",
        lambda: {"uncovered_capabilities": [], "by_disposition": {}},
    )
    built = critical.build()
    # Training finishing is not authorization to serve.
    assert built["launch_command"] != built["promotion_command"]
    assert "verification and adjudication" in built["promotion_precondition"]


# ── Durations are measured or absent ────────────────────────────────────


def test_training_duration_is_absent_rather_than_estimated(receipt):
    durations = receipt["measured_durations"]
    assert durations["training_seconds"] is None
    assert "rather than estimated" in durations["training_note"]


def test_the_decode_duration_names_the_checkpoint_it_was_measured_on(receipt):
    durations = receipt["measured_durations"]
    if durations["decode_seconds"] is None:
        pytest.skip("no measured reference installed")
    # The only measured run of this shape is on the 32B, and saying so is the
    # difference between a projection and a claim about the 27B.
    assert "32B" in str(durations["decode_checkpoint"])


# ── Concurrency is stated, not assumed ──────────────────────────────────


def test_no_model_active_stage_may_run_concurrently(receipt):
    for entry in receipt["stage_concurrency"]:
        if entry["needs_model"]:
            assert entry["may_run_concurrently_with"] == [], entry["stage"]


def test_the_model_active_stages_are_the_contiguous_five(receipt):
    assert receipt["remaining_model_active_stages"] == [
        "calibration",
        "training",
        "canary",
        "lesion_arms",
        "export",
    ]
    assert receipt["remaining_model_loads"] == 1


def test_post_unload_stages_do_not_need_the_model(receipt):
    after_unload = False
    for entry in receipt["stage_concurrency"]:
        if entry["stage"] == "unload":
            after_unload = True
            continue
        if after_unload:
            assert entry["needs_model"] is False, entry["stage"]


def test_the_decode_count_is_carried_from_the_measured_reference(receipt):
    counts = receipt["decode_calls"]
    if counts["measured_reference"] is None:
        pytest.skip("no measured reference installed")
    assert counts["planned"] == counts["measured_reference"]
