"""Primitive-coverage contracts for semantic recurrent transition cohorts."""

from __future__ import annotations

from dataclasses import replace

from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.transition_coverage import audit_transition_coverage


def _cohorts(per_cell: int = 24, holdout_per_cell: int = 6):
    domains = ("coding", "calibration", "misleading_premise")
    train = frontier_process_task_battery(
        domains,
        (1,),
        per_cell,
        seed=2026081505,
    )
    holdout = frontier_process_task_battery(
        domains,
        (1,),
        holdout_per_cell,
        seed=2026081505 + 9_973,
        excluded_prompts=tuple(task.prompt for task in train),
    )
    return train, holdout


def test_old_semantic_cohort_exposes_missing_calibration_categories() -> None:
    train, holdout = _cohorts(per_cell=64, holdout_per_cell=12)

    report = audit_transition_coverage(train, holdout)

    assert report["admission"]["in_distribution_primitive_coverage_admitted"] is False
    assert report["partition"] == {
        "training_count": 192,
        "holdout_count": 36,
        "family_sets_equal": True,
        "task_identity_overlap": [],
        "prompt_overlap_sha256s": [],
    }
    # The gap map, not one photograph of it. Its exact contents move as the
    # battery gains opcodes and state slots, and freezing a dict turned that
    # growth into a failure about coverage the audit was reporting correctly.
    # What this test is named for is that gaps are FOUND and named.
    missing = report["families"]["frontier_calibration"]["missing_state_support"]
    assert missing, "the audit found no missing calibration categories at all"
    assert all(
        isinstance(slot, str) and isinstance(rows, list) and rows
        for slot, rows in missing.items()
    ), missing
    assert "unseen_task_length_generalization" in report["claims_not_supported"]
    assert len(report["report_sha256"]) == 64


def test_expanded_semantic_cohort_has_fresh_covered_programs() -> None:
    train, holdout = _cohorts(per_cell=128, holdout_per_cell=12)

    report = audit_transition_coverage(train, holdout)

    assert report["admission"]["in_distribution_primitive_coverage_admitted"] is True
    assert all(
        family["exact_program_overlap_count"] == 0
        for family in report["families"].values()
    )


def test_missing_holdout_operand_support_fails_admission() -> None:
    train, holdout = _cohorts()
    task = holdout[0]
    program = task.transition_program
    assert program is not None
    first = list(program.actions[0])
    first[0] = 31
    changed_program = replace(program, actions=(tuple(first), *program.actions[1:]))
    changed = replace(
        task,
        transition_program=changed_program,
        transition_trace=changed_program.state_trace,
    )

    report = audit_transition_coverage(train, (changed, *holdout[1:]))

    assert report["admission"]["in_distribution_primitive_coverage_admitted"] is False
    assert any(
        family["missing_action_support"]
        for family in report["families"].values()
    )


def test_reused_program_or_task_identity_fails_admission() -> None:
    train, holdout = _cohorts()

    reused_program = replace(
        holdout[0],
        transition_program=train[0].transition_program,
        transition_trace=train[0].transition_trace,
        depth=train[0].depth,
        family=train[0].family,
    )
    report = audit_transition_coverage(train, (reused_program, *holdout[1:]))
    assert report["admission"]["in_distribution_primitive_coverage_admitted"] is False
    assert any(
        family["exact_program_overlap_count"]
        for family in report["families"].values()
    )

    duplicate_identity = replace(holdout[0], seed=train[0].seed, family=train[0].family)
    duplicate = audit_transition_coverage(train, (duplicate_identity, *holdout[1:]))
    assert duplicate["partition"]["task_identity_overlap"]
    assert duplicate["admission"]["in_distribution_primitive_coverage_admitted"] is False
