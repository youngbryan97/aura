from types import SimpleNamespace

import pytest

from tools.profile_semantic_source_order_misses import (
    _sha,
    failure_class,
    verified_validation_rows,
)


def program(*steps):
    return SimpleNamespace(instructions=tuple(SimpleNamespace(op=op, args=args)
                                              for op, args in steps))


def test_failure_class_separates_reach_operation_order_and_binding():
    target = program(("at", (0, 1)), ("sub", (2, 3)))
    assert failure_class(None, target) == "decode_failure"
    assert failure_class(program(("at", (0, 1)), ("idiv", (2, 3))), target) == "operation_choice"
    assert failure_class(program(("sub", (2, 3)), ("at", (0, 1))), target) == "operation_order"
    assert failure_class(program(("at", (1, 0)), ("sub", (2, 3))), target) == "argument_binding"
    assert failure_class(target, target) == "none"


def test_validation_profile_refuses_changed_receipt_or_cohort():
    body = {
        "candidate_receipt_sha256": "candidate",
        "source_order_plan_sha256": "plan",
        "validation_ids_sha256": "ids",
        "source_input_order": {"candidates": {"frozen": {"program_correct": [True, False]}}},
    }
    evaluation = {**body, "receipt_sha256": _sha(body)}
    model = SimpleNamespace(receipt_sha256="candidate")
    plan = {"report_sha256": "plan", "validation_example_ids_sha256": "ids"}
    assert verified_validation_rows(evaluation, model, plan, 2) == [True, False]
    with pytest.raises(ValueError, match="receipt"):
        verified_validation_rows({**evaluation, "validation_ids_sha256": "changed"}, model, plan, 2)
    with pytest.raises(ValueError, match="cohort"):
        verified_validation_rows(evaluation, model, plan, 3)
