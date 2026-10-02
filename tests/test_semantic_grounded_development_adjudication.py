from copy import deepcopy
from dataclasses import replace

import pytest

from tools.adjudicate_semantic_grounded_development import adjudicate_rows, adjudication_contract, grade_archive
from tools.semantic_grounded_development_archive import ARMS, digest


def example():
    from tests.test_semantic_program_shared_transducer import _examples

    return _examples()[0]


def row(item, ir=None, *, refusal=""):
    return {"source_id": item.ir.source_text_sha256, "source_token_ids": list(item.ir.source_token_ids),
        "public_inputs": list(item.public_inputs), "model_basis_sha256": item.ir.model_basis_receipt_sha256,
        "decoded_ir": None if refusal else (ir or item.ir).to_dict(), "decode": {"refusal": refusal},
        "receipt_sha256": "b" * 64}


def changed_ir(item, step, **changes):
    instructions = list(item.ir.instructions)
    instructions[step] = replace(instructions[step], **changes)
    return replace(item.ir, instructions=tuple(instructions))


def test_correct_program_has_separate_semantic_and_reference_answer_results():
    item = example()
    outcome = grade_archive(item, row(item))
    assert outcome["program_exact"] is outcome["program_equivalent"] is outcome["answer_correct"] is True
    assert outcome["decoded_value"] == outcome["annotated_value"] == 11
    assert outcome["status"] == "equivalent_interpretation"


def test_commuted_addition_is_equivalent_but_not_exact():
    item = example()
    ir = changed_ir(item, 0, args=(1, 0), argument_spans=tuple(reversed(item.ir.instructions[0].argument_spans)))
    outcome = grade_archive(item, row(item, ir))
    assert outcome["program_exact"] is False
    assert outcome["program_equivalent"] is outcome["answer_correct"] is True


def test_wrong_operand_roles_fail_semantics_even_when_numeric_answers_coincide():
    item = replace(example(), public_inputs=(2, 2, 4))
    ir = changed_ir(item, 1, args=(2, 3), argument_spans=tuple(reversed(item.ir.instructions[1].argument_spans)))
    outcome = grade_archive(item, row(item, ir))
    assert outcome["program_exact"] is outcome["program_equivalent"] is False
    assert outcome["answer_correct"] is True
    assert outcome["status"] == "program_mismatch"


def test_role_reversal_with_distinct_values_is_wrong_semantically_and_numerically():
    item = example()
    ir = changed_ir(item, 1, args=(2, 3), argument_spans=tuple(reversed(item.ir.instructions[1].argument_spans)))
    outcome = grade_archive(item, row(item, ir))
    assert outcome["program_equivalent"] is outcome["answer_correct"] is False
    assert outcome["decoded_value"] == -11


def test_undefined_annotation_is_unmeasured_not_equal_to_undefined_decode():
    item = example()
    item = replace(item, ir=changed_ir(item, 0, op="idiv"), public_inputs=(9, 0, 2))
    outcome = grade_archive(item, row(item))
    assert outcome["program_exact"] is True
    assert outcome["answer_correct"] is None
    assert outcome["status"] == "annotation_execution_unmeasured"


def test_undefined_decode_is_a_failure_when_annotation_executes():
    item = replace(example(), public_inputs=(9, 0, 2))
    outcome = grade_archive(item, row(item, changed_ir(item, 0, op="idiv")))
    assert outcome["answer_correct"] is False
    assert outcome["status"] == "decoded_execution_failed"
    assert outcome["annotated_value"] == 7


def test_refusal_does_not_invent_an_answer_measurement():
    item = example()
    outcome = grade_archive(item, row(item, refusal="budget_exhausted"))
    assert outcome["program_exact"] is outcome["program_equivalent"] is False
    assert outcome["answer_correct"] is None
    assert outcome["status"] == "decode_refused"


def test_different_input_anchors_are_not_accepted_as_the_same_interpretation():
    from core.learning.semantic_program_ir import TokenSpan

    item = example()
    ir = replace(item.ir, input_spans=(TokenSpan(0, 1), *item.ir.input_spans[1:]))
    outcome = grade_archive(item, row(item, ir))
    assert outcome["program_equivalent"] is False
    assert outcome["answer_correct"] is None
    assert outcome["status"] == "input_anchors_unaligned"


@pytest.mark.parametrize("field", ["source_id", "source_token_ids", "public_inputs", "model_basis_sha256",
    "decoded_ir", "refusal"])
def test_independent_grading_rejects_changed_public_request_or_program_identity(field):
    item = example()
    value = row(item)
    if field in {"source_id", "model_basis_sha256"}:
        value[field] = "c" * 64
    elif field in {"source_token_ids", "public_inputs"}:
        value[field][0] += 1
    elif field == "decoded_ir":
        value[field]["source_text_sha256"] = "c" * 64
    else:
        value["decode"]["refusal"] = "budget_exhausted"
    with pytest.raises(ValueError):
        grade_archive(item, value)


def measured_report(item):
    identity = item.ir.source_text_sha256
    report = {"receipt_sha256": "d" * 64, "plan": {"held_ids": [identity], "plan_sha256": "e" * 64},
        "comparison": {"candidates": {arm: {"program_correct": [True], "program_equivalent_correct": [True]}
            for arm in ARMS}}}
    rows = {arm: {identity: row(item)} for arm in ARMS}
    return report, rows


def test_replay_counters_preserve_denominators_groups_and_proof_boundaries():
    item = example()
    report, rows = measured_report(item)
    result = adjudicate_rows(report, [item], rows)
    assert result["population"] == 1
    assert result["adjudication_contract"] == adjudication_contract()
    assert result["receipt_sha256"] == digest({k: v for k, v in result.items() if k != "receipt_sha256"})
    for arm in ARMS:
        assert result["metrics"][arm] == {"program_exact": 1, "program_equivalent": 1,
            "answer_measured": 1, "answer_correct": 1, "refusals": 0}
        assert result["per_construction"][arm][item.construction_id]["population"] == 1
    assert result["semantic_interpretations_independently_replayed"] is True
    for name in ("backbone_loaded", "public_answer_generation_measured", "g03_complete", "fresh_transfer_proven",
            "qualification_evidence", "serving_authority"):
        assert result[name] is False


def test_replay_keeps_refused_requests_in_population_without_fabricated_answer_scores():
    item = example()
    report, rows = measured_report(item)
    report["comparison"]["candidates"]["joint_native"] = {
        "program_correct": [False], "program_equivalent_correct": [False]}
    rows["joint_native"][item.ir.source_text_sha256] = row(item, refusal="budget_exhausted")
    result = adjudicate_rows(report, [item], rows)
    assert result["population"] == 1
    assert result["metrics"]["joint_native"] == {"program_exact": 0, "program_equivalent": 0,
        "answer_measured": 0, "answer_correct": 0, "refusals": 1}
    assert result["per_construction"]["joint_native"][item.construction_id]["population"] == 1


@pytest.mark.parametrize("fault", ["contradiction", "missing_arm", "missing_row", "duplicate_source", "duplicate_id"])
def test_replay_rejects_contradictory_or_incomplete_report(fault):
    item = example()
    report, rows = measured_report(item)
    examples = [item]
    if fault == "contradiction":
        report = deepcopy(report)
        report["comparison"]["candidates"][ARMS[0]]["program_equivalent_correct"] = [False]
    elif fault == "missing_arm":
        rows.pop(ARMS[0])
    elif fault == "missing_row":
        rows[ARMS[0]].clear()
    elif fault == "duplicate_source":
        examples *= 2
    else:
        report["plan"]["held_ids"] *= 2
    with pytest.raises(ValueError):
        adjudicate_rows(report, examples, rows)
