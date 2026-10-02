from types import SimpleNamespace

import pytest

from core.learning.semantic_program_ir import TokenSpan
from tools.audit_semantic_grounded_operation_selection import operation_inventory_audit


def node(op, start, score):
    return SimpleNamespace(operation=op, span=TokenSpan(start, start + 1), score=score)


def request():
    return SimpleNamespace(ir=SimpleNamespace(source_token_ids=(1, 2, 3, 4),
        source_text_sha256="a" * 64, input_spans=(TokenSpan(0, 1), TokenSpan(1, 2)),
        instructions=(SimpleNamespace(op="add", operation_span=TokenSpan(2, 3)),
            SimpleNamespace(op="sub", operation_span=TokenSpan(3, 4)))),
        hidden_states="public-frozen-features", public_inputs=(4, 5), construction_id="held-form")


def model(charts, item):
    owner = SimpleNamespace(operation_length_penalty=1., inference_step_limit=lambda count: 2)
    def public_inventory(tokens, hidden, inputs, limit):
        assert (tokens, hidden, inputs, limit) == ((1, 2, 3, 4), "public-frozen-features", (4, 5), 2)
        return item.ir.input_spans, None, None, iter(charts)
    owner._runtime_operation_charts = public_inventory
    return owner


def test_audit_distinguishes_wrong_operation_ranking_from_missing_proposals():
    item = request()
    charts = ((node("sub", 3, 7.),), (node("add", 2, 3.), node("sub", 3, 3.)))
    row = operation_inventory_audit(model(charts, item), item)
    assert row["correct_count_proposed"] is row["correct_labels_proposed"] is row["correct_spans_proposed"] is True
    assert row["first_correct_label_index"] == 1
    assert row["best_correct_label_operation_margin"] == -2.
    assert row["annotations_used_for_proposals"] is row["backbone_loaded"] is False


def test_audit_does_not_call_count_match_correct_semantics():
    item = request()
    row = operation_inventory_audit(model(((node("sub", 2, 3.), node("add", 3, 3.)),), item), item)
    assert row["correct_count_proposed"] is True
    assert row["correct_labels_proposed"] is row["correct_spans_proposed"] is False
    assert row["first_correct_label_index"] is row["best_correct_label_operation_margin"] is None


def test_same_labels_with_different_spans_remain_separate_from_annotation_coverage():
    item = request()
    row = operation_inventory_audit(model(((node("add", 0, 3.), node("sub", 3, 3.)),), item), item)
    assert row["correct_labels_proposed"] is True
    assert row["correct_spans_proposed"] is False


def test_empty_inventory_is_a_measured_gap_not_a_pass():
    item = request()
    row = operation_inventory_audit(model((), item), item)
    assert row["chart_count"] == 0
    assert row["correct_count_proposed"] is row["correct_labels_proposed"] is False


def test_annotation_is_consulted_only_after_public_inventory_is_exhausted():
    item = request()
    witness = []
    class Source:
        source_token_ids = item.ir.source_token_ids
        source_text_sha256 = item.ir.source_text_sha256
        input_spans = item.ir.input_spans
        @property
        def instructions(self):
            assert witness == ["complete"]
            return item.ir.instructions
    def inventory():
        yield (node("add", 2, 3.), node("sub", 3, 3.))
        witness.append("complete")
    owner = model((), item)
    owner._runtime_operation_charts = lambda *public: (item.ir.input_spans, None, None, inventory())
    supplied = SimpleNamespace(**{**item.__dict__, "ir": Source()})
    assert operation_inventory_audit(owner, supplied)["correct_spans_proposed"] is True


def test_nonfinite_operation_scores_are_not_diagnostic_success():
    item = request()
    with pytest.raises(ValueError, match="finite"):
        operation_inventory_audit(model(((node("add", 2, float("nan")),),), item), item)
