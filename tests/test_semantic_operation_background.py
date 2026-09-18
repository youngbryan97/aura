"""Operation/background competition is learned, source-bound and executable."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from core.learning import semantic_operation_background as background
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_transducer import (
    OPERATION_BACKGROUND_LABEL,
    LinearClassifierHead,
    MultiViewClassifierHead,
    _sha,
)
from core.learning.semantic_program_transducer_fitting import _operation_nodes
from tests.test_semantic_program_shared_transducer import _examples, _grounding


@pytest.fixture(scope="module")
def parent():
    return fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())


@pytest.fixture(scope="module")
def candidate(parent):
    return background.refit_compositional_operation_background(parent, _examples())


def test_source_supervision_keeps_positives_and_excludes_direct_inputs(parent):
    for item in _examples():
        rows = background.operation_background_training_spans(item, parent.operation_pointer, parent.max_span_tokens)
        positives = {(instruction.operation_span, instruction.op) for instruction in item.ir.instructions}
        assert {row for row in rows if row[1] != OPERATION_BACKGROUND_LABEL} == positives
        negatives = [span for span, label in rows if label == OPERATION_BACKGROUND_LABEL]
        assert negatives
        assert all(span not in {one[0] for one in positives} for span in negatives)
        assert all(not (span.start < direct.end and direct.start < span.end)
                   for span in negatives for direct in item.ir.input_spans)


def test_background_penalizes_without_removing_operation_alternatives(parent):
    item = _examples()[0]
    width = parent.operation_head.heads[0].width
    def nodes(background_bias, limit):
        head = LinearClassifierHead((OPERATION_BACKGROUND_LABEL, "add", "sub"),
                                    np.zeros((3, width)), np.array([background_bias, 2., 1.]))
        return _operation_nodes(
            pointer=parent.operation_pointer,
            classifier=MultiViewClassifierHead(parent.operation_head.modes, (head,)),
            hidden=item.hidden_states, input_spans=item.ir.input_spans,
            max_span_tokens=parent.max_span_tokens, hidden_channels=parent.hidden_channels,
            hidden_channel_widths=parent.hidden_channel_widths, label_limit=limit,
        )
    low, high = nodes(-10., 1), nodes(10., 1)
    assert low and len(low) == len(high)
    assert [(node.operation, node.span) for node in low] == [(node.operation, node.span) for node in high]
    assert all(before.score > after.score for before, after in zip(low, high))
    assert {node.operation for node in nodes(10., 3)} == {"add", "sub"}


def test_fit_never_reads_held_out_features(parent, monkeypatch):
    examples = _examples()
    source = {id(item.hidden_states) for item in examples if item.split == "train"}
    original = background._operation_feature
    seen = []
    def capture(hidden, *args, **kwargs):
        assert id(hidden) in source
        seen.append(id(hidden))
        return original(hidden, *args, **kwargs)
    monkeypatch.setattr(background, "_operation_feature", capture)
    fitted = background.refit_compositional_operation_background(parent, examples)
    assert set(seen) == source
    record = fitted.training_receipt["operation_background_fit"]
    assert record["validation_used_for_fit"] is False and record["test_examples_used"] == 0


def test_receipt_roundtrip_and_lesion_cover_background(candidate, parent):
    assert candidate.receipt_sha256 != parent.receipt_sha256
    before, after = parent._coefficient_body(), candidate._coefficient_body()
    assert {key for key in before if before[key] != after[key]} == {"operation_head"}
    restored = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert restored.to_dict() == candidate.to_dict()
    assert OPERATION_BACKGROUND_LABEL in restored.operation_head.labels
    lesion = restored.coefficient_lesion()
    assert all(not np.count_nonzero(head.weight) for head in lesion.operation_head.heads)
    item = _examples()[0]
    result = restored.decode(
        source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
    )
    assert result.ir is not None
    assert all(instruction.op != OPERATION_BACKGROUND_LABEL for instruction in result.ir.instructions)


@pytest.mark.parametrize("defect", ["missing", "labels", "score", "test", "count", "hash"])
def test_corrupt_background_contract_rejected(candidate, defect):
    payload = copy.deepcopy(candidate.to_dict())
    receipt = payload["training_receipt"]
    record = receipt["operation_background_fit"]
    if defect == "missing":
        del receipt["operation_background_fit"]
    elif defect == "labels":
        record["labels"] = ["add"]
    elif defect == "score":
        record["score"] = "hard_reject"
    elif defect == "test":
        record["test_examples_used"] = 1
    elif defect == "count":
        record["background_spans"] = True
    else:
        record["targets_sha256"] = "z" * 64
    receipt["receipt_sha256"] = _sha({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError):
        compositional_semantic_program_transducer_from_dict(payload)


def test_duplicate_or_cross_split_source_rejected_before_fit(parent, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("invalid source reached fit")
    monkeypatch.setattr(background, "_fit_classifier", forbid)
    examples = _examples()
    item = next(item for item in examples if item.split == "train")
    for extra in (item, replace(item, split="validation")):
        with pytest.raises(ValueError, match="unique disjoint"):
            background.refit_compositional_operation_background(parent, (*examples, extra))


def test_background_odds_are_one_joint_score_not_two_evidence_scales(parent):
    item = _examples()[0]
    width = parent.operation_head.heads[0].width
    head = LinearClassifierHead((OPERATION_BACKGROUND_LABEL, "add", "sub"),
                                np.zeros((3, width)), np.array([0., 2., 1.]))
    nodes = _operation_nodes(pointer=parent.operation_pointer,
        classifier=MultiViewClassifierHead(parent.operation_head.modes, (head,)),
        hidden=item.hidden_states, input_spans=item.ir.input_spans,
        max_span_tokens=parent.max_span_tokens, hidden_channels=parent.hidden_channels,
        hidden_channel_widths=parent.hidden_channel_widths, label_limit=3, background_log_odds=True)
    assert nodes and len({node.pointer_score for node in nodes}) > 1
    assert all(node.score == pytest.approx(2. if node.operation == "add" else 1.) for node in nodes)


def test_joint_background_policy_survives_export_and_reaches_decode(parent, monkeypatch):
    model = background.refit_compositional_operation_background(parent, _examples(), background_log_odds=True)
    assert model.operation_length_penalty == 0.
    restored = compositional_semantic_program_transducer_from_dict(model.to_dict())
    import core.learning.semantic_program_compositional_transducer as runtime
    original = runtime._operation_nodes
    seen = []
    def capture(**kwargs):
        seen.append(kwargs["background_log_odds"])
        return original(**kwargs)
    monkeypatch.setattr(runtime, "_operation_nodes", capture)
    item = _examples()[0]
    restored.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256)
    assert seen == [True]
