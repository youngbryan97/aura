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


def test_graph_training_replays_runtime_background_odds_and_gradient(parent):
    from core.learning.semantic_operation_graph_learning import operation_graph_evidence

    model = background.refit_compositional_operation_background(parent, _examples(), background_log_odds=True)
    item = _examples()[0]
    nodes = _operation_nodes(pointer=model.operation_pointer, classifier=model.operation_head,
        hidden=item.hidden_states, input_spans=item.ir.input_spans,
        max_span_tokens=model.max_span_tokens, hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths, label_limit=3, background_log_odds=True)
    parameters = tuple(value.astype(np.float64) for head in model.operation_head.heads
                       for value in (head.weight, head.bias))
    for node, (bank, label) in zip(nodes[:5], operation_graph_evidence(model, item.hidden_states, nodes[:5]), strict=True):
        value, gradient = bank.score_gradient(label, parameters)
        assert value == pytest.approx(node.score, abs=1e-6)
        assert bank.score(label, parameters) == pytest.approx(value)
        rng = np.random.default_rng(42)
        direction = tuple(rng.normal(size=p.shape) for p in parameters)
        epsilon = 1e-5
        plus = tuple(p + epsilon * d for p, d in zip(parameters, direction, strict=True))
        minus = tuple(p - epsilon * d for p, d in zip(parameters, direction, strict=True))
        measured = (bank.score(label, plus) - bank.score(label, minus)) / (2 * epsilon)
        assert measured == pytest.approx(sum(np.sum(g * d) for g, d in zip(gradient, direction, strict=True)), abs=1e-6)


def test_annotated_graph_does_not_reintroduce_pointer_score_absent_from_runtime(parent):
    from core.learning.semantic_joint_graph_learning import score_annotated_graph

    model = background.refit_compositional_operation_background(parent, _examples(), background_log_odds=True)
    model = model.with_joint_definition_graph().with_categorical_relation_scores()
    measured = 0
    for item in _examples():
        outcome = model.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
            public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=model.model_basis_sha256)
        if outcome.ir is None:
            continue
        nodes = _operation_nodes(pointer=model.operation_pointer, classifier=model.operation_head,
            hidden=item.hidden_states, input_spans=outcome.ir.input_spans,
            max_span_tokens=model.max_span_tokens, hidden_channels=model.hidden_channels,
            hidden_channel_widths=model.hidden_channel_widths, label_limit=len(model.operation_head.labels),
            background_log_odds=True)
        scores = {(node.span, node.operation): node.score for node in nodes}
        expected = outcome.pointer_scores["argument_graph_total"] + sum(
            scores[(ins.operation_span, ins.op)] for ins in outcome.ir.instructions)
        graph = score_annotated_graph(model, item, outcome.ir.instructions, outcome.ir.input_spans,
                                      learn_operation_pointer=True)
        assert graph is not None and graph["score"] == pytest.approx(expected, abs=1e-5)
        assert not graph["argument_terms"]
        measured += 1
    assert measured > 0


@pytest.mark.parametrize("log_odds", [False, True])
def test_background_model_runs_the_real_retained_graph_training_loop(parent, log_odds, monkeypatch):
    from core.learning import semantic_joint_graph_learning as learning
    from core.learning.semantic_joint_graph_learning import (
        refit_compositional_joint_graphs, source_operation_constraints, source_operation_supervision,
    )

    examples = _examples()
    model = background.refit_compositional_operation_background(parent, examples, background_log_odds=log_odds)
    model = model.with_joint_definition_graph().with_categorical_relation_scores()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    training = tuple(item for item in examples if item.split == "train")
    supervision = source_operation_supervision(model, training)
    labels = [model.operation_head.labels[index] for index in supervision.labels]
    expected = [label for item in training for _, label in background.operation_background_training_spans(
        item, model.operation_pointer, model.max_span_tokens)]
    assert labels == expected
    assert OPERATION_BACKGROUND_LABEL in labels
    constraints = source_operation_constraints(model, supervision)
    assert len(constraints) == len(labels) * (len(model.operation_head.labels) - 1)
    original = learning.operation_graph_evidence
    source_features = {id(item.hidden_states) for item in training}
    measured = []

    def source_only(candidate, hidden, nodes):
        assert id(hidden) in source_features
        measured.append(id(hidden))
        return original(candidate, hidden, nodes)

    monkeypatch.setattr(learning, "operation_graph_evidence", source_only)
    fitted = refit_compositional_joint_graphs(model, examples, rounds=1, steps=1,
        constraint_learning=True, learn_arguments=True, learn_operation_pointer=True,
        retention_operation_charts=1)
    receipt = fitted.training_receipt["joint_graph_refit"]
    assert receipt["completed_rounds"] == 1
    assert receipt["source_supervision_includes_background"]
    assert receipt["source_operations"] == sum(label != OPERATION_BACKGROUND_LABEL for label in expected)
    assert receipt["source_training_spans"] == len(expected)
    assert set(measured) == source_features
    assert receipt["test_examples_used"] == 0 and not receipt["validation_used_for_fit"]
    assert not receipt["serving_authority"]
    assert fitted.training_receipt["operation_background_fit"] == model.training_receipt["operation_background_fit"]
    assert compositional_semantic_program_transducer_from_dict(fitted.to_dict()).receipt_sha256 == fitted.receipt_sha256
