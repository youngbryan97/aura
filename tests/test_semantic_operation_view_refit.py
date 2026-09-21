"""Operation views retain source split, tensor, serialization and lesion contracts."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from core.learning import semantic_operation_view_refit as refit
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_transducer import MultiViewClassifierHead, _sha
from core.learning.semantic_program_transducer_fitting import _operation_nodes
from tests.test_semantic_program_shared_transducer import _examples, _grounding


@pytest.fixture(scope="module")
def parent():
    return fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())


def two_views(parent):
    head = MultiViewClassifierHead(("contextual_mean", "contextual_last"), parent.operation_head.heads * 2)
    coefficient = parent._coefficient_body()
    coefficient["operation_head"] = head.to_dict()
    body = {k: v for k, v in parent.training_receipt.items() if k != "receipt_sha256"}
    body["coefficient_sha256"] = _sha(coefficient)
    body["operation_view_selection"] = {
        "schema": "aura.semantic_operation_view_selection.v1",
        "modes": list(head.modes), "objective": "source_validation_accuracy_then_cross_entropy_v1",
        "validation_used_for_fit": False, "test_examples_used": 0, "serving_authority": False,
        "candidates": [{"modes": list(head.modes), "selected": True, "correct": 1, "total": 1, "cross_entropy": 0.1}],
    }
    return replace(parent, operation_head=head, training_receipt={**body, "receipt_sha256": _sha(body)})


def test_multiview_roundtrip_and_lesion_cover_every_head(parent):
    model = two_views(parent)
    reloaded = compositional_semantic_program_transducer_from_dict(model.to_dict())
    assert reloaded.to_dict() == model.to_dict()
    lesion = reloaded.coefficient_lesion()
    assert lesion.operation_head.modes == model.operation_head.modes
    assert len(lesion.operation_head.heads) == 2
    assert all(np.count_nonzero(head.weight) == 0 for head in lesion.operation_head.heads)
    assert all(np.count_nonzero(head.weight) for head in model.operation_head.heads)


def test_runtime_supplies_all_declared_operation_views(parent, monkeypatch):
    model = two_views(parent)
    original = MultiViewClassifierHead.predict
    calls = []

    def capture(self, features):
        assert len(features) == len(self.modes) == 2
        calls.append(features)
        return original(self, features)

    monkeypatch.setattr(MultiViewClassifierHead, "predict", capture)
    item = _examples()[0]
    assert _operation_nodes(
        pointer=model.operation_pointer, classifier=model.operation_head,
        hidden=item.hidden_states, input_spans=item.ir.input_spans,
        max_span_tokens=model.max_span_tokens, hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
    )
    assert calls


@pytest.mark.parametrize("defect", ["missing", "modes", "test", "width", "nonfinite"])
def test_incompatible_view_receipts_and_geometry_are_rejected(parent, defect):
    payload = copy.deepcopy(two_views(parent).to_dict())
    body = payload["training_receipt"]
    if defect == "missing":
        del body["operation_view_selection"]
    elif defect == "modes":
        body["operation_view_selection"]["modes"] = ["contextual_mean"]
    elif defect == "test":
        body["operation_view_selection"]["test_examples_used"] = 1
    elif defect == "nonfinite":
        body["operation_view_selection"]["candidates"][0]["cross_entropy"] = float("inf")
        assert not refit.valid_operation_view_contract(
            two_views(parent).operation_head, body, parent.hidden_channels, parent.hidden_channel_widths,
        )
        with pytest.raises(ValueError):
            _sha(body)
        return
    else:
        for row in payload["operation_head"]["heads"][1]["weight"]:
            row.append(0.0)
        coefficient = two_views(parent)._coefficient_body()
        coefficient["operation_head"] = payload["operation_head"]
        body["coefficient_sha256"] = _sha(coefficient)
    body["receipt_sha256"] = _sha({k: v for k, v in body.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError):
        compositional_semantic_program_transducer_from_dict(payload)


def test_refit_uses_training_weights_and_validation_selection(parent, monkeypatch):
    examples = _examples()
    expected_rows = sum(len(item.ir.instructions) for item in examples if item.split == "train")
    original = refit._fit_classifier
    calls = []

    def capture(features, labels, **kwargs):
        assert len(features) == len(labels) == expected_rows
        calls.append(features.shape)
        return original(features, labels, **kwargs)

    monkeypatch.setattr(refit, "_fit_classifier", capture)
    model = refit.refit_compositional_operation_views(parent, examples)
    assert calls
    before, after = parent._coefficient_body(), model._coefficient_body()
    assert {k for k in before if before[k] != after[k]} <= {"operation_head", "operation_length_penalty"}
    receipt = model.training_receipt["operation_view_selection"]
    assert all("contextual_span_request_interaction" not in row["modes"] for row in receipt["candidates"])
    assert receipt["validation_used_for_fit"] is False
    assert receipt["test_examples_used"] == 0
    assert sum(row["selected"] for row in receipt["candidates"]) == 1
    assert compositional_semantic_program_transducer_from_dict(model.to_dict()).receipt_sha256 == model.receipt_sha256


def test_duplicate_source_rejected_before_fitting(parent, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("duplicate source reached fit")

    monkeypatch.setattr(refit, "_fit_classifier", forbid)
    examples = _examples()
    duplicate = next(item for item in examples if item.split == "train")
    with pytest.raises(ValueError, match="unique disjoint"):
        refit.refit_compositional_operation_views(parent, (*examples, duplicate))


@pytest.mark.parametrize("defect", ["losing_winner", "duplicate", "different_total", "negative_entropy", "unknown_view"])
def test_selection_receipt_checks_the_competing_views(parent, defect):
    model = two_views(parent)
    body = copy.deepcopy(model.training_receipt)
    rows = body["operation_view_selection"]["candidates"]
    other = {"modes": ["contextual_mean"], "selected": False, "correct": 0, "total": 1, "cross_entropy": 1.0}
    if defect == "losing_winner":
        other.update(correct=1, cross_entropy=0.0)
    elif defect == "duplicate":
        other["modes"] = rows[0]["modes"]
    elif defect == "different_total":
        other["total"] = 2
    elif defect == "negative_entropy":
        other["cross_entropy"] = -1
    else:
        other["modes"] = ["undeclared_channel"]
    rows.append(other)
    assert not refit.valid_operation_view_contract(model.operation_head, body, model.hidden_channels, model.hidden_channel_widths)


def test_legacy_combined_view_candidates_are_supported(parent):
    model = two_views(parent)
    body = copy.deepcopy(model.training_receipt)
    body["operation_view_selection"]["candidates"].append({
        "modes": ["lexical_mean_contextual_last"], "selected": False,
        "correct": 0, "total": 1, "cross_entropy": 1.0,
    })
    assert refit.valid_operation_view_contract(model.operation_head, body, model.hidden_channels, model.hidden_channel_widths)


@pytest.mark.parametrize("log_odds", [False, True])
@pytest.mark.parametrize("mode", ["contextual_span_request_interaction", "contextual_mean_transition"])
def test_new_view_preserves_background_supervision_and_runtime_scores(parent, monkeypatch, log_odds, mode):
    from core.learning import semantic_program_transducer_fitting as fitting
    from core.learning.semantic_operation_background import refit_compositional_operation_background
    from core.learning.semantic_program_transducer import OPERATION_BACKGROUND_LABEL

    examples = _examples()
    base = refit_compositional_operation_background(parent, examples, background_log_odds=log_odds)
    original = fitting._calibrate_operation_charts
    observations = []
    def capture(cached):
        for _item, charts in cached:
            for _score, nodes in charts:
                for node in nodes:
                    assert node.operation != OPERATION_BACKGROUND_LABEL
                    observations.append(node)
        return original(cached)
    monkeypatch.setattr(fitting, "_calibrate_operation_charts", capture)
    fitted = refit.refit_compositional_operation_views(base, examples, candidate_modes=(mode,))
    assert observations
    assert fitted.operation_head.modes == (mode,)
    assert fitted.operation_head.labels == base.operation_head.labels
    record = fitted.training_receipt["operation_background_fit"]
    old = base.training_receipt["operation_background_fit"]
    for field in ("targets_sha256", "score", "positive_spans", "background_spans", "training_ids_sha256"):
        assert record[field] == old[field]
    assert record["parent_transducer_receipt_sha256"] == base.receipt_sha256
    assert record["modes"] == [mode]
    assert compositional_semantic_program_transducer_from_dict(fitted.to_dict()).to_dict() == fitted.to_dict()
    runtime_nodes = []
    for item in (item for item in examples if item.split == "validation"):
        runtime_nodes.extend(_operation_nodes(pointer=fitted.operation_pointer,
            classifier=fitted.operation_head, hidden=item.hidden_states,
            input_spans=item.ir.input_spans, max_span_tokens=fitted.max_span_tokens,
            hidden_channels=fitted.hidden_channels, hidden_channel_widths=fitted.hidden_channel_widths,
            background_log_odds=log_odds))
    assert all(any(node == runtime for runtime in runtime_nodes) for node in observations)


@pytest.mark.parametrize("modes", [(), ("unknown",), ("contextual_mean", "contextual_mean"), "contextual_mean"])
def test_invalid_candidate_inventory_refuses_before_fitting(parent, monkeypatch, modes):
    monkeypatch.setattr(refit, "_fit_classifier", lambda *a, **k: pytest.fail("invalid inventory reached fit"))
    with pytest.raises(ValueError, match="unique supported"):
        refit.refit_compositional_operation_views(parent, _examples(), candidate_modes=modes)


@pytest.mark.parametrize("mode", ["contextual_span_request_interaction", "contextual_mean_transition"])
def test_request_view_fits_only_training_features_and_excludes_test(parent, monkeypatch, mode):
    examples = _examples()
    expected = np.stack([refit._operation_feature(item.hidden_states, ins.operation_span,
        mode=mode, hidden_channels=parent.hidden_channels, hidden_channel_widths=parent.hidden_channel_widths)
        for item in examples if item.split == "train" for ins in item.ir.instructions])
    original_fit, original_feature = refit._fit_classifier, refit._operation_feature
    seen = []
    def fit(features, labels, **kwargs):
        np.testing.assert_array_equal(features, expected)
        seen.append(len(labels))
        return original_fit(features, labels, **kwargs)
    forbidden = {id(item.hidden_states) for item in examples if item.split == "test"}
    def feature(hidden, *args, **kwargs):
        assert id(hidden) not in forbidden
        return original_feature(hidden, *args, **kwargs)
    monkeypatch.setattr(refit, "_fit_classifier", fit)
    monkeypatch.setattr(refit, "_operation_feature", feature)
    model = refit.refit_compositional_operation_views(parent, examples, candidate_modes=(mode,))
    assert seen == [len(expected)]
    from core.learning.semantic_operation_graph_learning import operation_graph_evidence
    item = examples[0]
    nodes = _operation_nodes(pointer=model.operation_pointer, classifier=model.operation_head,
        hidden=item.hidden_states, input_spans=item.ir.input_spans,
        max_span_tokens=model.max_span_tokens, hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths)
    parameters = tuple(value.astype(np.float64) for head in model.operation_head.heads
                       for value in (head.weight, head.bias))
    for node, (bank, label) in zip(nodes[:3], operation_graph_evidence(model, item.hidden_states, nodes[:3]), strict=True):
        value, gradient = bank.score_gradient(label, parameters)
        assert value + node.pointer_score == pytest.approx(node.score, abs=1e-6)
        direction = tuple(np.random.default_rng(12).normal(size=p.shape) for p in parameters)
        epsilon = 1e-5
        plus = tuple(p + epsilon * d for p, d in zip(parameters, direction, strict=True))
        minus = tuple(p - epsilon * d for p, d in zip(parameters, direction, strict=True))
        measured = (bank.score(label, plus) - bank.score(label, minus)) / (2 * epsilon)
        assert measured == pytest.approx(sum(np.sum(g * d) for g, d in zip(gradient, direction, strict=True)), abs=1e-6)
