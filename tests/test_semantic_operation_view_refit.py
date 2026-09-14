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
