"""Learned operation ambiguity survives into a source-bound typed chart."""

import copy

import numpy as np
import pytest

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_transducer import LinearClassifierHead, MultiViewClassifierHead, _sha
from core.learning.semantic_program_transducer_fitting import _operation_nodes, _operation_chart_candidates
from tests.test_semantic_program_shared_transducer import _examples, _grounding


@pytest.fixture(scope="module")
def parent():
    return fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())


def test_distribution_preserves_multiview_average_and_legacy_winner():
    heads = tuple(LinearClassifierHead(("add", "sub"), np.zeros((2, 1)), np.array(bias))
                  for bias in ((2.0, 0.0), (0.0, 1.0)))
    model = MultiViewClassifierHead(("contextual_mean", "contextual_last"), heads)
    features = (np.array([1.0]), np.array([2.0]))
    expected = np.mean([head.predict_probabilities(feature) for head, feature in zip(heads, features)], axis=0)
    assert np.array_equal(model.predict_probabilities(features), expected)
    assert model.predict(features) == (model.labels[np.argmax(expected)], float(max(expected)))


def test_alternative_labels_reach_chart_without_overlapping_themselves(parent):
    item = _examples()[0]
    kwargs = dict(pointer=parent.operation_pointer, classifier=parent.operation_head,
                  hidden=item.hidden_states, input_spans=item.ir.input_spans,
                  max_span_tokens=parent.max_span_tokens, hidden_channels=parent.hidden_channels,
                  hidden_channel_widths=parent.hidden_channel_widths)
    legacy = _operation_nodes(**kwargs)
    expanded = _operation_nodes(**kwargs, label_limit=2)
    assert len(expanded) == 2 * len(legacy)
    assert expanded[::2] == legacy
    for first, second in zip(expanded[::2], expanded[1::2]):
        assert first.span == second.span
        assert first.operation != second.operation
        assert first.score >= second.score
    charts = _operation_chart_candidates(expanded, max_steps=2, length_penalty=0, limit=16)
    assert charts
    for chart in charts:
        assert len({node.span for node in chart}) == len(chart)


def test_candidate_identity_roundtrip_and_runtime_use(parent, monkeypatch):
    model = parent.with_operation_label_alternatives(2)
    assert model.receipt_sha256 != parent.receipt_sha256
    assert "operation_label_limit" not in parent.training_receipt
    restored = compositional_semantic_program_transducer_from_dict(model.to_dict())
    assert restored.receipt_sha256 == model.receipt_sha256
    import core.learning.semantic_program_compositional_transducer as runtime
    original = runtime._operation_nodes
    observed = []

    def capture(**kwargs):
        observed.append(kwargs["label_limit"])
        return original(**kwargs)

    monkeypatch.setattr(runtime, "_operation_nodes", capture)
    item = _examples()[0]
    restored.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
                    public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
                    model_basis_sha256=item.ir.model_basis_receipt_sha256)
    assert observed == [2]


@pytest.mark.parametrize("value", [0, True, -1, 1000, "2"])
def test_invalid_policy_is_rejected_even_with_recomputed_receipt(parent, value):
    payload = copy.deepcopy(parent.to_dict())
    receipt = payload["training_receipt"]
    receipt["operation_label_limit"] = value
    receipt["receipt_sha256"] = _sha({key: val for key, val in receipt.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError):
        compositional_semantic_program_transducer_from_dict(payload)
