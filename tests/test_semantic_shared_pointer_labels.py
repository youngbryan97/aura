"""Shared pointer targets must not contradict another target of the same head."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning import semantic_program_transducer_fitting as fitting
from core.learning.semantic_program_ir import TokenSpan


def _example():
    instructions = tuple(
        SimpleNamespace(
            operation_span=TokenSpan(start, start + 2),
            argument_spans=(TokenSpan(start + 2, start + 3), TokenSpan(start + 4, start + 6)),
        )
        for start in (5, 12, 21)
    )
    return SimpleNamespace(
        hidden_states=np.eye(32, dtype=np.float32),
        ir=SimpleNamespace(
            n_inputs=2,
            input_spans=(TokenSpan(1, 2), TokenSpan(3, 4)),
            instructions=instructions,
        ),
    )


@pytest.mark.parametrize("kind", ("operation", "argument", "definition"))
def test_same_head_positive_boundaries_are_never_negative_labels(monkeypatch, kind):
    item = _example()
    if kind == "operation":
        spans = tuple(x.operation_span for x in item.ir.instructions)
    elif kind == "argument":
        spans = tuple(s for x in item.ir.instructions for s in x.argument_spans)
    else:
        spans = (*item.ir.input_spans, *(x.operation_span for x in item.ir.instructions))
    observed = []

    def capture(features, labels, **kwargs):
        indices = np.argmax(features, axis=1)
        positives = set(indices[labels == 1])
        negatives = set(indices[labels == 0])
        assert not positives & negatives, f"contradictory targets: {positives & negatives}"
        assert negatives
        assert np.isclose(np.sum(kwargs["sample_weight"]), len(labels))
        observed.append(positives)
        return np.zeros(features.shape[1], dtype=np.float32), 0.0

    monkeypatch.setattr(fitting, "_fit_binary_head", capture)
    fitting._fit_shared_pointer((item,), spans=lambda _item: spans)
    assert observed == [{s.start for s in spans}, {s.end - 1 for s in spans}]


def test_single_target_keeps_other_semantic_roles_as_hard_negatives(monkeypatch):
    item = _example()
    span = item.ir.instructions[0].operation_span
    negatives = []

    def capture(features, labels, **_kwargs):
        negatives.append(set(np.argmax(features[labels == 0], axis=1)))
        return np.zeros(features.shape[1], dtype=np.float32), 0.0

    monkeypatch.setattr(fitting, "_fit_binary_head", capture)
    fitting._fit_shared_pointer((item,), spans=lambda _item: (span,))
    assert {1, 3, 12, 21} <= negatives[0]
    assert {1, 3, 13, 22} <= negatives[1]
