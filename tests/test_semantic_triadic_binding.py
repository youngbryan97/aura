"""The source-trained triadic factor remains bound to the ordinary proposer."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_program_compositional_refits import (
    refit_compositional_triadic_bindings,
)
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_triadic_binding import (
    TriadicBindingHead,
    fit_triadic_binding_heads,
    triadic_binding_feature,
)
from tests.test_semantic_program_shared_transducer import _examples, _grounding


def test_feature_is_ordered_and_operation_conditioned():
    op = np.asarray([1, 2], dtype=np.float32)
    mention = np.asarray([2, 3], dtype=np.float32)
    definition = np.asarray([3, 5], dtype=np.float32)
    original = triadic_binding_feature(op, mention, definition)
    assert not np.array_equal(original, triadic_binding_feature(op, definition, mention))
    assert not np.array_equal(original, triadic_binding_feature(-op, mention, definition))
    with pytest.raises(ValueError, match="width"):
        triadic_binding_feature(op, mention, definition[:1])


def test_fit_requires_source_only_contrasts():
    examples = _examples()
    training = tuple(item for item in examples if item.split == "train")
    item = training[0]
    with pytest.raises(ValueError, match="source-only"):
        fit_triadic_binding_heads((replace(item, split="test"),), max_arity=2,
            hidden_channels=item.hidden_channels,
            hidden_channel_widths=item.hidden_channel_widths)
    heads, support = fit_triadic_binding_heads(training, max_arity=2,
        hidden_channels=item.hidden_channels,
        hidden_channel_widths=item.hidden_channel_widths)
    assert len(heads) == 2
    assert all(row["positive"] and row["negative"] for row in support["support"])
    assert all(np.isfinite(head.weight).all() for head in heads)


def test_refit_round_trip_and_isolated_lesion(monkeypatch):
    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    training = tuple(item for item in examples if item.split == "train")
    with monkeypatch.context() as patch:
        patch.setattr(type(parent), "decode", lambda self, **kwargs: SimpleNamespace(ir=None, refusal="test"))
        candidate = refit_compositional_triadic_bindings(parent, examples)
    assert candidate.triadic_binding_heads is not None
    assert candidate.training_receipt["triadic_binding_fit"]["serving_authority"] is False
    assert candidate.training_receipt["triadic_binding_fit"]["fit"]["source_count"] == len(training)
    replay = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert replay.receipt_sha256 == candidate.receipt_sha256
    assert replay.triadic_binding_heads is not None
    assert any(np.any(head.weight) for head in replay.triadic_binding_heads)
    lesion = replay.triadic_binding_lesion()
    assert all(not np.any(head.weight) for head in lesion.triadic_binding_heads)
    assert all(not np.any(head.weight) for head in replay.coefficient_lesion().triadic_binding_heads)
    assert all(not np.any(head.weight) for head in replay.relation_lesion().triadic_binding_heads)
    assert lesion.receipt_sha256 != replay.receipt_sha256
    calls = []
    original_score = TriadicBindingHead.score

    def observed_score(self, operation, mention, definition):
        calls.append((operation, mention, definition))
        return original_score(self, operation, mention, definition)

    with monkeypatch.context() as patch:
        patch.setattr(TriadicBindingHead, "score", observed_score)
        sample = next(item for item in examples if item.split == "test")
        replay.decode(
            source_token_ids=sample.ir.source_token_ids,
            hidden_states=sample.hidden_states,
            public_inputs=sample.public_inputs,
            source_text_sha256=sample.ir.source_text_sha256,
            model_basis_sha256=replay.model_basis_sha256,
        )
    assert calls
    with pytest.raises(ValueError, match="unfitted parent"):
        refit_compositional_triadic_bindings(candidate, examples)
    with pytest.raises(ValueError, match="source population"):
        refit_compositional_triadic_bindings(parent, (*examples, training[0]))


def test_head_rejects_nonfinite_coefficients():
    with pytest.raises(ValueError, match="invalid"):
        TriadicBindingHead(np.asarray([float("nan")] * 12), 0.0)
