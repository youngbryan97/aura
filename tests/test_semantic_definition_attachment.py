"""Learned definition ownership is source-bound and counted once in the graph."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from core.learning import semantic_definition_attachment as attachment
from core.learning.semantic_argument_optimization import optimize_argument_chart
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
    fit_compositional_semantic_program_transducer,
)
from core.learning.semantic_program_corpus import build_semantic_program_corpus
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import _sha
from core.learning.semantic_program_transducer_fitting import RegisterUseContract
from tests.test_semantic_program_shared_transducer import _examples, _grounding


def test_attachment_evidence_is_counted_once_for_repeated_uses():
    name = TokenSpan(10, 11)
    options = ((((1.0, 0, TokenSpan(0, 1)),), ((2.0, 0, TokenSpan(2, 3)),)),)
    result = optimize_argument_chart(
        options,
        n_inputs=1,
        contract=RegisterUseContract(2, 2, 0, 1, False),
        definition_options=(((name,), (name,)),),
        definition_scores={(0, name): 7.0},
    )
    assert result[0] == 10.0


def test_unselected_positive_attachment_cannot_inflate_score():
    left, right = TokenSpan(10, 11), TokenSpan(12, 13)
    options = ((((1.0, 0, TokenSpan(0, 1)), (-100.0, 1, TokenSpan(2, 3))),),)
    result = optimize_argument_chart(
        options,
        n_inputs=2,
        contract=RegisterUseContract(0, 1, 0, 1, False),
        definition_options=(((left, right),),),
        definition_scores={(0, left): 7.0, (1, right): 8.0},
    )
    assert result[0] == 8.0
    assert result[1] == ((0,),)


@pytest.mark.parametrize("scores", [{}, {(0, TokenSpan(10, 11)): float("nan")}])
def test_missing_or_nonfinite_attachment_evidence_is_rejected(scores):
    with pytest.raises(ValueError, match="attachment scores"):
        optimize_argument_chart(
            ((((1.0, 0, TokenSpan(0, 1)),),),),
            n_inputs=1,
            contract=RegisterUseContract(1, 1, 0, 1, False),
            definition_options=(((TokenSpan(10, 11),),),),
            definition_scores=scores,
        )


def test_fronted_input_definition_annotations_do_not_rewrite_the_task():
    old = build_semantic_program_corpus(examples_per_operation_pair=1)
    annotated = build_semantic_program_corpus(
        examples_per_operation_pair=1, annotate_register_definitions=True
    )
    changed = 0
    for before, after in zip(old, annotated, strict=True):
        assert replace(after, register_definition_spans=()) == before
        for anchor, definition in zip(
            after.input_spans, after.register_definition_spans[:3], strict=True
        ):
            if anchor != definition:
                changed += 1
                assert after.source_text[definition.start : definition.end] == "reserved operand"
                assert anchor.end < definition.start
    assert changed > 0


@pytest.fixture(scope="module")
def parent():
    return fit_compositional_semantic_program_transducer(_examples(), input_grounding=_grounding())


def test_refit_roundtrip_lesion_and_runtime_wiring(parent, monkeypatch):
    fitted_rows = []
    original_fit = attachment._fit_binary_head

    def record(features, labels, **kwargs):
        fitted_rows.append(len(labels))
        return original_fit(features, labels, **kwargs)

    monkeypatch.setattr(attachment, "_fit_binary_head", record)
    model = attachment.refit_definition_attachment(parent, _examples())
    fit = model.training_receipt["definition_attachment_fit"]
    assert fitted_rows == [fit["training_rows"]]
    assert fit["training_examples"] == sum(item.split == "train" for item in _examples())
    assert fit["validation_used_for_fit"] is False
    assert fit["test_examples_used"] == 0
    assert fit["objective"] == "conditional_anchor_owner_v2"
    assert fit["training_rows"] == sum(
        (item.ir.n_inputs + len(item.ir.instructions)) ** 2
        for item in _examples()
        if item.split == "train"
    )
    assert (
        compositional_semantic_program_transducer_from_dict(model.to_dict()).to_dict()
        == model.to_dict()
    )
    assert not np.any(model.coefficient_lesion().definition_attachment_head.weight)
    assert np.any(model.definition_attachment_head.weight)
    assert "definition_attachment_head" not in parent.to_dict()
    for key, value in parent._coefficient_body().items():
        assert model._coefficient_body()[key] == value
    calls = []
    original = attachment.attachment_hypotheses

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(attachment, "attachment_hypotheses", capture)
    item = _examples()[0]
    model.decode(
        source_token_ids=item.ir.source_token_ids,
        hidden_states=item.hidden_states,
        public_inputs=item.public_inputs,
        source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=item.ir.model_basis_receipt_sha256,
    )
    assert calls and calls[0][0] and calls[0][1]
    invalid = copy.deepcopy(model.to_dict())
    invalid["training_receipt"]["definition_attachment_fit"]["test_examples_used"] = 1
    body = {k: v for k, v in invalid["training_receipt"].items() if k != "receipt_sha256"}
    invalid["training_receipt"]["receipt_sha256"] = _sha(body)
    with pytest.raises(ValueError):
        compositional_semantic_program_transducer_from_dict(invalid)


def test_duplicate_source_is_rejected_before_attachment_fit(parent, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("invalid cohort reached fit")

    monkeypatch.setattr(attachment, "_fit_binary_head", forbid)
    items = _examples()
    with pytest.raises(ValueError, match="unique disjoint"):
        attachment.refit_definition_attachment(
            parent, (*items, next(x for x in items if x.split == "train"))
        )


def test_owner_probabilities_normalize_across_registers_not_alias_phrases():
    anchors = (TokenSpan(0, 1), TokenSpan(2, 3))
    alias = TokenSpan(4, 5)
    vectors = {
        anchors[0]: np.asarray([1.0, 0.0]),
        anchors[1]: np.asarray([0.0, 1.0]),
        alias: np.asarray([3.0, -2.0]),
    }

    class Head:
        def __init__(self, bias):
            self.bias = bias

        def score(self, definition, anchor):
            return float(np.dot(definition, anchor) + self.bias)

    scores = attachment._owner_scores(Head(0.0), vectors, anchors, conditional=True)
    shifted = attachment._owner_scores(Head(1000.0), vectors, anchors, conditional=True)
    for span in vectors:
        assert sum(np.exp(scores[register, span]) for register in range(2)) == pytest.approx(1.0)
        for register in range(2):
            assert scores[register, span] == pytest.approx(shifted[register, span])
    assert scores[0, alias] > scores[1, alias]
    vectors[TokenSpan(6, 7)] = vectors[alias]
    expanded = attachment._owner_scores(Head(0.0), vectors, anchors, conditional=True)
    assert expanded[0, alias] == scores[0, alias]
