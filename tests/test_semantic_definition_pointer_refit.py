"""Definition localization must learn anchor-only as well as named registers."""

from dataclasses import replace

import pytest

from core.learning import semantic_program_compositional_transducer as transducer
from core.learning.semantic_program_ir import TokenSpan
from tests.test_semantic_argument_pointer_refit import _item
from tests.test_semantic_argument_pointer_refit import parent as parent_fixture

parent = parent_fixture


def test_refit_includes_anchor_and_symbolic_targets_only_from_training(monkeypatch, parent):
    anchor, symbolic, validation, test = (
        _item(parent, split, identity)
        for split, identity in (("train", "a"), ("train", "b"), ("validation", "c"), ("test", "d"))
    )
    for item in (anchor, symbolic, validation, test):
        item.ir.n_inputs = 1
        item.ir.input_spans = (TokenSpan(0, 1),)
        item.ir.instructions[0].operation_span = TokenSpan(2, 3)
        item.register_definition_spans = ()
    symbolic.register_definition_spans = (TokenSpan(4, 5), TokenSpan(6, 7))
    pointer = replace(parent.definition_pointer, start_bias=parent.definition_pointer.start_bias + 1)

    def fit(examples, *, spans):
        assert examples == (anchor, symbolic)
        assert spans(anchor) == (TokenSpan(0, 1), TokenSpan(2, 3))
        assert spans(symbolic) == symbolic.register_definition_spans
        return pointer

    monkeypatch.setattr(transducer, "_fit_shared_pointer", fit)
    result = transducer.refit_compositional_definition_pointer(parent, (anchor, symbolic, test, validation))
    before, after = parent._coefficient_body(), result._coefficient_body()
    assert {key for key in before if before[key] != after[key]} == {"definition_pointer"}
    receipt = result.training_receipt["definition_pointer_refit"]
    assert receipt["training_examples"] == 2
    assert receipt["test_examples_used"] == receipt["validation_examples_used_for_fitting"] == 0
    assert not receipt["serving_authority"]
    assert result.definition_relation_head is parent.definition_relation_head
    assert transducer.compositional_semantic_program_transducer_from_dict(result.to_dict()).to_dict() == result.to_dict()


@pytest.mark.parametrize("problem", ("train", "validation", "overlap", "basis"))
def test_invalid_source_is_rejected_before_fitting(monkeypatch, parent, problem):
    train, validation = _item(parent, "train", "a"), _item(parent, "validation", "b")
    items = [train, validation]
    if problem in {"train", "validation"}:
        items = [item for item in items if item.split != problem]
    elif problem == "overlap":
        validation.ir.source_text_sha256 = train.ir.source_text_sha256
    else:
        validation.ir.model_basis_receipt_sha256 = "f" * 64

    def forbid(*args, **kwargs):
        raise AssertionError("invalid source reached fitting")

    monkeypatch.setattr(transducer, "_fit_shared_pointer", forbid)
    with pytest.raises(ValueError):
        transducer.refit_compositional_definition_pointer(parent, items)
