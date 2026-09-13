"""An operation-pointer refit must preserve the other learned mechanisms."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning import semantic_program_compositional_transducer as transducer
from core.learning.semantic_program_ir import TokenSpan


@pytest.fixture
def parent():
    path = Path(__file__).parents[1] / "artifacts/rlc/semantic_program_27b_frozen_path_v1/transducer.json"
    return transducer.compositional_semantic_program_transducer_from_dict(json.loads(path.read_text()))


def _item(parent, split, identity):
    return SimpleNamespace(
        split=split,
        hidden_channels=parent.hidden_channels,
        hidden_channel_widths=parent.hidden_channel_widths,
        tokenizer_identity_sha256=parent.input_grounding.tokenizer_identity_sha256,
        ir=SimpleNamespace(
            model_basis_receipt_sha256=parent.model_basis_sha256,
            source_text_sha256=identity * 64,
            instructions=(SimpleNamespace(operation_span=TokenSpan(2, 3)),),
        ),
    )


def test_refit_uses_training_for_weights_validation_for_length_and_no_test(monkeypatch, parent):
    train, validate, test = (_item(parent, s, i) for s, i in (
        ("train", "a"), ("validation", "b"), ("test", "c"),
    ))
    pointer = replace(parent.operation_pointer, start_bias=parent.operation_pointer.start_bias + 1)

    def fit(examples, *, spans):
        assert examples == (train,)
        assert spans(train) == (TokenSpan(2, 3),)
        return pointer

    def calibrate(examples, **kwargs):
        assert examples == (validate,)
        assert kwargs["pointer"] is pointer
        assert kwargs["classifier"] is parent.operation_head
        return parent.operation_length_penalty + 1, [{"measured": True}]

    monkeypatch.setattr(transducer, "_fit_shared_pointer", fit)
    monkeypatch.setattr(transducer, "_select_operation_length_penalty", calibrate)
    candidate = transducer.refit_compositional_operation_pointer(parent, (test, validate, train))
    before, after = parent._coefficient_body(), candidate._coefficient_body()
    assert {k for k in before if before[k] != after[k]} == {
        "operation_pointer", "operation_length_penalty",
    }
    receipt = candidate.training_receipt["operation_pointer_refit"]
    assert receipt["test_examples_used"] == 0
    assert receipt["training_examples"] == receipt["validation_examples"] == 1
    assert receipt["parent_transducer_receipt_sha256"] == parent.receipt_sha256
    assert not receipt["serving_authority"]
    assert "operation_pointer_refit" not in parent.training_receipt
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert transducer.compositional_semantic_program_transducer_from_dict(candidate.to_dict()).to_dict() == candidate.to_dict()


@pytest.mark.parametrize("problem", ("missing_train", "missing_validation", "overlap", "basis", "tokenizer", "channels"))
def test_refit_rejects_invalid_source_contract_before_fitting(monkeypatch, parent, problem):
    train, validate = _item(parent, "train", "a"), _item(parent, "validation", "b")
    items = [train, validate]
    if problem == "missing_train":
        items = [validate]
    elif problem == "missing_validation":
        items = [train]
    elif problem == "overlap":
        validate.ir.source_text_sha256 = train.ir.source_text_sha256
    elif problem == "basis":
        validate.ir.model_basis_receipt_sha256 = "f" * 64
    elif problem == "tokenizer":
        validate.tokenizer_identity_sha256 = "f" * 64
    else:
        validate.hidden_channel_widths = (1,)

    def forbid(*_args, **_kwargs):
        raise AssertionError("invalid source reached optimization")

    monkeypatch.setattr(transducer, "_fit_shared_pointer", forbid)
    with pytest.raises(ValueError):
        transducer.refit_compositional_operation_pointer(parent, items)
