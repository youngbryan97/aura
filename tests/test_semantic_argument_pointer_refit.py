"""Pointer-dependent proposal training must follow the refitted candidate surface."""

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
            instructions=(SimpleNamespace(argument_spans=(TokenSpan(2, 3), TokenSpan(5, 6))),),
        ),
    )


@pytest.mark.parametrize("refit_pointer", (False, True))
def test_proposals_and_calibration_use_the_selected_pointer(monkeypatch, parent, refit_pointer):
    train, validate, test = (_item(parent, s, i) for s, i in (
        ("train", "a"), ("validation", "b"), ("test", "c"),
    ))
    pointer = replace(parent.argument_pointer, start_bias=parent.argument_pointer.start_bias + 1)
    expected = pointer if refit_pointer else parent.argument_pointer
    heads = tuple(replace(head, bias=head.bias + 1) for head in parent.argument_proposal_heads)
    calls = []

    def fit_pointer(examples, *, spans):
        assert refit_pointer
        assert examples == (train,)
        assert spans(train) == (TokenSpan(2, 3), TokenSpan(5, 6))
        calls.append("pointer")
        return pointer

    def fit_proposals(examples, **kwargs):
        assert examples == (train,)
        assert kwargs["argument_pointer"] is expected
        calls.append("proposals")
        return heads, {"positive_rows": 2, "pointer_hard_negative_rows": 3}

    def calibrate(examples, **kwargs):
        assert examples == (validate,)
        assert kwargs["argument_pointer"] is expected
        assert kwargs["proposal_heads"] is heads
        assert kwargs["semantic_heads"] is parent.argument_role_heads
        calls.append("calibration")
        scale = parent.argument_proposal_scale + 1
        return scale, [{"selected": True, "proposal_scale": scale}]

    monkeypatch.setattr(transducer, "_fit_shared_pointer", fit_pointer)
    monkeypatch.setattr(transducer, "_fit_argument_proposal_heads", fit_proposals)
    monkeypatch.setattr(transducer, "_select_argument_proposal_scale", calibrate)
    candidate = transducer.refit_compositional_argument_proposals(
        parent, (test, validate, train), refit_pointer=refit_pointer,
    )
    assert calls == (["pointer"] if refit_pointer else []) + ["proposals", "calibration"]
    before, after = parent._coefficient_body(), candidate._coefficient_body()
    changed = {"argument_proposal_heads", "argument_proposal_scale"}
    if refit_pointer:
        changed.add("argument_pointer")
        receipt = candidate.training_receipt["argument_pointer_refit"]
        assert receipt["test_examples_used"] == 0
        assert receipt["training_examples"] == receipt["validation_examples"] == 1
        assert receipt["parent_transducer_receipt_sha256"] == parent.receipt_sha256
        assert receipt["dependent_proposal_heads_refitted"]
        assert not receipt["serving_authority"]
    else:
        assert "argument_pointer_refit" not in candidate.training_receipt
    assert {k for k in before if before[k] != after[k]} == changed
    assert "argument_pointer_refit" not in parent.training_receipt
    assert candidate.receipt_sha256 != parent.receipt_sha256
    assert transducer.compositional_semantic_program_transducer_from_dict(candidate.to_dict()).to_dict() == candidate.to_dict()


@pytest.mark.parametrize("problem", ("missing_train", "missing_validation", "overlap", "basis", "tokenizer", "channels"))
def test_invalid_source_cannot_reach_pointer_fitting(monkeypatch, parent, problem):
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
        transducer.refit_compositional_argument_proposals(parent, items, refit_pointer=True)


def test_real_pointer_refit_is_deterministic_and_excludes_test_data():
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = transducer.fit_compositional_semantic_program_transducer(
        examples, input_grounding=_grounding(),
    )
    candidate = transducer.refit_compositional_argument_proposals(
        parent, examples, refit_pointer=True,
    )
    repeated = transducer.refit_compositional_argument_proposals(
        parent, tuple(x for x in examples if x.split != "test"), refit_pointer=True,
    )
    assert candidate.receipt_sha256 == repeated.receipt_sha256
    # A clean full fit already uses the repaired sampler and proposal surface.
    assert candidate._coefficient_body() == parent._coefficient_body()
    assert candidate.training_receipt["argument_pointer_refit"]["test_examples_used"] == 0
