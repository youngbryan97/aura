"""Training/runtime boundary views retain source targets and split ownership."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_runtime_argument_views import (
    align_operation_views,
    runtime_argument_training_views,
)


def _instruction(op, start, end, args=(0, 1)):
    return SimpleNamespace(op=op, operation_span=TokenSpan(start, end), args=args)


def test_alignment_retains_gold_arguments_when_prediction_is_wrong():
    from tests.test_semantic_program_shared_transducer import _examples

    item = _examples()[0]
    gold = item.ir.instructions
    predicted = tuple(replace(i, args=tuple(reversed(i.args))) for i in gold)
    aligned = align_operation_views(gold, predicted)
    assert aligned == gold


def test_alignment_refuses_missing_ambiguous_and_cross_owned_nodes():
    gold = (_instruction("add", 1, 4), _instruction("add", 5, 8))
    assert align_operation_views(gold, ()) is None
    assert align_operation_views(gold, (_instruction("add", 2, 7), _instruction("add", 3, 6))) is None
    assert align_operation_views(gold, (_instruction("sub", 1, 4), _instruction("add", 5, 8))) is None


def test_views_preserve_every_source_and_never_decode_validation_or_test():
    from tests.test_semantic_program_shared_transducer import _examples

    training = tuple(x for x in _examples() if x.split == "train")
    calls = []

    def decode(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ir=None, refusal="test_refusal")

    model = SimpleNamespace(decode=decode, model_basis_sha256=training[0].ir.model_basis_receipt_sha256,
                            receipt_sha256="a" * 64)
    views, receipt = runtime_argument_training_views(model, training)
    assert views == training
    assert len(calls) == len(training)
    assert receipt["coverage"] == {"decode_refused": len(training)}
    assert receipt["predicted_arguments_used_as_targets"] is False
    for split in ("validation", "test"):
        calls.clear()
        with pytest.raises(ValueError, match="source training only"):
            runtime_argument_training_views(model, (replace(training[0], split=split),))
        assert calls == []


def test_augmented_view_changes_boundary_but_keeps_all_target_edges():
    from tests.test_semantic_program_shared_transducer import _examples

    item = next(x for x in _examples() if x.split == "train")
    first, *rest = item.ir.instructions
    span = first.operation_span
    changed = TokenSpan(max(0, span.start - 1), span.end + 1)
    predicted = replace(item.ir, instructions=(
        replace(first, operation_span=changed, args=tuple(reversed(first.args))), *rest,
    ))
    model = SimpleNamespace(
        decode=lambda **kwargs: SimpleNamespace(ir=predicted, refusal=""),
        model_basis_sha256=item.ir.model_basis_receipt_sha256, receipt_sha256="b" * 64,
    )
    views, receipt = runtime_argument_training_views(model, (item,))
    assert receipt["coverage"] == {"augmented": 1}
    assert len(views) == 2
    assert views[0] is item
    assert views[1].ir.instructions[0].operation_span == changed
    assert tuple(i.args for i in views[1].ir.instructions) == tuple(i.args for i in item.ir.instructions)
    assert views[1].ir.input_spans == item.ir.input_spans


def test_source_refit_records_views_and_roundtrips_without_serving_authority():
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
        refit_compositional_argument_rankings,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    candidate = refit_compositional_argument_rankings(model, examples, use_runtime_operation_views=True)
    receipt = candidate.training_receipt["argument_ranking_refit"]["runtime_operation_views"]
    assert receipt["source_examples"] == sum(x.split == "train" for x in examples)
    assert sum(receipt["coverage"].values()) == receipt["source_examples"]
    assert receipt["serving_authority"] is False
    assert compositional_semantic_program_transducer_from_dict(candidate.to_dict()).receipt_sha256 == candidate.receipt_sha256
    assert refit_compositional_argument_rankings(
        model, tuple(x for x in examples if x.split != "test"), use_runtime_operation_views=True,
    ).receipt_sha256 == candidate.receipt_sha256
