"""Training/runtime boundary views retain source targets and split ownership."""

from dataclasses import replace
from itertools import islice
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


def test_answer_blind_chart_views_keep_source_roles_and_deduplicate_boundaries():
    from tests.test_semantic_program_shared_transducer import _examples

    item = next(x for x in _examples() if x.split == "train")
    first, *rest = item.ir.instructions
    changed = TokenSpan(max(0, first.operation_span.start - 1), first.operation_span.end + 1)
    matching = tuple(SimpleNamespace(operation=instruction.op,
                                     span=changed if index == 0 else instruction.operation_span)
                     for index, instruction in enumerate(item.ir.instructions))
    wrong = tuple(SimpleNamespace(operation="sub" if instruction.op != "sub" else "add",
                                  span=instruction.operation_span)
                  for instruction in item.ir.instructions)
    calls = []

    def charts(*args):
        calls.append(args)
        return item.ir.input_spans, (), None, iter((matching, matching, wrong))

    model = SimpleNamespace(
        _runtime_operation_charts=charts,
        inference_step_limit=lambda n: len(item.ir.instructions),
        decode=lambda **kwargs: pytest.fail("chart views must not use selected decode"),
        receipt_sha256="d" * 64,
    )
    views, receipt = runtime_argument_training_views(
        model, (item,), max_operation_charts=3,
    )
    assert len(calls) == 1
    assert len(views) == 2
    assert views[0] is item
    assert views[1].ir.instructions[0].operation_span == changed
    assert tuple(i.args for i in views[1].ir.instructions) == tuple(
        i.args for i in item.ir.instructions
    )
    assert receipt["rows"][0]["charts_examined"] == 3
    assert receipt["rows"][0]["charts_aligned"] == 2
    assert receipt["rows"][0]["views_added"] == 1
    assert receipt["predicted_arguments_used_as_targets"] is False


def test_chart_views_reject_nontraining_before_search():
    from tests.test_semantic_program_shared_transducer import _examples

    item = next(x for x in _examples() if x.split == "train")
    model = SimpleNamespace(_runtime_operation_charts=lambda *args: pytest.fail("searched holdout"))
    with pytest.raises(ValueError, match="source training only"):
        runtime_argument_training_views(
            model, (replace(item, split="validation"),), max_operation_charts=8,
        )


def test_chart_views_remap_equal_valued_inputs_by_source_anchor():
    from tests.test_semantic_program_shared_transducer import _examples

    item = next(x for x in _examples() if x.split == "train")
    equal = replace(item, public_inputs=(item.public_inputs[0], item.public_inputs[0],
                                         *item.public_inputs[2:]))
    swapped = (item.ir.input_spans[1], item.ir.input_spans[0], *item.ir.input_spans[2:])
    nodes = tuple(SimpleNamespace(operation=i.op, span=i.operation_span)
                  for i in item.ir.instructions)
    model = SimpleNamespace(
        _runtime_operation_charts=lambda *args: (swapped, (), None, iter((nodes,))),
        inference_step_limit=lambda n: len(nodes), receipt_sha256="e" * 64,
    )
    views, receipt = runtime_argument_training_views(model, (equal,), max_operation_charts=1)
    assert len(views) == 2
    assert views[1].ir.input_spans == swapped
    assert views[1].ir.instructions[0].args == (1, 0)
    assert receipt["rows"][0]["source_to_runtime_input_registers"] == [1, 0, 2]


def test_runtime_chart_views_balance_sources_not_hypothesis_counts():
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )
    from core.learning.semantic_program_transducer_fitting import _argument_proposal_rows
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    first, second = islice((item for item in examples if item.split == "train" and
                            len(item.ir.instructions) == len(examples[0].ir.instructions)), 2)
    changed = replace(first.ir.instructions[0], operation_span=TokenSpan(
        max(0, first.ir.instructions[0].operation_span.start - 1),
        first.ir.instructions[0].operation_span.end + 1,
    ))
    extra = replace(first, ir=replace(first.ir, instructions=(changed, *first.ir.instructions[1:])))
    model = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    _, labels, weights, _, _ = _argument_proposal_rows(
        (first, extra, second), argument_pointer=model.argument_pointer, position=0,
        max_span_tokens=model.max_span_tokens,
        max_argument_span_tokens_by_type=model.max_argument_span_tokens_by_type,
        hidden_channels=model.hidden_channels,
        hidden_channel_widths=model.hidden_channel_widths,
        include_semantic_negatives=True, balance_source_views=True,
    )
    starts = [index for index, label in enumerate(labels) if label == 1]
    group_weights = [sum(weights[start:stop]) for start, stop in zip(
        starts, (*starts[1:], len(labels)), strict=True,
    )]
    steps = len(first.ir.instructions)
    assert sum(group_weights[:2 * steps]) == pytest.approx(sum(group_weights[2 * steps:]))


def test_chart_view_refit_reaches_the_ordinary_graph_selector(monkeypatch):
    import numpy as np

    from core.learning import semantic_program_compositional_transducer as transducer
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
        refit_compositional_argument_rankings,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    by_hidden = {id(item.hidden_states): item for item in examples if item.split == "train"}
    original_charts = type(parent)._runtime_operation_charts

    def source_charts(self, tokens, hidden, inputs, limit):
        item = by_hidden[id(hidden)]
        nodes = tuple(SimpleNamespace(
            operation=instruction.op,
            span=TokenSpan(instruction.operation_span.start, instruction.operation_span.end + 1),
        ) for instruction in item.ir.instructions)
        return item.ir.input_spans, (), None, iter((nodes,))

    monkeypatch.setattr(type(parent), "_runtime_operation_charts", source_charts)
    candidate = refit_compositional_argument_rankings(
        parent, examples, use_runtime_operation_views=True, runtime_operation_view_charts=1,
    )
    monkeypatch.setattr(type(parent), "_runtime_operation_charts", original_charts)
    receipt = candidate.training_receipt["argument_ranking_refit"]["runtime_operation_views"]
    assert receipt["training_views"] > receipt["source_examples"]
    assert any(not np.array_equal(before.weight, after.weight) for before, after in zip(
        parent.argument_role_heads, candidate.argument_role_heads, strict=True,
    ))
    seen = []
    original_assign = transducer._assign_typed_arguments

    def observe_selector(*args, **kwargs):
        seen.append(kwargs["model"])
        return original_assign(*args, **kwargs)

    monkeypatch.setattr(transducer, "_assign_typed_arguments", observe_selector)
    item = next(x for x in examples if x.split == "validation")
    candidate.decode(
        source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=candidate.model_basis_sha256,
    )
    assert seen and all(owner is candidate for owner in seen)


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
