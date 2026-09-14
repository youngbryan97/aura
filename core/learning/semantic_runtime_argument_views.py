"""Expose argument fitting to the parent's learned operation boundaries."""

from collections import Counter
from dataclasses import replace

from core.verify.invariants import invariant


def align_operation_views(gold, predicted):
    """Match labels and overlapping spans only when ownership is one-to-one.

    Predicted argument identities never become targets. Missing or ambiguous
    operation matches remain explicit coverage failures rather than invented labels.
    """
    if len(gold) != len(predicted):
        return None
    matches = []
    for instruction in gold:
        candidates = [index for index, other in enumerate(predicted)
                      if instruction.op == other.op
                      and instruction.operation_span.start < other.operation_span.end
                      and other.operation_span.start < instruction.operation_span.end]
        if len(candidates) != 1:
            return None
        matches.append(candidates[0])
    if len(set(matches)) != len(gold):
        return None
    return tuple(replace(instruction, operation_span=predicted[index].operation_span)
                 for instruction, index in zip(gold, matches, strict=True))


def runtime_argument_training_views(model, training, *, progress=None):
    """Keep every original target and add unambiguous on-policy boundary views."""
    if any(item.split != "train" for item in training):
        raise ValueError("runtime argument views may decode source training only")
    views, rows = [], []
    counts = Counter()
    for item in training:
        views.append(item)
        outcome = model.decode(
            source_token_ids=item.ir.source_token_ids,
            hidden_states=item.hidden_states,
            public_inputs=item.public_inputs,
            source_text_sha256=item.ir.source_text_sha256,
            model_basis_sha256=model.model_basis_sha256,
        )
        aligned = None if outcome.ir is None else align_operation_views(
            item.ir.instructions, outcome.ir.instructions,
        )
        if outcome.ir is None:
            status = "decode_refused"
        elif aligned is None:
            status = "alignment_unresolved"
        elif aligned == item.ir.instructions:
            status = "unchanged"
        else:
            status = "augmented"
            views.append(replace(item, ir=replace(item.ir, instructions=aligned)))
        counts[status] += 1
        rows.append({
            "source_text_sha256": item.ir.source_text_sha256,
            "status": status,
            "refusal": outcome.refusal,
            "gold_spans": [i.operation_span.to_dict() for i in item.ir.instructions],
            "predicted_spans": [] if outcome.ir is None else [
                i.operation_span.to_dict() for i in outcome.ir.instructions
            ],
        })
        if progress is not None and (len(rows) % 50 == 0 or len(rows) == len(training)):
            progress({"stage": "runtime_argument_views", "completed": len(rows),
                      "total": len(training), "coverage": dict(counts)})
    return tuple(views), {
        "schema": "aura.semantic_runtime_argument_views.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "source_examples": len(training),
        "training_views": len(views),
        "coverage": dict(counts),
        "rows": rows,
        "predicted_arguments_used_as_targets": False,
        "validation_used_for_fit": False,
        "test_examples_used": 0,
        "serving_authority": False,
    }


@invariant("semantic.runtime_views_preserve_targets", scope="learning",
           owner="core/learning/semantic_runtime_argument_views.py", observational=False)
def _runtime_views_preserve_targets():
    from core.learning.semantic_program_ir import SemanticIRInstruction, TokenSpan

    gold = SemanticIRInstruction("add", (0, 1), TokenSpan(3, 4),
                                 (TokenSpan(0, 1), TokenSpan(5, 6)), ())
    prediction = replace(gold, args=(1, 0), operation_span=TokenSpan(2, 4))
    aligned = align_operation_views((gold,), (prediction,))
    if aligned != (replace(gold, operation_span=prediction.operation_span),):
        raise AssertionError("runtime view changed a supervised target")
    return ()
