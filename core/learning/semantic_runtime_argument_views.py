"""Expose argument fitting to the parent's learned operation boundaries."""

from collections import Counter
from dataclasses import replace
from itertools import islice
from types import SimpleNamespace

from core.verify.invariants import invariant
from typing import Any


def align_operation_views(gold: tuple[Any, ...], predicted: tuple[Any, ...]) -> Any:
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


def runtime_argument_training_views(
    model: Any,
    training: Any,
    *,
    max_operation_charts: int=0,
    progress: Any=None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Keep every original target and add unambiguous on-policy boundary views."""
    if type(max_operation_charts) is not int or not 0 <= max_operation_charts <= 64:
        raise ValueError("runtime operation chart limit must be inside [0, 64]")
    if any(item.split != "train" for item in training):
        raise ValueError("runtime argument views may decode source training only")
    views, rows = [], []
    counts = Counter()
    for item in training:
        views.append(item)
        if max_operation_charts:
            from core.learning.semantic_joint_graph_learning import align_source_input_registers

            limit = model.inference_step_limit(len(item.public_inputs))
            if limit is None:
                status, detail = "unsupported_input_count", {}
            else:
                try:
                    spans, _, _, charts = model._runtime_operation_charts(
                        item.ir.source_token_ids, item.hidden_states, item.public_inputs, limit,
                    )
                    targets, mapping = align_source_input_registers(item, spans)
                except ValueError as exc:
                    status, detail = "runtime_chart_unavailable", {"reason": str(exc)}
                else:
                    seen = {(item.ir.input_spans, tuple(i.operation_span for i in item.ir.instructions),
                             tuple(i.args for i in item.ir.instructions))}
                    examined = matched = 0
                    accepted = []
                    for chart_index, nodes in enumerate(islice(charts, max_operation_charts)):
                        examined += 1
                        predicted = tuple(SimpleNamespace(op=node.operation, operation_span=node.span)
                                          for node in nodes)
                        aligned = align_operation_views(targets, predicted)
                        if aligned is None or tuple(i.operation_span for i in aligned) != tuple(
                            node.span for node in nodes
                        ):
                            continue
                        matched += 1
                        identity = (tuple(spans), tuple(i.operation_span for i in aligned),
                                    tuple(i.args for i in aligned))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        views.append(replace(item, ir=replace(
                            item.ir, input_spans=tuple(spans), instructions=aligned,
                        )))
                        accepted.append({"chart": chart_index, "operation_spans": [
                            instruction.operation_span.to_dict() for instruction in aligned
                        ]})
                    status = "augmented" if len(seen) > 1 else "unchanged"
                    detail = {"charts_examined": examined, "charts_aligned": matched,
                              "views_added": len(seen) - 1,
                              "accepted_views": accepted,
                              "source_to_runtime_input_registers": list(mapping)}
        else:
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
            detail = {"refusal": outcome.refusal,
                      "predicted_spans": [] if outcome.ir is None else [
                          i.operation_span.to_dict() for i in outcome.ir.instructions
                      ]}
        counts[status] += 1
        rows.append({
            "source_text_sha256": item.ir.source_text_sha256,
            "status": status,
            "gold_spans": [i.operation_span.to_dict() for i in item.ir.instructions],
            **detail,
        })
        if progress is not None and (len(rows) % 50 == 0 or len(rows) == len(training)):
            progress({"stage": "runtime_argument_views", "completed": len(rows),
                      "total": len(training), "coverage": dict(counts)})
    return tuple(views), {
        "schema": "aura.semantic_runtime_argument_views.v2" if max_operation_charts else
                  "aura.semantic_runtime_argument_views.v1",
        "parent_transducer_receipt_sha256": model.receipt_sha256,
        "max_operation_charts": max_operation_charts,
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
def _runtime_views_preserve_targets() -> tuple[()]:
    from core.learning.semantic_program_ir import SemanticIRInstruction, TokenSpan

    gold = SemanticIRInstruction("add", (0, 1), TokenSpan(3, 4),
                                 (TokenSpan(0, 1), TokenSpan(5, 6)), ())
    prediction = replace(gold, args=(1, 0), operation_span=TokenSpan(2, 4))
    aligned = align_operation_views((gold,), (prediction,))
    if aligned != (replace(gold, operation_span=prediction.operation_span),):
        raise AssertionError("runtime view changed a supervised target")
    return ()
