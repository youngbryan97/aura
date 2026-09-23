#!/usr/bin/env python3
"""Measure whether production span proposals can reach annotated relations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reach(model, item) -> dict:
    from core.learning.semantic_definition_attachment import attachment_hypotheses
    from core.learning.semantic_definition_candidates import (
        _LOCAL_DEFINITION_CANDIDATE_STRATEGY,
        _register_definition_candidates,
    )
    from core.learning.semantic_program_shared_transducer import _relation_span_vector
    from core.learning.semantic_program_transducer_fitting import (
        _argument_proposals_by_operation,
        _OperationNode,
    )

    nodes = tuple(_OperationNode(step.operation_span, step.op, 0., 0., 1.)
                  for step in item.ir.instructions)
    pointers = model.argument_pointer.score_sequence(item.hidden_states)
    mentions = _argument_proposals_by_operation(
        pointers, input_spans=item.ir.input_spans, operation_nodes=nodes,
        max_span_tokens=model.max_span_tokens, clause_local=True)
    anchors = (*item.ir.input_spans, *(node.span for node in nodes))
    definition_scores = model.definition_pointer.score_sequence(item.hidden_states)
    joint = model.training_receipt.get("definition_selection_policy") == "joint_graph_v1"
    definitions = _register_definition_candidates(
        anchors, input_count=item.ir.n_inputs, token_count=len(item.ir.source_token_ids),
        max_span_tokens=model.max_definition_span_tokens,
        pointer_scores=definition_scores,
        strategy=(_LOCAL_DEFINITION_CANDIDATE_STRATEGY if joint
                  else model.definition_candidate_strategy),
        source_ordered_boundaries=(model.training_receipt.get("definition_boundary_policy")
                                   in {"source_neighbors_v1", "source_neighborhood_v2"}),
        bidirectional_inputs=(model.training_receipt.get("definition_boundary_policy")
                              == "source_neighborhood_v2"),
    )
    if joint and model.definition_attachment_head is not None:
        hypotheses, _scores = attachment_hypotheses(
            model, item.hidden_states, anchors, definitions)
        definitions = tuple(tuple(span for register, span in hypotheses if register == index)
                            for index in range(len(anchors)))
    mention_hits = [span in {candidate for candidate, _score in proposals}
                    for instruction, proposals in zip(item.ir.instructions, mentions, strict=True)
                    for span in instruction.argument_spans]
    definition_hits = ([span in definitions[index]
                        for index, span in enumerate(item.register_definition_spans)]
                       if item.register_definition_origin == "explicit_annotation" else [])
    definition_top1 = []
    definition_cosine_top1 = []
    definition_selected_overlap = []
    if definition_hits:
        head = model.definition_relation_head

        def vector(span):
            return _relation_span_vector(
                item.hidden_states, span, hidden_channels=model.hidden_channels,
                hidden_channel_widths=model.hidden_channel_widths)

        for register, candidates in enumerate(definitions):
            references = [span for step in item.ir.instructions
                          for owner, span in zip(step.args, step.argument_spans, strict=True)
                          if owner == register]
            if not references or not candidates:
                definition_top1.append(None)
                definition_cosine_top1.append(None)
                definition_selected_overlap.append(None)
                continue
            scored = []
            cosine_scores = []
            for candidate in candidates:
                definition = vector(candidate)
                projected = definition @ head.definition_projection
                pointer = head.pointer_scale * definition_scores.score_span(candidate)
                value = sum(head.base_score(vector(span), definition)
                            + float((vector(span) @ head.query_projection) @ projected)
                            + pointer for span in references)
                scored.append((value, candidate))
                cosine_scores.append((sum(float(vector(span) @ definition)
                                          for span in references), candidate))
            selected = max(scored, key=lambda pair: (pair[0], -pair[1].start,
                                                      -pair[1].end))[1]
            cosine_selected = max(cosine_scores, key=lambda pair: (
                pair[0], -pair[1].start, -pair[1].end))[1]
            definition_top1.append(selected == item.register_definition_spans[register])
            definition_cosine_top1.append(
                cosine_selected == item.register_definition_spans[register])
            gold = item.register_definition_spans[register]
            definition_selected_overlap.append(
                selected.start < gold.end and gold.start < selected.end)
    return {"source": item.ir.source_text_sha256,
            "construction": item.construction_id,
            "split": item.split,
            "explicit_definitions": bool(definition_hits),
            "mention_found": sum(mention_hits), "mention_total": len(mention_hits),
            "all_mentions_found": all(mention_hits),
            "definition_found": sum(definition_hits),
            "definition_total": len(definition_hits),
            "definition_hits": definition_hits,
            "definition_top1": definition_top1,
            "definition_cosine_top1": definition_cosine_top1,
            "definition_selected_overlap": definition_selected_overlap,
            "all_definitions_found": all(definition_hits) if definition_hits else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transducer", "source-report", "feature-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bidirectional-inputs", action="store_true")
    parser.add_argument("--explicit-only", action="store_true")
    args = parser.parse_args()

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    source_raw = args.source_report.read_bytes()
    source_report = json.loads(source_raw)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    if args.bidirectional_inputs:
        model = model.with_bidirectional_input_definitions()
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]
               ["source_feature_manifest_sha256s"]]
    examples = load_source_examples(model, source_report, bundles)
    rows = [_reach(model, item) for item in examples
            if not args.explicit_only or item.register_definition_origin == "explicit_annotation"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split"], row["construction"]].append(row)
    summary = {}
    for (split, construction), members in sorted(grouped.items()):
        summary[f"{split}:{construction}"] = {
            "population": len(members),
            "all_mentions_found": sum(row["all_mentions_found"] for row in members),
            "mention_found": sum(row["mention_found"] for row in members),
            "mention_total": sum(row["mention_total"] for row in members),
            "explicit_definitions": sum(row["explicit_definitions"] for row in members),
            "all_definitions_found": sum(row["all_definitions_found"] is True
                                         for row in members),
            "definition_found": sum(row["definition_found"] for row in members),
            "definition_total": sum(row["definition_total"] for row in members),
            "definition_found_by_register": [
                sum(row["definition_hits"][index] for row in members
                    if len(row["definition_hits"]) > index)
                for index in range(max((len(row["definition_hits"]) for row in members),
                                       default=0))],
            "definition_top1_by_register": [
                sum(row["definition_top1"][index] is True for row in members
                    if len(row["definition_top1"]) > index)
                for index in range(max((len(row["definition_top1"]) for row in members),
                                       default=0))],
            "definition_cosine_top1_by_register": [
                sum(row["definition_cosine_top1"][index] is True for row in members
                    if len(row["definition_cosine_top1"]) > index)
                for index in range(max((len(row["definition_cosine_top1"]) for row in members),
                                       default=0))],
            "definition_selected_overlap_by_register": [
                sum(row["definition_selected_overlap"][index] is True for row in members
                    if len(row["definition_selected_overlap"]) > index)
                for index in range(max((len(row["definition_selected_overlap"]) for row in members),
                                       default=0))],
        }
    body = {"schema": "aura.semantic_relation_reach_probe.v6",
            "serving_authority": False, "development_only": True,
            "bidirectional_inputs": args.bidirectional_inputs,
            "explicit_only": args.explicit_only,
            "gold_operation_spans_used_for_stage_isolation": True,
            "gold_reference_spans_used_for_relation_upper_bound": True,
            "source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
            "transducer_sha256": hashlib.sha256(args.transducer.read_bytes()).hexdigest(),
            "rows": rows, "by_construction": summary}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    payload = json.dumps({**body, "receipt_sha256": digest}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("relation reach probe output already differs")
    args.output.write_bytes(payload)
    print(json.dumps({key: value for key, value in summary.items()
                      if key.startswith("validation:")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
