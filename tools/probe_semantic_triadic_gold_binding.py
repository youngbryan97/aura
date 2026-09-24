#!/usr/bin/env python3
"""Measure gold-span binding on a construction-held source fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--feature-schema", choices=("triple_product_v1", "joint_source_v2",
                                                     "joint_representation_v3", "projected_joint_v4"),
                        default="triple_product_v1")
    parser.add_argument("--runtime-operation-limit", type=int, default=0)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tools.probe_semantic_proposer_crossfit import crossfit_partition
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_refits import (
        attach_compositional_triadic_bindings,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_triadic_binding import (
        evaluate_triadic_gold_binding,
        fit_source_triadic_projection,
        fit_triadic_binding_heads,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    parent_raw, report_raw, folds_raw = (path.read_bytes() for path in
                                         (args.parent, args.source_report, args.folds))
    parent = compositional_semantic_program_transducer_from_dict(json.loads(parent_raw))
    report, folds = json.loads(report_raw), json.loads(folds_raw)
    bundles = [name + "=" + str(args.feature_root / name) for name in
               report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(parent, report, bundles)
    fit, calibration, held = crossfit_partition(examples, folds, args.fold, all_held=True)
    fit_ids = frozenset(item.ir.source_text_sha256 for item in fit)
    calibration_ids = frozenset(item.ir.source_text_sha256 for item in calibration)
    held_ids = frozenset(item.ir.source_text_sha256 for item in held)
    if fit_ids & calibration_ids or fit_ids & held_ids or calibration_ids & held_ids:
        raise ValueError("gold binding partition overlaps")
    projection_basis = projection_fit = None
    if args.feature_schema == "projected_joint_v4":
        projection_basis, projection_fit = fit_source_triadic_projection(
            fit, calibration, hidden_channels=parent.hidden_channels,
            hidden_channel_widths=parent.hidden_channel_widths)
    heads, training = fit_triadic_binding_heads(fit,
        max_arity=len(parent.argument_role_heads),
        hidden_channels=parent.hidden_channels,
        hidden_channel_widths=parent.hidden_channel_widths,
        feature_schema=args.feature_schema, projection_basis=projection_basis)
    result = evaluate_triadic_gold_binding(heads, held,
        hidden_channels=parent.hidden_channels,
        hidden_channel_widths=parent.hidden_channel_widths,
        training_source_ids=fit_ids)
    runtime_operation = None
    if args.runtime_operation_limit:
        if not 0 < args.runtime_operation_limit <= len(held):
            raise ValueError("runtime operation limit exceeds held sources")
        from core.learning.semantic_runtime_argument_views import runtime_argument_training_views

        selected = sorted(held, key=lambda item: hashlib.sha256(
            item.ir.source_text_sha256.encode("ascii")).hexdigest())[:args.runtime_operation_limit]
        views, view_receipt = runtime_argument_training_views(parent, selected)
        by_source = {}
        for view in views:
            by_source.setdefault(view.ir.source_text_sha256, []).append(view)
        aligned = []
        for row in view_receipt["rows"]:
            source_views = by_source[row["source_text_sha256"]]
            if row["status"] == "unchanged":
                aligned.append(source_views[0])
            elif row["status"] == "augmented":
                if len(source_views) != 2:
                    raise ValueError("runtime operation view is missing its aligned source")
                aligned.append(source_views[1])
        runtime_result = (evaluate_triadic_gold_binding(heads, aligned,
            hidden_channels=parent.hidden_channels,
            hidden_channel_widths=parent.hidden_channel_widths,
            training_source_ids=fit_ids) if aligned else None)
        if runtime_result is not None:
            runtime_result = {**runtime_result, "gold_operation_and_mention_spans": False,
                              "gold_mention_and_definition_spans": True}
        runtime_operation = {
            "selected_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in selected)),
            "selected_sources": len(selected), "aligned_sources": len(aligned),
            "coverage": view_receipt["coverage"], "binding_on_aligned": runtime_result,
            "operation_model_holdout": False,
        }
    lesions = None
    if args.feature_schema == "joint_source_v2":
        lesions = {}
        for component in ("geometry", "representation"):
            lesions[component] = evaluate_triadic_gold_binding(
                tuple(head.component_lesion(component) for head in heads), held,
                hidden_channels=parent.hidden_channels,
                hidden_channel_widths=parent.hidden_channel_widths,
                training_source_ids=fit_ids)
    elif args.feature_schema in {"joint_representation_v3", "projected_joint_v4"}:
        lesions = {}
        for role in ("operation", "mention", "definition"):
            lesions[role] = evaluate_triadic_gold_binding(
                tuple(head.role_lesion(role) for head in heads), held,
                hidden_channels=parent.hidden_channels,
                hidden_channel_widths=parent.hidden_channel_widths,
                training_source_ids=fit_ids)
    candidate_sha256 = None
    if args.candidate_output is not None:
        candidate = attach_compositional_triadic_bindings(
            parent, heads,
            training_source_ids=sorted(fit_ids),
            calibration_source_ids=sorted(calibration_ids),
            runtime_views={"schema": "aura.semantic_triadic_gold_source_views.v1",
                           "source_examples": len(fit), "training_views": len(fit)},
            fit=training, projection_fit=projection_fit,
            source_fold_provenance={"schema": "aura.semantic_triadic_source_fold_probe.v1",
                                    "held_ids_sha256": _sha(sorted(held_ids)),
                                    "qualification_evidence": False})
        candidate_payload = json.dumps(candidate.to_dict(), sort_keys=True,
                                       separators=(",", ":")).encode("ascii") + b"\n"
        candidate_sha256 = hashlib.sha256(candidate_payload).hexdigest()
        if not atomic_write_bytes_if_absent(args.candidate_output, candidate_payload, mode=0o400):
            if args.candidate_output.read_bytes() != candidate_payload:
                raise ValueError("source-fold candidate already exists with different evidence")
    body = {"schema": "aura.semantic_triadic_gold_probe.v1",
            "parent_receipt_sha256": parent.receipt_sha256,
            "parent_file_sha256": hashlib.sha256(parent_raw).hexdigest(),
            "source_report_file_sha256": hashlib.sha256(report_raw).hexdigest(),
            "folds_file_sha256": hashlib.sha256(folds_raw).hexdigest(),
            "fold": args.fold, "training_ids_sha256": _sha(sorted(fit_ids)),
            "feature_schema": args.feature_schema,
            "calibration_ids_sha256": _sha(sorted(calibration_ids)),
            "held_ids_sha256": _sha(sorted(held_ids)),
            "coefficient_sha256": _sha([head.to_dict() for head in heads]),
            "source_fold_candidate_file_sha256": candidate_sha256,
            "training": training, "projection_fit": projection_fit,
            "held_gold_binding": result, "runtime_operation_binding": runtime_operation,
            "held_component_lesions": lesions,
            "gold_spans_supplied": True, "test_examples_used_for_fit": 0,
            "qualification_evidence": False, "serving_authority": False}
    payload = json.dumps({**body, "receipt_sha256": _sha(body)}, sort_keys=True,
                         separators=(",", ":")).encode("ascii") + b"\n"
    if not atomic_write_bytes_if_absent(args.output, payload, mode=0o400):
        if args.output.read_bytes() != payload:
            raise ValueError("gold binding report already exists with different evidence")
    print(payload.decode("ascii"), end="", flush=True)


if __name__ == "__main__":
    main()
