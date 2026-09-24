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
    parser.add_argument("--feature-schema", choices=("triple_product_v1", "joint_source_v2"),
                        default="triple_product_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tools.probe_semantic_proposer_crossfit import crossfit_partition
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_triadic_binding import (
        evaluate_triadic_gold_binding,
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
    heads, training = fit_triadic_binding_heads(fit,
        max_arity=len(parent.argument_role_heads),
        hidden_channels=parent.hidden_channels,
        hidden_channel_widths=parent.hidden_channel_widths,
        feature_schema=args.feature_schema)
    result = evaluate_triadic_gold_binding(heads, held,
        hidden_channels=parent.hidden_channels,
        hidden_channel_widths=parent.hidden_channel_widths,
        training_source_ids=fit_ids)
    lesions = None
    if args.feature_schema == "joint_source_v2":
        lesions = {}
        for component in ("geometry", "representation"):
            lesions[component] = evaluate_triadic_gold_binding(
                tuple(head.component_lesion(component) for head in heads), held,
                hidden_channels=parent.hidden_channels,
                hidden_channel_widths=parent.hidden_channel_widths,
                training_source_ids=fit_ids)
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
            "training": training, "held_gold_binding": result,
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
