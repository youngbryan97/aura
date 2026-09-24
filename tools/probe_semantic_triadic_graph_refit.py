#!/usr/bin/env python3
"""Refit a frozen source-fold triadic candidate on complete source graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-parent", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tools.probe_semantic_proposer_crossfit import crossfit_partition
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment, load_source_examples,
    )

    configure_refit_environment(args.output)
    from core.learning.semantic_graph_margin import refit_compositional_graph_scales
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    raw = {name: path.read_bytes() for name, path in (
        ("candidate", args.candidate), ("source_parent", args.source_parent),
        ("source_report", args.source_report), ("folds", args.folds))}
    candidate = compositional_semantic_program_transducer_from_dict(json.loads(raw["candidate"]))
    source_parent = compositional_semantic_program_transducer_from_dict(json.loads(raw["source_parent"]))
    report = json.loads(raw["source_report"])
    folds = json.loads(raw["folds"])
    if (candidate.triadic_binding_heads is None
            or candidate.model_basis_sha256 != source_parent.model_basis_sha256
            or candidate.input_grounding != source_parent.input_grounding):
        raise ValueError("triadic source candidate differs from source feature owner")
    bundles = [name + "=" + str(args.feature_root / name) for name in
               report["representation_compatibility"]["source_feature_manifest_sha256s"]]
    examples = load_source_examples(source_parent, report, bundles)
    fit, calibration, held = crossfit_partition(examples, folds, args.fold, all_held=True)
    fit_ids = sorted(item.ir.source_text_sha256 for item in fit)
    calibration_ids = sorted(item.ir.source_text_sha256 for item in calibration)
    triadic_receipt = candidate.training_receipt["triadic_binding_fit"]
    if (candidate.training_receipt.get("training_example_ids_sha256") != _sha(fit_ids)
            or candidate.training_receipt.get("validation_example_ids_sha256") != _sha(calibration_ids)
            or triadic_receipt.get("training_example_ids_sha256") != _sha(fit_ids)
            or set(fit_ids) & {item.ir.source_text_sha256 for item in held}):
        raise ValueError("candidate source training partition differs from frozen fold")
    source_examples = (*fit, *(replace(item, split="validation") for item in calibration))
    def progress(event):
        if event["completed"] % 8 == 0 or event["completed"] == event["total"]:
            print(json.dumps({"stage": event["stage"], "completed": event["completed"],
                              "total": event["total"], "coverage": event["coverage"]},
                             sort_keys=True), flush=True)

    result = refit_compositional_graph_scales(candidate, source_examples, progress=progress)
    candidate_payload = json.dumps(result.to_dict(), sort_keys=True,
                                   separators=(",", ":")).encode("ascii") + b"\n"
    coverage = result.training_receipt["argument_graph_factor_refit"]
    body = {
        "schema": "aura.semantic_triadic_graph_source_probe.v1",
        "input_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        "fold": args.fold, "fit_ids_sha256": _sha(fit_ids),
        "calibration_ids_sha256": _sha(calibration_ids),
        "held_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in held)),
        "training_only_fit": True, "calibration_used_for_fit": False,
        "held_used_for_fit": False, "source_operation_spans_supplied": True,
        "candidate_file_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "graph_fit": coverage, "serving_authority": False,
    }
    payload = json.dumps({**body, "receipt_sha256": _sha(body)}, sort_keys=True,
                         separators=(",", ":")).encode("ascii") + b"\n"
    for path, value in ((args.candidate_output, candidate_payload), (args.output, payload)):
        if not atomic_write_bytes_if_absent(path, value, mode=0o400) and path.read_bytes() != value:
            raise ValueError(f"source graph result already differs: {path}")
    print(json.dumps({"coverage": coverage["coverage"], "fit": coverage["fit"],
                      "candidate": str(args.candidate_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
