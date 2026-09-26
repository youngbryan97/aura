#!/usr/bin/env python3
"""Repair a measured source-order fit's omitted model policy with lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def rebind_source_order_identity(model, report, plan, validation):
    """Issue a new identity only for an intact v2 fit and its measured cohort."""
    from core.learning.semantic_program_compositional_campaign import _sha as campaign_sha

    report_body = {key: value for key, value in report.items() if key != "report_sha256"}
    validation_body = {key: value for key, value in validation.items()
                       if key != "receipt_sha256"}
    receipt = model.training_receipt
    expected = ("training_example_count", "training_example_ids_sha256",
                "validation_example_count", "validation_example_ids_sha256")
    if (report.get("schema") != "aura.compositional_source_training.v2"
            or report.get("input_order_policy") != "source_token_order_v1"
            or report.get("fit_complete") is not True
            or report.get("serving_authority") is not False
            or report.get("report_sha256") != campaign_sha(report_body)
            or report.get("transducer_receipt_sha256") != model.receipt_sha256
            or receipt.get("input_order_policy") is not None
            or receipt.get("source_order_identity_rebind") is not None
            or plan.get("schema") != "aura.compositional_source_training_plan.v2"
            or plan.get("input_order_policy") != "source_token_order_v1"
            or any(report.get(key) != plan.get(key)
                   or receipt.get(key) != plan.get(key) for key in expected)
            or report.get("representation_compatibility") != plan.get("representation_compatibility")
            or report.get("cohort_split_counts") != plan.get("cohort_split_counts")
            or report.get("test_examples_available_to_fit") != 0
            or validation.get("schema") != "aura.semantic_source_order_regrade.v1"
            or validation.get("receipt_sha256") != _sha(validation_body)
            or validation.get("candidate_receipt_sha256") != model.receipt_sha256
            or validation.get("source_input_order", {}).get("candidates", {}).get(
                "frozen", {}).get("transducer_receipt_sha256") != model.receipt_sha256
            or validation.get("source_order_plan_sha256") != plan.get("report_sha256")
            or validation.get("validation_ids_sha256") != plan.get("validation_example_ids_sha256")
            or validation.get("serving_authority") is not False):
        raise ValueError("source-order identity rebind lacks matched fit and validation evidence")
    parent = model.receipt_sha256
    lineage = {"parent_transducer_receipt_sha256": parent,
               "parent_source_report_sha256": report["report_sha256"],
               "parent_validation_receipt_sha256": validation["receipt_sha256"],
               "coefficients_changed": False, "serving_authority": False}
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    body.update(input_order_policy="source_token_order_v1",
                source_order_identity_rebind=lineage)
    rebound = replace(model, training_receipt={**body, "receipt_sha256": campaign_sha(body)})
    new_report_body = dict(report_body)
    new_report_body.update(transducer_receipt_sha256=rebound.receipt_sha256,
                           candidate_identity_rebind=lineage)
    new_report = {**new_report_body, "report_sha256": campaign_sha(new_report_body)}
    if ({key: value for key, value in rebound.to_dict().items() if key != "training_receipt"}
            != {key: value for key, value in model.to_dict().items()
                if key != "training_receipt"}):
        raise ValueError("source-order identity rebind changed model coefficients")
    candidate_body = {
        "schema": "aura.semantic_source_order_identity_rebind.v1",
        "candidate": rebound.receipt_sha256,
        "source_report_sha256": new_report["report_sha256"],
        "parent_candidate_receipt_sha256": parent,
        "parent_validation_receipt_sha256": validation["receipt_sha256"],
        "coefficients_changed": False,
        "serving_authority": False,
    }
    candidate_report = {**candidate_body, "receipt_sha256": _sha(candidate_body)}
    return rebound, new_report, candidate_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "source-report", "validation", "output",
                 "report-output", "candidate-report-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    output_paths = (args.output, args.report_output, args.candidate_report_output)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("identity-rebound outputs must be distinct")
    os.environ.setdefault("AURA_LOG_DIR", str(args.output.parent / "logs"))
    os.environ.setdefault("AURA_STATE_ROOT", str(args.output.parent / "state"))

    from core.learning.semantic_program_compositional_campaign import (
        prepare_compositional_source_training,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.diagnose_compositional_semantic_transfer import _bundle_arguments

    bundles = {name: load_standard_semantic_feature_bundle(path)
               for name, path in _bundle_arguments(args.bundle).items()}
    _, plan = prepare_compositional_source_training(bundles, source_order_inputs=True)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.candidate.read_text("ascii")))
    report = json.loads(args.source_report.read_text("ascii"))
    validation = json.loads(args.validation.read_text("ascii"))
    rebound, new_report, candidate_report = rebind_source_order_identity(
        model, report, plan, validation)
    for path, value in ((args.output, rebound.to_dict()),
                        (args.report_output, new_report),
                        (args.candidate_report_output, candidate_report)):
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode("ascii")
        if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
            raise ValueError("identity-rebound output differs from measured lineage")
    print(json.dumps({"candidate": str(args.output), "source_report": str(args.report_output),
                      "candidate_report": str(args.candidate_report_output),
                      "parent": model.receipt_sha256, "rebound": rebound.receipt_sha256,
                      "coefficients_changed": False, "serving_authority": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
