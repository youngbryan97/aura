#!/usr/bin/env python3
"""Fit the existing compositional compiler from compatible measured source cohorts."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True, metavar="FAMILY=PATH")
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--expected-validation-receipt", type=Path)
    parser.add_argument("--source-order-inputs", action="store_true",
                        help="rebind every supervised input and IR register to source order")
    args = parser.parse_args()
    output, report_output = args.output.resolve(), args.report_output.resolve()
    if output == report_output or output.exists() or report_output.exists():
        raise FileExistsError("source training outputs must be distinct and absent")
    os.environ.setdefault("AURA_LOG_DIR", str(output.parent / "logs"))
    os.environ.setdefault("AURA_STATE_ROOT", str(output.parent / "state"))
    from core.learning.semantic_input_grounding import semantic_input_grounding_contract_from_dict
    from core.learning.semantic_program_compositional_campaign import (
        fit_compositional_source_campaign,
        prepare_compositional_source_training,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.diagnose_compositional_semantic_transfer import _bundle_arguments

    bundles = {name: load_standard_semantic_feature_bundle(path)
               for name, path in _bundle_arguments(args.bundle).items()}
    grounding = semantic_input_grounding_contract_from_dict(json.loads(args.grounding.read_text("ascii")))
    if args.expected_validation_receipt:
        expected = json.loads(args.expected_validation_receipt.read_text("ascii"))
        _examples, plan = prepare_compositional_source_training(
            bundles, source_order_inputs=args.source_order_inputs)
        for key in ("validation_example_count", "validation_example_ids_sha256"):
            if plan[key] != expected[key]:
                raise ValueError("source validation cohort changed during reacquisition")
        del _examples
    result = fit_compositional_source_campaign(bundles, input_grounding=grounding,
        source_order_inputs=args.source_order_inputs,
        progress=lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    for path, value in ((output, result.model.to_dict()), (report_output, result.report)):
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
            raise FileExistsError(path)
    print(json.dumps({"stage": "source_fit_complete", "model": str(output), "report": str(report_output),
                      "receipt_sha256": result.model.receipt_sha256, "serving_authority": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
