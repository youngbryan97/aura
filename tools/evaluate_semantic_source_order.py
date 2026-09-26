#!/usr/bin/env python3
"""Regrade a frozen semantic candidate under the live source-order input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("AURA_LOG_DIR", str(args.output.parent / "logs"))
    os.environ.setdefault("AURA_STATE_ROOT", str(args.output.parent / "state"))

    from core.learning.semantic_program_compositional_campaign import (
        prepare_compositional_source_training,
        select_compositional_program_candidate,
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
    original, original_plan = prepare_compositional_source_training(bundles)
    ordered, ordered_plan = prepare_compositional_source_training(
        bundles, source_order_inputs=True)
    if original_plan["validation_example_ids_sha256"] != ordered_plan["validation_example_ids_sha256"]:
        raise ValueError("source-order rebind changed validation identities")
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.candidate.read_text("ascii")))
    reports = {}
    for name, examples in (("corpus_input_order", original), ("source_input_order", ordered)):
        reports[name] = select_compositional_program_candidate(
            {"frozen": model}, examples, incumbent="frozen", scoring="register_indices_v1")
    old_rows = reports["corpus_input_order"]["candidates"]["frozen"]["program_correct"]
    new_rows = reports["source_input_order"]["candidates"]["frozen"]["program_correct"]
    body = {
        "schema": "aura.semantic_source_order_regrade.v1",
        "candidate_receipt_sha256": model.receipt_sha256,
        "old_plan_sha256": original_plan["report_sha256"],
        "source_order_plan_sha256": ordered_plan["report_sha256"],
        "validation_ids_sha256": ordered_plan["validation_example_ids_sha256"],
        "corpus_input_order": reports["corpus_input_order"],
        "source_input_order": reports["source_input_order"],
        "gains": sum(new and not old for old, new in zip(old_rows, new_rows, strict=True)),
        "regressions": sum(old and not new for old, new in zip(old_rows, new_rows, strict=True)),
        "new_weights_trained": False,
        "serving_authority": False,
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    if not atomic_write_bytes_if_absent(args.output, payload, mode=0o400):
        raise FileExistsError(args.output)
    print(json.dumps({"output": str(args.output), "receipt_sha256": receipt["receipt_sha256"],
                      "old_exact": sum(old_rows), "source_order_exact": sum(new_rows),
                      "gains": body["gains"], "regressions": body["regressions"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
