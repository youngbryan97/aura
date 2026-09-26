#!/usr/bin/env python3
"""Freeze construction-disjoint source folds for a signed candidate cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source_folds(examples, *, count: int = 3, seed: int = 0):
    from core.learning.semantic_construction_folds import construction_folds

    train = tuple(item for item in examples if item.split == "train")
    if not train or len(train) == len(examples):
        raise ValueError("frozen source folds require train and withheld examples")
    return construction_folds(train, count=count, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
        verify_candidate_report_identity,
    )

    configure_refit_environment(args.output)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    report = json.loads(args.source_report.read_bytes())
    candidate_report = json.loads(args.candidate_report.read_bytes())
    verify_candidate_report_identity(candidate_report, model, report)
    bundles = source_bundle_arguments(report, bundles=args.bundle)
    folds = source_folds(load_source_examples(model, report, bundles),
                         count=args.count, seed=args.seed)
    payload = (json.dumps(folds, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("ascii")
    if (not atomic_write_bytes_if_absent(args.output, payload, mode=0o400)
            and args.output.read_bytes() != payload):
        raise ValueError("frozen source folds differ from existing artifact")
    print(json.dumps({"output": str(args.output),
                      "receipt_sha256": folds["receipt_sha256"],
                      "population": len(folds["population"]),
                      "independent_groups": folds["independent_groups"],
                      "validation_used": False, "test_used": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
