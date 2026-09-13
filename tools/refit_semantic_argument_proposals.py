#!/usr/bin/env python3
"""Refit semantic evidence from the exact source cohort of a frozen parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def verify_source_splits(examples, receipt):
    """Reject both missing coverage and substituted examples before training."""
    for split, prefix in (("train", "training"), ("validation", "validation")):
        ids = sorted(x.ir.source_text_sha256 for x in examples if x.split == split)
        digest = hashlib.sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        if (
            not ids
            or len(ids) != receipt.get(f"{prefix}_example_count")
            or digest != receipt.get(f"{prefix}_example_ids_sha256")
        ):
            raise ValueError(f"{split} source cohort differs from frozen parent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--objective", choices=("binary_proposals", "pairwise_arguments", "operation_pointer", "argument_pointer"),
                        default="binary_proposals")
    args = parser.parse_args()
    from core.learning.semantic_program_basis import (
        bind_training_examples_to_shared_representation,
    )
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_compositional_campaign import (
        select_compositional_program_candidate,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        refit_compositional_argument_proposals,
        refit_compositional_argument_rankings,
        refit_compositional_operation_pointer,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validation_output is not None and (
        args.validation_output.exists()
        or args.validation_output.resolve() == args.output.resolve()
    ):
        raise FileExistsError(args.validation_output)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_text("ascii"))
    )
    report = json.loads(args.source_report.read_text("ascii"))
    compatibility = report["representation_compatibility"]
    expected = compatibility["source_feature_manifest_sha256s"]
    examples = {}
    for value in args.bundle:
        name, separator, path = value.partition("=")
        if not separator or name not in expected or name in examples:
            raise ValueError("source bundles must uniquely name every source family")
        bundle = load_standard_semantic_feature_bundle(Path(path).expanduser())
        if bundle.manifest["manifest_sha256"] != expected[name]:
            raise ValueError(f"source manifest differs: {name}")
        examples[name] = training_examples_from_feature_bundle(bundle)
    bound = bind_training_examples_to_shared_representation(
        examples, compatibility=compatibility
    )
    verify_source_splits(bound, model.training_receipt)
    refit = {
        "binary_proposals": refit_compositional_argument_proposals,
        "pairwise_arguments": refit_compositional_argument_rankings,
        "operation_pointer": refit_compositional_operation_pointer,
        "argument_pointer": refit_compositional_argument_proposals,
    }[args.objective]
    options = {"refit_pointer": True} if args.objective == "argument_pointer" else {}
    candidate = refit(model, bound, **options)
    payload = (json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
    if not atomic_write_bytes_if_absent(args.output, payload.encode("ascii"), mode=0o400):
        raise FileExistsError(args.output)
    if args.validation_output is not None:
        selection = select_compositional_program_candidate(
            {"incumbent": model, "refit": candidate}, bound, incumbent="incumbent"
        )
        payload = json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n"
        if not atomic_write_bytes_if_absent(
            args.validation_output, payload.encode("ascii"), mode=0o400
        ):
            raise FileExistsError(args.validation_output)
    print(json.dumps({"receipt_sha256": candidate.receipt_sha256, "serving_authority": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
