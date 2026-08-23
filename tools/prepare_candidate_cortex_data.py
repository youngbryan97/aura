#!/usr/bin/env python3
"""Prepare or verify immutable candidate-cortex persona and CRSM datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.candidate_cortex_data import (  # noqa: E402
    DEFAULT_MAX_CRSM_EXAMPLES,
    DEFAULT_RETENTION_EXAMPLES,
    DEFAULT_SPLIT_SEED,
    DEFAULT_VALID_FRACTION,
    CandidateCortexDataError,
    prepare_candidate_cortex_data,
    validate_candidate_cortex_data_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="publish one immutable generation")
    prepare.add_argument("--descriptor", type=Path, required=True)
    prepare.add_argument("--descriptor-sha256", required=True)
    prepare.add_argument("--persona-train", type=Path, required=True)
    prepare.add_argument("--persona-valid", type=Path, required=True)
    prepare.add_argument("--crsm-source", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--source-repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument("--valid-fraction", type=float, default=DEFAULT_VALID_FRACTION)
    prepare.add_argument("--max-crsm-examples", type=int, default=DEFAULT_MAX_CRSM_EXAMPLES)
    prepare.add_argument("--retention-examples", type=int, default=DEFAULT_RETENTION_EXAMPLES)
    prepare.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)

    verify = subcommands.add_parser("verify", help="verify a published receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--descriptor-sha256")
    verify.add_argument("--skip-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            receipt = prepare_candidate_cortex_data(
                descriptor_path=args.descriptor,
                expected_descriptor_sha256=args.descriptor_sha256,
                persona_train=args.persona_train,
                persona_valid=args.persona_valid,
                crsm_source=args.crsm_source,
                output_root=args.output_root,
                source_repo_root=args.source_repo_root,
                valid_fraction=args.valid_fraction,
                max_crsm_examples=args.max_crsm_examples,
                retention_examples=args.retention_examples,
                split_seed=args.split_seed,
            )
        else:
            receipt = validate_candidate_cortex_data_receipt(
                args.receipt,
                expected_descriptor_sha256=args.descriptor_sha256,
                verify_inputs=not args.skip_inputs,
            )
    except (CandidateCortexDataError, FileNotFoundError, OSError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
