#!/usr/bin/env python3
"""Build or verify an immutable compact candidate-cortex persona kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.candidate_cortex_kernel import (  # noqa: E402
    DEFAULT_SPLIT_SEED,
    DEFAULT_VALID_FRACTION,
    CandidateCortexKernelError,
    build_candidate_cortex_kernel,
    verify_candidate_cortex_kernel,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="publish one content-addressed kernel")
    build.add_argument("--descriptor", type=Path, required=True)
    build.add_argument("--descriptor-sha256", required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--source-repo-root", type=Path, default=REPO_ROOT)
    build.add_argument("--crsm", type=Path)
    build.add_argument("--valid-fraction", type=float, default=DEFAULT_VALID_FRACTION)
    build.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)

    verify = commands.add_parser("verify", help="verify a published kernel receipt")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--descriptor-sha256")
    verify.add_argument("--skip-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_candidate_cortex_kernel(
                descriptor_path=args.descriptor,
                expected_descriptor_sha256=args.descriptor_sha256,
                output_root=args.output_root,
                source_repo_root=args.source_repo_root,
                crsm_path=args.crsm,
                valid_fraction=args.valid_fraction,
                split_seed=args.split_seed,
            )
        else:
            receipt = verify_candidate_cortex_kernel(
                args.receipt,
                expected_descriptor_sha256=args.descriptor_sha256,
                verify_inputs=not args.skip_inputs,
            )
    except (CandidateCortexKernelError, FileNotFoundError, OSError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
