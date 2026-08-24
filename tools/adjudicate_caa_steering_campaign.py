#!/usr/bin/env python3
"""Build the strict CAA migration evaluation from independent replay evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluation.caa_causal_evaluation import (  # noqa: E402
    build_causal_evaluation,
    file_sha256,
)
from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402


def _read_json(path: Path) -> dict:
    payload = read_stable_bytes(path.expanduser().absolute(), max_bytes=16 * 1024 * 1024)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--independent-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evaluation = build_causal_evaluation(
            result=_read_json(args.result),
            metadata=_read_json(args.metadata),
            verifier_evidence=_read_json(args.independent_evidence),
            verifier_evidence_sha256=file_sha256(args.independent_evidence),
        )
        payload = json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
        with local_internal_governed_scope(
            "adjudicate_caa_steering_campaign.output", domain="file_write"
        ):
            get_file_write_gateway().write_text(
                args.out.expanduser().absolute(),
                payload,
                source="adjudicate_caa_steering_campaign.output",
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"CAA adjudication failed: {exc}", file=sys.stderr)
        return 2
    print(args.out)
    print(
        "qualified=True "
        f"({evaluation['treatment_successes']}/{evaluation['sample_count']} treatment)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
