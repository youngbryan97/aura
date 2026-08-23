#!/usr/bin/env python3
"""Reduce exact checkpoint measurements to candidate training admission JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.candidate_cortex_admission import (  # noqa: E402
    CandidateCortexAdmissionError,
    adjudicate_checkpoint_evidence,
)
from core.learning.candidate_cortex_training import (  # noqa: E402
    CandidateCortexTrainingError,
    load_and_verify_plan,
)


def _strict_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CandidateCortexAdmissionError("checkpoint_evidence_not_regular")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateCortexAdmissionError(
                    "checkpoint_evidence_duplicate_key"
                )
            result[key] = value
        return result

    raw = resolved.read_bytes()
    if not raw or len(raw) > 16 * 1024 * 1024:
        raise CandidateCortexAdmissionError("checkpoint_evidence_size_invalid")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateCortexAdmissionError("checkpoint_evidence_json_invalid") from exc
    if not isinstance(value, dict):
        raise CandidateCortexAdmissionError("checkpoint_evidence_schema_invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        plan = load_and_verify_plan(args.run_root, verify_full_model=True)
        evidence = _strict_json(args.evidence)
        result = adjudicate_checkpoint_evidence(
            evidence,
            plan=plan,
            stage_index=args.stage_index,
        )
    except (
        CandidateCortexAdmissionError,
        CandidateCortexTrainingError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
