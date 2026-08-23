#!/usr/bin/env python3
"""Append a source-bound correction for one preserved checkpoint measurement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_reconciliation import (  # noqa: E402
    CandidateCortexReconciliationError,
    reconcile_preserved_measurement,
)
from core.learning.candidate_cortex_training import (  # noqa: E402
    JOURNAL_FILE,
    STAGE_RECONCILIATION_SCHEMA,
    CandidateCortexTrainingError,
    append_authenticated_event,
    canonical_json_bytes,
    effective_stage_evidence,
    load_and_verify_plan,
    read_authenticated_journal,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from tools.measure_candidate_cortex_checkpoint import _strict_json  # noqa: E402


def _key(path: Path) -> bytes:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CandidateCortexReconciliationError("reconciliation_key_invalid")
    value = resolved.read_bytes()
    if len(value) < 32:
        raise CandidateCortexReconciliationError("reconciliation_key_invalid")
    return value


def _event(bundle: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "schema",
        "plan_sha256",
        "stage_index",
        "prior_admission_sha256",
        "prior_evidence_sha256",
        "detail_sha256",
        "evaluator_source_sha256",
        "reconciled_evidence_sha256",
        "admission",
        "reconciliation_sha256",
    }
    event = {key: bundle[key] for key in keys}
    if event["schema"] != STAGE_RECONCILIATION_SCHEMA:
        raise CandidateCortexReconciliationError("reconciliation_schema_invalid")
    return event


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with local_internal_governed_scope(
        "candidate_cortex_reconciliation.write", domain="file_write"
    ):
        created = get_file_write_gateway().write_bytes_if_absent(
            path,
            payload,
            mode=0o600,
            source="candidate_cortex_reconciliation.write",
        )
    if not created and path.read_bytes() != payload:
        raise CandidateCortexReconciliationError("reconciliation_output_conflict")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--journal-key", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = load_and_verify_plan(args.run_root, verify_full_model=True)
        key = _key(args.journal_key)
        journal = Path(str(plan["paths"]["journal"]))
        events = read_authenticated_journal(journal, key=key)
        _observations, admissions = effective_stage_evidence(events)
        if not 0 <= args.stage_index < len(admissions):
            raise CandidateCortexReconciliationError("reconciliation_stage_missing")
        bundle = reconcile_preserved_measurement(
            plan=plan,
            stage_index=args.stage_index,
            detail=_strict_json(args.detail),
            original_evidence=_strict_json(args.evidence),
            prior_admission=admissions[args.stage_index],
        )
        event = _event(bundle)
        output = {
            "event": event,
            "corrected_evidence": bundle["corrected_evidence"],
            "baseline_behavior": bundle["baseline_behavior"],
            "candidate_behavior": bundle["candidate_behavior"],
        }
        _write_once(args.output.expanduser().resolve(strict=False), output)
        existing = [
            row.get("payload")
            for row in events
            if row.get("event_type") == "stage_reconciled"
            and isinstance(row.get("payload"), dict)
            and row["payload"].get("stage_index") == args.stage_index
        ]
        if event not in existing:
            append_authenticated_event(
                journal,
                key=key,
                event_type="stage_reconciled",
                payload=event,
            )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (
        CandidateCortexReconciliationError,
        CandidateCortexTrainingError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
