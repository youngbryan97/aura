#!/usr/bin/env python3
"""Independently verify one frozen fresh-cohort semantic replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, *, max_bytes: int) -> Any:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(
        path.expanduser().resolve(strict=True),
        max_bytes=max_bytes,
    )
    return json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--replication-bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replication-report", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_replication_verification import (
            SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES,
            verify_frozen_semantic_replication,
        )
        from core.runtime.file_write_gateway import get_file_write_gateway

        training = load_standard_semantic_feature_bundle(args.training_bundle)
        replication = load_standard_semantic_feature_bundle(args.replication_bundle)
        model_payload = _load_json(args.model, max_bytes=16 * 1024 * 1024)
        report = _load_json(args.replication_report, max_bytes=32 * 1024 * 1024)
        source_sha256s = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES
        }
        verification = verify_frozen_semantic_replication(
            training_bundle=training,
            replication_bundle=replication,
            trained_model_payload=model_payload,
            stored_report=report,
            source_sha256s=source_sha256s,
        )
        with local_internal_governed_scope(
            "semantic_program_replication.verification",
            domain="file_write",
        ):
            get_file_write_gateway().write_bytes(
                args.verification_output,
                _canonical_bytes(verification),
                source="semantic_program_replication.verification",
            )
    except Exception as exc:  # noqa: BLE001 - verifier reports terminal evidence
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_replication_verifier_cli.v1",
                    "verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "schema": "aura.semantic_program_replication_verifier_cli.v1",
                "verified": True,
                "verification_sha256": verification["verification_sha256"],
                "held_out_treatment_answer_exact": verification[
                    "held_out_treatment_answer_exact"
                ],
                "held_out_total": verification["held_out_total"],
                "verification_output": str(
                    args.verification_output.expanduser().resolve()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
