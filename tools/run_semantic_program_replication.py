#!/usr/bin/env python3
"""Evaluate a frozen semantic transducer on one fresh resident cohort."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_replication import (
            FrozenTrainingCohort,
            evaluate_frozen_semantic_replication,
            load_reconstructed_semantic_cohort,
            semantic_replication_source_sha256s,
        )
        from core.runtime.file_write_gateway import get_file_write_gateway

        training = load_standard_semantic_feature_bundle(args.training_bundle)
        replication = load_reconstructed_semantic_cohort(args.replication_bundle)
        model_payload = _load_json(args.model, max_bytes=16 * 1024 * 1024)
        training_cohort = FrozenTrainingCohort(
            feature_manifest_sha256=training.manifest["manifest_sha256"],
            example_ids=tuple(
                str(item.metadata["example_id"]) for item in training.examples
            ),
        )
        report = evaluate_frozen_semantic_replication(
            replication,
            trained_model_payload=model_payload,
            training_cohort=training_cohort,
            training_manifest=training.manifest,
            source_sha256s=semantic_replication_source_sha256s(_REPO_ROOT),
        )
        with local_internal_governed_scope(
            "semantic_program_replication.report",
            domain="file_write",
        ):
            get_file_write_gateway().write_bytes(
                args.report_output,
                _canonical_bytes(report),
                source="semantic_program_replication.report",
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports terminal evidence
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_replication_cli.v1",
                    "complete": False,
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
                "schema": "aura.semantic_program_replication_cli.v1",
                "complete": True,
                "report_sha256": report["report_sha256"],
                "replication_example_count": report["replication_example_count"],
                "treatment_answer_exact": sum(
                    report["arms"][f"treatment:{split}"]["answer_exact"]
                    for split in ("train", "validation", "test")
                ),
                "report_output": str(args.report_output.expanduser().resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
