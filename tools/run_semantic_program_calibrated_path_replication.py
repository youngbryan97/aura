#!/usr/bin/env python3
"""Run the frozen calibrated-path replication through its mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
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


def _load(path: Path, *, max_bytes: int) -> tuple[Any, str]:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return (
        json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object),
        hashlib.sha256(payload).hexdigest(),
    )


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _validate_calibration_report(
    payload: Any,
    *,
    expected_report_sha256: Any,
) -> None:
    report = _mapping(payload, field="calibration report")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    digest = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if (
        report.get("report_sha256") != digest
        or report.get("report_sha256") != expected_report_sha256
        or report.get("admitted") is not True
    ):
        raise ValueError("calibration report identity or admission differs")


def _validated_output_path(output: Path, *, bundle: Path) -> Path:
    resolved_bundle = bundle.expanduser().resolve(strict=True)
    resolved_output = output.expanduser().resolve(strict=False)
    if resolved_output == resolved_bundle or resolved_output.is_relative_to(resolved_bundle):
        raise ValueError("replication output cannot modify the immutable feature bundle")
    return resolved_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--mixed-ensemble", type=Path, required=True)
    parser.add_argument("--mixed-calibration-report", type=Path, required=True)
    parser.add_argument("--source-only-ensemble", type=Path, required=True)
    parser.add_argument("--source-only-calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_calibrated_path_replication import (
            evaluate_calibrated_path_replication,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_path_ensemble import (
            semantic_program_path_ensemble_from_dict,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        preregistration, _preregistration_file_sha256 = _load(
            args.preregistration,
            max_bytes=1024 * 1024,
        )
        preregistration = _mapping(preregistration, field="preregistration")
        training_manifest, _training_manifest_file_sha256 = _load(
            args.training_manifest,
            max_bytes=16 * 1024 * 1024,
        )
        mixed_payload, mixed_file_sha256 = _load(
            args.mixed_ensemble,
            max_bytes=32 * 1024 * 1024,
        )
        mixed_report, mixed_report_file_sha256 = _load(
            args.mixed_calibration_report,
            max_bytes=16 * 1024 * 1024,
        )
        source_payload, source_file_sha256 = _load(
            args.source_only_ensemble,
            max_bytes=32 * 1024 * 1024,
        )
        source_report, source_report_file_sha256 = _load(
            args.source_only_calibration_report,
            max_bytes=16 * 1024 * 1024,
        )
        frozen = _mapping(preregistration.get("frozen_ensemble"), field="frozen ensemble")
        source_frozen = _mapping(
            preregistration.get("frozen_source_only_control"),
            field="frozen source-only control",
        )
        if (
            mixed_file_sha256 != frozen.get("file_sha256")
            or mixed_report_file_sha256 != frozen.get("calibration_report_file_sha256")
            or source_file_sha256 != source_frozen.get("file_sha256")
            or source_report_file_sha256 != source_frozen.get("calibration_report_file_sha256")
        ):
            raise ValueError("frozen calibrated artifact file identity differs")
        _validate_calibration_report(
            mixed_report,
            expected_report_sha256=frozen.get("calibration_report_sha256"),
        )
        _validate_calibration_report(
            source_report,
            expected_report_sha256=source_frozen.get("calibration_report_sha256"),
        )
        output = _validated_output_path(args.output, bundle=args.bundle)
        report = evaluate_calibrated_path_replication(
            bundle=load_standard_semantic_feature_bundle(args.bundle),
            training_manifest=_mapping(training_manifest, field="training manifest"),
            preregistration=preregistration,
            ensemble=semantic_program_path_ensemble_from_dict(mixed_payload),
            source_only_control=semantic_program_path_ensemble_from_dict(source_payload),
        )
        if not atomic_write_bytes_if_absent(output, _bytes(report), mode=0o400):
            raise FileExistsError("calibrated path replication output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_calibrated_path_replication_cli.v1",
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
                "schema": "aura.semantic_program_calibrated_path_replication_cli.v1",
                "complete": True,
                "verdict": report["verdict"],
                "improvements": report["improvements_over_incumbent"],
                "regressions": report["regressions_from_incumbent"],
                "result_sha256": report["result_sha256"],
                "output": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if report["verdict"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
