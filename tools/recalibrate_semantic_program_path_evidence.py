#!/usr/bin/env python3
"""Recalibrate semantic-path arbitration from signed, reusable evidence rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path, *, max_bytes: int) -> tuple[Any, str]:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return (
        json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object),
        hashlib.sha256(payload).hexdigest(),
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _bytes(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _verified_envelope(value: Any, *, hash_field: str, schema: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise ValueError(f"{schema} evidence envelope is invalid")
    body = {key: item for key, item in value.items() if key != hash_field}
    if value.get(hash_field) != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError(f"{schema} evidence hash is invalid")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _rows(value: Any, *, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{field} must be a non-empty sequence")
    return tuple(_mapping(row, field=f"{field} row") for row in value)


def _observation(
    row: Mapping[str, Any],
    *,
    source_namespace: str,
    calibration_split: str,
):
    from core.learning.semantic_program_path_calibration import (
        VerifiedSemanticPathObservation,
    )

    source_sha = row.get("source_text_sha256")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("semantic evidence source identity is invalid")
    return VerifiedSemanticPathObservation.from_mappings(
        incumbent=_mapping(
            row.get("incumbent_selection_values"),
            field="incumbent_selection_values",
        ),
        challenger=_mapping(
            row.get("challenger_selection_values"),
            field="challenger_selection_values",
        ),
        incumbent_correct=row.get("incumbent_correct"),
        challenger_correct=row.get("challenger_correct"),
        source_ref=f"{source_namespace}:{source_sha}",
        calibration_split=calibration_split,
        construction_id=row.get("construction_id"),
        topology_id=row.get("topology_id"),
    )


def _validate_source_calibration(
    report_payload: Any,
    ensemble: Any,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    report = _verified_envelope(
        report_payload,
        hash_field="report_sha256",
        schema="aura.semantic_program_path_calibration.v1",
    )
    rows = _rows(report.get("evidence_rows"), field="source calibration evidence_rows")
    if (
        report.get("admitted") is not True
        or report.get("expected_answers_available_to_paths") is not False
        or report.get("expected_answers_available_to_runtime") is not False
        or report.get("text_available_to_selector") is not False
        or ensemble.composition_receipt.get("calibration_report_sha256")
        != report["report_sha256"]
        or ensemble.model_basis_sha256 != report.get("model_basis_sha256")
        or ensemble.incumbent.receipt_sha256
        != report.get("incumbent_receipt_sha256")
        or ensemble.challenger.receipt_sha256
        != report.get("challenger_receipt_sha256")
    ):
        raise ValueError("source calibration binding is invalid")
    return report, rows


def _validate_development_evidence(
    result_payload: Any,
    ensemble: Any,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    result = _verified_envelope(
        result_payload,
        hash_field="result_sha256",
        schema="aura.semantic_program_path_selector_development_result.v1",
    )
    rows = _rows(result.get("rows"), field="development evidence rows")
    if (
        result.get("target_status") != "EXPOSED_DEVELOPMENT_TARGET"
        or result.get("expected_answers_available_to_paths_or_selector") is not False
        or result.get("expected_answers_available_to_evaluator") is not True
        or result.get("serving_authority") is not False
        or result.get("ensemble_receipt_sha256") != ensemble.receipt_sha256
    ):
        raise ValueError("development evidence binding is invalid")
    return result, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--source-ensemble", type=Path, required=True)
    parser.add_argument("--development-evaluation", type=Path, required=True)
    parser.add_argument("--development-ensemble", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_path_calibration import (
            calibrate_semantic_program_path_evidence,
        )
        from core.learning.semantic_program_path_ensemble import (
            build_calibrated_semantic_program_path_ensemble,
            semantic_program_path_ensemble_from_dict,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        source_report_payload, source_report_file_sha256 = _load(
            args.source_calibration,
            max_bytes=16 * 1024 * 1024,
        )
        source_ensemble_payload, source_ensemble_file_sha256 = _load(
            args.source_ensemble,
            max_bytes=32 * 1024 * 1024,
        )
        development_payload, development_file_sha256 = _load(
            args.development_evaluation,
            max_bytes=16 * 1024 * 1024,
        )
        development_ensemble_payload, development_ensemble_file_sha256 = _load(
            args.development_ensemble,
            max_bytes=32 * 1024 * 1024,
        )
        source_ensemble = semantic_program_path_ensemble_from_dict(
            source_ensemble_payload
        )
        development_ensemble = semantic_program_path_ensemble_from_dict(
            development_ensemble_payload
        )
        source_report, source_rows = _validate_source_calibration(
            source_report_payload,
            source_ensemble,
        )
        development_result, development_rows = _validate_development_evidence(
            development_payload,
            development_ensemble,
        )
        if (
            source_ensemble.model_basis_sha256
            != development_ensemble.model_basis_sha256
            or source_ensemble.incumbent.receipt_sha256
            != development_ensemble.incumbent.receipt_sha256
            or source_ensemble.challenger.receipt_sha256
            != development_ensemble.challenger.receipt_sha256
        ):
            raise ValueError("evidence sources refer to different semantic paths")

        source_identities = {row.get("source_text_sha256") for row in source_rows}
        development_identities = {
            row.get("source_text_sha256") for row in development_rows
        }
        if (
            len(source_identities) != len(source_rows)
            or len(development_identities) != len(development_rows)
            or source_identities & development_identities
        ):
            raise ValueError("semantic evidence sources overlap or contain duplicates")

        observations = [
            _observation(
                row,
                source_namespace=f"source:{source_report['report_sha256']}",
                calibration_split=str(row.get("calibration_split")),
            )
            for row in source_rows
        ]
        development_validation = tuple(
            row for row in development_rows if row.get("split") == "validation"
        )
        development_test = tuple(
            sorted(
                (row for row in development_rows if row.get("split") == "test"),
                key=lambda row: str(row.get("source_text_sha256")),
            )
        )
        if (
            len(development_validation) + len(development_test)
            != len(development_rows)
            or not development_validation
            or len(development_test) < 2
        ):
            raise ValueError("development evidence requires validation and test rows")
        observations.extend(
            _observation(
                row,
                source_namespace=f"development:{development_result['result_sha256']}",
                calibration_split="validation",
            )
            for row in development_validation
        )
        observations.extend(
            _observation(
                row,
                source_namespace=f"development:{development_result['result_sha256']}",
                calibration_split="tuning" if index % 2 == 0 else "admission",
            )
            for index, row in enumerate(development_test)
        )
        selector, report = calibrate_semantic_program_path_evidence(
            model_basis_sha256=source_ensemble.model_basis_sha256,
            incumbent_receipt_sha256=source_ensemble.incumbent.receipt_sha256,
            challenger_receipt_sha256=source_ensemble.challenger.receipt_sha256,
            observations=observations,
            evidence_source_receipts=(
                source_report["report_sha256"],
                development_result["result_sha256"],
            ),
        )

        output = args.output_directory.expanduser().resolve(strict=False)
        protected_directories = tuple(
            path.expanduser().resolve(strict=True).parent
            for path in (
                args.source_calibration,
                args.source_ensemble,
                args.development_evaluation,
                args.development_ensemble,
            )
        )
        if any(
            output == directory or output.is_relative_to(directory)
            for directory in protected_directories
        ):
            raise ValueError("recalibration output cannot modify input evidence")
        output.mkdir(parents=True, exist_ok=True)
        bundle_body = {
            "schema": "aura.semantic_program_path_evidence_bundle.v1",
            "complete": True,
            "admitted": selector is not None,
            "source_calibration_file_sha256": source_report_file_sha256,
            "source_ensemble_file_sha256": source_ensemble_file_sha256,
            "development_evaluation_file_sha256": development_file_sha256,
            "development_ensemble_file_sha256": development_ensemble_file_sha256,
            "calibration_report_sha256": report["report_sha256"],
        }
        writes = {
            output / "calibration_report.json": report,
            output / "bundle.json": {
                **bundle_body,
                "bundle_sha256": hashlib.sha256(_canonical(bundle_body)).hexdigest(),
            },
        }
        if selector is not None:
            ensemble = build_calibrated_semantic_program_path_ensemble(
                source_ensemble.incumbent,
                source_ensemble.challenger,
                selector=selector,
                calibration_report=report,
            )
            writes.update(
                {
                    output / "selector.json": selector.to_dict(),
                    output / "ensemble.json": ensemble.to_dict(),
                }
            )
        for path, payload in writes.items():
            if not atomic_write_bytes_if_absent(path, _bytes(payload), mode=0o400):
                raise FileExistsError(f"recalibration output already exists: {path}")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_path_evidence_cli.v1",
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
                "schema": "aura.semantic_program_path_evidence_cli.v1",
                "complete": True,
                "admitted": selector is not None,
                "reason": report["reason"],
                "report_sha256": report["report_sha256"],
                "output_directory": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if selector is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
