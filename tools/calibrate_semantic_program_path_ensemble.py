#!/usr/bin/env python3
"""Fit and freeze source-calibrated arbitration for two semantic paths."""

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


def _validated_output_directory(output: Path, *, bundle: Path) -> Path:
    resolved_bundle = bundle.expanduser().resolve(strict=True)
    resolved_output = output.expanduser().resolve(strict=False)
    if resolved_output == resolved_bundle or resolved_output.is_relative_to(resolved_bundle):
        raise ValueError("calibration output cannot modify the immutable feature bundle")
    return resolved_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_basis import (
            bind_examples_to_compatible_training_session,
            establish_semantic_representation_compatibility,
        )
        from core.learning.semantic_program_campaign import (
            training_examples_from_feature_bundle,
        )
        from core.learning.semantic_program_compositional_transducer import (
            compositional_semantic_program_transducer_from_dict,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_path_calibration import (
            calibrate_semantic_program_paths,
        )
        from core.learning.semantic_program_path_ensemble import (
            build_calibrated_semantic_program_path_ensemble,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        training_manifest, training_manifest_file_sha256 = _load(
            args.training_manifest,
            max_bytes=16 * 1024 * 1024,
        )
        incumbent_payload, incumbent_file_sha256 = _load(
            args.incumbent,
            max_bytes=32 * 1024 * 1024,
        )
        challenger_payload, challenger_file_sha256 = _load(
            args.challenger,
            max_bytes=32 * 1024 * 1024,
        )
        incumbent = compositional_semantic_program_transducer_from_dict(
            incumbent_payload
        )
        challenger = compositional_semantic_program_transducer_from_dict(
            challenger_payload
        )
        bundle = load_standard_semantic_feature_bundle(args.bundle)
        raw_examples = training_examples_from_feature_bundle(
            bundle,
            required_splits=frozenset({"validation", "test"}),
        )
        incumbent_compatibility = establish_semantic_representation_compatibility(
            model=incumbent,
            training_manifest=training_manifest,
            replication_manifest=bundle.manifest,
        )
        challenger_compatibility = establish_semantic_representation_compatibility(
            model=challenger,
            training_manifest=training_manifest,
            replication_manifest=bundle.manifest,
        )
        examples = bind_examples_to_compatible_training_session(
            raw_examples,
            compatibility=incumbent_compatibility,
        )
        selector, report = calibrate_semantic_program_paths(
            incumbent=incumbent,
            challenger=challenger,
            source_examples=examples,
        )
        output = _validated_output_directory(
            args.output_directory,
            bundle=args.bundle,
        )
        output.mkdir(parents=True, exist_ok=True)
        envelope_body = {
            "schema": "aura.semantic_program_path_calibration_bundle.v1",
            "complete": True,
            "admitted": selector is not None,
            "bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
            "training_manifest_file_sha256": training_manifest_file_sha256,
            "incumbent_file_sha256": incumbent_file_sha256,
            "challenger_file_sha256": challenger_file_sha256,
            "incumbent_representation_compatibility": incumbent_compatibility,
            "challenger_representation_compatibility": challenger_compatibility,
            "calibration_report_sha256": report.get("report_sha256"),
        }
        envelope = {
            **envelope_body,
            "bundle_sha256": hashlib.sha256(_bytes(envelope_body)).hexdigest(),
        }
        writes = {
            output / "calibration_report.json": report,
            output / "bundle.json": envelope,
        }
        if selector is not None:
            ensemble = build_calibrated_semantic_program_path_ensemble(
                incumbent,
                challenger,
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
                raise FileExistsError(f"calibration output already exists: {path}")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_path_calibration_cli.v1",
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
                "schema": "aura.semantic_program_path_calibration_cli.v1",
                "complete": True,
                "admitted": selector is not None,
                "reason": report["reason"],
                "report_sha256": report.get("report_sha256"),
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
