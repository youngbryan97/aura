#!/usr/bin/env python3
"""Evaluate a frozen calibrated semantic selector on a development target."""

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
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path, *, max_bytes: int) -> Any:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-status",
        choices=("EXPOSED_DEVELOPMENT_TARGET", "FRESH_RESERVED_TARGET"),
        required=True,
    )
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_basis import (
            bind_examples_to_compatible_training_session,
            establish_semantic_representation_compatibility,
        )
        from core.learning.semantic_program_campaign import (
            training_examples_from_feature_bundle,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_path_ensemble import (
            semantic_program_path_ensemble_from_dict,
        )
        from core.learning.semantic_program_path_selector_evaluation import (
            evaluate_calibrated_path_selector_development,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        training_manifest = _load(args.training_manifest, max_bytes=16 * 1024 * 1024)
        ensemble = semantic_program_path_ensemble_from_dict(
            _load(args.ensemble, max_bytes=32 * 1024 * 1024)
        )
        bundle = load_standard_semantic_feature_bundle(args.bundle)
        compatibility = establish_semantic_representation_compatibility(
            model=ensemble.incumbent,
            training_manifest=training_manifest,
            replication_manifest=bundle.manifest,
        )
        examples = bind_examples_to_compatible_training_session(
            training_examples_from_feature_bundle(
                bundle,
                required_splits=frozenset({"validation", "test"}),
            ),
            compatibility=compatibility,
        )
        report = evaluate_calibrated_path_selector_development(
            ensemble=ensemble,
            examples=examples,
            target_status=args.target_status,
        )
        output = args.output.expanduser().resolve(strict=False)
        bundle_path = args.bundle.expanduser().resolve(strict=True)
        if output == bundle_path or output.is_relative_to(bundle_path):
            raise ValueError("evaluation output cannot modify the immutable feature bundle")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not atomic_write_bytes_if_absent(output, _bytes(report), mode=0o400):
            raise FileExistsError("selector evaluation output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_path_selector_evaluation_cli.v1",
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
                "schema": "aura.semantic_program_path_selector_evaluation_cli.v1",
                "complete": True,
                "verdict": report["verdict"],
                "selected_correct": report["selected_correct"],
                "incumbent_correct": report["incumbent_correct"],
                "regressions": report["regressions_from_incumbent"],
                "output": str(output),
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if report["mechanism_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
