#!/usr/bin/env python3
"""Run the frozen semantic path-ensemble replication through its early gate."""

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
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_compositional_transducer import (
            compositional_semantic_program_transducer_from_dict,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_path_ensemble import (
            build_semantic_program_path_ensemble,
        )
        from core.learning.semantic_program_path_ensemble_replication import (
            evaluate_path_ensemble_replication,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        preregistration, _preregistration_file_sha256 = _load(
            args.preregistration,
            max_bytes=1024 * 1024,
        )
        training_manifest, _training_manifest_file_sha256 = _load(
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
        frozen = preregistration.get("frozen_paths", {})
        if (
            incumbent_file_sha256 != frozen.get("incumbent", {}).get("file_sha256")
            or challenger_file_sha256
            != frozen.get("challenger", {}).get("file_sha256")
        ):
            raise ValueError("frozen semantic path file identity differs")
        ensemble = build_semantic_program_path_ensemble(
            compositional_semantic_program_transducer_from_dict(incumbent_payload),
            compositional_semantic_program_transducer_from_dict(challenger_payload),
        )
        output = _validated_output_path(args.output, bundle=args.bundle)
        bundle = load_standard_semantic_feature_bundle(args.bundle)
        report = evaluate_path_ensemble_replication(
            bundle=bundle,
            training_manifest=training_manifest,
            preregistration=preregistration,
            ensemble=ensemble,
        )
        if not atomic_write_bytes_if_absent(output, _bytes(report), mode=0o400):
            raise FileExistsError("path ensemble replication output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_path_ensemble_replication_cli.v1",
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
                "schema": "aura.semantic_program_path_ensemble_replication_cli.v1",
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
