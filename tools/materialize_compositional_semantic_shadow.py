#!/usr/bin/env python3
"""Materialize the replicated 27B language-to-program shadow package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.compositional_semantic_qualification import (  # noqa: E402
    build_compositional_semantic_activation,
    canonical_document_bytes,
    compositional_semantic_activation_errors,
)
from core.learning.semantic_program_path_ensemble import (  # noqa: E402
    semantic_program_path_ensemble_from_dict,
)
from core.runtime.atomic_writer import atomic_write_bytes_if_absent  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402


def _read(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    raw = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=maximum_bytes)
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not an object")
    return value


def _install(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing package member differs: {path.name}")
        return
    if not atomic_write_bytes_if_absent(path, payload, mode=0o444):
        raise FileExistsError(f"package member raced: {path.name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--ordinary-primary", type=Path, required=True)
    parser.add_argument("--ordinary-sensitivity", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "artifacts/rlc/semantic_program_27b_frozen_path_v1"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve(strict=False)
    root = REPO_ROOT.resolve(strict=True)
    if not output.is_relative_to(root):
        raise ValueError("compositional semantic package must be inside the repository")
    output.mkdir(parents=True, exist_ok=True)

    ensemble = _read(args.ensemble, maximum_bytes=64 * 1024 * 1024)
    descriptor = _read(args.descriptor, maximum_bytes=2 * 1024 * 1024)
    preregistration = _read(args.preregistration, maximum_bytes=2 * 1024 * 1024)
    mechanism = _read(args.mechanism, maximum_bytes=4 * 1024 * 1024)
    ordinary_primary = _read(args.ordinary_primary, maximum_bytes=4 * 1024 * 1024)
    ordinary_sensitivity = _read(
        args.ordinary_sensitivity,
        maximum_bytes=4 * 1024 * 1024,
    )
    transducer = semantic_program_path_ensemble_from_dict(ensemble).challenger.to_dict()

    paths = {
        "transducer": output / "transducer.json",
        "mechanism": output / "mechanism_result.json",
        "ordinary_primary": output / "ordinary_primary.json",
        "ordinary_sensitivity": output / "ordinary_sensitivity.json",
        "activation": output / "activation.json",
    }
    _install(paths["transducer"], canonical_document_bytes(transducer))
    _install(paths["mechanism"], canonical_document_bytes(mechanism))
    _install(paths["ordinary_primary"], canonical_document_bytes(ordinary_primary))
    _install(paths["ordinary_sensitivity"], canonical_document_bytes(ordinary_sensitivity))

    _transducer, activation = build_compositional_semantic_activation(
        repo_root=root,
        preregistration=preregistration,
        mechanism=mechanism,
        ordinary_primary=ordinary_primary,
        ordinary_sensitivity=ordinary_sensitivity,
        descriptor=descriptor,
        ensemble=ensemble,
        transducer_path=paths["transducer"],
        preregistration_path=args.preregistration,
        mechanism_path=paths["mechanism"],
        ordinary_primary_path=paths["ordinary_primary"],
        ordinary_sensitivity_path=paths["ordinary_sensitivity"],
    )
    errors = compositional_semantic_activation_errors(
        activation,
        repo_root=root,
        selected_model_path=Path(str(activation["model"]["path"])),
    )
    if errors:
        raise RuntimeError(f"compositional semantic activation is invalid: {errors}")
    _install(paths["activation"], canonical_document_bytes(activation))
    print(
        json.dumps(
            {
                "activation": str(paths["activation"]),
                "activation_sha256": activation["activation_sha256"],
                "package_id": activation["package_id"],
                "serving_authority": activation["serving_authority"],
                "status": "materialized",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
