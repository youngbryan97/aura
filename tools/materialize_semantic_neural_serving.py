#!/usr/bin/env python3
"""Materialize an independently verified resident semantic runtime activation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.semantic_neural_serving import (  # noqa: E402
    ACTIVE_ACTIVATION_PATH,
    RESIDENT_ADJUDICATION_PATH,
    RESIDENT_RESULT_PATH,
    RESIDENT_VERIFICATION_PATH,
    build_semantic_neural_activation,
    semantic_neural_activation_errors,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resident-manifest",
        type=Path,
        # Anchored to the checkout, not to one machine's home directory:
        # the absolute literal made this tool runnable on exactly one host.
        default=REPO_ROOT / "training" / "fused-model" / "active.json",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--result", type=Path, default=RESIDENT_RESULT_PATH)
    parser.add_argument("--verification", type=Path, default=RESIDENT_VERIFICATION_PATH)
    parser.add_argument("--adjudication", type=Path, default=RESIDENT_ADJUDICATION_PATH)
    parser.add_argument("--out", type=Path, default=ACTIVE_ACTIVATION_PATH)
    parser.add_argument("--runtime-verification", type=Path)
    args = parser.parse_args()

    manifest = args.resident_manifest.expanduser().resolve(strict=True)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise RuntimeError("resident manifest is not an object")
    model = (
        args.model.expanduser().resolve(strict=True)
        if args.model is not None
        else Path(str(manifest_payload.get("active_model_path") or "")).resolve(strict=True)
    )
    activation = build_semantic_neural_activation(
        result_path=args.result,
        verification_path=args.verification,
        adjudication_path=args.adjudication,
        resident_manifest_path=manifest,
        model_path=model,
        runtime_verification_path=args.runtime_verification,
    )
    errors = semantic_neural_activation_errors(
        activation,
        model_path=model,
        require_runtime_qualification=args.runtime_verification is not None,
    )
    if errors:
        raise RuntimeError(f"materialized semantic activation is invalid: {errors}")
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        destination,
        json.dumps(activation, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(activation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
